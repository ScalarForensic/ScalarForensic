"""Image embedding backends: DINOv2 (HuggingFace), SSCD (TorchScript), and remote OpenAI-compat."""

import base64
import hashlib
import importlib.metadata
import io
import json
import os
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from scalar_forensic._model_hash import hash_dino_snapshot as _hash_dino_snapshot
from scalar_forensic.media_access import (  # noqa: F401 — compatibility exports
    _open_rgb,
    extract_exif,
    extract_exif_detailed,
    preprocess_batch,
)
from scalar_forensic.media_decode import (  # noqa: F401 — compatibility exports
    _EXIF_GPS_IFD_TAG,
    _EXIF_ORIENTATION_FORMATS,
    _EXIF_ORIENTATION_TAG,
    _MAX_IMAGE_PIXELS_ENV,
    ExifInfo,
    _cap_short_side,
    _configure_max_image_pixels_from_env,
)

_SSCD_INPUT_SIZE = 288
# SSCD requires short-side ≥ 331 px before the 288×288 center-crop.
_SSCD_SCALE = 331

# Reapply overrides when this compatibility module is explicitly reloaded.
_configure_max_image_pixels_from_env()

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_library_versions() -> dict[str, str]:
    libs = ["torch", "transformers", "torchvision", "qdrant-client", "Pillow", "av"]
    versions: dict[str, str] = {"python": sys.version}
    for lib in libs:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "unknown"
    return versions


def hash_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest of already-loaded bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_bytes_md5(data: bytes) -> str:
    """Return MD5 hex digest of already-loaded bytes."""
    return hashlib.md5(data).hexdigest()  # noqa: S324


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return SHA-256 hex digest of a file, reading it in chunks.

    Uses O(chunk_size) memory regardless of file size — suitable for large
    video files where ``read_bytes()`` would cause a memory spike.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def hash_file_both(path: Path, chunk_size: int = 1 << 20) -> tuple[str, str]:
    """Return ``(sha256, md5)`` hex digests of a file in a single read pass."""
    h_sha = hashlib.sha256()
    h_md5 = hashlib.md5()  # noqa: S324
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h_sha.update(buf)
            h_md5.update(buf)
    return h_sha.hexdigest(), h_md5.hexdigest()


def effective_preprocessing_cap(normalize_size: int) -> int:
    """Short-side pixel cap satisfying both SSCD (≥_SSCD_SCALE) and DINOv2 (≥normalize_size)."""
    return max(_SSCD_SCALE, normalize_size)


def write_thumbnail(img: Image.Image, dest: Path, size: int) -> None:
    """Save a thumbnail JPEG at *dest* (size×size, aspect-ratio preserved), atomically.

    Uses a temp-file + os.replace strategy so concurrent readers never see a
    partially-written file.
    """
    import contextlib
    import tempfile

    thumb = img.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    if thumb.mode not in ("RGB", "L"):
        thumb = thumb.convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=dest.parent, suffix=dest.suffix or ".jpg"
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        thumb.save(tmp_path, format="JPEG", quality=85, optimize=True)
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
        raise


class HashCache:
    """Persistent SHA-256 cache keyed by (normalized absolute real-path, mtime_ns, size).

    On construction the entire table is loaded into an in-memory dict so every
    lookup is a plain dict.get — no SQLite round-trip per file.  New entries
    are collected in a pending list and flushed to disk in one bulk INSERT
    transaction by flush() / close(), so there is no per-file I/O overhead
    during the parallel hash pass.

    Thread-safety: reads use an unsynchronised dict.get (GIL-safe in CPython).
    Writes hold a lock.  Two threads racing on the same cold key will both
    compute the hash and the second write silently overwrites with the same
    value — harmless.

    Disable the cache at runtime by setting SFN_HASH_CACHE_PATH= (empty).
    """

    @staticmethod
    def _cache_key(path: Path) -> str:
        """Return a stable, normalized absolute path string for use as a cache key."""
        return str(path.resolve())

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS file_hashes (
                path     TEXT    NOT NULL PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size     INTEGER NOT NULL,
                sha256   TEXT    NOT NULL,
                md5      TEXT    NOT NULL DEFAULT ''
            )"""
        )
        self._conn.commit()
        rows = self._conn.execute("SELECT path, mtime_ns, size, sha256, md5 FROM file_hashes")
        self._mem: dict[str, tuple[int, int, str, str]] = {
            r[0]: (r[1], r[2], r[3], r[4]) for r in rows
        }
        self._pending: list[tuple[str, int, int, str, str]] = []
        self._lock = threading.Lock()

    def get_or_hash(self, path: Path) -> tuple[str, bool]:
        """Return ``(sha256, was_cached)``.  Thread-safe.

        MD5 is computed alongside SHA-256 at no extra I/O cost and stored in
        the cache so that ``get_or_hash_both`` can serve it without a disk read.
        """
        key = self._cache_key(path)
        st = path.stat()
        mtime_ns: int = st.st_mtime_ns
        size: int = st.st_size
        entry = self._mem.get(key)
        if entry is not None and entry[0] == mtime_ns and entry[1] == size:
            return entry[2], True
        sha, md5 = hash_file_both(path)
        with self._lock:
            self._mem[key] = (mtime_ns, size, sha, md5)
            self._pending.append((key, mtime_ns, size, sha, md5))
        return sha, False

    def get_or_hash_both(self, path: Path) -> tuple[str, str, bool]:
        """Return ``(sha256, md5, was_cached)``.  Thread-safe."""
        key = self._cache_key(path)
        st = path.stat()
        mtime_ns: int = st.st_mtime_ns
        size: int = st.st_size
        entry = self._mem.get(key)
        if entry is not None and entry[0] == mtime_ns and entry[1] == size and entry[3]:
            return entry[2], entry[3], True
        sha, md5 = hash_file_both(path)
        with self._lock:
            self._mem[key] = (mtime_ns, size, sha, md5)
            self._pending.append((key, mtime_ns, size, sha, md5))
        return sha, md5, False

    def peek(self, path: Path) -> str | None:
        """Return the cached SHA-256 for *path*, or None — never hashes.

        For callers that want the digest only if it is free: a request handler
        that must start streaming bytes now cannot afford the whole-file read a
        miss would cost.  None means "not cached", never "does not match".
        """
        entry = self._mem.get(self._cache_key(path))
        if entry is None:
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        if entry[0] != st.st_mtime_ns or entry[1] != st.st_size:
            return None
        return entry[2]

    def flush(self) -> int:
        """Persist pending entries to SQLite.  Returns the number of rows written."""
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return 0
        self._conn.executemany(
            "INSERT OR REPLACE INTO file_hashes (path, mtime_ns, size, sha256, md5)"
            " VALUES (?,?,?,?,?)",
            pending,
        )
        self._conn.commit()
        return len(pending)

    def close(self) -> None:
        self.flush()
        self._conn.close()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess_pil_batch(images: list[Image.Image], cap: int) -> list[Image.Image]:
    """Apply ``_cap_short_side`` to already-decoded PIL Images.

    Video frames arrive from PyAV as PIL Images, so there is no I/O or
    format-decoding step.  This function provides the same size-capping that
    ``preprocess_batch`` applies internally, without the bytes-open overhead.
    ICC and EXIF corrections are not applied here — video frames come from
    PyAV already in sRGB-equivalent colour space with correct geometry.

    Unlike ``preprocess_batch``, this function never returns exceptions —
    callers are responsible for ensuring all inputs are valid RGB Images.

    Pass ``max(_SSCD_SCALE, settings.normalize_size)`` as *cap*.
    """
    return [_cap_short_side(img, cap) for img in images]


class DINOv2Embedder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        normalize_size: int = 224,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.normalize_size = normalize_size
        self.device = _resolve_device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            size={"shortest_edge": normalize_size},
            crop_size={"height": normalize_size, "width": normalize_size},
        )
        actual_size = getattr(self.processor, "size", None)
        actual_crop = getattr(self.processor, "crop_size", None)
        if actual_size != {"shortest_edge": normalize_size} or actual_crop != {
            "height": normalize_size,
            "width": normalize_size,
        }:
            raise ValueError(
                f"DINOv2 processor geometry does not match normalize_size={normalize_size}: "
                f"size={actual_size}, crop_size={actual_crop}. Refusing unverifiable embeddings."
            )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name, dtype=dtype, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()
        # torch.compile has known issues on ROCm for certain GPU architectures (e.g. gfx1201):
        # 'KernelMetadata' object has no attribute 'cluster_dims' (TorchDynamo bug in ROCm builds).
        # Detect ROCm via torch.version.hip and skip compilation to avoid silent batch failures.
        _is_rocm = getattr(torch.version, "hip", None) is not None
        self.compiled = self.device == "cuda" and not _is_rocm
        if self.compiled:
            self.model = torch.compile(self.model)
        self._model_hash: str | None = None

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size  # type: ignore[no-any-return]

    @property
    def inference_dtype(self) -> str:
        return "float16" if self.device == "cuda" else "float32"

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            local = Path(self.model_name)
            if local.is_dir():
                snapshot_path = local
            else:
                from huggingface_hub import snapshot_download

                snapshot_path = Path(snapshot_download(self.model_name, local_files_only=True))
            self._model_hash = _hash_dino_snapshot(snapshot_path)
        return self._model_hash

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"]
        expected_shape = (len(images), 3, self.normalize_size, self.normalize_size)
        if tuple(pixels.shape) != expected_shape:
            raise ValueError(
                f"DINOv2 effective tensor geometry {tuple(pixels.shape)} does not match "
                f"{expected_shape}; refusing embeddings with misleading preprocessing provenance."
            )
        inputs = inputs.to(self.device)
        if self.device == "cuda":
            inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].float()
        cls = F.normalize(cls, p=2, dim=1)
        return cls.cpu().tolist()


def _sscd_resize(img: Image.Image) -> Image.Image:
    """Proportionally resize *img* so the short side is exactly ``_SSCD_SCALE`` (331 px).

    Returns *img* unchanged when the short side is already ``_SSCD_SCALE`` —
    the common case after ``_cap_short_side(_SSCD_SCALE)``.  For images smaller
    than 331 px it upscales as SSCD requires.
    """
    w, h = img.size
    short = min(w, h)
    if short == _SSCD_SCALE:
        return img
    if w <= h:
        new_w, new_h = _SSCD_SCALE, round(h * _SSCD_SCALE / w)
    else:
        new_w, new_h = round(w * _SSCD_SCALE / h), _SSCD_SCALE
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _sscd_crops(img: Image.Image, n_crops: int) -> list[Image.Image]:
    """Return *n_crops* 288×288 crops from *img* (already at ``_SSCD_SCALE`` short side).

    ``n_crops=1`` — center crop only (matches previous single-crop behaviour).
    ``n_crops=5`` — center crop plus the four corner crops, giving full spatial
    coverage for images where the semantically relevant content is off-centre
    (e.g. surveillance stills, padded composites).
    """
    size = _SSCD_INPUT_SIZE
    w, h = img.size
    cx, cy = (w - size) // 2, (h - size) // 2
    crops = [img.crop((cx, cy, cx + size, cy + size))]  # center
    if n_crops == 5:
        crops += [
            img.crop((0, 0, size, size)),  # top-left
            img.crop((w - size, 0, w, size)),  # top-right
            img.crop((0, h - size, size, h)),  # bottom-left
            img.crop((w - size, h - size, w, h)),  # bottom-right
        ]
    return crops


class SSCDEmbedder:
    def __init__(self, model_name: str, device: str = "auto", n_crops: int = 1) -> None:
        if n_crops not in (1, 5):
            raise ValueError(
                f"SSCDEmbedder n_crops={n_crops!r} is not supported. "
                "Allowed values: 1 (center crop only) or 5 (center + 4 corners)."
            )
        self.model_name = model_name
        self.normalize_size = _SSCD_INPUT_SIZE
        self.n_crops = n_crops
        self.device = _resolve_device(device)
        if not Path(model_name).exists():
            raise FileNotFoundError(
                f"SSCD checkpoint not found: {model_name}\n"
                "Download it from:\n"
                "  https://github.com/facebookresearch/sscd-copy-detection/releases\n"
                "Then set SFN_MODEL_SSCD=/path/to/sscd_disc_mixup.torchscript.pt in .env"
            )
        self._model = torch.jit.load(model_name, map_location=self.device)
        self._model.eval()
        _is_rocm = getattr(torch.version, "hip", None) is not None
        self.compiled = self.device == "cuda" and not _is_rocm
        if self.compiled:
            # NOTE: .half() produces degenerate constant vectors with this TorchScript checkpoint.
            # Keep SSCD in fp32 even on CUDA; torch.compile still gives a meaningful speedup.
            self._model = torch.compile(self._model)
        self._to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
        self._model_hash: str | None = None

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def inference_dtype(self) -> str:
        return "float32"  # fp16 causes degenerate embeddings with this TorchScript checkpoint

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            h = hashlib.sha256()
            with open(self.model_name, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            self._model_hash = h.hexdigest()
        return self._model_hash

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed *images* using the SSCD model.

        Each image is first resized to ``_SSCD_SCALE`` (331 px) short side, then
        ``self.n_crops`` crops of ``_SSCD_INPUT_SIZE``×``_SSCD_INPUT_SIZE`` (288×288)
        are generated.  All crops are embedded in a single forward pass.

        When ``n_crops == 1`` the center crop result is returned directly.
        When ``n_crops == 5``, per-crop embeddings are L2-normalised, averaged
        across crops per source image, then L2-normalised again — producing a
        single unit-norm vector per image that covers the full spatial extent.
        """
        n_orig = len(images)
        resized = [_sscd_resize(img) for img in images]
        crop_lists = [_sscd_crops(img, self.n_crops) for img in resized]
        flat_crops = [c for crops in crop_lists for c in crops]

        batch = torch.stack([self._to_tensor(c) for c in flat_crops]).to(self.device)
        # autocast runs conv2d in FP16 (Tensor Cores / matrix units) while keeping
        # BatchNorm in FP32 — avoiding the degenerate-embedding issue that a full
        # .half() conversion causes for this checkpoint.  Enabled only on CUDA
        # (covers both NVIDIA and ROCm); CPU autocast uses BF16 which is a
        # separate concern and not enabled here.
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device == "cuda"):
            flat_embs = self._model(batch).float()  # (n_orig * n_crops, 512)

        if self.n_crops == 1:
            return flat_embs.cpu().tolist()

        # L2-normalise per crop → average over crops per source image → L2-normalise
        flat_embs = F.normalize(flat_embs, p=2, dim=1)
        embs = flat_embs.view(n_orig, self.n_crops, -1).mean(dim=1)
        embs = F.normalize(embs, p=2, dim=1)
        return embs.cpu().tolist()


class RemoteEmbedder:
    """Calls an OpenAI-compatible POST /v1/embeddings endpoint.

    Images are PNG-encoded and sent as base64 data-URIs (``data:image/png;base64,…``)
    in the ``input`` array.  PNG is lossless: it avoids DCT block artefacts on
    screenshots and digital evidence, and eliminates double-compression when the
    source is already JPEG.  Required configuration: endpoint URL
    (``SFN_EMBEDDING_ENDPOINT``), model name (``SFN_EMBEDDING_MODEL``), and embedding
    dimension (``SFN_EMBEDDING_DIM``, must match the dimension the remote model
    actually produces).  An optional Bearer API key may be provided via
    ``SFN_EMBEDDING_API_KEY``.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        embedding_dim: int,
        api_key: str | None = None,
        normalize_size: int = 224,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self._embedding_dim = embedding_dim
        self._api_key = api_key
        self.normalize_size = normalize_size
        self.device = "remote"
        self._model_hash: str | None = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def inference_dtype(self) -> str:
        return "float32"

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            h = hashlib.sha256()
            h.update(self.endpoint.encode())
            h.update(self.model_name.encode())
            h.update(str(self._embedding_dim).encode())
            # Prefix signals that this is a config-derived hash, not a weights
            # hash, so consumers (UI, audit logs) can surface the distinction.
            self._model_hash = "config:" + h.hexdigest()
        return self._model_hash

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs: list[str] = []
        for img in images:
            buf = io.BytesIO()
            # PNG is lossless: avoids DCT block artefacts on screenshots and
            # digital evidence, and eliminates double-compression if the source
            # was already JPEG.  The payload size increase is acceptable given
            # that this path is only used for remote embedding servers.
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            inputs.append(f"data:image/png;base64,{b64}")

        payload = json.dumps({"model": self.model_name, "input": inputs}).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            f"{self.endpoint}/v1/embeddings",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read(4096).decode(errors="replace")
            raise RuntimeError(f"Embedding endpoint returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding endpoint connection failed: {exc.reason}") from exc

        raw_data = data.get("data")
        if not isinstance(raw_data, list):
            excerpt = str(data)[:200]
            raise RuntimeError(f"Embedding response missing 'data' list: {excerpt}")
        items = sorted(raw_data, key=lambda x: x["index"])
        expected = len(images)
        if len(items) != expected:
            raise RuntimeError(
                f"Embedding response returned {len(items)} items for {expected} inputs"
            )
        indices = {item["index"] for item in items}
        if indices != set(range(expected)):
            raise RuntimeError(
                f"Embedding response indices {sorted(indices)} do not cover 0..{expected - 1}"
            )
        embeddings = [item["embedding"] for item in items]
        for i, emb in enumerate(embeddings):
            if len(emb) != self._embedding_dim:
                raise RuntimeError(
                    f"Embedding [{i}] has length {len(emb)}, expected {self._embedding_dim}"
                )
        return embeddings


AnyEmbedder = DINOv2Embedder | SSCDEmbedder | RemoteEmbedder


def load_embedder(
    model: str,
    use_sscd: bool,
    device: str = "auto",
    normalize_size: int = 224,
    *,
    remote_endpoint: str | None = None,
    remote_api_key: str | None = None,
    embedding_dim: int = 0,
    local_files_only: bool = False,
    n_crops: int = 1,
) -> AnyEmbedder:
    """Load the embedder for the selected backend.

    When *remote_endpoint* is set the remote OpenAI-compatible endpoint is used
    and *use_sscd* / *device* are ignored.  *embedding_dim* must be > 0 in that
    case (set ``SFN_EMBEDDING_DIM`` to the dimension the remote model produces).

    *local_files_only* is forwarded to :class:`DINOv2Embedder` and prevents the
    HuggingFace SDK from fetching model files or metadata from the Hub.

    *n_crops* is forwarded to :class:`SSCDEmbedder` and controls the number of
    spatial crops used per image during embedding (1 = center only, 5 = center +
    four corners).  Ignored for DINOv2 and remote backends.
    """
    if remote_endpoint is not None:
        if embedding_dim <= 0:
            raise ValueError(
                "SFN_EMBEDDING_DIM must be set to a positive integer "
                "when SFN_EMBEDDING_ENDPOINT is configured."
            )
        return RemoteEmbedder(
            endpoint=remote_endpoint,
            model_name=model,
            embedding_dim=embedding_dim,
            api_key=remote_api_key,
            normalize_size=normalize_size,
        )
    if use_sscd:
        return SSCDEmbedder(model_name=model, device=device, n_crops=n_crops)
    return DINOv2Embedder(
        model_name=model,
        device=device,
        normalize_size=normalize_size,
        local_files_only=local_files_only,
    )
