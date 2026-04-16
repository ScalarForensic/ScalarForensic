"""Image embedding backends: DINOv2 (HuggingFace), SSCD (TorchScript), and remote OpenAI-compat."""

import base64
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

_SSCD_INPUT_SIZE = 288
# SSCD requires short-side ≥ 331 px before the 288×288 center-crop.
_SSCD_SCALE = 331
# Backward-compat alias — callers that don't pass an explicit cap still get 331 px.
_SHARED_CAP = _SSCD_SCALE

_MAX_IMAGE_PIXELS_ENV = "SFN_MAX_IMAGE_PIXELS"
_LEGACY_MAX_IMAGE_PIXELS_ENV = "SCALAR_FORENSIC_MAX_IMAGE_PIXELS"


def _configure_max_image_pixels_from_env() -> None:
    """Configure Pillow's decompression-bomb guard from an explicit env override.

    By default, leave Pillow's built-in MAX_IMAGE_PIXELS limit unchanged so web
    and other untrusted-image paths retain decompression-bomb protection.
    Set SFN_MAX_IMAGE_PIXELS to:
      - a positive integer to allow a larger finite pixel count; or
      - "none" / "disable" / "disabled" to turn the guard off explicitly for
        trusted ingestion runs.

    For backward compatibility, the legacy SCALAR_FORENSIC_MAX_IMAGE_PIXELS key
    is also accepted when SFN_MAX_IMAGE_PIXELS is unset.
    """
    source_env = _MAX_IMAGE_PIXELS_ENV
    raw_value = os.getenv(_MAX_IMAGE_PIXELS_ENV)
    if raw_value is None:
        source_env = _LEGACY_MAX_IMAGE_PIXELS_ENV
        raw_value = os.getenv(_LEGACY_MAX_IMAGE_PIXELS_ENV)
    if raw_value is None:
        return
    value = raw_value.strip()
    if not value:
        return
    if value.lower() in {"none", "disable", "disabled"}:
        Image.MAX_IMAGE_PIXELS = None
        return
    try:
        max_pixels = int(value)
    except ValueError:
        raise ValueError(
            f"{source_env} must be a positive integer or one of 'none', 'disable', 'disabled';"
            f" got {raw_value!r}"
        ) from None
    if max_pixels <= 0:
        raise ValueError(
            f"{source_env} must be a positive integer or one of 'none', 'disable', 'disabled'"
        )
    Image.MAX_IMAGE_PIXELS = max_pixels


_configure_max_image_pixels_from_env()

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_EXIF_GPS_IFD_TAG = 34853  # PIL ExifTags.Base.GPSInfo


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


def hash_file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return MD5 hex digest of a file, reading it in chunks."""
    h = hashlib.md5()  # noqa: S324
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


class ExifInfo(TypedDict):
    exif: bool
    exif_geo_data: bool


def extract_exif(data: bytes) -> ExifInfo:
    """Return EXIF presence flags from raw image bytes.

    Opens the image without converting to RGB — colour conversion strips EXIF on
    some formats. Falls back to exif=False on any error.
    """
    try:
        img = Image.open(io.BytesIO(data))
        exif_obj = img.getexif()
        return ExifInfo(exif=len(exif_obj) > 0, exif_geo_data=_EXIF_GPS_IFD_TAG in exif_obj)
    except Exception:  # noqa: BLE001
        return ExifInfo(exif=False, exif_geo_data=False)


def extract_exif_detailed(data: bytes) -> dict:
    """Return detailed file + EXIF metadata dict for display in the UI."""
    result: dict = {"exif": False, "exif_geo_data": False}
    try:
        img = Image.open(io.BytesIO(data))
        result["width"], result["height"] = img.size
        result["format"] = img.format or ""
        result["size_bytes"] = len(data)

        exif_obj = img.getexif()
        result["exif"] = len(exif_obj) > 0
        result["exif_geo_data"] = _EXIF_GPS_IFD_TAG in exif_obj

        if result["exif"]:
            # Common tags by ID (stable across Pillow versions)
            _TAG_NAMES = {271: "make", 272: "model", 306: "datetime", 305: "software"}
            for tag_id, key in _TAG_NAMES.items():
                val = exif_obj.get(tag_id)
                if val:
                    result[key] = str(val).strip("\x00").strip()

        if result["exif_geo_data"]:
            gps_ifd = exif_obj.get_ifd(_EXIF_GPS_IFD_TAG)
            try:
                lat_dms = gps_ifd.get(2)  # GPSLatitude
                lat_ref = gps_ifd.get(1)  # GPSLatitudeRef 'N'/'S'
                lon_dms = gps_ifd.get(4)  # GPSLongitude
                lon_ref = gps_ifd.get(3)  # GPSLongitudeRef 'E'/'W'
                if lat_dms and lon_dms:

                    def _dms(dms: tuple, ref: str) -> float:
                        d, m, s = [float(x) for x in dms]
                        v = d + m / 60.0 + s / 3600.0
                        return -v if ref in ("S", "W") else v

                    result["gps_lat"] = round(_dms(lat_dms, lat_ref or "N"), 6)
                    result["gps_lon"] = round(_dms(lon_dms, lon_ref or "E"), 6)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return result


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_EXIF_ORIENTATION_TAG = 0x0112  # EXIF tag 274
_EXIF_ORIENTATION_FORMATS = frozenset({"JPEG", "TIFF", "WEBP", "MPO", "HEIF"})


def _open_rgb(data: bytes, cap: int = _SHARED_CAP) -> Image.Image:
    """Decode image bytes to a normalised RGB PIL Image.

    ICC colour profiles are ignored — images are treated as sRGB regardless of
    any embedded profile.  Wide-gamut sources (AdobeRGB, ProPhoto) would
    produce slightly shifted pixel values, but such images are rare in forensic
    datasets and the semantic embedding models are robust to small colour shifts.

    EXIF orientation is applied so that phone photos stored with a non-trivial
    Orientation tag are embedded in the orientation the user sees, not the raw
    sensor orientation.  The check is skipped entirely for formats that never
    carry EXIF (PNG, BMP, GIF …) and short-circuits for the common Orientation=1
    case to avoid a full decode+rotate on every image.

    For JPEG inputs, ``draft()`` is called before any pixel decode to enable
    libjpeg's built-in shrink-on-load: the decoder picks the largest 1/N scale
    (N ∈ {2, 4, 8}) that still produces both dimensions ≥ *cap*, then
    ``_cap_short_side`` does the final resize in software.  A 4000×3000 JPEG
    shrunk to 1/8 (500×375) before Lanczos resizing to 331 px is decoded in a
    fraction of the time, with negligible quality difference for semantic
    embedding.  Images already close to *cap* decode at full size (no-op).
    """
    img = Image.open(io.BytesIO(data))
    # draft() must be called before any operation that triggers full pixel
    # decode (including exif_transpose and convert).  getexif() only reads
    # the file header so it is safe to call before draft().
    if img.format == "JPEG":
        img.draft("RGB", (cap, cap))
    if img.format in _EXIF_ORIENTATION_FORMATS:
        if img.getexif().get(_EXIF_ORIENTATION_TAG, 1) != 1:
            img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _cap_short_side(img: Image.Image, cap: int = _SHARED_CAP) -> Image.Image:
    """Scale down proportionally if the short side exceeds *cap*; leave smaller images untouched."""
    w, h = img.size
    short = min(w, h)
    if short <= cap:
        return img
    scale = cap / short
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def preprocess_batch(
    image_data: list[bytes], cap: int = _SHARED_CAP
) -> list[Image.Image | Exception]:
    """Open RGB (with EXIF orientation correction), cap short side to *cap* px.

    Parallelised over CPU cores.  Returns one entry per input image: an
    ``Image.Image`` on success or the raised ``Exception`` on failure.
    Individual futures are used instead of ``pool.map()`` so that a single
    corrupt or oversized file does not abort the rest of the batch — callers
    should check each result with ``isinstance(result, Exception)``.

    *cap* defaults to ``_SHARED_CAP`` (331 px) for backward compatibility.
    Pass ``max(_SSCD_SCALE, settings.normalize_size)`` from call sites so that
    DINOv2 receives images at its configured resolution.
    """

    def _process(data: bytes) -> Image.Image:
        return _cap_short_side(_open_rgb(data, cap), cap)

    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(_process, data) for data in image_data]
        results: list[Image.Image | Exception] = []
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results


def preprocess_pil_batch(images: list[Image.Image], cap: int = _SHARED_CAP) -> list[Image.Image]:
    """Apply ``_cap_short_side`` to already-decoded PIL Images.

    Video frames arrive from PyAV as PIL Images, so there is no I/O or
    format-decoding step.  This function provides the same size-capping that
    ``preprocess_batch`` applies internally, without the bytes-open overhead.
    ICC and EXIF corrections are not applied here — video frames come from
    PyAV already in sRGB-equivalent colour space with correct geometry.

    Unlike ``preprocess_batch``, this function never returns exceptions —
    callers are responsible for ensuring all inputs are valid RGB Images.

    *cap* defaults to ``_SHARED_CAP`` for backward compatibility; pass
    ``max(_SSCD_SCALE, settings.normalize_size)`` from call sites.
    """
    return [_cap_short_side(img, cap) for img in images]


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------


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
        # Warn if the processor silently ignored the resolution kwargs — this can
        # happen with non-standard DINOv2 variants that use a different processor class.
        actual_size = getattr(self.processor, "size", {})
        if isinstance(actual_size, dict) and actual_size.get("shortest_edge", 0) != normalize_size:
            import warnings

            warnings.warn(
                f"DINOv2 processor did not accept normalize_size={normalize_size}; "
                f"actual size config: {actual_size}. "
                "Embeddings will be produced at the processor's built-in default resolution.",
                stacklevel=2,
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
            h = hashlib.sha256()
            for file in sorted(snapshot_path.rglob("*")):
                if not file.is_file():
                    continue
                h.update(file.name.encode())
                with file.open("rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
            self._model_hash = h.hexdigest()
        return self._model_hash

    def normalize_batch_bytes(self, images: list[Image.Image]) -> list[Image.Image]:
        return images  # already pre-capped; AutoImageProcessor handles the resize to 224×224

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].float()
        cls = F.normalize(cls, p=2, dim=1)
        return cls.cpu().tolist()


# ---------------------------------------------------------------------------
# SSCD
# ---------------------------------------------------------------------------


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

    def normalize_batch_bytes(self, images: list[Image.Image]) -> list[Image.Image]:
        # SSCD-specific transforms (resize + crop) are performed inside embed_images
        # so that multi-crop ensembling can expand the batch before inference.
        return images

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
        with torch.no_grad(), torch.autocast(self.device, enabled=self.device == "cuda"):
            flat_embs = self._model(batch).float()  # (n_orig * n_crops, 512)

        if self.n_crops == 1:
            return flat_embs.cpu().tolist()

        # L2-normalise per crop → average over crops per source image → L2-normalise
        flat_embs = F.normalize(flat_embs, p=2, dim=1)
        embs = flat_embs.view(n_orig, self.n_crops, -1).mean(dim=1)
        embs = F.normalize(embs, p=2, dim=1)
        return embs.cpu().tolist()


# ---------------------------------------------------------------------------
# Remote (OpenAI-compatible embeddings endpoint)
# ---------------------------------------------------------------------------


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
            self._model_hash = h.hexdigest()
        return self._model_hash

    def normalize_batch_bytes(self, images: list[Image.Image]) -> list[Image.Image]:
        return images  # server-side preprocessing assumed

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

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
