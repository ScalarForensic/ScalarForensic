"""Image embedding backends: DINOv2 (HuggingFace), SSCD (TorchScript), and remote OpenAI-compat."""

import base64
import hashlib
import importlib.metadata
import io
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

_SSCD_INPUT_SIZE = 288
# Short-side cap applied once before both models: SSCD needs 331 px, DINOv2 needs 256 px.
_SHARED_CAP = 331
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_EXIF_GPS_IFD_TAG = 34853  # PIL ExifTags.Base.GPSInfo


def get_library_versions() -> dict[str, str]:
    libs = ["torch", "transformers", "torchvision", "qdrant-client", "Pillow"]
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


def _open_rgb(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _cap_short_side(img: Image.Image) -> Image.Image:
    """Scale down if the short side exceeds _SHARED_CAP; leave smaller images untouched."""
    w, h = img.size
    short = min(w, h)
    if short <= _SHARED_CAP:
        return img
    scale = _SHARED_CAP / short
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def preprocess_batch(image_data: list[bytes]) -> list[Image.Image]:
    """Shared pre-step: open RGB, cap short side to _SHARED_CAP px, parallelised over CPU cores."""

    def _process(data: bytes) -> Image.Image:
        return _cap_short_side(_open_rgb(data))

    with ThreadPoolExecutor() as pool:
        return list(pool.map(_process, image_data))


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------


class DINOv2Embedder:
    def __init__(self, model_name: str, device: str = "auto", normalize_size: int = 512) -> None:
        self.model_name = model_name
        self.normalize_size = normalize_size
        self.device = _resolve_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()
        if self.device == "cuda":
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
        return outputs.last_hidden_state[:, 0, :].float().cpu().tolist()


# ---------------------------------------------------------------------------
# SSCD
# ---------------------------------------------------------------------------


def _sscd_crop(img: Image.Image) -> Image.Image:
    """Aspect-ratio resize to 331 px short side then center-crop to 288×288.

    For images pre-capped at 331 px the resize is a near-no-op; for images
    smaller than 331 px (never capped) it upscales as the model requires.
    """
    size = _SSCD_INPUT_SIZE
    scale = int(size * 1.15)  # 331
    w, h = img.size
    new_w, new_h = (scale, int(h * scale / w)) if w <= h else (int(w * scale / h), scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


class SSCDEmbedder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.normalize_size = _SSCD_INPUT_SIZE
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
        if self.device == "cuda":
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
        return [_sscd_crop(img) for img in images]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        batch = torch.stack([self._to_tensor(img) for img in images]).to(self.device)
        with torch.no_grad():
            embeddings = self._model(batch)
        return embeddings.float().cpu().tolist()


# ---------------------------------------------------------------------------
# Remote (OpenAI-compatible embeddings endpoint)
# ---------------------------------------------------------------------------


class RemoteEmbedder:
    """Calls an OpenAI-compatible POST /v1/embeddings endpoint.

    Images are JPEG-encoded and sent as base64 data-URIs in the ``input`` array,
    which is the convention used by servers such as Infinity and similar multimodal
    embedding APIs.  Required configuration: endpoint URL (``SFN_EMBEDDING_ENDPOINT``),
    model name (``SFN_EMBEDDING_MODEL``), and embedding dimension (``SFN_EMBEDDING_DIM``,
    must match the dimension the remote model actually produces).  An optional Bearer
    API key may be provided via ``SFN_EMBEDDING_API_KEY``.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        embedding_dim: int,
        api_key: str | None = None,
        normalize_size: int = 512,
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
            self._model_hash = h.hexdigest()
        return self._model_hash

    def normalize_batch_bytes(self, images: list[Image.Image]) -> list[Image.Image]:
        return images  # server-side preprocessing assumed

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs: list[str] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            b64 = base64.b64encode(buf.getvalue()).decode()
            inputs.append(f"data:image/jpeg;base64,{b64}")

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
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"Embedding endpoint returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding endpoint connection failed: {exc.reason}") from exc

        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AnyEmbedder = DINOv2Embedder | SSCDEmbedder | RemoteEmbedder


def load_embedder(
    model: str,
    use_sscd: bool,
    device: str = "auto",
    normalize_size: int = 512,
    *,
    remote_endpoint: str | None = None,
    remote_api_key: str | None = None,
    embedding_dim: int = 0,
) -> AnyEmbedder:
    """Load the embedder for the selected backend.

    When *remote_endpoint* is set the remote OpenAI-compatible endpoint is used
    and *use_sscd* / *device* are ignored.  *embedding_dim* must be > 0 in that
    case (set ``SFN_EMBEDDING_DIM`` to the dimension the remote model produces).
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
        return SSCDEmbedder(model_name=model, device=device)
    return DINOv2Embedder(model_name=model, device=device, normalize_size=normalize_size)
