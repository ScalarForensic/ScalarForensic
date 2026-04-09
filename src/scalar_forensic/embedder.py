"""Image embedding backends: DINOv2 (HuggingFace) and SSCD (TorchScript)."""

import hashlib
import importlib.metadata
import io
import sys
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

_SSCD_INPUT_SIZE = 288
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


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------


class DINOv2Embedder:
    def __init__(self, model_name: str, device: str = "auto", normalize_size: int = 512) -> None:
        self.model_name = model_name
        self.normalize_size = normalize_size
        self.device = _resolve_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._model_hash: str | None = None

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size  # type: ignore[no-any-return]

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

    def normalize_batch_bytes(self, image_data: list[bytes]) -> list[Image.Image]:
        size = self.normalize_size
        return [_open_rgb(d).resize((size, size), Image.Resampling.LANCZOS) for d in image_data]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().tolist()


# ---------------------------------------------------------------------------
# SSCD
# ---------------------------------------------------------------------------


def _sscd_preprocess(data: bytes) -> Image.Image:
    """Aspect-ratio-preserving resize to 115 % then center-crop to 288×288."""
    img = _open_rgb(data)
    size = _SSCD_INPUT_SIZE
    scale = int(size * 1.15)
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
    def model_hash(self) -> str:
        if self._model_hash is None:
            h = hashlib.sha256()
            with open(self.model_name, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            self._model_hash = h.hexdigest()
        return self._model_hash

    def normalize_batch_bytes(self, image_data: list[bytes]) -> list[Image.Image]:
        return [_sscd_preprocess(d) for d in image_data]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        batch = torch.stack([self._to_tensor(img) for img in images]).to(self.device)
        with torch.no_grad():
            embeddings = self._model(batch)
        return embeddings.cpu().tolist()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AnyEmbedder = DINOv2Embedder | SSCDEmbedder


def load_embedder(
    model: str,
    use_sscd: bool,
    device: str = "auto",
    normalize_size: int = 512,
) -> AnyEmbedder:
    """Load the embedder for the explicitly selected backend."""
    if use_sscd:
        return SSCDEmbedder(model_name=model, device=device)
    return DINOv2Embedder(model_name=model, device=device, normalize_size=normalize_size)
