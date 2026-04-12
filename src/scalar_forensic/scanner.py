"""Recursive image and container file discovery."""

from collections.abc import Iterator
from pathlib import Path

from scalar_forensic.extractor import CONTAINER_EXTENSIONS

IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
        ".jp2",
        ".ico",
        ".psd",
    }
)

_HEIF_EXTENSIONS = frozenset({".heic", ".heif"})
_HEIF_AVAILABLE: bool = False


def _maybe_register_heif() -> None:
    global _HEIF_AVAILABLE
    try:
        import pillow_heif  # noqa: PLC0415

        pillow_heif.register_heif_opener()
        _HEIF_AVAILABLE = True
    except ImportError:
        pass


_maybe_register_heif()


def scan_images(root: Path) -> Iterator[Path]:
    """Recursively yield image file paths under root."""
    extensions = IMAGE_EXTENSIONS | (_HEIF_EXTENSIONS if _HEIF_AVAILABLE else frozenset())
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def scan_all_files(root: Path) -> Iterator[tuple[Path, str]]:
    """Recursively yield *(path, file_type)* for every regular file under *root*.

    *file_type* is one of:

    * ``"image"`` — a directly-supported raster image format.
    * ``"container"`` — a container format (ZIP, PDF, DOCX, ODF) that may hold
      embedded images and will be processed by the extractor.
    * ``"other"`` — unrecognised format; skipped during ingestion.
    """
    image_extensions = IMAGE_EXTENSIONS | (_HEIF_EXTENSIONS if _HEIF_AVAILABLE else frozenset())
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in image_extensions:
            yield path, "image"
        elif suffix in CONTAINER_EXTENSIONS:
            yield path, "container"
        else:
            yield path, "other"
