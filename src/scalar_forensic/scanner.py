"""Recursive image and video file discovery."""

from collections.abc import Iterator
from pathlib import Path

from scalar_forensic.video import VIDEO_EXTENSIONS

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


def scan_all_files(root: Path) -> Iterator[tuple[Path, str]]:
    """Recursively yield *(path, file_type)* for every regular file under *root*.

    ``file_type`` is one of ``"image"``, ``"video"``, or ``"unsupported"``.
    """
    image_extensions = IMAGE_EXTENSIONS | (_HEIF_EXTENSIONS if _HEIF_AVAILABLE else frozenset())
    for path in root.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower()
            if ext in image_extensions:
                yield path, "image"
            elif ext in VIDEO_EXTENSIONS:
                yield path, "video"
            else:
                yield path, "unsupported"
