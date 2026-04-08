"""Recursive image file discovery."""

from collections.abc import Iterator
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})


def scan_images(root: Path) -> Iterator[Path]:
    """Recursively yield image file paths under root."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
