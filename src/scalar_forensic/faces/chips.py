"""Face chip store: lossless aligned crop + human review chip + thumbnail (spec §7.3).

The PNG holds the exact 112x112 RGB tensor fed to the embedder
(pre-normalisation) and chip_hash() covers those exact bytes, so the
stored file authenticates the model input.  The review JPEG is the
unwarped, dilated source crop the examiner actually looks at.

Artefact roles: the PNG and the review JPEG are evidentiary; the
thumbnail is derived, non-evidentiary and regenerable — it never enters
chip_hash() and carries no provenance of its own.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

_REVIEW_QUALITY = 95


def chip_hash(aligned_rgb: np.ndarray) -> str:
    h, w = aligned_rgb.shape[:2]
    hasher = hashlib.sha256(f"{h}x{w}:".encode())
    hasher.update(np.ascontiguousarray(aligned_rgb).tobytes())
    return hasher.hexdigest()


def chip_paths(store_dir: Path, chash: str) -> tuple[Path, Path, Path]:
    shard = store_dir / chash[:2]
    return (
        shard / f"{chash}.png",
        shard / f"{chash}.review.jpg",
        shard / f"{chash}.thumb.jpg",
    )


def dilated_clamped_bbox(
    bbox: tuple[float, float, float, float], dilation: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    dx, dy = w * dilation, h * dilation
    x0, y0 = max(0, int(x - dx)), max(0, int(y - dy))
    x1, y1 = min(img_w, int(x + w + dx)), min(img_h, int(y + h + dy))
    return x0, y0, x1 - x0, y1 - y0


def write_thumbnail(review_path: Path, thumb_path: Path, thumb_size: int) -> None:
    """Downscale the review chip's long side to *thumb_size*; never upscale.

    Standalone so the browse endpoint can regenerate a deleted thumbnail
    without re-running the pipeline.
    """
    with Image.open(review_path) as img:
        img = img.convert("RGB")
        if max(img.size) > thumb_size:
            img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(thumb_path, format="JPEG", quality=_REVIEW_QUALITY)


def write_chips(
    store_dir: Path,
    aligned_rgb: np.ndarray,
    source_rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    dilation: float,
    thumb_size: int,
) -> str:
    """Write the three chip artefacts, returning the chip hash.

    Idempotent: existing files are kept (matching the frame-store reuse
    pattern), so re-indexing the same face costs no rewrites.
    """
    chash = chip_hash(aligned_rgb)
    png, jpg, thumb = chip_paths(store_dir, chash)
    png.parent.mkdir(parents=True, exist_ok=True)
    if not png.exists():
        Image.fromarray(aligned_rgb).save(png, format="PNG")
    if not jpg.exists():
        x, y, w, h = dilated_clamped_bbox(bbox, dilation, source_rgb.shape[1], source_rgb.shape[0])
        if w > 0 and h > 0:  # bbox fully off-image after clamping: skip review chip
            Image.fromarray(source_rgb[y : y + h, x : x + w]).save(
                jpg, format="JPEG", quality=_REVIEW_QUALITY
            )
    if jpg.exists() and not thumb.exists():
        write_thumbnail(jpg, thumb, thumb_size)
    return chash
