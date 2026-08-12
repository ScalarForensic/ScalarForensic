"""Face chip store: lossless aligned crop + human review chip + thumbnail (spec §7.3).

The PNG holds the exact 112x112 RGB tensor fed to the embedder
(pre-normalisation) and aligned_chip_hash() covers those exact bytes, so
the stored file authenticates the model input.  The review JPEG is the
unwarped, dilated source crop the examiner actually looks at, and it is
content-addressed separately by review_chip_hash().

Hash domains are separated because the same dimension-prefixed RGB array
can legitimately arise once as an aligned crop and once as a native review
crop; paths are chosen by hash plus suffix alone, so a shared domain would
let a review-only observation be served another observation's aligned PNG.

Artefact roles: the PNG and the review JPEG are evidentiary; the
thumbnail is derived, non-evidentiary and regenerable — it never enters
either hash and carries no provenance of its own.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

_REVIEW_QUALITY = 95

_ALIGNED_DOMAIN = b"aligned-rgb-v1\0"
_REVIEW_DOMAIN = b"review-source-rgb-v1\0"


def _domain_hash(domain: bytes, arr: np.ndarray) -> str:
    # dtype and the *full* shape enter the prefix, not just height x width:
    # (6,6) uint8 and (6,6,1) uint8 have byte-identical buffers, as do
    # (2,2,4) uint8 and (2,2,1) uint32.  Every array reaching here today is
    # HxWx3 uint8, so neither collision is live -- but these digests become
    # payload fields and filenames, so widening the prefix afterwards would
    # be a schema-and-filesystem migration.
    shape = "x".join(str(n) for n in arr.shape)
    hasher = hashlib.sha256(domain + f"{arr.dtype.str}:{shape}:".encode())
    hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


def aligned_chip_hash(aligned_rgb: np.ndarray) -> str:
    """Identity of the exact 112x112 tensor fed to the embedder."""
    return _domain_hash(_ALIGNED_DOMAIN, aligned_rgb)


def review_chip_hash(crop_rgb: np.ndarray) -> str:
    """Identity of the native-resolution source crop an examiner reviews.

    Domain-separated from aligned_chip_hash: the same pixel array can arise
    in both roles, and the chip endpoints resolve files by hash + suffix.
    """
    return _domain_hash(_REVIEW_DOMAIN, crop_rgb)


def chip_paths(store_dir: Path, chash: str) -> tuple[Path, Path, Path]:
    shard = store_dir / chash[:2]
    return (
        shard / f"{chash}.png",
        shard / f"{chash}.review.jpg",
        shard / f"{chash}.thumb.jpg",
    )


def review_chip_paths(store_dir: Path, chash: str) -> tuple[Path, Path]:
    shard = store_dir / chash[:2]
    return shard / f"{chash}.review.jpg", shard / f"{chash}.thumb.jpg"


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


def write_review_chips(
    store_dir: Path,
    source_rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    dilation: float,
    thumb_size: int,
) -> str | None:
    """Write the review JPEG and thumbnail for a review-only observation.

    Returns None when the dilated bbox clamps to zero area: a review-only
    observation whose crop does not exist is useless, so the caller rejects
    it rather than storing a hash for files that were never written.
    """
    x, y, w, h = dilated_clamped_bbox(bbox, dilation, source_rgb.shape[1], source_rgb.shape[0])
    if w <= 0 or h <= 0:
        return None
    crop = source_rgb[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    chash = review_chip_hash(crop)
    jpg, thumb = review_chip_paths(store_dir, chash)
    jpg.parent.mkdir(parents=True, exist_ok=True)
    if not jpg.exists():
        Image.fromarray(crop).save(jpg, format="JPEG", quality=_REVIEW_QUALITY)
    if not thumb.exists():
        write_thumbnail(jpg, thumb, thumb_size)
    return chash


def write_aligned_chips(
    store_dir: Path,
    aligned_rgb: np.ndarray,
    source_rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    dilation: float,
    thumb_size: int,
) -> tuple[str, str | None]:
    """Write the aligned PNG plus the review artefacts.

    Returns (aligned_hash, review_hash).  The PNG is content-addressed in the
    aligned domain because it authenticates the exact model input; the review
    JPEG lives in the review domain so both observation kinds resolve review
    artefacts by the same rule.

    Idempotent: existing files are kept (matching the frame-store reuse
    pattern), so re-indexing the same face costs no rewrites.
    """
    ahash = aligned_chip_hash(aligned_rgb)
    png, _, _ = chip_paths(store_dir, ahash)
    png.parent.mkdir(parents=True, exist_ok=True)
    if not png.exists():
        Image.fromarray(aligned_rgb).save(png, format="PNG")
    rhash = write_review_chips(store_dir, source_rgb, bbox, dilation, thumb_size)
    return ahash, rhash
