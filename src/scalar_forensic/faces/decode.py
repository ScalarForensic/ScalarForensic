"""Full-resolution decode for face detection.

Deliberately NOT scalar_forensic.embedder._open_rgb: that path calls
PIL's JPEG draft() against the 331 px embedding cap, which destroys
small faces.  Faces are detected at full oriented resolution and the
detector applies its own input cap (spec §6.1).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps


def load_for_detection(data: bytes) -> np.ndarray:
    """Decode *data* to oriented full-resolution RGB uint8 (H, W, 3).

    Pillow's MAX_IMAGE_PIXELS decompression-bomb guard stays active here
    (deliberate: this path decodes at full resolution with no 331 px cap
    to protect it; SFN_MAX_IMAGE_PIXELS overrides for trusted ingestion,
    same as everywhere else).
    """
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)
