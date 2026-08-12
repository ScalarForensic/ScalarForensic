import io

import numpy as np
from PIL import Image

from scalar_forensic.faces.decode import load_for_detection


def _jpeg_bytes(img: Image.Image, exif: Image.Exif | None = None) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, exif=exif or Image.Exif())
    return buf.getvalue()


def test_full_resolution_no_draft_downscale():
    # 2000px wide: _open_rgb's draft() path would decode this at reduced scale.
    img = Image.new("RGB", (2000, 1400), (10, 20, 30))
    out = load_for_detection(_jpeg_bytes(img))
    assert out.shape == (1400, 2000, 3)
    assert out.dtype == np.uint8


def test_exif_orientation_applied():
    img = Image.new("RGB", (100, 60), (0, 0, 0))
    px = img.load()
    px[0, 0] = (255, 0, 0)  # top-left marker
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 270 deg CW to display upright
    out = load_for_detection(_jpeg_bytes(img, exif))
    assert out.shape[:2] == (100, 60)  # oriented: dims swap


def test_non_rgb_modes_converted():
    img = Image.new("L", (50, 50), 128)
    out = load_for_detection(_jpeg_bytes(img))
    assert out.shape == (50, 50, 3)
