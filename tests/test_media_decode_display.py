import io

import pytest
from PIL import Image

from scalar_forensic import media_decode


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_thumbnail_resize_precedes_color_conversion(mode):
    image = Image.new(mode, (32, 16))
    # Alpha/palette/grayscale take different Pillow resize and JPEG paths.
    if mode == "RGBA":
        image.paste((200, 20, 40, 80), (8, 0, 24, 16))
    elif mode == "P":
        image.putpalette([i for i in range(256) for _ in range(3)])
        image.paste(127, (8, 0, 24, 16))
    else:
        image.paste(127 if mode == "L" else (200, 20, 40), (8, 0, 24, 16))
    source = io.BytesIO()
    image.save(source, format="PNG")
    expected = image.copy()
    expected.thumbnail((8, 8), Image.Resampling.LANCZOS)
    if expected.mode not in ("RGB", "L"):
        expected = expected.convert("RGB")
    actual = media_decode.thumbnail_input(source.getvalue(), 8)
    assert actual.size == (8, 4)
    assert actual.mode == expected.mode
    assert actual.tobytes() == expected.tobytes()


def test_display_decode_does_not_apply_exif_or_jpeg_draft():
    image = Image.new("RGB", (1600, 800), (10, 20, 30))
    exif = Image.Exif()
    exif[274] = 6
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", exif=exif)
    decoded = media_decode.display_rgb(encoded.getvalue())
    assert decoded.size == (1600, 800)
    assert decoded.mode == "RGB"
