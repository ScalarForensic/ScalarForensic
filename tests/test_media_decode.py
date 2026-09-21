"""CPU-only decoder contract, including native versus embedding geometry."""

import importlib
import io
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageOps


def _decoder():
    assert importlib.util.find_spec("scalar_forensic.media_decode") is not None, (
        "CPU-only media_decode module is missing"
    )
    return importlib.import_module("scalar_forensic.media_decode")


def _encoded(fmt="PNG", orientation=1, size=(1600, 1200)):
    y, x = np.indices((size[1], size[0]))
    pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype("uint8")
    exif = Image.Exif()
    exif[274] = orientation
    exif[271] = " Camera Maker "
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format=fmt, exif=exif)
    return buf.getvalue()


def test_import_does_not_load_models_application_or_video():
    _decoder()
    script = """
import sys
import scalar_forensic.media_decode
assert isinstance(scalar_forensic.media_decode.register_heif(), bool)
blocked = ('torch', 'torchvision', 'transformers', 'cv2', 'onnxruntime', 'av',
           'scalar_forensic.config', 'scalar_forensic.embedder',
           'scalar_forensic.scanner', 'scalar_forensic.faces.indexing')
assert not [name for name in sys.modules
            if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "TIFF", "WEBP"])
@pytest.mark.parametrize("orientation", [1, 6])
def test_native_and_embedding_pixels_keep_distinct_decode_paths(fmt, orientation):
    decoder = _decoder()
    data = _encoded(fmt, orientation)
    native, shared = decoder.decode_shared(data, 331)
    reference = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    np.testing.assert_array_equal(native, np.asarray(reference))
    np.testing.assert_array_equal(decoder.load_for_detection(data), native)
    assert native.shape == ((1600, 1200, 3) if orientation == 6 else (1200, 1600, 3))
    expected = Image.open(io.BytesIO(data))
    if fmt == "JPEG":
        expected.draft("RGB", (331, 331))
    if fmt != "PNG" and orientation == 6:
        expected = ImageOps.exif_transpose(expected)
    expected = expected.convert("RGB")
    target = (331, 441) if orientation == 6 and fmt != "PNG" else (441, 331)
    expected = expected.resize(target, Image.Resampling.LANCZOS)
    actual = decoder.preprocess_batch([data], 331)[0]
    assert actual.size == target
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    if fmt == "JPEG" or (fmt == "PNG" and orientation != 1):
        assert shared is None
    else:
        np.testing.assert_array_equal(np.asarray(shared), np.asarray(actual))


def test_small_png_pixels_and_batch_error_order():
    decoder = _decoder()
    data = _encoded(size=(3, 2))
    results = decoder.preprocess_batch([data, b"broken", data], 331)
    assert len(results) == 3
    assert isinstance(results[1], Exception)
    assert results[0].size == (3, 2)
    assert [tuple(pixel) for pixel in np.asarray(results[0]).reshape(-1, 3)] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 2),
        (0, 1, 1),
        (1, 1, 2),
        (2, 1, 3),
    ]
    assert results[0].tobytes() == results[2].tobytes()
    assert decoder.preprocess_batch([], 331) == []


def test_exif_metadata_and_invalid_input():
    decoder = _decoder()
    data = _encoded(size=(3, 2))
    assert decoder.extract_exif(data) == {"exif": True, "exif_geo_data": False}
    assert decoder.extract_exif_detailed(data) == {
        "exif": True,
        "exif_geo_data": False,
        "width": 3,
        "height": 2,
        "format": "PNG",
        "size_bytes": len(data),
        "make": "Camera Maker",
    }
    assert decoder.extract_exif(b"broken") == {"exif": False, "exif_geo_data": False}
    assert decoder.extract_exif_detailed(b"broken") == {
        "exif": False,
        "exif_geo_data": False,
    }


def test_pixel_limit_applies_on_decoder_and_embedder_reload():
    _decoder()
    script = """
import importlib
import os
from PIL import Image
from scalar_forensic import media_decode
assert Image.MAX_IMAGE_PIXELS == 1234
os.environ['SFN_MAX_IMAGE_PIXELS'] = 'disabled'
importlib.reload(media_decode)
assert Image.MAX_IMAGE_PIXELS is None
from scalar_forensic import embedder
os.environ['SFN_MAX_IMAGE_PIXELS'] = '5678'
importlib.reload(embedder)
assert Image.MAX_IMAGE_PIXELS == 5678
os.environ['SFN_MAX_IMAGE_PIXELS'] = '0'
try:
    importlib.reload(media_decode)
except ValueError:
    pass
else:
    raise AssertionError('zero pixel limit was accepted')
"""
    env = {**os.environ, "SFN_MAX_IMAGE_PIXELS": "1234"}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
