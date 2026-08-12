import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scalar_forensic.faces.detect import YuNetDetector, _scaled_size

REAL_MODEL = os.environ.get("SFN_FACE_DETECTOR_MODEL", "")


def test_scaled_size_caps_long_side_and_reports_scale():
    w, h, scale = _scaled_size(3200, 2400, max_size=1600)
    assert (w, h) == (1600, 1200)
    assert scale == pytest.approx(0.5)
    # Small images are not upscaled.
    assert _scaled_size(640, 480, max_size=1600) == (640, 480, 1.0)


def _fake_backend(rows: np.ndarray):
    """Patch the _create_yunet seam (cv2's C-extension types reject setattr)."""
    stub = MagicMock()
    stub.detect.return_value = (1, rows)
    return patch("scalar_forensic.faces.detect._create_yunet", return_value=stub), stub


def test_detect_scales_coords_back_and_emits_canonical_order(tmp_path):
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"weights")
    # One face at detector scale 0.5, bbox (10,20,30,40).  Landmark columns are
    # written in whatever _YUNET_TO_CANONICAL was empirically derived to expect;
    # this test pins scaling and internal consistency.  The *correctness* of the
    # map against reality is enforced by assert_canonical_landmarks inside
    # detect() plus the real-model test below — not by this synthetic row.
    # Row below assumes the identity map ([0..4]); adjust if derivation differs.
    rows = np.array([[10, 20, 30, 40, 40, 10, 60, 10, 50, 20, 42, 30, 58, 30, 0.94]], np.float32)
    ctx, stub = _fake_backend(rows)
    with ctx:
        det = YuNetDetector(model, max_size=1600)
        img = np.zeros((2400, 3200, 3), dtype=np.uint8)  # -> scale 0.5
        faces = det.detect(img)
    assert len(faces) == 1
    f = faces[0]
    assert f.bbox == pytest.approx((20, 40, 60, 80))  # /0.5 back to source px
    assert f.detect_scale == pytest.approx(0.5)
    np.testing.assert_allclose(  # canonical: le, re, nose, lm, rm
        f.landmarks,
        np.array([[80, 20], [120, 20], [100, 40], [84, 60], [116, 60]], np.float32),
    )
    stub.setInputSize.assert_called_once_with((1600, 1200))


def test_detect_drops_noncanonical_output_and_counts_it(tmp_path):
    # A row whose eye pair comes out swapped after the map is dropped (not
    # emitted, not raised — a rotated real face must not crash a run) and
    # counted, so a wholesale-wrong map is loudly visible in the stats.
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"weights")
    rows = np.array([[10, 20, 30, 40, 60, 10, 40, 10, 50, 20, 42, 30, 58, 30, 0.94]], np.float32)
    ctx, _ = _fake_backend(rows)
    with ctx:
        det = YuNetDetector(model, max_size=1600)
        assert det.detect(np.zeros((100, 100, 3), np.uint8)) == []
        assert det.n_dropped_noncanonical == 1


def test_detect_converts_rgb_to_bgr_for_yunet(tmp_path):
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"weights")
    ctx, stub = _fake_backend(np.empty((0, 15), np.float32))
    with ctx:
        det = YuNetDetector(model, max_size=1600)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[..., 0] = 200  # red channel in RGB
        det.detect(img)
    passed = stub.detect.call_args[0][0]
    assert passed[0, 0, 2] == 200 and passed[0, 0, 0] == 0  # red now in BGR channel 2


def test_no_faces_returns_empty(tmp_path):
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"weights")
    stub = MagicMock()
    stub.detect.return_value = (1, None)  # YuNet returns None for no faces
    with patch("scalar_forensic.faces.detect._create_yunet", return_value=stub):
        assert YuNetDetector(model, max_size=1600).detect(np.zeros((10, 10, 3), np.uint8)) == []


@pytest.mark.skipif(
    not Path(REAL_MODEL or "/nonexistent").exists(), reason="YuNet model not present"
)
def test_real_model_landmark_order_on_real_face():
    # Committed fixture: any permissively-licensed photo with one clear frontal
    # face (tests/fixtures/faces/real_face.jpg — implementer supplies, e.g. a
    # self-taken photo).  Asserts the ordering invariant on REAL detector output,
    # which is the only thing that can catch a wrong _YUNET_TO_CANONICAL map.
    from PIL import Image as _Image

    img = np.array(_Image.open(Path("tests/fixtures/faces/real_face.jpg")).convert("RGB"))
    det = YuNetDetector(Path(REAL_MODEL), max_size=1600)
    faces = det.detect(img)
    assert faces, "fixture must contain a detectable face"
    for f in faces:
        assert f.landmarks[0, 0] < f.landmarks[1, 0]  # left eye left of right eye
        assert f.landmarks[3, 0] < f.landmarks[4, 0]  # left mouth left of right mouth
