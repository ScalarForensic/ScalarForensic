import numpy as np
import pytest

from scalar_forensic.faces.quality import pose_ratio, post_align_gate, pre_align_gate
from scalar_forensic.faces.types import FaceDetection

FRONTAL = np.array([[30, 30], [70, 30], [50, 55], [35, 75], [65, 75]], np.float32)


def _det(conf=0.95, size=100.0, lm=FRONTAL):
    return FaceDetection(bbox=(0, 0, size, size), landmarks=lm, confidence=conf, detect_scale=1.0)


def test_pre_gate_passes_good_face():
    r = pre_align_gate(_det(), min_conf=0.8, min_size=64, max_pose=0.35)
    assert r.passed and r.reason is None
    assert set(r.subscores) == {"confidence", "size", "pose"}


def test_pre_gate_rejects_low_confidence():
    r = pre_align_gate(_det(conf=0.5), min_conf=0.8, min_size=64, max_pose=0.35)
    assert not r.passed and r.reason == "confidence"


def test_pre_gate_rejects_small_face():
    # min side in detector-input px: bbox 40px at detect_scale 1.0
    r = pre_align_gate(_det(size=40.0), min_conf=0.8, min_size=64, max_pose=0.35)
    assert not r.passed and r.reason == "size"


def test_pre_gate_rejects_strong_profile():
    # Nose far outside the eye span -> strong yaw.
    profile = np.array([[30, 30], [45, 30], [70, 50], [33, 75], [48, 75]], np.float32)
    r = pre_align_gate(_det(lm=profile), min_conf=0.8, min_size=64, max_pose=0.35)
    assert not r.passed and r.reason == "pose"


def test_pose_ratio_zero_for_symmetric_face():
    assert pose_ratio(FRONTAL) == pytest.approx(0.0, abs=0.05)


def test_post_gate_rejects_flat_crop_as_blurry():
    flat = np.full((80, 80), 128, dtype=np.uint8)
    r = post_align_gate(flat, min_sharpness=25.0, max_clipped_frac=0.6)
    assert not r.passed and r.reason == "sharpness"


def test_post_gate_rejects_clipped_exposure():
    sharp_but_clipped = np.zeros((80, 80), dtype=np.uint8)
    sharp_but_clipped[::2] = 255  # high Laplacian variance, but 100% clipped pixels
    r = post_align_gate(sharp_but_clipped, min_sharpness=25.0, max_clipped_frac=0.6)
    assert not r.passed and r.reason == "exposure"


def test_post_gate_passes_textured_crop():
    rng = np.random.default_rng(3)
    textured = rng.integers(60, 200, size=(80, 80), dtype=np.uint8)
    r = post_align_gate(textured, min_sharpness=25.0, max_clipped_frac=0.6)
    assert r.passed and set(r.subscores) == {"sharpness", "exposure"}
