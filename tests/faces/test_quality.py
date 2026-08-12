import numpy as np
import pytest

from scalar_forensic.faces.quality import (
    pose_ratio,
    post_align_gate,
    pre_align_gate,
    review_gate,
)
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


def _rdet(conf=0.9, w=100.0, h=100.0, scale=1.0, lm=FRONTAL):
    return FaceDetection(bbox=(0.0, 0.0, w, h), landmarks=lm, confidence=conf, detect_scale=scale)


def test_review_gate_passes_a_small_but_confident_face():
    # 40px: below the 64px embedding floor, above a 24px review floor.
    r = review_gate(_rdet(conf=0.93, w=40.0, h=49.0), min_conf=0.6, min_size=24)
    assert r.passed is True
    assert r.reason is None
    assert r.subscores["size"] == 40.0


def test_review_gate_rejects_below_size():
    r = review_gate(_rdet(w=20.0, h=20.0), min_conf=0.6, min_size=24)
    assert r.passed is False
    assert r.reason == "size"


def test_review_gate_rejects_below_confidence():
    r = review_gate(_rdet(conf=0.55), min_conf=0.6, min_size=24)
    assert r.passed is False
    assert r.reason == "confidence"


def test_review_gate_measures_size_in_detector_input_px():
    # A large face in a downscaled image is small to the detector.
    r = review_gate(_rdet(w=100.0, h=100.0, scale=0.2), min_conf=0.6, min_size=24)
    assert r.passed is False
    assert r.subscores["size"] == 20.0


def test_review_gate_ignores_pose():
    # Pose is an embedding concern only: a profile face is still worth a look.
    # Nose far outside the eye span -> pose_ratio ~2.2, well past any max_pose.
    profile = np.array([[30, 30], [45, 30], [70, 50], [33, 75], [48, 75]], np.float32)
    assert pose_ratio(profile) > 0.35
    assert review_gate(_rdet(lm=profile), min_conf=0.6, min_size=24).passed is True
    # Same landmarks would be rejected by the embedding gate.
    assert pre_align_gate(_rdet(lm=profile), min_conf=0.6, min_size=24, max_pose=0.35).reason == (
        "pose"
    )
