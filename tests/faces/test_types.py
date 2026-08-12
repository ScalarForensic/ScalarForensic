import numpy as np
import pytest

from scalar_forensic.faces.types import LANDMARK_ORDER, FaceDetection, assert_canonical_landmarks


def test_landmark_order_is_the_documented_contract():
    assert LANDMARK_ORDER == (
        "left_eye",
        "right_eye",
        "nose_tip",
        "left_mouth",
        "right_mouth",
    )


def test_face_detection_holds_canonical_shapes():
    lm = np.array([[10, 20], [30, 20], [20, 30], [12, 40], [28, 40]], dtype=np.float32)
    det = FaceDetection(
        bbox=(5.0, 10.0, 30.0, 35.0), landmarks=lm, confidence=0.93, detect_scale=0.5
    )
    assert det.landmarks.shape == (5, 2)
    assert det.confidence == pytest.approx(0.93)


def test_assert_canonical_rejects_swapped_eyes():
    # Right eye left of left eye => mirrored order => must raise.
    lm = np.array([[30, 20], [10, 20], [20, 30], [12, 40], [28, 40]], dtype=np.float32)
    with pytest.raises(ValueError, match="landmark order"):
        assert_canonical_landmarks(lm)


def test_assert_canonical_rejects_bad_shape():
    with pytest.raises(ValueError, match="5x2"):
        assert_canonical_landmarks(np.zeros((4, 2), dtype=np.float32))
