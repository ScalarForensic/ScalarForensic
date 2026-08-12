import json
from pathlib import Path

import numpy as np
from PIL import Image

from scalar_forensic.faces.align import (
    ALIGNED_SIZE,
    ALIGNMENT_VERSION,
    ARCFACE_DST,
    align_face,
    umeyama,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "faces"


def test_reference_points_match_insightface_arcface_dst():
    expected = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(ARCFACE_DST, expected, atol=1e-4)
    assert ALIGNMENT_VERSION == "arcface-112-v1"
    assert ALIGNED_SIZE == 112


def test_umeyama_recovers_a_known_similarity_transform():
    rng = np.random.default_rng(7)
    src = rng.uniform(0, 100, size=(5, 2))
    theta, scale, tx, ty = 0.3, 1.7, 12.0, -5.0
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = scale * src @ rot.T + [tx, ty]
    m = umeyama(src, dst)  # float64 in — float32 downcast costs ~6e-6 and fails tighter tols
    src_h = np.hstack([src, np.ones((5, 1))])
    np.testing.assert_allclose(src_h @ m.T, dst, atol=1e-4)


def test_umeyama_never_reflects():
    # Mirrored destination must still produce det(R) > 0 (reflection corrected).
    src = ARCFACE_DST.astype(np.float64)
    dst = src.copy()
    dst[:, 0] = 112 - dst[:, 0]
    m = umeyama(src, dst)
    assert np.linalg.det(m[:, :2]) > 0


def test_identity_landmarks_produce_identity_crop():
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255
    out = align_face(img, ARCFACE_DST.copy())
    assert out.shape == (112, 112, 3)
    diff = np.abs(out.astype(int) - img.astype(int))
    assert diff.max() <= 1  # warp with ~identity matrix; allow 1 ULP of interpolation


def test_umeyama_matrix_matches_independent_reference():
    # Reference derived via a genuinely different formulation: the optimal
    # non-reflective 2-D similarity is w = a*z + b over complex numbers,
    # solved with plain lstsq (no SVD, no Umeyama).  Both minimise the same
    # objective, so the optima coincide; literals computed once, offline.
    src = np.array([[260.0, 210.0], [340.0, 205.0], [300.0, 260.0], [270.0, 310.0], [335.0, 305.0]])
    m = umeyama(src, ARCFACE_DST.astype(np.float64))
    reference = np.array(
        [
            [0.4187296417, -0.0148068730, -66.1912889251],
            [0.0148068730, 0.4187296417, -40.5883363192],
        ]
    )
    np.testing.assert_allclose(m, reference, atol=1e-6)


def test_golden_fixture():
    meta = json.loads((FIXTURES / "golden_landmarks.json").read_text())
    src_img = np.array(Image.open(FIXTURES / meta["source_png"]))
    lm = np.array(meta["landmarks"], dtype=np.float32)
    expected = np.array(Image.open(FIXTURES / "golden_aligned.png"))
    out = align_face(src_img, lm)
    assert np.abs(out.astype(int) - expected.astype(int)).max() <= 1
