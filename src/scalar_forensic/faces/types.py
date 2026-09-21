"""Shared detection types and the canonical landmark contract.

Canonical landmark order (spec §6.1): left eye, right eye, nose tip,
left mouth corner, right mouth corner — "left" meaning image-left on an
upright face (the subject's right side), retained as the face rotates.
Every detector adapter MUST reorder its
native output into this order; a swapped eye pair produces a mirrored
alignment that silently degrades matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LANDMARK_ORDER: tuple[str, ...] = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_mouth",
    "right_mouth",
)


# eq=False: the default dataclass __eq__/__hash__ choke on the ndarray field
# ("truth value of an array is ambiguous"); identity comparison is all we need.
@dataclass(frozen=True, eq=False)
class FaceDetection:
    """One detected face in oriented source-pixel coordinates."""

    bbox: tuple[float, float, float, float]  # x, y, w, h
    landmarks: np.ndarray  # (5, 2) float32, canonical order
    confidence: float
    detect_scale: float  # source px * detect_scale = detector-input px


def assert_canonical_landmarks(lm: np.ndarray) -> None:
    """Reject mirrored/degenerate ordering without rejecting in-plane rotation."""
    if lm.shape != (5, 2):
        raise ValueError(f"landmarks must be 5x2, got {lm.shape}")
    if not np.isfinite(lm).all():
        raise ValueError("landmarks must be finite")
    eye_to_mouth = (lm[3] + lm[4] - lm[0] - lm[1]) / 2.0
    # In image coordinates, each left-to-right pair and the eye-to-mouth
    # direction have positive winding. Rotation preserves it; swapping a
    # pair reverses it. Do not sort by x: landmark identities must survive roll.
    for name, axis in (("eyes", lm[1] - lm[0]), ("mouth", lm[4] - lm[3])):
        winding = axis[0] * eye_to_mouth[1] - axis[1] * eye_to_mouth[0]
        if winding <= 0:
            raise ValueError(f"bad landmark order: {name} are mirrored or degenerate")
