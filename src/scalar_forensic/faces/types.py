"""Shared detection types and the canonical landmark contract.

Canonical landmark order (spec §6.1): left eye, right eye, nose tip,
left mouth corner, right mouth corner — "left" meaning image-left
(the subject's right side).  Every detector adapter MUST reorder its
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
    """Sanity-check the canonical ordering invariants that survive pose.

    Checks are deliberately loose (faces rotate) but catch the two silent
    killers: swapped eye pair and swapped mouth pair.
    """
    if lm.shape != (5, 2):
        raise ValueError(f"landmarks must be 5x2, got {lm.shape}")
    if lm[0, 0] >= lm[1, 0]:
        raise ValueError("bad landmark order: left eye is not left of right eye")
    if lm[3, 0] >= lm[4, 0]:
        raise ValueError("bad landmark order: left mouth is not left of right mouth")
