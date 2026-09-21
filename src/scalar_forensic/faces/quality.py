"""Two-stage quality gate (spec §6.2) — the primary false-positive lever.

Pre-alignment: detector confidence, size, pose-from-landmarks (cheap).
Post-alignment: sharpness and exposure on the native-resolution source
crop — NOT the 112x112 resample, whose Laplacian variance mostly
re-encodes the resize factor.

All thresholds are bootstrap values passed by the caller; the Phase 1b
face-calibration record supersedes them (spec §10.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from scalar_forensic.faces.types import FaceDetection


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None = None
    subscores: dict[str, float] = field(default_factory=dict)


def pose_ratio(landmarks: np.ndarray) -> float:
    """Yaw proxy: nose offset along the eye axis, divided by eye separation.

    Using the face's eye axis makes the score invariant to in-plane rotation.
    This remains a coarse profile gate, not an angle estimate.
    """
    left_eye, right_eye, nose = np.asarray(landmarks[:3], dtype=np.float64)
    eye_axis = right_eye - left_eye
    span_squared = float(eye_axis @ eye_axis)
    if span_squared <= 1.0:
        return 1.0
    offset = nose - (left_eye + right_eye) / 2.0
    ratio = float(abs(offset @ eye_axis) / span_squared)
    return ratio if np.isfinite(ratio) else 1.0


def pre_align_gate(
    det: FaceDetection, *, min_conf: float, min_size: int, max_pose: float
) -> GateResult:
    min_side_input_px = min(det.bbox[2], det.bbox[3]) * det.detect_scale
    pose = pose_ratio(det.landmarks)
    subs = {"confidence": det.confidence, "size": min_side_input_px, "pose": pose}
    if det.confidence < min_conf:
        return GateResult(False, "confidence", subs)
    if min_side_input_px < min_size:
        return GateResult(False, "size", subs)
    if pose > max_pose:
        return GateResult(False, "pose", subs)
    return GateResult(True, None, subs)


def review_gate(det: FaceDetection, *, min_conf: float, min_size: int) -> GateResult:
    """Admit a detection for hand review (spec: gate-split design).

    Deliberately weaker than pre_align_gate: no pose check.  Pose degrades an
    *alignment*, and this path never aligns — the examiner looks at the
    unwarped source crop, where a profile face is perfectly examinable.
    """
    min_side_input_px = min(det.bbox[2], det.bbox[3]) * det.detect_scale
    subs = {"confidence": det.confidence, "size": min_side_input_px}
    if det.confidence < min_conf:
        return GateResult(False, "confidence", subs)
    if min_side_input_px < min_size:
        return GateResult(False, "size", subs)
    return GateResult(True, None, subs)


def post_align_gate(
    source_crop_gray: np.ndarray, *, min_sharpness: float, max_clipped_frac: float
) -> GateResult:
    sharpness = float(cv2.Laplacian(source_crop_gray, cv2.CV_64F).var())
    clipped = float(np.mean((source_crop_gray <= 2) | (source_crop_gray >= 253)))
    subs = {"sharpness": sharpness, "exposure": clipped}
    # Exposure before sharpness: a clipped strobe frame can have huge
    # Laplacian variance and would otherwise pass as "sharp".
    if clipped > max_clipped_frac:
        return GateResult(False, "exposure", subs)
    if sharpness < min_sharpness:
        return GateResult(False, "sharpness", subs)
    return GateResult(True, None, subs)
