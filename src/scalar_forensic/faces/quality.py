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
    """Yaw proxy: horizontal nose offset from the eye midpoint over eye span.

    0.0 = frontal; grows toward ±1 in profile.  Coarse by design — its job
    is rejecting strong profiles, not estimating angles.
    """
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_span = float(right_eye[0] - left_eye[0])
    if eye_span <= 1.0:
        return 1.0
    mid_x = (left_eye[0] + right_eye[0]) / 2.0
    return float(abs(nose[0] - mid_x) / eye_span)


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
