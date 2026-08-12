"""Face detectors.  Default: YuNet via cv2.FaceDetectorYN (local ONNX, no network).

Adapter contract (spec §6.1): output in oriented source pixels, landmarks
reordered into the canonical order, RGB in / BGR handled internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from scalar_forensic.embedder import hash_file
from scalar_forensic.faces.types import FaceDetection, assert_canonical_landmarks

# Map from YuNet's native landmark column order to the canonical order.
# DERIVED EMPIRICALLY against the real 2023mar model on 10 real faces from
# data/sample_images (varied pose and scale): column 0 was left of column 1
# and column 3 left of column 4 in every case.  OpenCV documents these as
# the *subject's* right eye/mouth corner, which is image-left — so the
# identity map is already canonical.  assert_canonical_landmarks in detect()
# keeps guarding it at runtime; see tests/faces/test_detect.py
# ::test_real_model_landmark_order_on_real_face.
_YUNET_TO_CANONICAL = [0, 1, 2, 3, 4]


def _create_yunet(model_path: str, score_threshold: float):
    """Seam for tests — cv2's extension types cannot be mock-patched directly."""
    return cv2.FaceDetectorYN.create(model_path, "", (0, 0), score_threshold=score_threshold)


class FaceDetector(Protocol):
    def detect(self, img_rgb: np.ndarray) -> list[FaceDetection]: ...

    @property
    def detector_id(self) -> str: ...

    @property
    def model_hash(self) -> str: ...


def _scaled_size(w: int, h: int, max_size: int) -> tuple[int, int, float]:
    long_side = max(w, h)
    if long_side <= max_size:
        return w, h, 1.0
    scale = max_size / long_side
    return round(w * scale), round(h * scale), scale


class YuNetDetector:
    def __init__(self, model_path: Path, max_size: int, score_threshold: float = 0.5) -> None:
        self._model_path = model_path
        self._max_size = max_size
        self._model_hash = hash_file(model_path)
        self._score_threshold = score_threshold  # recorded in provenance (PipelineConfig)
        self.n_dropped_noncanonical = 0
        self._net = _create_yunet(str(model_path), score_threshold)

    @property
    def detector_id(self) -> str:
        return "yunet"

    @property
    def model_hash(self) -> str:
        return self._model_hash

    def detect(self, img_rgb: np.ndarray) -> list[FaceDetection]:
        h, w = img_rgb.shape[:2]
        dw, dh, scale = _scaled_size(w, h, self._max_size)
        small = (
            img_rgb if scale == 1.0 else cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        )
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        self._net.setInputSize((dw, dh))
        _, rows = self._net.detect(bgr)
        if rows is None or len(rows) == 0:
            return []
        out: list[FaceDetection] = []
        for row in rows:
            x, y, bw, bh = (float(v) / scale for v in row[0:4])
            native = row[4:14].reshape(5, 2) / scale
            lm = native[_YUNET_TO_CANONICAL].astype(np.float32)
            # Runtime guard against a wrong _YUNET_TO_CANONICAL map: drop
            # non-canonical output rather than raise — one legitimately rotated
            # face must not crash a whole indexing run.  A wholesale-wrong map
            # shows up as ~100% "landmark_order" rejections in the marker stats
            # and CLI summary, which is loud enough to catch immediately.
            try:
                assert_canonical_landmarks(lm)
            except ValueError:
                self.n_dropped_noncanonical += 1
                continue
            out.append(
                FaceDetection(
                    bbox=(x, y, bw, bh),
                    landmarks=lm,
                    confidence=float(row[14]),
                    detect_scale=scale,
                )
            )
        return out
