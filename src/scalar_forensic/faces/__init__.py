"""Optional face modality (spec: docs/specs/face-pipeline.md).

Core modules never import this package at module level; availability is
probed via scalar_forensic.faces availability helpers.
"""

from scalar_forensic.faces.types import LANDMARK_ORDER, FaceDetection, assert_canonical_landmarks

__all__ = ["LANDMARK_ORDER", "FaceDetection", "assert_canonical_landmarks"]
