"""Pipeline provenance recorded on every face point (spec §7.1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

_VERSION_INFO_FIELDS = {"sfn_version", "cv2_version", "ort_version"}


@dataclass(frozen=True)
class PipelineConfig:
    detector_id: str
    detector_model_hash: str
    detector_score_threshold: float
    detect_max_size: int
    embedder_model_name: str
    embedder_model_hash: str
    manifest_hash: str
    embedder_dim: int
    alignment_version: str
    normalization_id: str
    min_conf: float
    min_size: int
    min_sharpness: float
    max_clipped: float
    max_pose: float
    crop_dilation: float
    sfn_version: str
    cv2_version: str
    ort_version: str

    @property
    def config_hash(self) -> str:
        # Version-info fields are recorded but excluded from the hash:
        # a library upgrade alone must not orphan a collection.  Hard
        # comparability is enforced field-by-field in safeguards.
        hashed = {k: v for k, v in asdict(self).items() if k not in _VERSION_INFO_FIELDS}
        canon = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    def to_payload(self) -> dict:
        return {**asdict(self), "pipeline_config_hash": self.config_hash}
