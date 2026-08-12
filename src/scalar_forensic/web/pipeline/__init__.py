"""Analysis and query pipeline for the web frontend.

Imports from the existing backend — no modifications to those modules.
"""

from __future__ import annotations

from scalar_forensic.web.pipeline._embedders import _embedder_cache, _get_embedder
from scalar_forensic.web.pipeline.analysis import (
    ProgressEvent,
    _analyze_file,
    _analyze_video_file,
    _video_frame_batch,
    analyze_session,
)
from scalar_forensic.web.pipeline.faces_query import (
    QueryFaceResult,
    detect_query_faces,
    query_embedder_block,
)
from scalar_forensic.web.pipeline.faces_search import (
    MODEL_REFERENCE_NOTE,
    MODEL_REFERENCE_THRESHOLD,
    UNCALIBRATED_BANNER,
    calibration_block,
    search_query_faces,
)
from scalar_forensic.web.pipeline.faces_stats import (
    FaceScoreStats,
    face_audit,
    face_score_stats,
)
from scalar_forensic.web.pipeline.modes import get_available_modes
from scalar_forensic.web.pipeline.provenance import (
    _PROVENANCE_FIELD_NAMES,
    _VECTOR_MODE_MAP,
    _payload_model_provenance,
    get_hit_qdrant_provenance,
)
from scalar_forensic.web.pipeline.query import (
    _MODE_PRIORITY,
    FileResult,
    Hit,
    MatchedVideoFrame,
    QueryProvenance,
    _group_video_hits,
    _hit_sort_key,
    _merge_hit,
    _query_exact,
    _query_exact_video,
    _query_vector,
    _unmerged_sort_key,
    query_session,
)
from scalar_forensic.web.pipeline.stats import (
    _HIST_BUCKETS,
    _STATS_SAMPLE,
    SemanticStats,
    query_semantic_stats,
)

__all__ = [
    "_HIST_BUCKETS",
    "_MODE_PRIORITY",
    "_PROVENANCE_FIELD_NAMES",
    "_STATS_SAMPLE",
    "_VECTOR_MODE_MAP",
    "MODEL_REFERENCE_NOTE",
    "MODEL_REFERENCE_THRESHOLD",
    "UNCALIBRATED_BANNER",
    "FaceScoreStats",
    "FileResult",
    "Hit",
    "MatchedVideoFrame",
    "ProgressEvent",
    "QueryFaceResult",
    "QueryProvenance",
    "SemanticStats",
    "_analyze_file",
    "_analyze_video_file",
    "_embedder_cache",
    "_get_embedder",
    "_group_video_hits",
    "_hit_sort_key",
    "_merge_hit",
    "_payload_model_provenance",
    "_query_exact",
    "_query_exact_video",
    "_query_vector",
    "_unmerged_sort_key",
    "_video_frame_batch",
    "analyze_session",
    "calibration_block",
    "detect_query_faces",
    "face_audit",
    "face_score_stats",
    "get_available_modes",
    "get_hit_qdrant_provenance",
    "query_embedder_block",
    "query_semantic_stats",
    "search_query_faces",
    "query_session",
]
