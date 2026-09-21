"""Analysis and query pipeline for the web frontend.

Imports from the existing backend — no modifications to those modules.
"""

from __future__ import annotations

from scalar_forensic.web.pipeline.analysis import (
    ProgressEvent,
    _video_frame_batch,
    analyze_session,
)
from scalar_forensic.web.pipeline.faces_query import (
    detect_query_faces,
    query_embedder_block,
)
from scalar_forensic.web.pipeline.faces_search import (
    MODEL_REFERENCE_NOTE,
    MODEL_REFERENCE_THRESHOLD,
    calibration_block,
    search_query_faces,
)
from scalar_forensic.web.pipeline.faces_stats import (
    face_audit,
    face_score_stats,
)
from scalar_forensic.web.pipeline.modes import get_available_modes
from scalar_forensic.web.pipeline.provenance import get_hit_qdrant_provenance
from scalar_forensic.web.pipeline.query import (
    Hit,
    MatchedVideoFrame,
    QueryProvenance,
    _group_video_hits,
    _merge_hit,
    _query_exact_video,
    _query_vector,
    query_session,
)
from scalar_forensic.web.pipeline.stats import query_semantic_stats

__all__ = [
    "MODEL_REFERENCE_NOTE",
    "MODEL_REFERENCE_THRESHOLD",
    "Hit",
    "MatchedVideoFrame",
    "ProgressEvent",
    "QueryProvenance",
    "_group_video_hits",
    "_merge_hit",
    "_query_exact_video",
    "_query_vector",
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
