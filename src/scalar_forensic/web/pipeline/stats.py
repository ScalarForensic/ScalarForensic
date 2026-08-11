"""Semantic score distribution statistics for a single uploaded file."""

from __future__ import annotations

import logging
import statistics as _statistics
from dataclasses import dataclass

from qdrant_client import QdrantClient

from scalar_forensic.config import Settings
from scalar_forensic.web.session import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic score distribution stats (on-demand, per uploaded file)
# ---------------------------------------------------------------------------

_STATS_SAMPLE = 10_000
_HIST_BUCKETS = 20  # 0.05-wide buckets covering normalised [0.0, 1.0] (cosine [-1,1] → [0,1])


@dataclass
class SemanticStats:
    sample_size: int  # how many points were requested
    count: int  # how many were actually returned
    min_score: float
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    max_score: float
    mean: float
    stdev: float
    # _HIST_BUCKETS counts on normalised [0,1] scale; bucket i covers [i*0.05, (i+1)*0.05)
    histogram: list[int]


def query_semantic_stats(
    session: Session,
    file_id: str,
    settings: Settings,
    sample_size: int = _STATS_SAMPLE,
) -> tuple[SemanticStats | None, str | None]:
    """Return score-distribution stats for one uploaded file using the DINOv2 named vector.

    Queries the top-*sample_size* most-similar points (using="dino") with no score threshold
    so the result covers the full relevant tail.  Returns (stats, None) on success or
    (None, error_message) on failure.
    """
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None:
        return None, "file not found in session"
    if entry.error:
        return None, f"analysis failed: {entry.error}"
    if not entry.dino_embedding:
        return None, "no semantic embedding — was semantic mode selected during analysis?"

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    try:
        result = client.query_points(
            collection_name=settings.collection,
            query=entry.dino_embedding,
            using="dino",
            limit=sample_size,
            with_payload=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic stats query failed: %s", exc)
        return None, f"Qdrant query failed: {type(exc).__name__}"

    scores = [r.score for r in result.points]
    if not scores:
        return None, "no points returned from collection (collection may be empty)"

    n = len(scores)
    mean = sum(scores) / n
    stdev = _statistics.stdev(scores) if n >= 2 else 0.0

    if n >= 2:
        cuts = _statistics.quantiles(scores, n=100, method="inclusive")
        p10 = cuts[9]
        p25 = cuts[24]
        median = cuts[49]
        p75 = cuts[74]
        p90 = cuts[89]
    else:
        p10 = p25 = median = p75 = p90 = scores[0]

    histogram = [0] * _HIST_BUCKETS
    for s in scores:
        # Normalise cosine score from [-1, 1] to [0, 1] before bucketing.
        # Note: the numeric stats (min/max/percentiles) are on the raw cosine scale;
        # the histogram intentionally uses the normalised scale so the UI threshold
        # marker (also normalised before bucket lookup) aligns correctly.
        normalised = (max(-1.0, min(s, 1.0)) + 1.0) / 2.0
        idx = min(max(int(normalised * _HIST_BUCKETS), 0), _HIST_BUCKETS - 1)
        histogram[idx] += 1

    return SemanticStats(
        sample_size=sample_size,
        count=n,
        min_score=min(scores),
        p10=p10,
        p25=p25,
        median=median,
        p75=p75,
        p90=p90,
        max_score=max(scores),
        mean=mean,
        stdev=stdev,
        histogram=histogram,
    ), None
