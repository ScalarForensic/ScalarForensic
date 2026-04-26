"""NumPy triplet scoring for query images against indexed Tag references.

Used by ``POST /api/triage/query-images`` to evaluate uncommitted session
embeddings against a Tag's positive and negative reference vectors without
sending those vectors through Qdrant.  Only DINOv2 vectors are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scalar_forensic.discovery import MAX_CONTEXT_PAIRS, pair_indices

__all__ = [
    "MAX_CONTEXT_PAIRS",
    "QueryEvalHit",
    "score_query_entries",
    "score_query_vector",
]


@dataclass
class QueryEvalHit:
    """Ranked result for one uploaded query image evaluated against a Tag.

    ``raw_score`` is the cosine similarity to the best positive reference.
    Mirrors the Recommend-mode meaning of :attr:`DiscoveryHit.raw_score`
    so both paths expose the same field to callers.
    """

    file_id: str
    filename: str
    triplet_score: int | None = None
    raw_score: float = 0.0


def _cosine_sims(query: list[float], refs: list[list[float]]) -> np.ndarray:
    """Cosine similarities between one query vector and all ref vectors."""
    q = np.array(query, dtype=np.float32)
    r = np.array(refs, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn < 1e-12:
        return np.zeros(len(refs), dtype=np.float32)
    rn = np.linalg.norm(r, axis=1)
    return (r @ q) / (rn * qn + 1e-12)


# Re-exported for backward compatibility — the canonical implementation
# lives in :mod:`scalar_forensic.discovery` so the Qdrant and NumPy paths
# cannot drift in pair ordering.
_pair_indices = pair_indices


def score_query_vector(
    query_vec: list[float],
    pos_vecs: list[list[float]],
    neg_vecs: list[list[float]],
) -> tuple[int | None, float]:
    """Return ``(triplet_score, raw_score)`` for a single query vector.

    *raw_score* is the max cosine similarity to any positive reference.
    *triplet_score* is ``None`` when the tag has no negatives.
    """
    if not pos_vecs:
        return None, 0.0
    pos_sims = _cosine_sims(query_vec, pos_vecs)
    raw_score = float(pos_sims.max())
    if not neg_vecs:
        return None, raw_score
    neg_sims = _cosine_sims(query_vec, neg_vecs)
    pairs = pair_indices(len(pos_vecs), len(neg_vecs))
    score = int(sum(1 for pi, ni in pairs if pos_sims[pi] > neg_sims[ni]))
    return score, raw_score


def score_query_entries(
    entries: list[tuple[str, str, list[float] | None]],
    pos_vecs: list[list[float]],
    neg_vecs: list[list[float]],
    limit: int = 50,
) -> list[QueryEvalHit]:
    """Score ``(file_id, filename, dino_vec)`` entries against dino references.

    Returns hits sorted by triplet score then raw score, limited to *limit*.

    Filtering logic:
    - Discovery mode (negatives present): entries with ts == 0 are included —
      they are on the fully-benign side of the boundary and useful for review.
      Only entries where *both* ts == 0 and rs == 0 (orthogonal to all positives)
      are excluded as genuinely uninformative.
    - Recommend mode (no negatives, ts is None): entries are included when
      rs > 0, i.e. there is some positive-side cosine signal.
    """
    results: list[QueryEvalHit] = []
    for file_id, filename, dino_vec in entries:
        if dino_vec is None:
            continue
        ts, rs = score_query_vector(dino_vec, pos_vecs, neg_vecs)
        if (ts is not None and ts > 0) or rs > 0:
            results.append(
                QueryEvalHit(file_id=file_id, filename=filename, triplet_score=ts, raw_score=rs)
            )

    results.sort(key=lambda h: (h.triplet_score or 0, h.raw_score), reverse=True)
    return results[:limit]
