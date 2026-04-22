"""NumPy triplet scoring for query images against indexed Tag references.

Used by ``POST /api/triage/query-images`` to evaluate uncommitted session
embeddings against a Tag's positive and negative reference vectors without
sending those vectors through Qdrant.  Only DINOv2 vectors are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Must match scalar_forensic.discovery._MAX_CONTEXT_PAIRS.
_MAX_CONTEXT_PAIRS = 64


@dataclass
class QueryEvalHit:
    """Ranked result for one uploaded query image evaluated against a Tag."""

    file_id: str
    filename: str
    triplet_score: int | None = None
    cosine_margin: float = 0.0


def _cosine_sims(query: list[float], refs: list[list[float]]) -> np.ndarray:
    """Cosine similarities between one query vector and all ref vectors."""
    q = np.array(query, dtype=np.float32)
    r = np.array(refs, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn < 1e-12:
        return np.zeros(len(refs), dtype=np.float32)
    rn = np.linalg.norm(r, axis=1)
    return (r @ q) / (rn * qn + 1e-12)


def _pair_indices(n_pos: int, n_neg: int) -> list[tuple[int, int]]:
    """Return pair indices in the same diagonal-first order as _build_context_pairs."""
    limit = min(n_pos * n_neg, _MAX_CONTEXT_PAIRS)
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    short = min(n_pos, n_neg)
    for i in range(short):
        seen.add((i, i))
        result.append((i, i))
        if len(result) >= limit:
            return result
    for p in range(n_pos):
        for n in range(n_neg):
            if (p, n) in seen:
                continue
            result.append((p, n))
            if len(result) >= limit:
                return result
    return result


def score_query_vector(
    query_vec: list[float],
    pos_vecs: list[list[float]],
    neg_vecs: list[list[float]],
) -> tuple[int | None, float]:
    """Return ``(triplet_score, cosine_margin)`` for a single query vector.

    *cosine_margin* is the max cosine similarity to any positive reference.
    *triplet_score* is ``None`` when the tag has no negatives.
    """
    if not pos_vecs:
        return None, 0.0
    pos_sims = _cosine_sims(query_vec, pos_vecs)
    cosine_margin = float(pos_sims.max())
    if not neg_vecs:
        return None, cosine_margin
    neg_sims = _cosine_sims(query_vec, neg_vecs)
    pairs = _pair_indices(len(pos_vecs), len(neg_vecs))
    score = int(sum(1 for pi, ni in pairs if pos_sims[pi] > neg_sims[ni]))
    return score, cosine_margin


def score_query_entries(
    entries: list[tuple[str, str, list[float] | None, list[float] | None]],
    pos_vecs: list[list[float]],
    neg_vecs: list[list[float]],
    limit: int = 50,
) -> list[QueryEvalHit]:
    """Score ``(file_id, filename, dino_vec, _)`` entries against dino references.

    Returns hits sorted by triplet score then cosine margin, limited to *limit*.
    Entries with triplet_score == 0 and cosine_margin == 0 are excluded.
    """
    results: list[QueryEvalHit] = []
    for file_id, filename, dino_vec, _sscd_vec in entries:
        if dino_vec is None:
            continue
        ts, cm = score_query_vector(dino_vec, pos_vecs, neg_vecs)
        if (ts is not None and ts > 0) or cm > 0:
            results.append(QueryEvalHit(
                file_id=file_id, filename=filename, triplet_score=ts, cosine_margin=cm
            ))

    results.sort(key=lambda h: (h.triplet_score or 0, h.cosine_margin), reverse=True)
    return results[:limit]
