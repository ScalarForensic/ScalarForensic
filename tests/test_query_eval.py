"""Tests for the NumPy triplet scoring in query_eval.

Verifies that the pure-NumPy path produces scores consistent with the
Qdrant Discovery logic implemented in discovery.py.
"""

from __future__ import annotations

import math

import pytest

from scalar_forensic.discovery import _build_context_pairs
from scalar_forensic.query_eval import (
    MAX_CONTEXT_PAIRS,
    _cosine_sims,
    pair_indices,
    score_query_entries,
    score_query_vector,
)


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ── _cosine_sims ─────────────────────────────────────────────────────────────


def test_cosine_sims_identical_vectors_score_one():
    q = _unit([1.0, 0.0, 0.0])
    refs = [_unit([1.0, 0.0, 0.0])]
    sims = _cosine_sims(q, refs)
    assert abs(sims[0] - 1.0) < 1e-5


def test_cosine_sims_orthogonal_vectors_score_zero():
    q = _unit([1.0, 0.0])
    refs = [_unit([0.0, 1.0])]
    sims = _cosine_sims(q, refs)
    assert abs(sims[0]) < 1e-5


def test_cosine_sims_zero_query_returns_zeros():
    q = [0.0, 0.0, 0.0]
    refs = [[1.0, 0.0, 0.0]]
    sims = _cosine_sims(q, refs)
    assert sims[0] == 0.0


# ── pair_indices ─────────────────────────────────────────────────────────────


def testpair_indices_covers_all_pairs_under_cap():
    pairs = pair_indices(3, 2)
    assert len(pairs) == 6  # 3 × 2 = 6, below cap
    assert set(pairs) == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}


def testpair_indices_caps_at_max():
    pairs = pair_indices(10, 10)
    assert len(pairs) == MAX_CONTEXT_PAIRS


def testpair_indices_diagonal_first():
    # First min(n_pos, n_neg) pairs should be (i, i) for i in range(short)
    pairs = pair_indices(3, 3)
    assert pairs[0] == (0, 0)
    assert pairs[1] == (1, 1)
    assert pairs[2] == (2, 2)


# ── cross-module parity ──────────────────────────────────────────────────────
# The NumPy path (pair_indices) and the Qdrant path (_build_context_pairs)
# MUST emit pairs in the same order: the Qdrant server scores the Nth pair
# using the Nth (positive, negative) reference on its side, and our NumPy
# mirror uses the same indices to score uploaded-but-unindexed images.  Any
# drift between the two orderings silently changes scoring semantics.


@pytest.mark.parametrize(
    "n_pos, n_neg",
    [(1, 1), (1, 5), (5, 1), (3, 3), (4, 7), (10, 10), (16, 16)],
)
def test_pair_ordering_matches_discovery_builder(n_pos, n_neg):
    positives = [f"p{i}" for i in range(n_pos)]
    negatives = [f"n{i}" for i in range(n_neg)]

    # Discovery emits ContextPair objects; dereference to (positive, negative) strings.
    discovery_pairs = [(p.positive, p.negative) for p in _build_context_pairs(positives, negatives)]

    # NumPy emits index tuples; map via positives/negatives to the same string pairs.
    numpy_pairs = [(positives[pi], negatives[ni]) for pi, ni in pair_indices(n_pos, n_neg)]

    assert numpy_pairs == discovery_pairs, (
        "pair ordering drift: _build_context_pairs (discovery.py) and "
        "pair_indices (query_eval.py) must emit pairs in the same order"
    )


# ── score_query_vector ────────────────────────────────────────────────────────


def test_score_query_vector_all_pairs_satisfied():
    # Query is identical to positive, far from negative
    pos = [_unit([1.0, 0.0, 0.0])]
    neg = [_unit([0.0, 1.0, 0.0])]
    query = _unit([1.0, 0.0, 0.0])
    score, margin = score_query_vector(query, pos, neg)
    assert score == 1
    assert margin > 0.9


def test_score_query_vector_no_pairs_satisfied():
    # Query is closer to negative
    pos = [_unit([0.0, 1.0, 0.0])]
    neg = [_unit([1.0, 0.0, 0.0])]
    query = _unit([1.0, 0.0, 0.0])
    score, margin = score_query_vector(query, pos, neg)
    assert score == 0


def test_score_query_vector_no_negatives_returns_none_score():
    pos = [_unit([1.0, 0.0, 0.0])]
    query = _unit([0.8, 0.6, 0.0])
    score, margin = score_query_vector(query, pos, [])
    assert score is None
    assert margin > 0


def test_score_query_vector_no_positives_returns_none_zero():
    neg = [_unit([1.0, 0.0, 0.0])]
    query = _unit([1.0, 0.0, 0.0])
    score, margin = score_query_vector(query, [], neg)
    assert score is None
    assert margin == 0.0


def test_score_query_vector_three_pairs_two_satisfied():
    # 2 positives, 1 negative → 2 pairs
    pos = [_unit([1.0, 0.0]), _unit([0.9, 0.1])]
    neg = [_unit([0.0, 1.0])]
    query = _unit([1.0, 0.0])  # close to both positives, far from negative
    score, _ = score_query_vector(query, pos, neg)
    assert score == 2


# ── score_query_entries ───────────────────────────────────────────────────────


def test_score_query_entries_ranks_by_triplet_score():
    pos_dino = [_unit([1.0, 0.0])]
    neg_dino = [_unit([0.0, 1.0])]

    # File A: close to positive → triplet_score=1
    # File B: close to negative → triplet_score=0, lower margin
    entries = [
        ("fileB", "b.jpg", _unit([0.0, 1.0])),  # far from pos
        ("fileA", "a.jpg", _unit([1.0, 0.0])),  # close to pos
    ]
    hits = score_query_entries(entries, pos_dino, neg_dino, limit=10)
    assert hits[0].file_id == "fileA"
    assert hits[0].triplet_score == 1


def test_score_query_entries_excludes_zero_score():
    """Entries orthogonal to every positive (ts=0, cm=0) must not be returned."""
    pos_dino = [_unit([1.0, 0.0])]
    neg_dino = [_unit([0.0, 1.0])]
    entries = [
        # Orthogonal to pos: raw_score == 0; closer to neg: triplet_score == 0.
        ("fileA", "a.jpg", _unit([0.0, 1.0])),
    ]
    hits = score_query_entries(entries, pos_dino, neg_dino, limit=10)
    assert hits == [], "zero-score entries must be excluded from the result list"


def test_score_query_entries_respects_limit():
    pos_dino = [_unit([1.0, 0.0])]
    neg_dino = [_unit([0.0, 1.0])]
    entries = [(f"file{i}", f"f{i}.jpg", _unit([1.0, 0.0])) for i in range(10)]
    hits = score_query_entries(entries, pos_dino, neg_dino, limit=3)
    assert len(hits) <= 3


def test_score_query_entries_skips_entries_without_dino_vec():
    pos_dino = [_unit([1.0, 0.0])]
    neg_dino = [_unit([0.0, 1.0])]
    entries = [("f", "f.jpg", None)]
    hits = score_query_entries(entries, pos_dino, neg_dino, limit=10)
    assert hits == []
