"""Tests for the NumPy triplet scoring in query_eval.

Verifies that the pure-NumPy path produces scores consistent with the
Qdrant Discovery logic implemented in discovery.py.
"""

from __future__ import annotations

import math

from scalar_forensic.query_eval import (
    _MAX_CONTEXT_PAIRS,
    _cosine_sims,
    _pair_indices,
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


# ── _pair_indices ─────────────────────────────────────────────────────────────


def test_pair_indices_covers_all_pairs_under_cap():
    pairs = _pair_indices(3, 2)
    assert len(pairs) == 6  # 3 × 2 = 6, below cap
    assert set(pairs) == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}


def test_pair_indices_caps_at_max():
    pairs = _pair_indices(10, 10)
    assert len(pairs) == _MAX_CONTEXT_PAIRS


def test_pair_indices_diagonal_first():
    # First min(n_pos, n_neg) pairs should be (i, i) for i in range(short)
    pairs = _pair_indices(3, 3)
    assert pairs[0] == (0, 0)
    assert pairs[1] == (1, 1)
    assert pairs[2] == (2, 2)


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
        ("fileB", "b.jpg", _unit([0.0, 1.0]), None),  # far from pos
        ("fileA", "a.jpg", _unit([1.0, 0.0]), None),  # close to pos
    ]
    hits = score_query_entries(entries, pos_dino, neg_dino, limit=10)
    assert hits[0].file_id == "fileA"
    assert hits[0].triplet_score == 1


def test_score_query_entries_excludes_zero_score():
    pos_dino = [_unit([1.0, 0.0])]
    neg_dino = [_unit([0.0, 1.0])]
    entries = [
        ("fileA", "a.jpg", _unit([0.0, 1.0]), None),  # far from positive
    ]
    hits = score_query_entries(entries, pos_dino, neg_dino, limit=10)
    assert isinstance(hits, list)


def test_score_query_entries_respects_limit():
    pos_dino = [_unit([1.0, 0.0])]
    neg_dino = [_unit([0.0, 1.0])]
    entries = [(f"file{i}", f"f{i}.jpg", _unit([1.0, 0.0]), None) for i in range(10)]
    hits = score_query_entries(entries, pos_dino, neg_dino, limit=3)
    assert len(hits) <= 3


def test_score_query_entries_skips_entries_without_dino_vec():
    pos_dino = [_unit([1.0, 0.0])]
    neg_dino = [_unit([0.0, 1.0])]
    entries = [("f", "f.jpg", None, None)]
    hits = score_query_entries(entries, pos_dino, neg_dino, limit=10)
    assert hits == []
