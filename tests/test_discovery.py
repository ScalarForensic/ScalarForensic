"""Tests for the Qdrant Discovery / Recommend query builder.

Verifies that :mod:`scalar_forensic.discovery` picks the correct query
kind (Discovery with context pairs vs Recommend with positives only),
handles reverse polarity, excludes the concept's own reference points,
and fuses dual-vector rankings so items agreed on by both embedding
spaces outrank items from only one.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client.models import (
    ContextPair,
    ContextQuery,
    DiscoverQuery,
    Filter,
    HasIdCondition,
    RecommendQuery,
)

from scalar_forensic.concepts import Concept
from scalar_forensic.discovery import (
    _MAX_CONTEXT_PAIRS,
    _build_context_pairs,
    run_discovery,
    run_discovery_dual,
    run_triage,
)


def _concept(
    *,
    positives: list[str] | None = None,
    negatives: list[str] | None = None,
    target: str | None = None,
) -> Concept:
    return Concept(
        concept_id="test-concept",
        name="test",
        positive_ids=list(positives or []),
        negative_ids=list(negatives or []),
        target_id=target,
    )


def _client_returning(points: list) -> MagicMock:
    """Build a MagicMock QdrantClient whose query_points returns *points*."""
    client = MagicMock()
    result = MagicMock()
    result.points = points
    client.query_points.return_value = result
    return client


def _pt(pid: str, score: float, *, payload: dict | None = None) -> MagicMock:
    m = MagicMock()
    m.id = pid
    m.score = score
    m.payload = payload or {}
    return m


# ---------------------------------------------------------------------------
# _build_context_pairs
# ---------------------------------------------------------------------------


def test_build_pairs_cartesian_product_when_small():
    pairs = _build_context_pairs(["p1", "p2"], ["n1", "n2"])
    # 2 × 2 = 4 pairs, all under the cap.
    assert len(pairs) == 4
    all_positives = {p.positive for p in pairs}
    all_negatives = {p.negative for p in pairs}
    assert all_positives == {"p1", "p2"}
    assert all_negatives == {"n1", "n2"}


def test_build_pairs_covers_every_reference_under_cap():
    """Every positive and every negative must appear at least once even when the cap bites."""
    positives = [f"p{i}" for i in range(10)]
    negatives = [f"n{i}" for i in range(10)]
    pairs = _build_context_pairs(positives, negatives)
    assert len(pairs) <= _MAX_CONTEXT_PAIRS
    assert {p.positive for p in pairs} == set(positives)
    assert {p.negative for p in pairs} == set(negatives)


def test_build_pairs_empty_when_one_side_missing():
    assert _build_context_pairs([], ["n1"]) == []
    assert _build_context_pairs(["p1"], []) == []


# ---------------------------------------------------------------------------
# run_discovery — query construction
# ---------------------------------------------------------------------------


def test_context_only_without_target_uses_context_query():
    """Concept with pairs but no target ⇒ ContextQuery (not DiscoverQuery).

    ``DiscoverInput.target`` is required and rejects ``None``, so context-only
    triage must go through ``ContextQuery``.  This test locks that contract in.
    """
    client = _client_returning([_pt("hit1", 3.0)])
    concept = _concept(positives=["p1", "p2"], negatives=["n1"])
    hits = run_discovery(
        client, "sfn", concept, vector_name="dino", limit=5
    )
    assert len(hits) == 1
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], ContextQuery)
    # 2 positives × 1 negative = 2 pairs.
    assert len(kwargs["query"].context) == 2
    assert all(isinstance(p, ContextPair) for p in kwargs["query"].context)
    # using=<vec> must be passed so Qdrant picks the right named vector.
    assert kwargs["using"] == "dino"


def test_discovery_with_target_and_pairs_uses_discover_with_target():
    client = _client_returning([])
    concept = _concept(positives=["p1"], negatives=["n1"], target="t1")
    run_discovery(client, "sfn", concept, vector_name="dino", limit=5)
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], DiscoverQuery)
    assert kwargs["query"].discover.target == "t1"


def test_recommend_fallback_when_no_negatives():
    """No negatives ⇒ no pairs ⇒ Recommend query."""
    client = _client_returning([])
    concept = _concept(positives=["p1", "p2"], negatives=[])
    run_discovery(client, "sfn", concept, vector_name="sscd", limit=5)
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], RecommendQuery)
    recommend = kwargs["query"].recommend
    assert list(recommend.positive) == ["p1", "p2"]


def test_recommend_fallback_with_only_target():
    """Only a target, nothing else ⇒ Recommend with target as the single positive."""
    client = _client_returning([])
    concept = _concept(target="t1")
    run_discovery(client, "sfn", concept, vector_name="dino", limit=5)
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], RecommendQuery)
    assert list(kwargs["query"].recommend.positive) == ["t1"]


def test_raises_when_concept_has_only_negatives():
    """Qdrant Recommend requires ≥1 positive — error out rather than silently return empty."""
    client = _client_returning([])
    concept = _concept(positives=[], negatives=["n1"])
    with pytest.raises(ValueError):
        run_discovery(client, "sfn", concept, vector_name="dino", limit=5)


def test_reverse_polarity_swaps_pair_roles():
    """When reverse=True, what was positive becomes negative in the built pairs."""
    client = _client_returning([])
    concept = _concept(positives=["p1"], negatives=["n1"])
    run_discovery(
        client, "sfn", concept, vector_name="dino", limit=5, reverse=True
    )
    _, kwargs = client.query_points.call_args
    # Context-only (no target) ⇒ ContextQuery with a top-level ``context`` list.
    pair = kwargs["query"].context[0]
    # Swap: the original negative now appears in the positive slot.
    assert pair.positive == "n1"
    assert pair.negative == "p1"


def test_reference_points_are_excluded_from_results():
    """The concept's own references must not appear in its own discovery hits."""
    client = _client_returning([])
    concept = _concept(positives=["p1"], negatives=["n1"], target="t1")
    run_discovery(client, "sfn", concept, vector_name="dino", limit=5)
    _, kwargs = client.query_points.call_args
    query_filter: Filter = kwargs["query_filter"]
    assert query_filter is not None
    # All three reference IDs must be in the must_not HasIdCondition.
    must_not_ids: set[str] = set()
    for cond in query_filter.must_not or []:
        if isinstance(cond, HasIdCondition):
            must_not_ids.update(cond.has_id)
    assert must_not_ids == {"p1", "n1", "t1"}


def test_exclude_references_can_be_disabled():
    client = _client_returning([])
    concept = _concept(positives=["p1"], negatives=["n1"])
    run_discovery(
        client,
        "sfn",
        concept,
        vector_name="dino",
        limit=5,
        exclude_references=False,
    )
    _, kwargs = client.query_points.call_args
    assert kwargs["query_filter"] is None


def test_reference_collection_plumbs_lookup_from():
    """Cross-collection reference vectors must reach Qdrant as lookup_from."""
    client = _client_returning([])
    concept = _concept(positives=["p1"], negatives=["n1"])
    run_discovery(
        client,
        "sfn",
        concept,
        vector_name="dino",
        limit=5,
        reference_collection="sfn_references",
    )
    _, kwargs = client.query_points.call_args
    lookup = kwargs["lookup_from"]
    assert lookup is not None
    assert lookup.collection == "sfn_references"
    assert lookup.vector == "dino"


def test_discovery_score_is_rounded_to_integer_triplet():
    """Qdrant returns the triplet count as a float — callers want an int."""
    client = _client_returning([_pt("a", 3.0), _pt("b", 2.0)])
    concept = _concept(positives=["p1"], negatives=["n1"])
    hits = run_discovery(client, "sfn", concept, vector_name="dino", limit=5)
    assert [h.triplet_score for h in hits] == [3, 2]
    assert all(isinstance(h.triplet_score, int) for h in hits)


def test_recommend_score_keeps_float_and_none_triplet():
    """Recommend path ⇒ score is a cosine similarity, not an integer count."""
    client = _client_returning([_pt("a", 0.88)])
    concept = _concept(positives=["p1"])
    hits = run_discovery(client, "sfn", concept, vector_name="dino", limit=5)
    assert hits[0].triplet_score is None
    assert hits[0].cosine_margin == pytest.approx(0.88)


def test_query_failure_is_swallowed_and_returns_empty_list():
    """Individual vector failures must not abort the dual-vector fusion."""
    client = MagicMock()
    client.query_points.side_effect = RuntimeError("qdrant down")
    concept = _concept(positives=["p1"], negatives=["n1"])
    hits = run_discovery(client, "sfn", concept, vector_name="dino", limit=5)
    assert hits == []


# ---------------------------------------------------------------------------
# run_discovery_dual — fusion
# ---------------------------------------------------------------------------


def test_dual_ranks_dual_agreement_above_single_vector_hits():
    """The core innovation: items in BOTH vector rankings must rank higher."""
    # 'shared' appears in both dino and sscd; 'dino-only' appears only in dino.
    shared_in_dino = _pt("shared", 2.0, payload={"image_path": "/s.jpg"})
    dino_only = _pt("dino-only", 5.0, payload={"image_path": "/d.jpg"})
    shared_in_sscd = _pt("shared", 1.0, payload={"image_path": "/s.jpg"})

    # Dino first, SSCD second (run_discovery_dual's default vector_names order).
    call_results = [
        [dino_only, shared_in_dino],  # dino result
        [shared_in_sscd],  # sscd result
    ]
    client = MagicMock()

    def _query(**kwargs):
        res = MagicMock()
        res.points = call_results.pop(0)
        return res

    client.query_points.side_effect = _query

    concept = _concept(positives=["p1"], negatives=["n1"])
    fused = run_discovery_dual(client, "sfn", concept, limit=10)

    assert fused[0].point_id == "shared"
    assert fused[0].matched_modes == ["dino", "sscd"]
    assert fused[0].triplet_score_dino == 2
    assert fused[0].triplet_score_sscd == 1
    assert fused[0].fused_triplet_score == 3

    assert fused[1].point_id == "dino-only"
    assert fused[1].matched_modes == ["dino"]


def test_dual_falls_back_to_triplet_score_when_both_single_vector():
    """Two dino-only hits: rank by their individual triplet score."""
    call_results = [
        [_pt("high", 7.0), _pt("low", 1.0)],  # dino
        [],  # sscd empty
    ]
    client = MagicMock()

    def _query(**kwargs):
        res = MagicMock()
        res.points = call_results.pop(0)
        return res

    client.query_points.side_effect = _query

    concept = _concept(positives=["p1"], negatives=["n1"])
    fused = run_discovery_dual(client, "sfn", concept, limit=10)
    assert [h.point_id for h in fused] == ["high", "low"]


def test_run_triage_single_mode_returns_fused_hit_shape():
    """Single-mode run_triage must keep the same envelope as dual for callers."""
    client = _client_returning([_pt("a", 4.0)])
    concept = _concept(positives=["p1"], negatives=["n1"])
    fused = run_triage(client, "sfn", concept, mode="dino", limit=5)
    assert len(fused) == 1
    assert fused[0].matched_modes == ["dino"]
    assert fused[0].triplet_score_dino == 4
    assert fused[0].triplet_score_sscd is None
    assert fused[0].fused_triplet_score == 4
