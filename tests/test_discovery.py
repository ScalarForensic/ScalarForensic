"""Tests for the Qdrant Discovery / Recommend query builder.

Verifies that :mod:`scalar_forensic.discovery` picks the correct query
kind (Discovery vs Recommend), applies the auto-target rule, handles
reverse mode, excludes the tag's own reference points, and fuses
dual-vector rankings correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client.models import (
    DiscoverQuery,
    Filter,
    HasIdCondition,
    RecommendQuery,
)

from scalar_forensic.discovery import (
    _MAX_CONTEXT_PAIRS,
    _build_context_pairs,
    run_discovery,
    run_triage,
)
from scalar_forensic.tags import Tag


def _tag(
    *,
    positives: list[str] | None = None,
    negatives: list[str] | None = None,
    target: str | None = None,
) -> Tag:
    return Tag(
        tag_id="test-tag",
        name="test",
        positive_ids=list(positives or []),
        negative_ids=list(negatives or []),
        target_id=target,
    )


def _client_returning(points: list) -> MagicMock:
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
    assert len(pairs) == 4
    all_positives = {p.positive for p in pairs}
    all_negatives = {p.negative for p in pairs}
    assert all_positives == {"p1", "p2"}
    assert all_negatives == {"n1", "n2"}


def test_build_pairs_covers_every_reference_under_cap():
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
# run_discovery — auto-target
# ---------------------------------------------------------------------------


def test_auto_target_uses_discover_query_when_positives_present():
    """Tag with pairs but no explicit target ⇒ auto-target fires DiscoverQuery."""
    client = _client_returning([_pt("hit1", 3.0)])
    tag = _tag(positives=["p1", "p2"], negatives=["n1"])
    hits = run_discovery(client, "sfn", tag, vector_name="dino", limit=5)
    assert len(hits) == 1
    _, kwargs = client.query_points.call_args
    # Auto-target uses positives[0] = "p1" ⇒ DiscoverQuery, not ContextQuery
    assert isinstance(kwargs["query"], DiscoverQuery)
    assert kwargs["query"].discover.target == "p1"
    assert len(kwargs["query"].discover.context) == 2  # 2 positives × 1 negative
    assert kwargs["using"] == "dino"


def test_explicit_target_overrides_auto_target():
    """Explicit target_id beats the implicit first-positive anchor."""
    client = _client_returning([])
    tag = _tag(positives=["p1"], negatives=["n1"], target="t1")
    run_discovery(client, "sfn", tag, vector_name="dino", limit=5)
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], DiscoverQuery)
    assert kwargs["query"].discover.target == "t1"


def test_context_query_fires_when_only_negatives_reversed():
    """reverse=True on a tag with only positives stored ⇒ positives=[n-side] only;
    no pairs possible, falls back to Recommend."""
    client = _client_returning([])
    # After reverse, effective positives=[], negatives=["p1"] — error expected
    tag = _tag(positives=[], negatives=["n1"])
    with pytest.raises(ValueError):
        run_discovery(client, "sfn", tag, vector_name="dino", limit=5)


def test_recommend_fallback_when_no_negatives():
    """No negatives ⇒ no pairs ⇒ Recommend query (auto-target doesn't help here)."""
    client = _client_returning([])
    tag = _tag(positives=["p1", "p2"], negatives=[])
    run_discovery(client, "sfn", tag, vector_name="sscd", limit=5)
    _, kwargs = client.query_points.call_args
    # No negatives → no context pairs; auto-target sets target=p1, but pairs=[],
    # so we fall through to RecommendQuery (target is included in positives).
    assert isinstance(kwargs["query"], RecommendQuery)


def test_recommend_fallback_with_only_target():
    client = _client_returning([])
    tag = _tag(target="t1")
    run_discovery(client, "sfn", tag, vector_name="dino", limit=5)
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], RecommendQuery)
    assert list(kwargs["query"].recommend.positive) == ["t1"]


def test_raises_when_tag_has_only_negatives():
    client = _client_returning([])
    tag = _tag(positives=[], negatives=["n1"])
    with pytest.raises(ValueError):
        run_discovery(client, "sfn", tag, vector_name="dino", limit=5)


def test_reverse_polarity_swaps_pair_roles():
    """When reverse=True, what was positive becomes negative in the built pairs."""
    client = _client_returning([])
    tag = _tag(positives=["p1"], negatives=["n1"])
    run_discovery(client, "sfn", tag, vector_name="dino", limit=5, reverse=True)
    _, kwargs = client.query_points.call_args
    # Reversed: effective positives=["n1"], negatives=["p1"]
    # Auto-target = "n1"; DiscoverQuery with context [(n1, p1)]
    assert isinstance(kwargs["query"], DiscoverQuery)
    pair = kwargs["query"].discover.context[0]
    assert pair.positive == "n1"
    assert pair.negative == "p1"


def test_reference_points_are_excluded_from_results():
    client = _client_returning([])
    tag = _tag(positives=["p1"], negatives=["n1"], target="t1")
    run_discovery(client, "sfn", tag, vector_name="dino", limit=5)
    _, kwargs = client.query_points.call_args
    query_filter: Filter = kwargs["query_filter"]
    assert query_filter is not None
    must_not_ids: set[str] = set()
    for cond in query_filter.must_not or []:
        if isinstance(cond, HasIdCondition):
            must_not_ids.update(cond.has_id)
    assert must_not_ids == {"p1", "n1", "t1"}


def test_auto_target_excluded_only_once():
    """When auto-target = positives[0], it must not be listed twice in exclude."""
    client = _client_returning([])
    tag = _tag(positives=["p1"], negatives=["n1"])
    run_discovery(client, "sfn", tag, vector_name="dino", limit=5)
    _, kwargs = client.query_points.call_args
    must_not_ids: list[str] = []
    for cond in (kwargs["query_filter"].must_not or []):
        if isinstance(cond, HasIdCondition):
            must_not_ids.extend(cond.has_id)
    # "p1" should appear exactly once even though it is both a positive and the auto-target
    assert must_not_ids.count("p1") == 1


def test_exclude_references_can_be_disabled():
    client = _client_returning([])
    tag = _tag(positives=["p1"], negatives=["n1"])
    run_discovery(client, "sfn", tag, vector_name="dino", limit=5, exclude_references=False)
    _, kwargs = client.query_points.call_args
    assert kwargs["query_filter"] is None


def test_reference_collection_plumbs_lookup_from():
    client = _client_returning([])
    tag = _tag(positives=["p1"], negatives=["n1"])
    run_discovery(
        client, "sfn", tag, vector_name="dino", limit=5,
        reference_collection="sfn_references",
    )
    _, kwargs = client.query_points.call_args
    lookup = kwargs["lookup_from"]
    assert lookup is not None
    assert lookup.collection == "sfn_references"
    assert lookup.vector == "dino"


def test_discovery_score_is_rounded_to_integer_triplet():
    client = _client_returning([_pt("a", 3.0), _pt("b", 2.0)])
    tag = _tag(positives=["p1"], negatives=["n1"])
    hits = run_discovery(client, "sfn", tag, vector_name="dino", limit=5)
    assert [h.triplet_score for h in hits] == [3, 2]
    assert all(isinstance(h.triplet_score, int) for h in hits)


def test_recommend_score_keeps_float_and_none_triplet():
    client = _client_returning([_pt("a", 0.88)])
    tag = _tag(positives=["p1"])
    hits = run_discovery(client, "sfn", tag, vector_name="dino", limit=5)
    assert hits[0].triplet_score is None
    assert hits[0].cosine_margin == pytest.approx(0.88)


def test_query_failure_is_swallowed_and_returns_empty_list():
    client = MagicMock()
    client.query_points.side_effect = RuntimeError("qdrant down")
    tag = _tag(positives=["p1"], negatives=["n1"])
    hits = run_discovery(client, "sfn", tag, vector_name="dino", limit=5)
    assert hits == []


# ---------------------------------------------------------------------------
# run_triage — dino-only wrapper
# ---------------------------------------------------------------------------


def test_run_triage_returns_discovery_hits_via_dino():
    client = _client_returning([_pt("a", 4.0)])
    tag = _tag(positives=["p1"], negatives=["n1"])
    hits = run_triage(client, "sfn", tag, limit=5)
    assert len(hits) == 1
    assert hits[0].point_id == "a"
    assert hits[0].triplet_score == 4
    assert hits[0].vector_name == "dino"
    _, kwargs = client.query_points.call_args
    assert kwargs["using"] == "dino"
