"""Tests for the Concept persistence layer.

All Qdrant interaction is stubbed via :class:`unittest.mock.MagicMock`,
mirroring the style already used in :mod:`test_get_available_modes`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from scalar_forensic.concepts import Concept, ConceptStore


def _fresh_client() -> MagicMock:
    """Build a MagicMock QdrantClient that pretends the sidecar already exists.

    Pretending the collection already exists keeps the ConceptStore
    constructor from calling ``create_collection`` so tests don't have
    to assert on that call when it's not what they are verifying.
    """
    client = MagicMock()
    existing = MagicMock()
    existing.name = "sfn_concepts"
    client.get_collections.return_value.collections = [existing]
    info = MagicMock()
    info.payload_schema = {
        "concept_id": True,
        "name": True,
        "polarity": True,
        "is_concept": True,
    }
    client.get_collection.return_value = info
    return client


def _store() -> tuple[ConceptStore, MagicMock]:
    client = _fresh_client()
    store = ConceptStore(client, "sfn_concepts")
    return store, client


def test_derived_id_is_stable_per_name():
    """Two calls with the same name must produce the same concept_id."""
    a = ConceptStore.derive_concept_id("case-42-adult-content")
    b = ConceptStore.derive_concept_id("case-42-adult-content")
    c = ConceptStore.derive_concept_id("case-42-other")
    assert a == b
    assert a != c


def test_create_writes_payload_only_point_with_expected_shape():
    store, client = _store()
    concept = store.create(
        "case-42",
        positive_ids=["p1", "p2"],
        negative_ids=["n1"],
        target_id="t1",
        polarity="incriminating",
        notes="initial set",
    )

    assert concept.concept_id == ConceptStore.derive_concept_id("case-42")
    assert concept.positive_ids == ["p1", "p2"]
    assert concept.negative_ids == ["n1"]
    assert concept.target_id == "t1"
    assert concept.polarity == "incriminating"
    # Timestamps are ISO 8601 UTC.
    datetime.fromisoformat(concept.created_at)
    assert concept.created_at.endswith("+00:00")

    # Upsert was called exactly once with one PointStruct carrying the payload.
    assert client.upsert.call_count == 1
    kwargs = client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "sfn_concepts"
    (point,) = kwargs["points"]
    assert point.id == concept.concept_id
    assert point.payload["is_concept"] is True
    assert point.payload["positive_ids"] == ["p1", "p2"]
    assert point.payload["negative_ids"] == ["n1"]
    assert point.payload["polarity"] == "incriminating"
    # Payload-only point — the vector field is an empty dict, never a real embedding.
    assert point.vector == {}


def test_get_returns_none_for_missing_record():
    store, client = _store()
    client.retrieve.return_value = []
    assert store.get("missing") is None


def test_get_ignores_non_concept_records():
    """Belt-and-braces: if the sidecar ever holds a non-concept point, filter it out."""
    store, client = _store()
    rec = MagicMock()
    rec.payload = {"is_concept": False, "concept_id": "x", "name": "not-a-concept"}
    client.retrieve.return_value = [rec]
    assert store.get("x") is None


def test_mark_positive_moves_from_negative_list():
    """A point already in negative must be moved — the latest label wins."""
    store, client = _store()

    original = Concept(
        concept_id="c1",
        name="n",
        positive_ids=["p1"],
        negative_ids=["p2"],
        created_at=datetime.now(UTC).isoformat(),
        updated_at=datetime.now(UTC).isoformat(),
    )
    rec = MagicMock()
    rec.payload = original.to_payload()
    client.retrieve.return_value = [rec]

    updated = store.mark("c1", "p2", "positive")
    assert "p2" in updated.positive_ids
    assert "p2" not in updated.negative_ids
    # updated_at must advance.
    assert updated.updated_at >= original.updated_at


def test_mark_is_idempotent():
    store, client = _store()
    original = Concept(
        concept_id="c1",
        name="n",
        positive_ids=["p1"],
        negative_ids=[],
        created_at="2020-01-01T00:00:00+00:00",
        updated_at="2020-01-01T00:00:00+00:00",
    )
    rec = MagicMock()
    rec.payload = original.to_payload()
    client.retrieve.return_value = [rec]

    updated = store.mark("c1", "p1", "positive")
    # No duplicate.
    assert updated.positive_ids == ["p1"]


def test_mark_raises_lookup_error_for_missing_concept():
    store, client = _store()
    client.retrieve.return_value = []
    with pytest.raises(LookupError):
        store.mark("does-not-exist", "p1", "positive")


def test_delete_returns_false_when_concept_missing():
    store, client = _store()
    client.retrieve.return_value = []
    deleted = store.delete("missing")
    assert deleted is False
    client.delete.assert_not_called()


def test_delete_returns_true_and_calls_qdrant_when_concept_exists():
    store, client = _store()
    rec = MagicMock()
    rec.payload = Concept(concept_id="c1", name="n", created_at="", updated_at="").to_payload()
    client.retrieve.return_value = [rec]
    deleted = store.delete("c1")
    assert deleted is True
    client.delete.assert_called_once()


def test_to_payload_and_from_payload_roundtrip():
    original = Concept(
        concept_id="c1",
        name="name",
        positive_ids=["a", "b"],
        negative_ids=["c"],
        target_id="t",
        polarity="exculpatory",
        notes="abc",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-02T00:00:00+00:00",
    )
    restored = Concept.from_payload(original.to_payload())
    assert restored == original
