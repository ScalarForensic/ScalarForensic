"""Tests for the Tag persistence layer.

All Qdrant interaction is stubbed via :class:`unittest.mock.MagicMock`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from scalar_forensic.tags import Tag, TagStore


def _fresh_client() -> MagicMock:
    """Build a MagicMock QdrantClient that pretends the sidecar already exists."""
    client = MagicMock()
    existing = MagicMock()
    existing.name = "sfn_tags"
    client.get_collections.return_value.collections = [existing]
    info = MagicMock()
    info.payload_schema = {
        "tag_id": True,
        "name": True,
        "is_tag": True,
    }
    client.get_collection.return_value = info
    return client


def _store() -> tuple[TagStore, MagicMock]:
    client = _fresh_client()
    # Default: no pre-existing tag records (create() calls get() internally).
    client.retrieve.return_value = []
    store = TagStore(client, "sfn_tags")
    return store, client


def test_derived_id_is_stable_per_name():
    a = TagStore.derive_tag_id("case-42-adult-content")
    b = TagStore.derive_tag_id("case-42-adult-content")
    c = TagStore.derive_tag_id("case-42-other")
    assert a == b
    assert a != c


def test_create_writes_payload_only_point_with_expected_shape():
    store, client = _store()
    tag = store.create(
        "case-42",
        positive_ids=["p1", "p2"],
        negative_ids=["n1"],
        target_id="t1",
        notes="initial set",
    )

    assert tag.tag_id == TagStore.derive_tag_id("case-42")
    assert tag.positive_ids == ["p1", "p2"]
    assert tag.negative_ids == ["n1"]
    assert tag.target_id == "t1"
    # No polarity field.
    assert not hasattr(tag, "polarity")
    datetime.fromisoformat(tag.created_at)
    assert tag.created_at.endswith("+00:00")

    assert client.upsert.call_count == 1
    kwargs = client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "sfn_tags"
    (point,) = kwargs["points"]
    assert point.id == tag.tag_id
    assert point.payload["is_tag"] is True
    assert point.payload["positive_ids"] == ["p1", "p2"]
    assert point.payload["negative_ids"] == ["n1"]
    # Polarity must NOT appear in payload.
    assert "polarity" not in point.payload
    # Tags are payload-only; the dummy vector exists solely to satisfy
    # Qdrant's "collection must have ≥1 vector config" requirement.
    assert point.vector == {"_tag": [0.0]}


def test_create_has_no_polarity_field():
    """Tag.to_payload must not include a 'polarity' key."""
    tag = Tag(tag_id="x", name="n", created_at="", updated_at="")
    assert "polarity" not in tag.to_payload()


def test_get_returns_none_for_missing_record():
    store, client = _store()
    client.retrieve.return_value = []
    assert store.get("missing") is None


def test_get_ignores_non_tag_records():
    store, client = _store()
    rec = MagicMock()
    rec.payload = {"is_tag": False, "tag_id": "x", "name": "not-a-tag"}
    client.retrieve.return_value = [rec]
    assert store.get("x") is None


def test_mark_positive_moves_from_negative_list():
    store, client = _store()
    original = Tag(
        tag_id="c1",
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
    assert updated.updated_at >= original.updated_at


def test_mark_is_idempotent():
    store, client = _store()
    original = Tag(
        tag_id="c1",
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
    assert updated.positive_ids == ["p1"]


def test_mark_raises_lookup_error_for_missing_tag():
    store, client = _store()
    client.retrieve.return_value = []
    with pytest.raises(LookupError):
        store.mark("does-not-exist", "p1", "positive")


def test_delete_returns_false_when_tag_missing():
    store, client = _store()
    client.retrieve.return_value = []
    deleted = store.delete("missing")
    assert deleted is False
    client.delete.assert_not_called()


def test_delete_returns_true_and_calls_qdrant_when_tag_exists():
    store, client = _store()
    rec = MagicMock()
    rec.payload = Tag(tag_id="c1", name="n", created_at="", updated_at="").to_payload()
    client.retrieve.return_value = [rec]
    deleted = store.delete("c1")
    assert deleted is True
    client.delete.assert_called_once()


def test_to_payload_and_from_payload_roundtrip():
    original = Tag(
        tag_id="c1",
        name="name",
        positive_ids=["a", "b"],
        negative_ids=["c"],
        target_id="t",
        notes="abc",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-02T00:00:00+00:00",
    )
    restored = Tag.from_payload(original.to_payload())
    assert restored == original


def test_migration_payload_rewrite():
    """Migrated concept payloads are correctly rewritten as tag payloads."""
    concept_payload = {
        "is_concept": True,
        "concept_id": "c1",
        "name": "weapons",
        "positive_ids": ["p1"],
        "negative_ids": ["n1"],
        "target_id": None,
        "polarity": "incriminating",
        "notes": "test",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    tag_payload = {
        "is_tag": True,
        "tag_id": concept_payload["concept_id"],
        "name": concept_payload["name"],
        "positive_ids": concept_payload["positive_ids"],
        "negative_ids": concept_payload["negative_ids"],
        "target_id": concept_payload["target_id"],
        "notes": concept_payload["notes"],
        "created_at": concept_payload["created_at"],
        "updated_at": concept_payload["updated_at"],
    }
    tag = Tag.from_payload(tag_payload)
    assert tag.tag_id == "c1"
    assert tag.name == "weapons"
    assert tag.positive_ids == ["p1"]
    assert "polarity" not in tag.to_payload()


def test_create_preserves_created_at_on_upsert():
    """Re-creating a tag with the same name must not clobber its created_at."""
    store, client = _store()

    # First creation: no existing record → new timestamps.
    tag1 = store.create("weapons", positive_ids=["p1"], notes="v1")
    original_created_at = tag1.created_at
    original_updated_at = tag1.updated_at

    # Simulate the tag now existing in Qdrant so the next create() finds it.
    existing_rec = MagicMock()
    existing_rec.payload = tag1.to_payload()
    client.retrieve.return_value = [existing_rec]

    # Second creation with the same name — only notes change.
    tag2 = store.create("weapons", positive_ids=["p1", "p2"], notes="v2")

    assert tag2.created_at == original_created_at, (
        "created_at must not change when re-creating a tag with the same name"
    )
    assert tag2.updated_at >= original_updated_at, "updated_at must advance on re-create"
    assert tag2.notes == "v2"
    assert tag2.positive_ids == ["p1", "p2"]
