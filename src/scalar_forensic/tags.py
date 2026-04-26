"""Tag persistence for investigator-in-the-loop Discovery triage.

A *Tag* is a named, reusable set of positive and negative reference point IDs
(and optionally a single target anchor) used as input to Qdrant's Discovery
and Recommendation APIs.  Tags are stored as payload-only points in a sidecar
Qdrant collection so they outlive any single review session — mirroring the
payload-only-anchor pattern used for per-video metadata records.

Polarity ("incriminating" vs "exculpatory") has been removed; the ``reverse``
flag on each triage run is the real exculpatory knob.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from scalar_forensic.indexer import qdrant_scroll_all

# Qdrant does not allow a collection with zero named vectors, so the
# sidecar collection is created with a single 1-D dummy vector that is
# never populated on any tag point (tags are payload-only).
_DUMMY_VECTOR_NAME = "_tag"
_DUMMY_VECTOR_DIM = 1

# Deterministic UUID namespace for tag IDs — derived from user-supplied
# names so calling create_tag twice with the same name upserts the same
# record rather than creating duplicates.  Same value as the legacy concept
# namespace so existing names map to the same IDs after migration.
_TAG_NAMESPACE = uuid.UUID("8b5c0c6a-6d2a-4b4b-9a7f-c0ffee5fbe57")


@dataclass
class Tag:
    """A named pair of positive and negative reference point ID lists.

    ``positive_ids`` and ``negative_ids`` are Qdrant point IDs (strings or
    integers) that already exist in the case collection (or in the optional
    reference collection).  They are *not* vectors — they are resolved to
    vectors server-side by Qdrant at query time, which means the tag is
    automatically in sync with any vector re-indexing.

    ``target_id`` is an optional explicit anchor.  When absent and at least
    one positive exists, the first positive is used as an implicit anchor at
    query time so DiscoverQuery fires immediately without requiring the user
    to call "Set from hit".
    """

    tag_id: str
    name: str
    positive_ids: list[str | int] = field(default_factory=list)
    negative_ids: list[str | int] = field(default_factory=list)
    target_id: str | int | None = None
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_payload(self) -> dict:
        return {
            "is_tag": True,
            "tag_id": self.tag_id,
            "name": self.name,
            "positive_ids": list(self.positive_ids),
            "negative_ids": list(self.negative_ids),
            "target_id": self.target_id,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> Tag:
        return cls(
            tag_id=payload["tag_id"],
            name=payload.get("name", ""),
            positive_ids=list(payload.get("positive_ids") or []),
            negative_ids=list(payload.get("negative_ids") or []),
            target_id=payload.get("target_id"),
            notes=payload.get("notes", ""),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
        )


class TagStore:
    """Persistence layer for :class:`Tag` objects backed by Qdrant.

    One payload-only point per tag in the configured sidecar collection.
    The point ID is a UUIDv5 derived from the tag name so tag identity is
    stable across re-creates with the same name.

    Mutation methods (:meth:`mark`, :meth:`unmark`, :meth:`set_target`) use
    read-modify-write without distributed locking.  This is intentional — the
    expected deployment model is a single forensic investigator per machine.
    Concurrent writes from multiple browser tabs would produce a last-writer-
    wins race; that risk is accepted for this use case.
    """

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    _DUMMY_VECTOR_NAME: VectorParams(
                        size=_DUMMY_VECTOR_DIM, distance=Distance.COSINE
                    )
                },
            )
        info = self.client.get_collection(self.collection)
        schema = info.payload_schema or {}
        for field_name in ("tag_id", "name"):
            if field_name not in schema:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        if "is_tag" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="is_tag",
                field_schema=PayloadSchemaType.BOOL,
            )

    @staticmethod
    def derive_tag_id(name: str) -> str:
        """Return the deterministic UUIDv5 used as the tag's Qdrant point ID."""
        return str(uuid.uuid5(_TAG_NAMESPACE, name))

    def create(
        self,
        name: str,
        *,
        positive_ids: list[str | int] | None = None,
        negative_ids: list[str | int] | None = None,
        target_id: str | int | None = None,
        notes: str = "",
    ) -> Tag:
        """Create or replace a tag by *name*.

        Name-derived IDs mean re-running with the same name overwrites the
        existing tag in place — intentional for scripted workflows.  The
        original ``created_at`` timestamp is preserved across re-creates so
        that editing a tag's notes or IDs via the UI does not destroy its
        creation time.
        """
        now = datetime.now(UTC).isoformat()
        tag_id = self.derive_tag_id(name)
        existing = self.get(tag_id)
        created_at = existing.created_at if existing else now
        tag = Tag(
            tag_id=tag_id,
            name=name,
            positive_ids=list(positive_ids or []),
            negative_ids=list(negative_ids or []),
            target_id=target_id,
            notes=notes,
            created_at=created_at,
            updated_at=now,
        )
        self._upsert(tag)
        return tag

    def get(self, tag_id: str) -> Tag | None:
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[tag_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        if not payload.get("is_tag"):
            return None
        return Tag.from_payload(payload)

    def list(self) -> list[Tag]:
        out: list[Tag] = []
        for r in qdrant_scroll_all(
            self.client,
            self.collection,
            scroll_filter=Filter(must=[FieldCondition(key="is_tag", match=MatchValue(value=True))]),
            limit=256,
            with_payload=True,
            with_vectors=False,
        ):
            payload = r.payload or {}
            if payload.get("is_tag"):
                out.append(Tag.from_payload(payload))
        out.sort(key=lambda t: t.updated_at, reverse=True)
        return out

    def mark(
        self,
        tag_id: str,
        point_id: str | int,
        role: Literal["positive", "negative"],
    ) -> Tag:
        """Append *point_id* to the tag's positive or negative list.

        No-op if already present in that role.  A point in the other role
        is moved — the most recent mark wins.  Raises :class:`LookupError`
        if the tag does not exist.
        """
        tag = self.get(tag_id)
        if tag is None:
            raise LookupError(f"Tag not found: {tag_id}")
        if role == "positive":
            if point_id in tag.negative_ids:
                tag.negative_ids.remove(point_id)
            if point_id not in tag.positive_ids:
                tag.positive_ids.append(point_id)
        else:
            if point_id in tag.positive_ids:
                tag.positive_ids.remove(point_id)
            if point_id not in tag.negative_ids:
                tag.negative_ids.append(point_id)
        tag.updated_at = datetime.now(UTC).isoformat()
        self._upsert(tag)
        return tag

    def unmark(self, tag_id: str, point_id: str | int) -> Tag:
        tag = self.get(tag_id)
        if tag is None:
            raise LookupError(f"Tag not found: {tag_id}")
        if point_id in tag.positive_ids:
            tag.positive_ids.remove(point_id)
        if point_id in tag.negative_ids:
            tag.negative_ids.remove(point_id)
        tag.updated_at = datetime.now(UTC).isoformat()
        self._upsert(tag)
        return tag

    def set_target(self, tag_id: str, target_id: str | int | None) -> Tag:
        tag = self.get(tag_id)
        if tag is None:
            raise LookupError(f"Tag not found: {tag_id}")
        tag.target_id = target_id
        tag.updated_at = datetime.now(UTC).isoformat()
        self._upsert(tag)
        return tag

    def delete(self, tag_id: str) -> bool:
        """Return True if the tag existed and was deleted."""
        if self.get(tag_id) is None:
            return False
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[tag_id]),
        )
        return True

    def _upsert(self, tag: Tag) -> None:
        # Tags are payload-only.  The collection is created with a single
        # named dummy vector (``_DUMMY_VECTOR_NAME``) because Qdrant
        # collections cannot have zero vector configs.  Write a constant
        # all-zero vector for that name on every upsert so the point is
        # unambiguously valid in a multi-vector collection.
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=tag.tag_id,
                    vector={_DUMMY_VECTOR_NAME: [0.0] * _DUMMY_VECTOR_DIM},
                    payload=tag.to_payload(),
                )
            ],
        )
