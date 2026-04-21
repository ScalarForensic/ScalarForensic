"""Concept persistence for investigator-in-the-loop Discovery triage.

A *Concept* is a named, reusable set of positive and negative reference
point IDs (and optionally a single target anchor) used as input to
Qdrant's Discovery and Recommendation APIs.  Concepts are stored as
payload-only points in a sidecar Qdrant collection so they outlive any
single review session and can be shared across cases, mirroring the
payload-only-anchor pattern already used for per-video metadata records
(:meth:`scalar_forensic.indexer.Indexer.upsert_video_records`).

The concept's polarity (``"incriminating"`` vs ``"exculpatory"``) is
stored alongside the IDs.  Reverse triage simply swaps the roles of the
positive and negative lists at query time — see
:mod:`scalar_forensic.discovery`.
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

Polarity = Literal["incriminating", "exculpatory"]

# Qdrant does not allow a collection with zero named vectors, so the
# sidecar collection is created with a single 1-D dummy vector that is
# never populated on any concept point (concepts are payload-only).
_DUMMY_VECTOR_NAME = "_concept"
_DUMMY_VECTOR_DIM = 1

# Deterministic UUID namespace for concept IDs — derived from user-supplied
# names so calling create_concept twice with the same name upserts the same
# record rather than creating duplicates.  Fixed random UUID, never changes.
_CONCEPT_NAMESPACE = uuid.UUID("8b5c0c6a-6d2a-4b4b-9a7f-c0ffee5fbe57")


@dataclass
class Concept:
    """A named pair of positive and negative reference point ID lists.

    ``positive_ids`` and ``negative_ids`` are Qdrant point IDs (strings or
    integers) that already exist in the case collection (or, if
    :attr:`reference_collection` is configured, in that sidecar).  They
    are *not* vectors — they are resolved to vectors server-side by
    Qdrant at query time, which means the concept is automatically in
    sync with any vector re-indexing of the underlying points.

    ``target_id`` is an optional single anchor.  When present the query
    uses Qdrant's Discovery search (target + context); when absent it
    uses context-only search (triplet ranking over pairs).

    ``polarity`` is metadata only; it does not change the stored pair
    roles.  Discovery callers inspect it to decide whether to swap
    positive/negative at query time for "reverse / exculpatory" mode.
    """

    concept_id: str
    name: str
    positive_ids: list[str | int] = field(default_factory=list)
    negative_ids: list[str | int] = field(default_factory=list)
    target_id: str | int | None = None
    polarity: Polarity = "incriminating"
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_payload(self) -> dict:
        """Render the concept as a Qdrant payload dict."""
        return {
            "is_concept": True,
            "concept_id": self.concept_id,
            "name": self.name,
            "positive_ids": list(self.positive_ids),
            "negative_ids": list(self.negative_ids),
            "target_id": self.target_id,
            "polarity": self.polarity,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> Concept:
        return cls(
            concept_id=payload["concept_id"],
            name=payload.get("name", ""),
            positive_ids=list(payload.get("positive_ids") or []),
            negative_ids=list(payload.get("negative_ids") or []),
            target_id=payload.get("target_id"),
            polarity=payload.get("polarity", "incriminating"),
            notes=payload.get("notes", ""),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
        )


class ConceptStore:
    """Persistence layer for :class:`Concept` objects backed by Qdrant.

    One payload-only point per concept in the configured sidecar
    collection.  The point ID is a UUIDv5 derived from the concept name
    so concept identity is stable across re-creates with the same name.
    """

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            # The single dummy vector is required by Qdrant schema but is
            # never set on concept points — they are payload-only.
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
        for field_name in ("concept_id", "name", "polarity"):
            if field_name not in schema:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        if "is_concept" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="is_concept",
                field_schema=PayloadSchemaType.BOOL,
            )

    @staticmethod
    def derive_concept_id(name: str) -> str:
        """Return the deterministic UUIDv5 used as the concept's Qdrant point ID."""
        return str(uuid.uuid5(_CONCEPT_NAMESPACE, name))

    def create(
        self,
        name: str,
        *,
        positive_ids: list[str | int] | None = None,
        negative_ids: list[str | int] | None = None,
        target_id: str | int | None = None,
        polarity: Polarity = "incriminating",
        notes: str = "",
    ) -> Concept:
        """Create or replace a concept by *name*.

        Name-derived IDs mean re-running with the same name overwrites
        the existing concept in place — intentional for scripted
        workflows.  Use :meth:`update` / :meth:`mark` to amend instead.
        """
        now = datetime.now(UTC).isoformat()
        concept = Concept(
            concept_id=self.derive_concept_id(name),
            name=name,
            positive_ids=list(positive_ids or []),
            negative_ids=list(negative_ids or []),
            target_id=target_id,
            polarity=polarity,
            notes=notes,
            created_at=now,
            updated_at=now,
        )
        self._upsert(concept)
        return concept

    def get(self, concept_id: str) -> Concept | None:
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[concept_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        if not payload.get("is_concept"):
            return None
        return Concept.from_payload(payload)

    def list(self) -> list[Concept]:
        out: list[Concept] = []
        for r in qdrant_scroll_all(
            self.client,
            self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="is_concept", match=MatchValue(value=True))]
            ),
            limit=256,
            with_payload=True,
            with_vectors=False,
        ):
            payload = r.payload or {}
            if payload.get("is_concept"):
                out.append(Concept.from_payload(payload))
        out.sort(key=lambda c: c.updated_at, reverse=True)
        return out

    def mark(
        self,
        concept_id: str,
        point_id: str | int,
        role: Literal["positive", "negative"],
    ) -> Concept:
        """Append *point_id* to the concept's positive or negative list.

        No-op if the ID is already present in that role.  A point that
        is already in the *other* role is moved — the most recent mark
        wins.  Raises :class:`LookupError` if the concept does not exist.
        """
        concept = self.get(concept_id)
        if concept is None:
            raise LookupError(f"Concept not found: {concept_id}")
        if role == "positive":
            if point_id in concept.negative_ids:
                concept.negative_ids.remove(point_id)
            if point_id not in concept.positive_ids:
                concept.positive_ids.append(point_id)
        else:
            if point_id in concept.positive_ids:
                concept.positive_ids.remove(point_id)
            if point_id not in concept.negative_ids:
                concept.negative_ids.append(point_id)
        concept.updated_at = datetime.now(UTC).isoformat()
        self._upsert(concept)
        return concept

    def unmark(self, concept_id: str, point_id: str | int) -> Concept:
        concept = self.get(concept_id)
        if concept is None:
            raise LookupError(f"Concept not found: {concept_id}")
        if point_id in concept.positive_ids:
            concept.positive_ids.remove(point_id)
        if point_id in concept.negative_ids:
            concept.negative_ids.remove(point_id)
        concept.updated_at = datetime.now(UTC).isoformat()
        self._upsert(concept)
        return concept

    def set_target(self, concept_id: str, target_id: str | int | None) -> Concept:
        concept = self.get(concept_id)
        if concept is None:
            raise LookupError(f"Concept not found: {concept_id}")
        concept.target_id = target_id
        concept.updated_at = datetime.now(UTC).isoformat()
        self._upsert(concept)
        return concept

    def delete(self, concept_id: str) -> bool:
        """Return True if the concept existed and was deleted."""
        if self.get(concept_id) is None:
            return False
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[concept_id]),
        )
        return True

    def _upsert(self, concept: Concept) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=concept.concept_id,
                    vector={},
                    payload=concept.to_payload(),
                )
            ],
        )
