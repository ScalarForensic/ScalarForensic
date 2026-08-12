"""Face observation store: case-scoped Qdrant sidecar collection (spec §7)."""

from __future__ import annotations

import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from scalar_forensic.faces.align import ALIGNMENT_VERSION
from scalar_forensic.faces.provenance import PipelineConfig
from scalar_forensic.indexer import qdrant_scroll_all

FACE_VECTOR_NAME = "face"
_META_POINT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "sfn:face-collection-meta"))
_SCROLL_LIMIT = 256

# Hard fields gate embedding comparability: a mismatch means the stored
# vectors and the ones we would write are not in the same space.
_HARD_FIELDS = (
    "embedder_model_hash",
    "manifest_hash",
    "embedder_dim",
    "alignment_version",
    "normalization_id",
)
# Soft fields change *which* faces get in, not what the vectors mean.
_SOFT_FIELDS = (
    "detector_id",
    "detector_model_hash",
    "detect_max_size",
    "min_conf",
    "min_size",
    "review_min_conf",
    "review_min_size",
)


@dataclass(frozen=True)
class PurgeResult:
    n_points: int
    chip_hashes: list[str]


class FaceStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        case_collection: str,
        embedder_dim: int,
    ) -> None:
        # Injected client, matching TagStore — callers own client lifetime.
        self.client = client
        self.collection = collection
        self.case_collection = case_collection
        self.embedder_dim = embedder_dim

    # --- identity -------------------------------------------------------

    def observation_key(self, image_hash: str, frame_timecode_ms: int | None, bbox) -> str:
        # Model-independent: no alignment version — labels reference this and
        # must survive re-alignment/re-detection under new versions.
        x, y, w, h = (round(v) for v in bbox)
        ts = "" if frame_timecode_ms is None else str(frame_timecode_ms)
        return f"{image_hash}:{ts}:{x}:{y}:{w}:{h}"

    def face_point_id(self, image_hash: str, frame_timecode_ms: int | None, bbox) -> str:
        # ALIGNMENT_VERSION is part of the ID so a future arcface-112-v2
        # re-index coexists with v1 points instead of overwriting them.
        key = self.observation_key(image_hash, frame_timecode_ms, bbox)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"face:{key}:{ALIGNMENT_VERSION}"))

    # --- collection lifecycle -------------------------------------------

    def collection_is_new(self) -> bool:
        existing = {c.name for c in self.client.get_collections().collections}
        return self.collection not in existing

    def ensure_collection(
        self, cfg: PipelineConfig, examiner_id: str, authorization_ref: str | None
    ) -> None:
        if not self.collection_is_new():
            # Leave meta as recorded at creation; check_compat() judges it.
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                FACE_VECTOR_NAME: VectorParams(size=self.embedder_dim, distance=Distance.COSINE)
            },
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
            ),
            on_disk_payload=True,
        )
        for field, schema in (
            ("image_hash", PayloadSchemaType.KEYWORD),
            ("image_path", PayloadSchemaType.KEYWORD),
            ("video_hash", PayloadSchemaType.KEYWORD),
            ("video_path", PayloadSchemaType.KEYWORD),
            ("is_face", PayloadSchemaType.KEYWORD),
            ("group_id", PayloadSchemaType.KEYWORD),
            ("quality", PayloadSchemaType.FLOAT),
            ("frame_timecode_ms", PayloadSchemaType.INTEGER),
        ):
            self.client.create_payload_index(
                collection_name=self.collection, field_name=field, field_schema=schema
            )
        meta_payload = {
            "is_face_meta": True,
            "case_collection": self.case_collection,
            "enablement": {
                "examiner_id": examiner_id,
                "enabled_at": datetime.now(UTC).isoformat(),
                "authorization_ref": authorization_ref,
            },
            # Informational only (spec §7.3): case-handover context, never
            # part of the comparability tuple.
            "app_host": socket.gethostname(),
            **{f: getattr(cfg, f) for f in _HARD_FIELDS + _SOFT_FIELDS},
        }
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=_META_POINT_ID, vector={}, payload=meta_payload)],
        )

    def _meta_payload(self) -> dict:
        records = self.client.retrieve(
            collection_name=self.collection, ids=[_META_POINT_ID], with_payload=True
        )
        if not records:
            return {}
        return records[0].payload or {}

    def check_compat(self, cfg: PipelineConfig) -> list[str]:
        """Raise on hard mismatches, return warning strings for soft ones.

        Fields absent from the stored meta are "unknown", not a mismatch —
        the same stance safeguards.py takes for older collections.
        """
        meta = self._meta_payload()
        if not meta:
            return []
        stored_case = meta.get("case_collection")
        if stored_case is not None and stored_case != self.case_collection:
            raise ValueError(
                f"Face collection {self.collection!r} belongs to case collection "
                f"{stored_case!r}, but this run targets {self.case_collection!r}.\n"
                "  Face observations are case-scoped: refusing to mix biometric data\n"
                "  across cases.  Set SFN_FACE_COLLECTION explicitly if this is intended."
            )
        hard_mismatches = [
            f"{f}: collection has {meta[f]!r}, this run has {getattr(cfg, f)!r}"
            for f in _HARD_FIELDS
            if f in meta and meta[f] != getattr(cfg, f)
        ]
        if hard_mismatches:
            raise ValueError(
                "Face embeddings in this collection are not comparable with the "
                "current configuration:\n  - " + "\n  - ".join(hard_mismatches)
            )
        return [
            f"{f}: collection has {meta[f]!r}, this run has {getattr(cfg, f)!r}"
            for f in _SOFT_FIELDS
            if f in meta and meta[f] != getattr(cfg, f)
        ]

    # --- writes ---------------------------------------------------------

    def upsert_faces(self, points: list[PointStruct]) -> None:
        if not points:
            return
        self.client.upsert(collection_name=self.collection, points=points)

    def marker_point(
        self,
        image_hash: str,
        video_hash: str | None,
        cfg_hash: str,
        n_detected: int,
        n_kept: int,
        rejected: dict[str, int],
        n_review_only: int = 0,
        review_only_reasons: dict[str, int] | None = None,
        n_dropped_noncanonical: int = 0,
    ) -> PointStruct:
        """One marker per *processed unit* — per image, per video frame.

        Keying on the frame's own hash rather than the video's is what keeps
        processed_hashes() from skipping every frame after the first.
        """
        return PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"face-marker:{image_hash}")),
            vector={},
            payload={
                "is_face_marker": True,
                "image_hash": image_hash,
                "video_hash": video_hash,
                "faces_processed_at": datetime.now(UTC).isoformat(),
                "faces_pipeline_config_hash": cfg_hash,
                "n_detected": n_detected,
                "n_kept": n_kept,
                "n_rejected": rejected,
                # n_kept still means "embedded".  Without these two the marker
                # understates how many biometric crops the run wrote, and
                # n_detected stops reconciling with its own breakdown.
                "n_review_only": n_review_only,
                "review_only_reasons": review_only_reasons or {},
                # Silently subtracted from n_detected before this: the detector
                # drops rows whose landmarks are non-canonical.
                "n_dropped_noncanonical": n_dropped_noncanonical,
            },
        )

    def video_rollup_point(
        self,
        video_hash: str,
        cfg_hash: str,
        n_detected: int,
        n_kept: int,
        rejected: dict[str, int],
        n_frames: int,
        n_review_only: int = 0,
        review_only_reasons: dict[str, int] | None = None,
        n_dropped_noncanonical: int = 0,
    ) -> PointStruct:
        """Aggregate marker for a whole video, written once after its frames.

        Distinct ID namespace from the per-frame markers so it can never
        overwrite one — processed_hashes() keys on the frame markers.
        """
        return PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"face-marker-video:{video_hash}")),
            vector={},
            payload={
                "is_face_video_rollup": True,
                "video_hash": video_hash,
                "faces_processed_at": datetime.now(UTC).isoformat(),
                "faces_pipeline_config_hash": cfg_hash,
                "n_frames": n_frames,
                "n_detected": n_detected,
                "n_kept": n_kept,
                "n_rejected": rejected,
                "n_review_only": n_review_only,
                "review_only_reasons": review_only_reasons or {},
                "n_dropped_noncanonical": n_dropped_noncanonical,
            },
        )

    # --- reads ----------------------------------------------------------

    def processed_hashes(self, cfg_hash: str) -> set[str]:
        flt = Filter(
            must=[
                FieldCondition(key="is_face_marker", match=MatchValue(value=True)),
                FieldCondition(key="faces_pipeline_config_hash", match=MatchValue(value=cfg_hash)),
            ]
        )
        return {
            rec.payload["image_hash"]
            for rec in qdrant_scroll_all(
                self.client,
                self.collection,
                scroll_filter=flt,
                limit=_SCROLL_LIMIT,
                with_payload=["image_hash"],
            )
            if rec.payload and rec.payload.get("image_hash")
        }

    def get_face(self, point_id: str) -> dict | None:
        """Return one stored observation's payload, or None if absent."""
        records = self.client.retrieve(
            collection_name=self.collection, ids=[point_id], with_payload=True
        )
        if not records:
            return None
        return {"id": records[0].id, **(records[0].payload or {})}

    def get_marker(self, image_hash: str) -> dict | None:
        """Return the processed marker for one medium, or None if absent."""
        marker_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"face-marker:{image_hash}"))
        records = self.client.retrieve(
            collection_name=self.collection, ids=[marker_id], with_payload=True
        )
        if not records:
            return None
        return records[0].payload or {}

    def list_faces(self, image_hash: str) -> list[dict]:
        flt = Filter(
            must=[
                FieldCondition(key="is_face", match=MatchValue(value=True)),
                FieldCondition(key="image_hash", match=MatchValue(value=image_hash)),
            ]
        )
        return [
            {"id": rec.id, **(rec.payload or {})}
            for rec in qdrant_scroll_all(
                self.client,
                self.collection,
                scroll_filter=flt,
                limit=_SCROLL_LIMIT,
                with_payload=True,
            )
        ]

    # --- purge ----------------------------------------------------------

    def clear_face_vector(self, point_ids: list[str]) -> None:
        """Remove the named face vector from points, keeping their payloads.

        Called when an observation is demoted to review-only on re-index.
        Point IDs are stable across runs (face_point_id excludes config_hash)
        while idempotency keys on config_hash, so a threshold change rewrites
        existing points in place.  An upsert with no vector must not be
        trusted to clear a previously stored one: a review-only point that
        kept its vector would still be returned by similarity search.

        Precondition: every id must already exist in the collection.  qdrant
        returns 404 for an unknown id rather than treating it as a no-op
        (verified live in tests/faces/test_store_integration.py), and the
        error is deliberately not swallowed — a silently ignored clear could
        leave a review-only observation searchable.  The CLI satisfies this by
        upserting the points immediately before, with the client's default
        wait=True.
        """
        if not point_ids:
            return
        self.client.delete_vectors(
            collection_name=self.collection,
            vectors=[FACE_VECTOR_NAME],
            points=list(point_ids),
        )

    def unreferenced_chip_hashes(self, hashes: list[str]) -> list[str]:
        """Of *hashes*, those no surviving point still references.

        Chip files are content-addressed and therefore shared between
        observations with byte-identical crops — common for review chips
        across exact-duplicate media, and between an embedded observation and
        a review-only one that produced the same source crop.  Unlinking a
        shared chip would break a surviving observation's evidence.
        """
        if not hashes:
            return []
        wanted = set(hashes)
        still_referenced: set[str] = set()
        for rec in qdrant_scroll_all(
            self.client,
            self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="is_face", match=MatchValue(value=True))]
            ),
            limit=_SCROLL_LIMIT,
            with_payload=["aligned_chip_hash", "review_chip_hash"],
        ):
            payload = rec.payload or {}
            for key in ("aligned_chip_hash", "review_chip_hash"):
                value = payload.get(key)
                if value in wanted:
                    still_referenced.add(value)
        # Order-preserving dedup: a shared hash appears once per referencing
        # observation, and the caller's unlink count is audit-facing.
        return [h for h in dict.fromkeys(hashes) if h not in still_referenced]

    _STALE_FIELDS = [
        "observation_key",
        "embedding_status",
        "embedding_exclusion_reason",
        "pipeline_config_hash",
        "indexed_at",
        "bbox",
        "aligned_chip_hash",
        "review_chip_hash",
    ]

    def stale_face_points(self, image_hash: str, produced_ids: set[str]) -> list[dict]:
        """Face points for one medium that the current run did not produce.

        Point IDs derive from image_hash + timecode + rounded bbox + alignment
        version — not from any threshold — so a threshold change rewrites a
        point in place and leaves nothing stale.  Two cases do leave something
        behind: a face that drops below the review gate produces no point at
        all, and a detector change shifts bboxes so the new observation lands
        on a different id.  In both the old point survives, still carrying its
        old provenance and, if it was embedded, still in the search space.

        Scoped to ``is_face`` deliberately: the medium's marker and any video
        rollup are not observations, and sweeping them up here would delete the
        very counts that describe the medium.
        """
        flt = Filter(
            must=[
                FieldCondition(key="is_face", match=MatchValue(value=True)),
                FieldCondition(key="image_hash", match=MatchValue(value=image_hash)),
            ]
        )
        return [
            {"id": rec.id, **(rec.payload or {})}
            for rec in qdrant_scroll_all(
                self.client,
                self.collection,
                scroll_filter=flt,
                limit=_SCROLL_LIMIT,
                with_payload=list(self._STALE_FIELDS),
            )
            if rec.id not in produced_ids
        ]

    def delete_face_points(self, point_ids: list[str]) -> PurgeResult:
        """Delete named observations, reporting the chip hashes they freed.

        The caller must still filter those hashes through
        unreferenced_chip_hashes() before unlinking: chips are content
        addressed and a surviving observation may share one.
        """
        if not point_ids:
            return PurgeResult(n_points=0, chip_hashes=[])
        chip_hashes: list[str] = []
        for rec in self.client.retrieve(
            collection_name=self.collection, ids=list(point_ids), with_payload=True
        ):
            payload = rec.payload or {}
            for key in ("aligned_chip_hash", "review_chip_hash"):
                chash = payload.get(key)
                if chash:
                    chip_hashes.append(chash)
        self.client.delete(
            collection_name=self.collection, points_selector=PointIdsList(points=list(point_ids))
        )
        return PurgeResult(n_points=len(point_ids), chip_hashes=list(dict.fromkeys(chip_hashes)))

    def _purge_by_filter(self, flt: Filter) -> PurgeResult:
        ids: list = []
        chip_hashes: list[str] = []
        for rec in qdrant_scroll_all(
            self.client, self.collection, scroll_filter=flt, limit=_SCROLL_LIMIT, with_payload=True
        ):
            ids.append(rec.id)
            payload = rec.payload or {}
            for key in ("aligned_chip_hash", "review_chip_hash"):
                chash = payload.get(key)
                if chash:
                    chip_hashes.append(chash)
        self.client.delete(
            collection_name=self.collection, points_selector=PointIdsList(points=ids)
        )
        # Deduplicated: one review JPEG is shared by every observation whose
        # source crop is byte-identical, so the raw list repeats hashes and the
        # caller's unlinked-file count goes into the audit record.
        return PurgeResult(n_points=len(ids), chip_hashes=list(dict.fromkeys(chip_hashes)))

    def purge_media(self, image_hash: str) -> PurgeResult:
        """Delete every face point for one medium; caller unlinks the chips."""
        flt = Filter(must=[FieldCondition(key="image_hash", match=MatchValue(value=image_hash))])
        return self._purge_by_filter(flt)

    def purge_all(self) -> PurgeResult:
        """Delete all face and marker points, preserving the meta point.

        Never delete_collection: the enablement record is an auditable act
        and must survive routine purges.
        """
        flt = Filter(
            should=[
                FieldCondition(key="is_face", match=MatchValue(value=True)),
                FieldCondition(key="is_face_marker", match=MatchValue(value=True)),
                FieldCondition(key="is_face_video_rollup", match=MatchValue(value=True)),
            ]
        )
        return self._purge_by_filter(flt)
