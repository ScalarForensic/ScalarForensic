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
_SOFT_FIELDS = ("detector_id", "detector_model_hash", "detect_max_size", "min_conf", "min_size")


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

    def _purge_by_filter(self, flt: Filter) -> PurgeResult:
        ids: list = []
        chip_hashes: list[str] = []
        for rec in qdrant_scroll_all(
            self.client, self.collection, scroll_filter=flt, limit=_SCROLL_LIMIT, with_payload=True
        ):
            ids.append(rec.id)
            chash = (rec.payload or {}).get("chip_hash")
            if chash:
                chip_hashes.append(chash)
        self.client.delete(
            collection_name=self.collection, points_selector=PointIdsList(points=ids)
        )
        return PurgeResult(n_points=len(ids), chip_hashes=chip_hashes)

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
            ]
        )
        return self._purge_by_filter(flt)
