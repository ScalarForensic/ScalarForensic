"""Qdrant collection management and vector upsert."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


class Indexer:
    def __init__(
        self, url: str, collection: str, embedding_dim: int, api_key: str | None = None
    ) -> None:
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self._ensure_collection(embedding_dim)

    def _ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            info = self.client.get_collection(self.collection)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                raise ValueError(
                    f"Collection '{self.collection}' uses named vectors — not supported."
                )
            existing_dim = vectors_config.size  # type: ignore[union-attr]
            if existing_dim != dim:
                raise ValueError(
                    f"Collection '{self.collection}' already exists with dim={existing_dim}, "
                    f"but the current model produces dim={dim}. "
                    f"Use a different collection or the matching backend."
                )
        # Ensure payload indexes exist (idempotent).
        info = self.client.get_collection(self.collection)
        schema = info.payload_schema or {}
        if "image_hash" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="image_hash",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        if "image_path" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="image_path",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        if "image_hash_md5" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="image_hash_md5",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        if "video_hash" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="video_hash",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        if "video_path" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="video_path",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        if "frame_timecode_ms" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="frame_timecode_ms",
                field_schema=PayloadSchemaType.INTEGER,
            )
        if "is_video_frame" not in schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="is_video_frame",
                field_schema=PayloadSchemaType.BOOL,
            )

    def get_indexed_hashes(self, hashes: list[str]) -> set[str]:
        """Return the subset of hashes that are already in the collection."""
        if not hashes:
            return set()
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                should=[FieldCondition(key="image_hash", match=MatchValue(value=h)) for h in hashes]
            ),
            limit=len(hashes),
            with_payload=["image_hash"],
            with_vectors=False,
        )
        return {r.payload["image_hash"] for r in results}

    def get_indexed_paths(self, paths: list[str]) -> set[str]:
        """Return the subset of absolute path strings already stored in the collection."""
        if not paths:
            return set()
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                should=[FieldCondition(key="image_path", match=MatchValue(value=p)) for p in paths]
            ),
            limit=len(paths),
            with_payload=["image_path"],
            with_vectors=False,
        )
        return {r.payload["image_path"] for r in results}

    def upsert_batch(
        self,
        image_paths: list[Path],
        image_hashes: list[str],
        embeddings: list[list[float]],
        shared_metadata: dict,
        exif_payloads: dict[Path, dict] | None = None,
        image_hashes_md5: list[str] | None = None,
        video_metadata: list[dict | None] | None = None,
    ) -> None:
        """Upsert vectors with full forensic metadata payload.

        For video frames, pass ``video_metadata`` as a list of per-point dicts
        (or ``None`` entries for non-video points in a mixed batch).  When a
        dict is present for point ``i`` it must contain: ``video_hash``,
        ``video_path``, ``frame_timecode_ms``, ``frame_index``,
        ``extraction_fps``, ``pyav_version``.  The point ID is derived from
        ``video_hash + ":" + str(frame_timecode_ms)`` to ensure per-video-frame
        uniqueness across different source files with identical frame content.
        """
        if not len(image_paths) == len(image_hashes) == len(embeddings):
            raise ValueError(
                f"Batch length mismatch: paths={len(image_paths)}, "
                f"hashes={len(image_hashes)}, embeddings={len(embeddings)}"
            )
        if image_hashes_md5 is not None and len(image_hashes_md5) != len(image_hashes):
            raise ValueError(
                f"MD5 hash list length mismatch: "
                f"sha256={len(image_hashes)}, md5={len(image_hashes_md5)}"
            )
        if video_metadata is not None and len(video_metadata) != len(image_hashes):
            raise ValueError(
                f"video_metadata length mismatch: "
                f"expected={len(image_hashes)}, got={len(video_metadata)}"
            )
        indexed_at = datetime.now(UTC).isoformat()
        points = []
        for i, (image_path, image_hash, embedding) in enumerate(
            zip(image_paths, image_hashes, embeddings)
        ):
            vmeta = video_metadata[i] if video_metadata is not None else None

            # Video frames get a unique ID per video+timecode so two different
            # videos with an identical frame produce two separate Qdrant points.
            if vmeta is not None:
                point_id_key = vmeta["video_hash"] + ":" + str(vmeta["frame_timecode_ms"])
            else:
                point_id_key = image_hash
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id_key))

            payload: dict = {
                # Forensic identifiers
                "image_hash": image_hash,
                **({"image_hash_md5": image_hashes_md5[i]} if image_hashes_md5 else {}),
                # Video frames carry a virtual path string (already absolute).
                # Regular images are resolved to an absolute path so that
                # get_indexed_paths() and /api/hit-image lookups always match.
                "image_path": (
                    str(image_path) if vmeta is not None else str(Path(image_path).resolve())
                ),
                "indexed_at": indexed_at,
                # Model & library provenance
                "model_name": shared_metadata["model_name"],
                "model_hash": shared_metadata["model_hash"],
                "embedding_dim": shared_metadata["embedding_dim"],
                "normalize_size": shared_metadata["normalize_size"],
                "library_versions": shared_metadata["library_versions"],
                # EXIF flags (only present when extraction is enabled)
                **(exif_payloads.get(image_path, {}) if exif_payloads else {}),
            }

            # Video-frame provenance fields
            if vmeta is not None:
                payload["is_video_frame"] = True
                payload["video_hash"] = vmeta["video_hash"]
                payload["video_path"] = vmeta["video_path"]
                payload["frame_timecode_ms"] = vmeta["frame_timecode_ms"]
                payload["frame_index"] = vmeta["frame_index"]
                payload["extraction_fps"] = vmeta["extraction_fps"]
                payload["pyav_version"] = vmeta["pyav_version"]

            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        self.client.upsert(collection_name=self.collection, points=points)
