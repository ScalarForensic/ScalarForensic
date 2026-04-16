"""Qdrant collection management and vector upsert."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasVectorCondition,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    PointVectors,
    VectorParams,
)


class Indexer:
    def __init__(
        self,
        url: str,
        collection: str,
        vector_name: str,
        embedding_dim: int,
        api_key: str | None = None,
        initial_vectors_config: dict[str, VectorParams] | None = None,
    ) -> None:
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.vector_name = vector_name
        self._ensure_collection(vector_name, embedding_dim, initial_vectors_config)

    def _ensure_collection(
        self,
        vector_name: str,
        dim: int,
        initial_vectors_config: dict[str, VectorParams] | None,
    ) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            # Use the full multi-vector config when provided so all vector types
            # are registered in one shot; fall back to just this vector otherwise.
            create_config = initial_vectors_config or {
                vector_name: VectorParams(size=dim, distance=Distance.COSINE)
            }
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=create_config,
            )
            info = self.client.get_collection(self.collection)
        else:
            info = self.client.get_collection(self.collection)
            vectors_config = info.config.params.vectors
            if not isinstance(vectors_config, dict):
                raise ValueError(
                    f"Collection '{self.collection}' uses a legacy single-vector format. "
                    "Drop it (e.g. via qdrant-client or the Qdrant dashboard) and re-index."
                )
            if vector_name not in vectors_config:
                # Qdrant supports adding new named vector types to an existing
                # collection without touching existing data.  This enables
                # incremental indexing: index --dino in one run, add --sscd later.
                self.client.update_collection(
                    collection_name=self.collection,
                    vectors_config={
                        vector_name: VectorParams(size=dim, distance=Distance.COSINE)
                    },
                )
                info = self.client.get_collection(self.collection)
            else:
                existing_dim = vectors_config[vector_name].size
                if existing_dim != dim:
                    raise ValueError(
                        f"Vector '{vector_name}' in collection '{self.collection}' already "
                        f"exists with dim={existing_dim}, but the current model produces "
                        f"dim={dim}. Use a different collection or the matching model."
                    )
        # Ensure payload indexes exist (idempotent). info is available from both branches above.
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

    def get_all_indexed_hashes(self) -> set[str]:
        """Return the set of all image_hash values for points that have this vector.

        Performs a single paginated scroll rather than one query per batch,
        building an in-memory set for O(1) per-item lookup during ingestion.
        Only considers points that carry the vector type this Indexer manages.
        """
        result: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[HasVectorCondition(has_vector=self.vector_name)]),
                limit=10_000,
                with_payload=["image_hash"],
                with_vectors=False,
                offset=offset,
            )
            for r in records:
                if r.payload and "image_hash" in r.payload:
                    result.add(r.payload["image_hash"])
            if not records or offset is None:
                break
        return result

    def get_all_indexed_paths(self) -> set[str]:
        """Return the set of all image_path values for points that have this vector.

        Performs a single paginated scroll rather than one query per batch,
        building an in-memory set for O(1) per-item lookup during ingestion.
        Only considers points that carry the vector type this Indexer manages.
        """
        result: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[HasVectorCondition(has_vector=self.vector_name)]),
                limit=10_000,
                with_payload=["image_path"],
                with_vectors=False,
                offset=offset,
            )
            for r in records:
                if r.payload and "image_path" in r.payload:
                    result.add(r.payload["image_path"])
            if not records or offset is None:
                break
        return result

    def get_all_video_info(self) -> dict[str, dict]:
        """Return one payload record per distinct video_hash stored in the collection.

        The returned dict maps ``video_hash → {extraction_fps, max_frames_cap, complete}``
        where *complete* is True iff ``video_frames_total`` is present in the payload
        (written by :meth:`mark_video_complete` after a successful full index).

        Performs a single paginated scroll so the caller can evaluate video
        completeness locally from the returned metadata without per-video
        Qdrant queries.  Only considers points that carry the vector type this
        Indexer manages.
        """
        seen: dict[str, dict] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="is_video_frame", match=MatchValue(value=True)),
                        HasVectorCondition(has_vector=self.vector_name),
                    ]
                ),
                limit=10_000,
                with_payload=[
                    "video_hash",
                    "extraction_fps",
                    "max_frames_cap",
                    "video_frames_total",
                ],
                with_vectors=False,
                offset=offset,
            )
            for r in records:
                p = r.payload or {}
                vh = p.get("video_hash")
                if vh and vh not in seen:
                    seen[vh] = {
                        "extraction_fps": p.get("extraction_fps"),
                        "max_frames_cap": p.get("max_frames_cap"),
                        "complete": "video_frames_total" in p,
                    }
            if not records or offset is None:
                break
        return seen

    def mark_video_complete(self, video_hash: str, frame_count: int) -> None:
        """Set video_frames_total on every frame of this video as a completion marker.

        Called after all frames have been successfully upserted.  Its absence
        tells :meth:`get_all_video_info` that a previous run was interrupted and the
        video must be re-indexed.
        """
        self.client.set_payload(
            collection_name=self.collection,
            payload={"video_frames_total": frame_count},
            points=Filter(
                must=[FieldCondition(key="video_hash", match=MatchValue(value=video_hash))]
            ),
        )

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

        Each point stores its embedding under the named vector ``self.vector_name``
        (``"dino"`` or ``"sscd"``).  Model-specific provenance fields are prefixed
        with the vector name (e.g. ``dino_model_name``) so a single point can carry
        provenance for both models when indexed by both pipelines.

        Points that already exist in the collection (e.g. because the other model was
        indexed first) are updated in-place: only the new named vector and its provenance
        are written; the core payload and any other named vectors are left untouched.

        For video frames, pass ``video_metadata`` as a list of per-point dicts
        (or ``None`` entries for non-video points in a mixed batch).  When a
        dict is present for point ``i`` it must contain: ``video_hash``,
        ``video_path``, ``frame_timecode_ms``, ``frame_index``,
        ``extraction_fps``, ``max_frames_cap``, ``pyav_version``.  The point ID is derived from
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
        vn = self.vector_name
        indexed_at = datetime.now(UTC).isoformat()

        # Model-specific provenance — prefixed with the vector name so a single
        # point can hold provenance for multiple models side by side.
        model_provenance: dict = {
            f"{vn}_model_name": shared_metadata["model_name"],
            f"{vn}_model_hash": shared_metadata["model_hash"],
            f"{vn}_embedding_dim": shared_metadata["embedding_dim"],
            f"{vn}_normalize_size": shared_metadata["normalize_size"],
            f"{vn}_inference_dtype": shared_metadata["inference_dtype"],
            "library_versions": shared_metadata["library_versions"],
            f"{vn}_indexed_at": indexed_at,
            **(
                {"sscd_n_crops": shared_metadata["sscd_n_crops"]}
                if "sscd_n_crops" in shared_metadata
                else {}
            ),
        }

        # Build all point IDs and per-point data in one pass.
        point_ids: list[str] = []
        core_payloads: list[dict] = []
        vector_list: list[list[float]] = []

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
            point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, point_id_key)))

            core: dict = {
                # Forensic identifiers
                "image_hash": image_hash,
                **({"image_hash_md5": image_hashes_md5[i]} if image_hashes_md5 else {}),
                # Video frames carry a virtual path string (already absolute).
                # Regular images are resolved to an absolute path so that
                # get_indexed_paths() and /api/hit-image lookups always match.
                "image_path": (
                    str(image_path) if vmeta is not None else str(Path(image_path).resolve())
                ),
                # EXIF flags (only present when extraction is enabled)
                **(exif_payloads.get(image_path, {}) if exif_payloads else {}),
            }

            # Video-frame provenance fields
            if vmeta is not None:
                core["is_video_frame"] = True
                core["video_hash"] = vmeta["video_hash"]
                core["video_path"] = vmeta["video_path"]
                core["frame_timecode_ms"] = vmeta["frame_timecode_ms"]
                core["frame_index"] = vmeta["frame_index"]
                core["extraction_fps"] = vmeta["extraction_fps"]
                core["max_frames_cap"] = vmeta["max_frames_cap"]
                core["pyav_version"] = vmeta["pyav_version"]

            core_payloads.append(core)
            vector_list.append(embedding)

        # Determine which points already exist so we can do targeted updates
        # rather than full replacements (which would wipe the other model's vector).
        retrieved = self.client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_vectors=False,
            with_payload=False,
        )
        existing_ids = {r.id for r in retrieved}

        new_points: list[PointStruct] = []
        existing_vector_updates: list[PointVectors] = []
        existing_point_ids: list[str] = []

        for pid, core, emb in zip(point_ids, core_payloads, vector_list):
            if pid not in existing_ids:
                new_points.append(
                    PointStruct(
                        id=pid,
                        vector={vn: emb},
                        payload={**core, **model_provenance},
                    )
                )
            else:
                existing_vector_updates.append(PointVectors(id=pid, vector={vn: emb}))
                existing_point_ids.append(pid)

        if new_points:
            self.client.upsert(collection_name=self.collection, points=new_points)

        if existing_vector_updates:
            self.client.update_vectors(
                collection_name=self.collection, points=existing_vector_updates
            )
            # model_provenance is identical for every point in the batch — one call suffices.
            self.client.set_payload(
                collection_name=self.collection,
                payload=model_provenance,
                points=existing_point_ids,
            )
