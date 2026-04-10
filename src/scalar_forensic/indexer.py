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
    ) -> None:
        """Upsert vectors with full forensic metadata payload."""
        if not len(image_paths) == len(image_hashes) == len(embeddings):
            raise ValueError(
                f"Batch length mismatch: paths={len(image_paths)}, "
                f"hashes={len(image_hashes)}, embeddings={len(embeddings)}"
            )
        if image_hashes_md5 is not None and len(image_hashes_md5) != len(image_hashes):
            raise ValueError(
                f"MD5 hash list length mismatch: sha256={len(image_hashes)}, md5={len(image_hashes_md5)}"
            )
        indexed_at = datetime.now(UTC).isoformat()
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, image_hash)),
                vector=embedding,
                payload={
                    # Forensic identifiers
                    "image_hash": image_hash,
                    **({"image_hash_md5": image_hashes_md5[i]} if image_hashes_md5 else {}),
                    "image_path": str(image_path.resolve()),
                    "indexed_at": indexed_at,
                    # Model & library provenance
                    "model_name": shared_metadata["model_name"],
                    "model_hash": shared_metadata["model_hash"],
                    "embedding_dim": shared_metadata["embedding_dim"],
                    "normalize_size": shared_metadata["normalize_size"],
                    "library_versions": shared_metadata["library_versions"],
                    # EXIF flags (only present when extraction is enabled)
                    **(exif_payloads.get(image_path, {}) if exif_payloads else {}),
                },
            )
            for i, (image_path, image_hash, embedding) in enumerate(
                zip(image_paths, image_hashes, embeddings)
            )
        ]
        self.client.upsert(collection_name=self.collection, points=points)
