"""Analysis and query pipeline for the web frontend.

Imports from the existing backend — no modifications to those modules.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings
from scalar_forensic.embedder import (
    AnyEmbedder,
    hash_bytes,
    hash_bytes_md5,
    load_embedder,
    preprocess_batch,
)
from scalar_forensic.web.session import FileEntry, Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedder cache — models are expensive to load; keep them alive per process
# ---------------------------------------------------------------------------

_embedder_cache: dict[str, AnyEmbedder] = {}


def _get_embedder(key: str, settings: Settings) -> AnyEmbedder:
    if key not in _embedder_cache:
        if key == "sscd":
            local_model = settings.model_sscd
            use_sscd = True
        else:
            local_model = settings.model_dino
            use_sscd = False
        if settings.embedding_endpoint:
            if not settings.embedding_model:
                raise ValueError(
                    "SFN_EMBEDDING_MODEL must be set when SFN_EMBEDDING_ENDPOINT is configured."
                )
            effective_model = settings.embedding_model
        else:
            effective_model = local_model
        _embedder_cache[key] = load_embedder(
            model=effective_model,
            use_sscd=use_sscd,
            device=settings.device,
            normalize_size=settings.normalize_size,
            remote_endpoint=settings.embedding_endpoint,
            remote_api_key=settings.embedding_api_key,
            embedding_dim=settings.embedding_dim,
        )
    return _embedder_cache[key]


# ---------------------------------------------------------------------------
# Phase 1: Analysis
# ---------------------------------------------------------------------------


@dataclass
class ProgressEvent:
    type: str  # "progress" | "file_done" | "error" | "done"
    current: int = 0
    total: int = 0
    filename: str = ""
    file_id: str = ""
    message: str = ""
    session_id: str = ""


def analyze_session(
    session: Session,
    modes: list[str],
    settings: Settings,
) -> Generator[ProgressEvent]:
    """Hash and embed every file in the session. Yields progress events."""
    need_sscd = "altered" in modes
    need_dino = "semantic" in modes

    embedders: dict[str, AnyEmbedder] = {}
    if need_sscd:
        try:
            embedders["sscd"] = _get_embedder("sscd", settings)
        except Exception as exc:  # noqa: BLE001
            yield ProgressEvent(type="error", message=f"SSCD model load failed: {exc}")
    if need_dino:
        try:
            embedders["dino"] = _get_embedder("dino", settings)
        except Exception as exc:  # noqa: BLE001
            yield ProgressEvent(type="error", message=f"DINOv2 model load failed: {exc}")

    total = len(session.files)
    for i, entry in enumerate(session.files):
        yield ProgressEvent(
            type="progress",
            current=i,
            total=total,
            filename=entry.filename,
            file_id=entry.file_id,
        )
        try:
            _analyze_file(entry, embedders)
            yield ProgressEvent(
                type="file_done",
                current=i + 1,
                total=total,
                filename=entry.filename,
                file_id=entry.file_id,
            )
        except Exception as exc:  # noqa: BLE001
            entry.error = str(exc)
            yield ProgressEvent(
                type="error",
                current=i + 1,
                total=total,
                filename=entry.filename,
                file_id=entry.file_id,
                message=str(exc),
            )

    yield ProgressEvent(type="done", total=total, session_id=session.session_id)


def _analyze_file(entry: FileEntry, embedders: dict[str, AnyEmbedder]) -> None:
    data = entry.temp_path.read_bytes()
    entry.file_hash = hash_bytes(data)
    entry.file_hash_md5 = hash_bytes_md5(data)
    if not embedders:
        return
    pre_images = preprocess_batch([data])
    for key, embedder in embedders.items():
        norm_images = embedder.normalize_batch_bytes(pre_images)
        emb = embedder.embed_images(norm_images)[0]
        if key == "sscd":
            entry.sscd_embedding = emb
        else:
            entry.dino_embedding = emb


# ---------------------------------------------------------------------------
# Phase 2: Query
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    path: str
    scores: dict  # mode → score, e.g. {"exact": 1.0, "altered": 0.97, "semantic": 0.93}
    exif: bool | None = None
    exif_geo_data: bool | None = None

    def best_score(self) -> float:
        return max(self.scores.values(), default=0.0)


@dataclass
class FileResult:
    file_id: str
    filename: str
    hits: list[Hit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class QueryProvenance:
    modes: list[str]
    threshold_altered: float
    threshold_semantic: float
    limit: int
    timestamp: str  # ISO 8601 UTC, e.g. "2026-04-10T14:32:00.123456+00:00"


def query_session(
    session: Session,
    modes: list[str],
    threshold_altered: float,
    threshold_semantic: float,
    limit: int,
    settings: Settings,
) -> list[FileResult]:
    """Query Qdrant for every file in the session using stored embeddings.

    Fast — no re-embedding. Called on every slider change.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    results: list[FileResult] = []

    for entry in session.files:
        if entry.error:
            results.append(
                FileResult(
                    file_id=entry.file_id,
                    filename=entry.filename,
                    errors=[f"analysis failed: {entry.error}"],
                )
            )
            continue

        file_result = FileResult(file_id=entry.file_id, filename=entry.filename)
        all_hits: list[Hit] = []

        if "exact" in modes and entry.file_hash:
            exact_hits, errs = _query_exact(client, entry.file_hash, entry.file_hash_md5, settings)
            all_hits.extend(exact_hits[:limit])
            file_result.errors.extend(errs)

        if "altered" in modes and entry.sscd_embedding:
            hits, errs = _query_vector(
                client,
                collection=settings.collection_sscd,
                vector=entry.sscd_embedding,
                mode="altered",
                threshold=threshold_altered,
                limit=limit,
            )
            seen_altered: dict[str, Hit] = {}
            for h in hits:
                if h.path not in seen_altered:
                    seen_altered[h.path] = h
            all_hits.extend(
                sorted(seen_altered.values(), key=lambda h: (-h.best_score(), h.path))[:limit]
            )
            file_result.errors.extend(errs)

        if "semantic" in modes and entry.dino_embedding:
            hits, errs = _query_vector(
                client,
                collection=settings.collection_dino,
                vector=entry.dino_embedding,
                mode="semantic",
                threshold=threshold_semantic,
                limit=limit,
            )
            seen_semantic: dict[str, Hit] = {}
            for h in hits:
                if h.path not in seen_semantic:
                    seen_semantic[h.path] = h
            all_hits.extend(
                sorted(seen_semantic.values(), key=lambda h: (-h.best_score(), h.path))[:limit]
            )
            file_result.errors.extend(errs)

        # Sort combined list deterministically: best score descending, then path for tie-breaking
        # No global limit — each mode independently contributed up to `limit` hits
        file_result.hits = sorted(all_hits, key=lambda h: (-h.best_score(), h.path))
        results.append(file_result)

    return results


def _query_exact(
    client: QdrantClient,
    image_hash: str,
    image_hash_md5: str | None,
    settings: Settings,
) -> tuple[list[Hit], list[str]]:
    """Return exact SHA-256 hash matches and detect MD5 collisions."""
    hits: list[Hit] = []
    errors: list[str] = []
    collision_paths: set[str] = set()
    for collection in (settings.collection_sscd, settings.collection_dino):
        try:
            records, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="image_hash", match=MatchValue(value=image_hash))]
                ),
                limit=50,
                with_payload=["image_path", "exif", "exif_geo_data"],
                with_vectors=False,
            )
            for r in records:
                path = r.payload.get("image_path", "")
                if not any(h.path == path for h in hits):
                    hits.append(
                        Hit(
                            path=path,
                            scores={"exact": 1.0},
                            exif=r.payload.get("exif"),
                            exif_geo_data=r.payload.get("exif_geo_data"),
                        )
                    )

            # Collision detection: find images with same MD5 but different SHA-256
            if image_hash_md5:
                md5_records, _ = client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="image_hash_md5", match=MatchValue(value=image_hash_md5)
                            )
                        ]
                    ),
                    limit=50,
                    with_payload=["image_path", "image_hash"],
                    with_vectors=False,
                )
                for r in md5_records:
                    stored_sha256 = r.payload.get("image_hash", "")
                    if stored_sha256 and stored_sha256 != image_hash:
                        path = r.payload.get("image_path", "")
                        if path not in collision_paths:
                            collision_paths.add(path)
                            errors.append(
                                f"MD5 collision: '{path}' has the same MD5 ({image_hash_md5}) "
                                f"but a different SHA-256 ({stored_sha256})"
                            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exact query failed on %s: %s", collection, exc)
            errors.append(f"exact query failed: {type(exc).__name__}")
    return hits, errors


def _query_vector(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    mode: str,
    threshold: float,
    limit: int,
) -> tuple[list[Hit], list[str]]:
    try:
        result = client.query_points(
            collection_name=collection,
            query=vector,
            score_threshold=threshold,
            limit=limit,
            with_payload=["image_path", "exif", "exif_geo_data"],
        )
        return [
            Hit(
                path=r.payload.get("image_path", ""),
                scores={mode: r.score},
                exif=r.payload.get("exif"),
                exif_geo_data=r.payload.get("exif_geo_data"),
            )
            for r in result.points
        ], []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector query failed on %s (%s): %s", collection, mode, exc)
        return [], [f"{mode} query failed: {type(exc).__name__}"]


# ---------------------------------------------------------------------------
# Collection availability
# ---------------------------------------------------------------------------


async def get_available_modes(settings: Settings) -> tuple[list[str], str | None]:
    """Return which query modes are usable based on existing Qdrant collections.

    Retries up to 4 times (initial + 3 retries) with exponential backoff (1s/2s/4s)
    so transient startup delays don't immediately surface as errors.
    Returns a tuple of (modes, error_message); error_message is None on success.
    """
    _delays = [1, 2, 4]
    last_exc: Exception | None = None

    for attempt in range(4):
        if attempt > 0:
            await asyncio.sleep(_delays[attempt - 1])
        try:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            existing = {c.name for c in client.get_collections().collections}
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Qdrant connection attempt %d/4 failed at %s: %s",
                attempt + 1,
                settings.qdrant_url,
                exc,
            )
    else:
        return [], str(last_exc)

    if not existing:
        return [], None

    modes: list[str] = ["exact"]  # exact works as long as any collection exists
    if settings.collection_sscd in existing:
        modes.append("altered")
    if settings.collection_dino in existing:
        modes.append("semantic")
    return modes, None
