"""Analysis and query pipeline for the web frontend.

Imports from the existing backend — no modifications to those modules.
"""

from __future__ import annotations

import asyncio
import logging
import statistics as _statistics
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings
from scalar_forensic.embedder import (
    AnyEmbedder,
    hash_bytes,
    hash_bytes_md5,
    hash_file,
    hash_file_md5,
    load_embedder,
    preprocess_batch,
    preprocess_pil_batch,
)
from scalar_forensic.video import VIDEO_EXTENSIONS, extract_frames
from scalar_forensic.web.session import FileEntry, Session, VideoFrameEntry

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
            local_files_only=not settings.allow_online,
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
            if Path(entry.filename).suffix.lower() in VIDEO_EXTENSIONS:
                yield from _analyze_video_file(entry, embedders)
            else:
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
    pre_results = preprocess_batch([data])
    result = pre_results[0]
    if isinstance(result, Exception):
        raise result.with_traceback(result.__traceback__)
    pre_images = [result]
    for key, embedder in embedders.items():
        norm_images = embedder.normalize_batch_bytes(pre_images)
        emb = embedder.embed_images(norm_images)[0]
        if key == "sscd":
            entry.sscd_embedding = emb
        else:
            entry.dino_embedding = emb


_VIDEO_FRAME_BATCH = 32  # frames processed per embedding batch in the web pipeline


def _analyze_video_file(
    entry: FileEntry, embedders: dict[str, AnyEmbedder]
) -> Generator[ProgressEvent, None, None]:
    """Extract frames from an uploaded video temp file and embed each one.

    Frames are extracted via the seek-based generator and processed in batches
    of ``_VIDEO_FRAME_BATCH`` so peak memory is bounded regardless of video
    length.  The video file is hashed in a streaming pass rather than loaded
    into RAM.

    Yields ``video_progress`` ProgressEvents after each batch so the web UI
    can show how many frames have been processed while a long video is running.
    """
    from itertools import batched as _batched

    from scalar_forensic.config import Settings

    # Chunked hash — avoids reading the whole upload into memory just to hash it
    entry.file_hash = hash_file(entry.temp_path)
    entry.file_hash_md5 = hash_file_md5(entry.temp_path)
    entry.is_video = True

    settings = Settings()
    frame_entries: list[VideoFrameEntry] = []

    try:
        gen = extract_frames(
            entry.temp_path, fps=settings.video_fps, max_frames=settings.video_max_frames
        )
        for raw_batch in _batched(gen, _VIDEO_FRAME_BATCH):
            frames = list(raw_batch)
            if not frames:
                continue

            pil_images = preprocess_pil_batch([f.image for f in frames])

            # Create / extend frame_entries for this batch
            batch_start = len(frame_entries)
            for frame in frames:
                frame_entries.append(
                    VideoFrameEntry(
                        frame_index=frame.frame_index,
                        timecode_ms=frame.timecode_ms,
                        frame_hash=frame.frame_hash,
                    )
                )

            for key, embedder in embedders.items():
                norm_images = embedder.normalize_batch_bytes(pil_images)
                embeddings = embedder.embed_images(norm_images)
                for j, emb in enumerate(embeddings):
                    fe = frame_entries[batch_start + j]
                    if key == "sscd":
                        fe.sscd_embedding = emb
                    else:
                        fe.dino_embedding = emb

            yield ProgressEvent(
                type="video_progress",
                current=len(frame_entries),
                total=settings.video_max_frames,
                filename=entry.filename,
                file_id=entry.file_id,
            )

    except Exception as exc:
        raise RuntimeError(f"Video frame extraction failed: {exc}") from exc

    if not frame_entries:
        raise RuntimeError("No frames could be extracted from the video")

    entry.video_frames = frame_entries

    # The first frame's embeddings serve as the entry's top-level vector.
    # query_session() iterates entry.video_frames to query with each frame.
    entry.sscd_embedding = frame_entries[0].sscd_embedding
    entry.dino_embedding = frame_entries[0].dino_embedding


# ---------------------------------------------------------------------------
# Phase 2: Query
# ---------------------------------------------------------------------------


@dataclass
class MatchedVideoFrame:
    timecode_ms: int
    frame_hash: str
    scores: dict  # mode → score


@dataclass
class Hit:
    path: str
    scores: dict  # mode → score, e.g. {"exact": 1.0, "altered": 0.97, "semantic": 0.93}
    exif: bool | None = None
    exif_geo_data: bool | None = None
    image_hash: str | None = None  # SHA-256, used to request /api/thumbnail/{hash}
    model_provenance: dict = field(default_factory=dict)  # mode → {name, hash} from Qdrant payload
    # Video-frame fields (only set when is_video_frame is True)
    is_video_frame: bool = False
    video_path: str | None = None
    video_hash: str | None = None
    frame_timecode_ms: int | None = None
    # After grouping: one Hit per video with all matched frames attached
    matched_frames: list[MatchedVideoFrame] | None = None
    # Query-video frame timecodes (ms) that generated this hit.  Set only for
    # video uploads; None for single-image queries and exact-hash matches.
    query_timecodes: list[int] | None = None

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


_MODE_PRIORITY: dict[str, int] = {"exact": 0, "altered": 1, "semantic": 2}


def _hit_sort_key(h: Hit) -> tuple:
    """Unified sort: exact first, then best non-exact score desc, then path."""
    has_exact = "exact" in h.scores
    non_exact_scores = [v for k, v in h.scores.items() if k != "exact"]
    best_non_exact = max(non_exact_scores, default=0.0)
    return (0 if has_exact else 1, -best_non_exact, h.path)


def _unmerged_sort_key(h: Hit) -> tuple:
    """Unmerged sort: group by mode (exact → altered → semantic), then score desc, then path."""
    mode = next(iter(h.scores), "")
    priority = _MODE_PRIORITY.get(mode, 99)
    score = next(iter(h.scores.values()), 0.0)
    return (priority, -score, h.path)


def _merge_hit(h: Hit, dest: dict[str, Hit]) -> None:
    """Merge a hit into the accumulator, combining scores from multiple modes.

    Scores are merged with max() so a later, lower score for the same mode
    never downgrades an earlier, higher one.  Model provenance uses setdefault
    so the first-seen value is kept; it doesn't change across duplicate points.
    query_timecodes are accumulated (union, insertion-ordered).
    """
    if h.path in dest:
        existing = dest[h.path]
        for mode, score in h.scores.items():
            existing.scores[mode] = max(existing.scores.get(mode, score), score)
        if h.image_hash and not existing.image_hash:
            existing.image_hash = h.image_hash
        for mode, provenance in h.model_provenance.items():
            existing.model_provenance.setdefault(mode, provenance)
        if h.query_timecodes:
            if existing.query_timecodes is None:
                existing.query_timecodes = list(h.query_timecodes)
            else:
                for tc in h.query_timecodes:
                    if tc not in existing.query_timecodes:
                        existing.query_timecodes.append(tc)
    else:
        dest[h.path] = h


def _group_video_hits(hits: list[Hit]) -> list[Hit]:
    """Collapse multiple video-frame hits from the same source video into one Hit.

    Non-video hits pass through unchanged.  For each unique ``video_path``,
    keeps the frame with the best score as the representative Hit and attaches
    all matched frames in ``matched_frames`` for the timeline UI.
    """
    non_video: list[Hit] = []
    video_groups: dict[str, list[Hit]] = {}

    for h in hits:
        if h.is_video_frame and h.video_path:
            video_groups.setdefault(h.video_path, []).append(h)
        else:
            non_video.append(h)

    grouped_video: list[Hit] = []
    for vpath, group in video_groups.items():
        # Best frame = highest best_score
        representative = max(group, key=lambda h: h.best_score())
        matched = [
            MatchedVideoFrame(
                timecode_ms=h.frame_timecode_ms or 0,
                frame_hash=h.image_hash or "",
                scores=h.scores,
            )
            for h in sorted(group, key=lambda h: h.frame_timecode_ms or 0)
        ]
        # Union of query-frame timecodes across every hit in the group
        all_qtc: list[int] = []
        for h in group:
            for tc in h.query_timecodes or []:
                if tc not in all_qtc:
                    all_qtc.append(tc)
        grouped_video.append(
            Hit(
                path=representative.path,
                scores=representative.scores,
                exif=representative.exif,
                exif_geo_data=representative.exif_geo_data,
                image_hash=representative.image_hash,
                model_provenance=representative.model_provenance,
                is_video_frame=True,
                video_path=vpath,
                video_hash=representative.video_hash,
                frame_timecode_ms=representative.frame_timecode_ms,
                matched_frames=matched,
                query_timecodes=all_qtc if all_qtc else None,
            )
        )

    # Re-sort combined list
    return sorted(non_video + grouped_video, key=_hit_sort_key)


def query_session(
    session: Session,
    modes: list[str],
    threshold_altered: float,
    threshold_semantic: float,
    limit: int,
    settings: Settings,
    unify: bool = True,
) -> tuple[list[FileResult], dict[str, dict]]:
    """Query Qdrant for every file in the session using stored embeddings.

    Fast — no re-embedding. Called on every slider change.

    When *unify* is True (default) hits from different modes are merged by path
    so each DB image produces one result row with all applicable scores.
    When *unify* is False each mode contributes its own rows independently,
    so the same image may appear multiple times with different scores.

    Returns a tuple of (results, embedding_models) where embedding_models maps
    mode → {name, hash} for each embedder currently loaded in the web process.
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
        merged: dict[str, Hit] = {}
        unmerged: list[Hit] = []

        if "exact" in modes and entry.file_hash:
            exact_hits, errs = _query_exact(client, entry.file_hash, entry.file_hash_md5, settings)
            for h in exact_hits:
                if unify:
                    _merge_hit(h, merged)
                else:
                    unmerged.append(h)
            file_result.errors.extend(errs)

        # For video uploads query with every frame's embedding so all relevant
        # matches surface, not just matches for the first frame.
        # For images use the single top-level embedding as before.
        # Each entry is (vector, query_timecode_ms); timecode is None for images.
        sscd_vecs: list[tuple[list[float], int | None]] = []
        dino_vecs: list[tuple[list[float], int | None]] = []
        if entry.is_video and entry.video_frames:
            for vf in entry.video_frames:
                if vf.sscd_embedding:
                    sscd_vecs.append((vf.sscd_embedding, vf.timecode_ms))
                if vf.dino_embedding:
                    dino_vecs.append((vf.dino_embedding, vf.timecode_ms))
        else:
            if entry.sscd_embedding:
                sscd_vecs.append((entry.sscd_embedding, None))
            if entry.dino_embedding:
                dino_vecs.append((entry.dino_embedding, None))

        if "altered" in modes:
            for vec, qtc in sscd_vecs:
                hits, errs = _query_vector(
                    client,
                    collection=settings.collection_sscd,
                    vector=vec,
                    mode="altered",
                    threshold=threshold_altered,
                    limit=limit,
                )
                for h in hits:
                    if qtc is not None:
                        h.query_timecodes = [qtc]
                    if unify:
                        _merge_hit(h, merged)
                    else:
                        unmerged.append(h)
                file_result.errors.extend(errs)

        if "semantic" in modes:
            for vec, qtc in dino_vecs:
                hits, errs = _query_vector(
                    client,
                    collection=settings.collection_dino,
                    vector=vec,
                    mode="semantic",
                    threshold=threshold_semantic,
                    limit=limit,
                )
                for h in hits:
                    if qtc is not None:
                        h.query_timecodes = [qtc]
                    if unify:
                        _merge_hit(h, merged)
                    else:
                        unmerged.append(h)
                file_result.errors.extend(errs)

        if unify:
            flat_hits = sorted(merged.values(), key=_hit_sort_key)[:limit]
        else:
            flat_hits = sorted(unmerged, key=_unmerged_sort_key)
        file_result.hits = _group_video_hits(flat_hits)
        results.append(file_result)

    # Collect model provenance for the embedders currently loaded in this process
    embedding_models: dict[str, dict] = {}
    if "altered" in modes:
        emb = _embedder_cache.get("sscd")
        if emb is not None:
            embedding_models["altered"] = {"name": emb.model_name, "hash": emb.model_hash}
    if "semantic" in modes:
        emb = _embedder_cache.get("dino")
        if emb is not None:
            embedding_models["semantic"] = {"name": emb.model_name, "hash": emb.model_hash}

    return results, embedding_models


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
    # Map each collection to the mode label used in model_provenance
    collection_mode = {
        settings.collection_sscd: "altered",
        settings.collection_dino: "semantic",
    }
    for collection in (settings.collection_sscd, settings.collection_dino):
        mode_key = collection_mode[collection]
        try:
            records, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="image_hash", match=MatchValue(value=image_hash))]
                ),
                limit=50,
                with_payload=[
                    "image_path",
                    "image_hash",
                    "exif",
                    "exif_geo_data",
                    "model_name",
                    "model_hash",
                ],
                with_vectors=False,
            )
            for r in records:
                path = r.payload.get("image_path", "")
                mn = r.payload.get("model_name", "")
                mh = r.payload.get("model_hash", "")
                mp: dict = {mode_key: {"name": mn, "hash": mh}} if (mn or mh) else {}
                existing = next((h for h in hits if h.path == path), None)
                if existing is None:
                    hits.append(
                        Hit(
                            path=path,
                            scores={"exact": 1.0},
                            exif=r.payload.get("exif"),
                            exif_geo_data=r.payload.get("exif_geo_data"),
                            image_hash=r.payload.get("image_hash"),
                            model_provenance=mp,
                        )
                    )
                else:
                    # Image found in a second collection — merge its model provenance
                    existing.model_provenance.update(mp)

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
            msg = str(exc).lower()
            if "not found" in msg or "doesn't exist" in msg:
                # Collection not yet indexed — not an error, just skip it
                logger.debug("Exact query skipped non-existent collection %s", collection)
            else:
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
            with_payload=[
                "image_path",
                "image_hash",
                "exif",
                "exif_geo_data",
                "model_name",
                "model_hash",
                "is_video_frame",
                "video_path",
                "video_hash",
                "frame_timecode_ms",
            ],
        )
        hits = []
        for r in result.points:
            mp: dict = {}
            mn = r.payload.get("model_name", "")
            mh = r.payload.get("model_hash", "")
            if mn or mh:
                mp[mode] = {"name": mn, "hash": mh}
            hits.append(
                Hit(
                    path=r.payload.get("image_path", ""),
                    scores={mode: r.score},
                    exif=r.payload.get("exif"),
                    exif_geo_data=r.payload.get("exif_geo_data"),
                    image_hash=r.payload.get("image_hash"),
                    model_provenance=mp,
                    is_video_frame=bool(r.payload.get("is_video_frame")),
                    video_path=r.payload.get("video_path"),
                    video_hash=r.payload.get("video_hash"),
                    frame_timecode_ms=r.payload.get("frame_timecode_ms"),
                )
            )
        return hits, []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector query failed on %s (%s): %s", collection, mode, exc)
        return [], [f"{mode} query failed: {type(exc).__name__}"]


# ---------------------------------------------------------------------------
# Semantic score distribution stats (on-demand, per uploaded file)
# ---------------------------------------------------------------------------

_STATS_SAMPLE = 10_000
_HIST_BUCKETS = 20  # 0.05-wide buckets covering normalised [0.0, 1.0] (cosine [-1,1] → [0,1])


@dataclass
class SemanticStats:
    sample_size: int  # how many points were requested
    count: int  # how many were actually returned
    min_score: float
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    max_score: float
    mean: float
    stdev: float
    # _HIST_BUCKETS counts on normalised [0,1] scale; bucket i covers [i*0.05, (i+1)*0.05)
    histogram: list[int]


def query_semantic_stats(
    session: Session,
    file_id: str,
    settings: Settings,
    sample_size: int = _STATS_SAMPLE,
) -> tuple[SemanticStats | None, str | None]:
    """Return score-distribution stats for one uploaded file against the DINOv2 collection.

    Queries the top-*sample_size* most-similar points with no score threshold so the
    result covers the full relevant tail.  Returns (stats, None) on success or
    (None, error_message) on failure.
    """
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None:
        return None, "file not found in session"
    if entry.error:
        return None, f"analysis failed: {entry.error}"
    if not entry.dino_embedding:
        return None, "no semantic embedding — was semantic mode selected during analysis?"

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    try:
        result = client.query_points(
            collection_name=settings.collection_dino,
            query=entry.dino_embedding,
            limit=sample_size,
            with_payload=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic stats query failed: %s", exc)
        return None, f"Qdrant query failed: {type(exc).__name__}"

    scores = [r.score for r in result.points]
    if not scores:
        return None, "no points returned from collection (collection may be empty)"

    n = len(scores)
    mean = sum(scores) / n
    stdev = _statistics.stdev(scores) if n >= 2 else 0.0

    if n >= 2:
        cuts = _statistics.quantiles(scores, n=100, method="inclusive")
        p10 = cuts[9]
        p25 = cuts[24]
        median = cuts[49]
        p75 = cuts[74]
        p90 = cuts[89]
    else:
        p10 = p25 = median = p75 = p90 = scores[0]

    histogram = [0] * _HIST_BUCKETS
    for s in scores:
        # Normalise cosine score from [-1, 1] to [0, 1] before bucketing.
        # Note: the numeric stats (min/max/percentiles) are on the raw cosine scale;
        # the histogram intentionally uses the normalised scale so the UI threshold
        # marker (also normalised before bucket lookup) aligns correctly.
        normalised = (max(-1.0, min(s, 1.0)) + 1.0) / 2.0
        idx = min(max(int(normalised * _HIST_BUCKETS), 0), _HIST_BUCKETS - 1)
        histogram[idx] += 1

    return SemanticStats(
        sample_size=sample_size,
        count=n,
        min_score=min(scores),
        p10=p10,
        p25=p25,
        median=median,
        p75=p75,
        p90=p90,
        max_score=max(scores),
        mean=mean,
        stdev=stdev,
        histogram=histogram,
    ), None


# ---------------------------------------------------------------------------
# Full indexing provenance for a single image (on-demand, for Audit modal)
# ---------------------------------------------------------------------------

_PROVENANCE_FIELDS = [
    "model_name",
    "model_hash",
    "indexed_at",
    "library_versions",
    "inference_dtype",
    "normalize_size",
    "embedding_dim",
]


def get_hit_qdrant_provenance(image_hash: str, settings: Settings) -> dict[str, dict]:
    """Fetch full indexing-time provenance for one image hash from Qdrant.

    Returns a dict keyed by mode ("altered", "semantic") containing all
    provenance fields stored in the point payload when the image was indexed.
    Missing collections are silently skipped.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    collection_mode = {
        settings.collection_sscd: "altered",
        settings.collection_dino: "semantic",
    }
    result: dict[str, dict] = {}
    for collection, mode in collection_mode.items():
        try:
            records, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="image_hash", match=MatchValue(value=image_hash))]
                ),
                limit=1,
                with_payload=_PROVENANCE_FIELDS,
                with_vectors=False,
            )
            if records:
                result[mode] = {k: records[0].payload.get(k) for k in _PROVENANCE_FIELDS}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Provenance query failed on %s: %s", collection, exc)
    return result


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
