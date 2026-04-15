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
                yield from _analyze_video_file(entry, embedders, settings)
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


def _video_frame_batch(settings: Settings) -> int:
    """Return the effective video-frame batch size.

    Resolution order: explicit ``SFN_BATCH_SIZE`` > calibration cache > 32.
    """
    if settings.batch_size is not None:
        return settings.batch_size
    from scalar_forensic.calibration import load_cached_batch_size

    return load_cached_batch_size() or 32


def _analyze_video_file(
    entry: FileEntry, embedders: dict[str, AnyEmbedder], settings: Settings
) -> Generator[ProgressEvent, None, None]:
    """Extract frames from an uploaded video temp file and embed each one.

    Frames are extracted via the seek-based generator and processed in batches
    whose size is resolved via :func:`_video_frame_batch` (explicit config,
    calibration cache, or default 32) so peak memory is bounded regardless of
    video length.  The video file is hashed in a streaming pass rather than
    loaded into RAM.

    Yields ``video_progress`` ProgressEvents after each batch so the web UI
    can show how many frames have been processed while a long video is running.
    """
    from itertools import batched as _batched

    # Chunked hash — avoids reading the whole upload into memory just to hash it
    entry.file_hash = hash_file(entry.temp_path)
    entry.file_hash_md5 = hash_file_md5(entry.temp_path)
    entry.is_video = True
    frame_entries: list[VideoFrameEntry] = []
    _batch_sz = _video_frame_batch(settings)

    try:
        gen = extract_frames(
            entry.temp_path, fps=settings.video_fps, max_frames=settings.video_max_frames
        )
        for raw_batch in _batched(gen, _batch_sz):
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
    # A merged hit (after the final unify pass) may be associated with multiple
    # query frames, so this list can contain more than one timecode.
    query_timecodes: list[int] | None = None
    # Timecode (ms) of the query frame whose score was the highest among all
    # query frames that contributed to this hit.  Useful for the frontend to
    # auto-navigate to the most relevant query frame for a merged video hit.
    best_query_timecode_ms: int | None = None

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


def _merge_hit(h: Hit, dest: dict[str, Hit], key: str | None = None) -> None:
    """Merge a hit into an accumulator keyed by *key* (defaults to ``h.path``).

    Used in two contexts:
    1. Per-query-entity (within one image or video frame): combines ALTER and
       SEMAN scores for the same dataset path onto one card.
    2. Final unify pass (video queries only): folds hits for the same dataset
       video across different query frames; the caller passes ``video_path`` as
       the key so all query-frame × dataset-video pairs collapse to one card.

    Scores are kept at their highest observed value (max) so a later, lower
    score for the same mode never downgrades an earlier, higher one.
    """
    k = key if key is not None else h.path
    if k in dest:
        existing = dest[k]
        pre_merge_best = existing.best_score()

        for mode, score in h.scores.items():
            if mode not in existing.scores or score > existing.scores[mode]:
                existing.scores[mode] = score
                if mode in h.model_provenance:
                    existing.model_provenance[mode] = h.model_provenance[mode]

        if h.image_hash and not existing.image_hash:
            existing.image_hash = h.image_hash

        if h.query_timecodes:
            tc = h.query_timecodes[0]
            if existing.query_timecodes is None:
                existing.query_timecodes = [tc]
                existing._query_timecodes_seen: set = {tc}  # type: ignore[attr-defined]
                existing.best_query_timecode_ms = tc
            else:
                seen: set = getattr(
                    existing, "_query_timecodes_seen", set(existing.query_timecodes)
                )
                if tc not in seen:
                    existing.query_timecodes.append(tc)
                    seen.add(tc)
                    existing._query_timecodes_seen = seen  # type: ignore[attr-defined]
                # Update best_query_timecode_ms when the incoming hit contributes
                # a score higher than the current maximum across all modes.
                if h.best_score() > pre_merge_best:
                    existing.best_query_timecode_ms = tc

        if h.matched_frames:
            if existing.matched_frames is None:
                existing.matched_frames = list(h.matched_frames)
            else:
                # Merge by timecode: for duplicate timecodes keep the max score
                # per mode rather than discarding the new entry entirely.
                # Defer sort — caller is responsible for sorting after all merges.
                tc_to_mf = {mf.timecode_ms: mf for mf in existing.matched_frames}
                for mf in h.matched_frames:
                    if mf.timecode_ms not in tc_to_mf:
                        tc_to_mf[mf.timecode_ms] = mf
                    else:
                        existing_mf = tc_to_mf[mf.timecode_ms]
                        for mode, score in mf.scores.items():
                            if mode not in existing_mf.scores or score > existing_mf.scores[mode]:
                                existing_mf.scores[mode] = score
                existing.matched_frames = list(tc_to_mf.values())
    else:
        if h.query_timecodes:
            h.best_query_timecode_ms = h.query_timecodes[0]
        dest[k] = h


def _group_video_hits(hits: list[Hit]) -> list[Hit]:
    """Collapse dataset video-frame hits from the same source video into one Hit.

    Called once per query entity (one image or one video frame), so all hits
    in ``hits`` come from a single query vector.  Non-video hits pass through
    unchanged.  For each unique ``video_path`` the frame with the highest
    score becomes the representative; its score is the exact Qdrant similarity
    for that one comparison — nothing is mathematically combined.  All matched
    dataset frames are preserved in ``matched_frames`` with their individual
    exact scores.
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
        # Representative = frame with highest overall score for this query entity.
        # Its image_hash/timecode are used for the thumbnail.
        representative = max(group, key=lambda h: h.best_score())
        matched = [
            MatchedVideoFrame(
                timecode_ms=h.frame_timecode_ms or 0,
                frame_hash=h.image_hash or "",
                scores=h.scores,  # exact score from this one comparison
            )
            for h in sorted(group, key=lambda h: h.frame_timecode_ms or 0)
        ]
        # Build per-mode best scores across all dataset frames in this group.
        # Each score IS the exact Qdrant similarity from one specific 1:1
        # comparison — we pick the best-scoring frame for each mode so that
        # the grouped hit remains visible under all applicable mode filters,
        # even when the best ALTER frame and best SEMAN frame differ.
        group_scores: dict[str, float] = {}
        group_provenance: dict[str, dict] = {}
        for h in group:
            for mode, score in h.scores.items():
                if mode not in group_scores or score > group_scores[mode]:
                    group_scores[mode] = score
                    if mode in h.model_provenance:
                        group_provenance[mode] = h.model_provenance[mode]
        # query_timecodes: all hits in this group come from the same query
        # entity so they share the same single timecode (or None for images).
        qtc = representative.query_timecodes
        grouped_video.append(
            Hit(
                path=representative.path,
                scores=group_scores,
                exif=representative.exif,
                exif_geo_data=representative.exif_geo_data,
                image_hash=representative.image_hash,
                model_provenance=group_provenance,
                is_video_frame=True,
                video_path=vpath,
                video_hash=representative.video_hash,
                frame_timecode_ms=representative.frame_timecode_ms,
                matched_frames=matched,
                query_timecodes=qtc,
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

        # Exact hash matches are file-level (not frame-level).  Collect once
        # and pass through without merging with embedding-based results.
        # For video uploads use video_hash (file SHA-256); for images use
        # image_hash (pixel hash).  The two hash spaces never collide.
        all_flat_hits: list[Hit] = []
        if "exact" in modes and entry.file_hash:
            if entry.is_video:
                exact_hits, errs = _query_exact_video(client, entry.file_hash, settings)
            else:
                exact_hits, errs = _query_exact(
                    client, entry.file_hash, entry.file_hash_md5, settings
                )
            all_flat_hits.extend(exact_hits)
            file_result.errors.extend(errs)

        # Build (vector, query_timecode_ms) pairs.
        # For images timecode is None; for video uploads each frame contributes
        # its own pair so every frame is queried independently.
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

        # Pre-index vectors by timecode for O(n) per-frame dispatch.
        sscd_by_qtc: dict[int | None, list[list[float]]] = {}
        for vec, tc in sscd_vecs:
            sscd_by_qtc.setdefault(tc, []).append(vec)
        dino_by_qtc: dict[int | None, list[list[float]]] = {}
        for vec, tc in dino_vecs:
            dino_by_qtc.setdefault(tc, []).append(vec)

        # Collect all unique query timecodes (insertion-ordered).
        seen_qtcs: set = set()
        all_qtcs: list[int | None] = []
        for _, tc in sscd_vecs + dino_vecs:
            if tc not in seen_qtcs:
                all_qtcs.append(tc)
                seen_qtcs.add(tc)

        # Query each timecode (frame or image) in complete isolation.
        # Within one query entity, altered and semantic hits for the same
        # dataset path are combined onto one hit card (unify=True) so both
        # measurements are visible side by side — both scores originate from
        # the exact same query→dataset pair, just via different models.
        # After the per-qtc loop, the final unify pass may merge hits for the
        # same dataset path across different query frames (video queries); when
        # that happens the reported score is the max across those comparisons
        # and query_timecodes accumulates all contributing query timecodes.
        for qtc in all_qtcs:
            frame_merged: dict[str, Hit] = {}  # used when unify=True
            frame_unmerged: list[Hit] = []  # used when unify=False

            if "altered" in modes:
                for vec in sscd_by_qtc.get(qtc, []):
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
                            _merge_hit(h, frame_merged)
                        else:
                            frame_unmerged.append(h)
                    file_result.errors.extend(errs)

            if "semantic" in modes:
                for vec in dino_by_qtc.get(qtc, []):
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
                            _merge_hit(h, frame_merged)
                        else:
                            frame_unmerged.append(h)
                    file_result.errors.extend(errs)

            # Group dataset-video frames found by this query entity, then
            # append to the shared results list.  Because grouping is done
            # per query entity, a dataset video appears as a separate Hit for
            # each query frame that matched it — scores are never combined.
            if unify:
                all_flat_hits.extend(_group_video_hits(list(frame_merged.values())))
            else:
                # unify=False: keep modes separate — group video frames per mode
                # independently so a database video can appear once per mode,
                # consistent with unify=False semantics for image hits.
                # Calling _group_video_hits on the mixed list would silently
                # re-unify altered and semantic scores despite unify=False.
                for mode_hits in (
                    [h for h in frame_unmerged if "altered" in h.scores],
                    [h for h in frame_unmerged if "semantic" in h.scores],
                ):
                    if mode_hits:
                        all_flat_hits.extend(_group_video_hits(mode_hits))

        # Final merge pass (unify only): exact hits and vector hits for the
        # same dataset path must end up on one row.  Exact hits were added to
        # all_flat_hits before the per-qtc loop; vector hits were appended
        # after it.  _merge_hit folds both into one Hit per path — every score
        # still originates from its own independent 1:1 comparison.
        if unify:
            final_merged: dict[str, Hit] = {}
            for h in all_flat_hits:
                # Video hits: key by video_path so all query-frame × dataset-video
                # pairs collapse into one card regardless of which frame was chosen
                # as the representative.  Image hits: key by path as before.
                key = h.video_path if h.is_video_frame and h.video_path else None
                _merge_hit(h, final_merged, key=key)
            # Sort matched_frames once after all merges (deferred from _merge_hit).
            for h in final_merged.values():
                if h.matched_frames:
                    h.matched_frames.sort(key=lambda mf: mf.timecode_ms)
            sorted_hits = sorted(final_merged.values(), key=_hit_sort_key)
            # Video queries: do not apply limit — each Hit represents one
            # distinct query-frame × database-video match, and forensic
            # review requires every match to be visible, not just the top N.
            file_result.hits = sorted_hits if entry.is_video else sorted_hits[:limit]
        else:
            file_result.hits = sorted(all_flat_hits, key=_unmerged_sort_key)

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


def _query_exact_video(
    client: QdrantClient,
    video_hash: str,
    settings: Settings,
) -> tuple[list[Hit], list[str]]:
    """Return exact video-hash matches for a video query.

    Searches both collections for indexed frames whose ``video_hash`` equals
    the query video's SHA-256, groups them by ``video_path`` into one Hit per
    matching database video, and populates ``matched_frames`` with all matched
    frames sorted by timecode.  Each matching database video gets score
    ``{"exact": 1.0}``.
    """
    hits: list[Hit] = []
    errors: list[str] = []
    collection_mode = {
        settings.collection_sscd: "altered",
        settings.collection_dino: "semantic",
    }
    # video_path → {timecode_ms → payload dict}, provenance dict
    video_frames: dict[str, dict[int, dict]] = {}
    video_provenance: dict[str, dict] = {}
    video_video_hash: dict[str, str] = {}

    # Page size: use video_max_frames when set (0 = no cap → fall back to 2000).
    _page_size = settings.video_max_frames if settings.video_max_frames > 0 else 2000

    for collection in (settings.collection_sscd, settings.collection_dino):
        mode_key = collection_mode[collection]
        try:
            scroll_filter = Filter(
                must=[FieldCondition(key="video_hash", match=MatchValue(value=video_hash))]
            )
            payload_fields = [
                "image_path",
                "image_hash",
                "video_path",
                "video_hash",
                "frame_timecode_ms",
                "model_name",
                "model_hash",
            ]
            offset = None
            while True:
                records, offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=scroll_filter,
                    limit=_page_size,
                    offset=offset,
                    with_payload=payload_fields,
                    with_vectors=False,
                )
                for r in records:
                    vpath = r.payload.get("video_path", "")
                    if not vpath:
                        continue
                    if vpath not in video_frames:
                        video_frames[vpath] = {}
                        video_provenance[vpath] = {}
                        video_video_hash[vpath] = r.payload.get("video_hash", video_hash)
                    mn = r.payload.get("model_name", "")
                    mh = r.payload.get("model_hash", "")
                    if (mn or mh) and mode_key not in video_provenance[vpath]:
                        video_provenance[vpath][mode_key] = {"name": mn, "hash": mh}
                    tc: int = r.payload.get("frame_timecode_ms") or 0
                    if tc not in video_frames[vpath]:
                        video_frames[vpath][tc] = {
                            "timecode_ms": tc,
                            "frame_hash": r.payload.get("image_hash", ""),
                            "image_path": r.payload.get("image_path", ""),
                        }
                if not records or offset is None:
                    break
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "not found" in msg or "doesn't exist" in msg:
                logger.debug("Exact video query skipped non-existent collection %s", collection)
            else:
                logger.warning("Exact video query failed on %s: %s", collection, exc)
                errors.append(f"exact video query failed: {type(exc).__name__}")

    for vpath, frames_by_tc in video_frames.items():
        sorted_frames = sorted(frames_by_tc.values(), key=lambda f: f["timecode_ms"])
        representative = sorted_frames[0] if sorted_frames else {}
        matched = [
            MatchedVideoFrame(
                timecode_ms=f["timecode_ms"],
                frame_hash=f["frame_hash"],
                scores={"exact": 1.0},
            )
            for f in sorted_frames
        ]
        hits.append(
            Hit(
                path=representative.get("image_path", vpath),
                scores={"exact": 1.0},
                image_hash=representative.get("frame_hash"),
                model_provenance=video_provenance.get(vpath, {}),
                is_video_frame=True,
                video_path=vpath,
                video_hash=video_video_hash.get(vpath, video_hash),
                frame_timecode_ms=representative.get("timecode_ms"),
                matched_frames=matched,
            )
        )

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
