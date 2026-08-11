"""Phase 2 of the web pipeline: querying Qdrant with the stored embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings
from scalar_forensic.indexer import qdrant_scroll_all
from scalar_forensic.web.pipeline._embedders import _embedder_cache
from scalar_forensic.web.pipeline.provenance import _payload_model_provenance
from scalar_forensic.web.session import Session

logger = logging.getLogger(__name__)


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
    # Dedup set for query_timecodes — kept in sync with the list for O(1)
    # membership tests during the merge pass.  Excluded from repr and compare
    # so it is invisible to callers that iterate over Hit fields.
    _query_timecodes_seen: set[int] = field(default_factory=set, repr=False, compare=False)
    is_reference: bool = False

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
                existing._query_timecodes_seen.add(tc)
                existing.best_query_timecode_ms = tc
            else:
                if tc not in existing._query_timecodes_seen:
                    existing.query_timecodes.append(tc)
                    existing._query_timecodes_seen.add(tc)
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
            tc = h.query_timecodes[0]
            h._query_timecodes_seen = {tc}
            h.best_query_timecode_ms = tc
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
        # Propagate is_reference from the group: every hit in a group comes
        # from the same source video, so all entries share the same value.
        # Using any() instead of representative.is_reference is a defensive
        # safeguard against future callers that mix sources before grouping.
        is_reference = any(h.is_reference for h in group)
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
                is_reference=is_reference,
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
    include_reference: bool = False,
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
        # NOTE: the per-mode structure below (altered / semantic branches, unify
        # merge/append, _group_video_hits) is mirrored by the reference overlay
        # loop that follows it.  If you add or change a mode branch here, apply
        # the same change to the reference overlay loop and vice versa.
        for qtc in all_qtcs:
            frame_merged: dict[str, Hit] = {}  # used when unify=True
            frame_unmerged: list[Hit] = []  # used when unify=False

            if "altered" in modes:
                for vec in sscd_by_qtc.get(qtc, []):
                    hits, errs = _query_vector(
                        client,
                        collection=settings.collection,
                        vector=vec,
                        mode="altered",
                        threshold=threshold_altered,
                        limit=limit,
                        vector_name="sscd",
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
                        collection=settings.collection,
                        vector=vec,
                        mode="semantic",
                        threshold=threshold_semantic,
                        limit=limit,
                        vector_name="dino",
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

        # Reference collection overlay: query the reference collection with the
        # same per-mode embeddings used for the main case search and append hits
        # as is_reference=True.  Mirrors the per-qtc + merge/group structure of
        # the case-collection loop above so reference hits honour unify=True
        # (mode-merge per path), unify=False (per-mode rows), and propagate
        # query_timecodes from video-query frames.
        # NOTE: keep in sync with the case-collection loop above.
        if include_reference and settings.reference_collection:
            for qtc in all_qtcs:
                ref_frame_merged: dict[str, Hit] = {}  # used when unify=True
                ref_frame_unmerged: list[Hit] = []  # used when unify=False

                if "altered" in modes:
                    for vec in sscd_by_qtc.get(qtc, []):
                        ref_hits, ref_errs = _query_vector(
                            client,
                            collection=settings.reference_collection,
                            vector=vec,
                            mode="altered",
                            threshold=threshold_altered,
                            limit=limit,
                            vector_name="sscd",
                            is_reference_result=True,
                        )
                        for h in ref_hits:
                            if qtc is not None:
                                h.query_timecodes = [qtc]
                            if unify:
                                _merge_hit(h, ref_frame_merged)
                            else:
                                ref_frame_unmerged.append(h)
                        file_result.errors.extend(ref_errs)

                if "semantic" in modes:
                    for vec in dino_by_qtc.get(qtc, []):
                        ref_hits, ref_errs = _query_vector(
                            client,
                            collection=settings.reference_collection,
                            vector=vec,
                            mode="semantic",
                            threshold=threshold_semantic,
                            limit=limit,
                            vector_name="dino",
                            is_reference_result=True,
                        )
                        for h in ref_hits:
                            if qtc is not None:
                                h.query_timecodes = [qtc]
                            if unify:
                                _merge_hit(h, ref_frame_merged)
                            else:
                                ref_frame_unmerged.append(h)
                        file_result.errors.extend(ref_errs)

                if unify:
                    all_flat_hits.extend(_group_video_hits(list(ref_frame_merged.values())))
                else:
                    for mode_hits in (
                        [h for h in ref_frame_unmerged if "altered" in h.scores],
                        [h for h in ref_frame_unmerged if "semantic" in h.scores],
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
                # Prefix with is_reference so an accidental path overlap between
                # the case and reference collections cannot collapse the two
                # onto one row — the overlay is always kept distinct.
                base = h.video_path if h.is_video_frame and h.video_path else h.path
                key = f"{'ref' if h.is_reference else 'case'}:{base}"
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
    try:
        records, _ = client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="image_hash", match=MatchValue(value=image_hash))]
            ),
            limit=50,
            with_payload=[
                "image_path",
                "image_hash",
                "exif",
                "exif_geo_data",
                "dino_model_name",
                "dino_model_hash",
                "sscd_model_name",
                "sscd_model_hash",
                "is_video_frame",
                "video_path",
                "video_hash",
                "frame_timecode_ms",
            ],
            with_vectors=False,
        )
        for r in records:
            path = r.payload.get("image_path", "")
            mp = _payload_model_provenance(r.payload)
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
                        is_video_frame=bool(r.payload.get("is_video_frame")),
                        video_path=r.payload.get("video_path"),
                        video_hash=r.payload.get("video_hash"),
                        frame_timecode_ms=r.payload.get("frame_timecode_ms"),
                    )
                )
            else:
                existing.model_provenance.update(mp)
        _query_exact_ok = True
    except Exception as _query_exact_exc:  # noqa: BLE001
        msg = str(_query_exact_exc).lower()
        if "not found" in msg or "doesn't exist" in msg:
            logger.debug("Exact query skipped non-existent collection %s", settings.collection)
        else:
            logger.warning(
                "Exact hash query failed on %s: %s", settings.collection, _query_exact_exc
            )
            errors.append(f"exact query failed: {type(_query_exact_exc).__name__}")
        _query_exact_ok = False

    # Collision detection: find images with same MD5 but different SHA-256
    if _query_exact_ok and image_hash_md5:
        try:
            md5_records, _ = client.scroll(
                collection_name=settings.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="image_hash_md5", match=MatchValue(value=image_hash_md5))
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
            logger.warning("MD5 collision query failed on %s: %s", settings.collection, exc)
    return hits, errors


def _query_exact_video(
    client: QdrantClient,
    video_hash: str,
    settings: Settings,
) -> tuple[list[Hit], list[str]]:
    """Return exact video-hash matches for a video query.

    Searches the unified collection for indexed frames whose ``video_hash``
    equals the query video's SHA-256, groups them by ``video_path`` into one
    Hit per matching database video, and populates ``matched_frames`` with all
    matched frames sorted by timecode.  Each matching database video gets score
    ``{"exact": 1.0}``.
    """
    hits: list[Hit] = []
    errors: list[str] = []
    # video_path → {timecode_ms → payload dict}, provenance dict
    video_frames: dict[str, dict[int, dict]] = {}
    video_provenance: dict[str, dict] = {}
    video_video_hash: dict[str, str] = {}

    # Page size: use video_max_frames when set (0 = no cap → fall back to 2000).
    _page_size = settings.video_max_frames if settings.video_max_frames > 0 else 2000

    try:
        scroll_filter = Filter(
            must=[
                FieldCondition(key="video_hash", match=MatchValue(value=video_hash)),
                FieldCondition(key="is_video_frame", match=MatchValue(value=True)),
            ]
        )
        payload_fields = [
            "image_path",
            "image_hash",
            "video_path",
            "video_hash",
            "frame_timecode_ms",
            "dino_model_name",
            "dino_model_hash",
            "sscd_model_name",
            "sscd_model_hash",
        ]
        for r in qdrant_scroll_all(
            client,
            settings.collection,
            scroll_filter=scroll_filter,
            limit=_page_size,
            with_payload=payload_fields,
        ):
            vpath = r.payload.get("video_path", "")
            if not vpath:
                continue
            tc_raw = r.payload.get("frame_timecode_ms")
            if tc_raw is None:
                continue
            try:
                tc: int = int(tc_raw)
            except (TypeError, ValueError):
                continue
            if vpath not in video_frames:
                video_frames[vpath] = {}
                video_provenance[vpath] = {}
                video_video_hash[vpath] = r.payload.get("video_hash", video_hash)
            for mode, entry in _payload_model_provenance(r.payload).items():
                if mode not in video_provenance[vpath]:
                    video_provenance[vpath][mode] = entry
            if tc not in video_frames[vpath]:
                video_frames[vpath][tc] = {
                    "timecode_ms": tc,
                    "frame_hash": r.payload.get("image_hash", ""),
                    "image_path": r.payload.get("image_path", ""),
                }
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "not found" in msg or "doesn't exist" in msg:
            logger.debug(
                "Exact video query skipped non-existent collection %s", settings.collection
            )
        else:
            logger.warning("Exact video query failed on %s: %s", settings.collection, exc)
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
    vector_name: str = "dino",
    is_reference_result: bool = False,
) -> tuple[list[Hit], list[str]]:
    try:
        result = client.query_points(
            collection_name=collection,
            query=vector,
            using=vector_name,
            score_threshold=threshold,
            limit=limit,
            with_payload=[
                "image_path",
                "image_hash",
                "exif",
                "exif_geo_data",
                f"{vector_name}_model_name",
                f"{vector_name}_model_hash",
                "is_video_frame",
                "video_path",
                "video_hash",
                "frame_timecode_ms",
                "is_reference",
            ],
        )
        hits = []
        for r in result.points:
            mp = _payload_model_provenance(r.payload)
            # Source of truth for is_reference is the point's payload.  Fall back
            # to the caller-supplied flag when the payload does not carry the
            # field (case-collection points indexed before --reference existed).
            payload_is_ref = bool(r.payload.get("is_reference"))
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
                    is_reference=payload_is_ref or is_reference_result,
                )
            )
        return hits, []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector query failed on %s (%s): %s", collection, mode, exc)
        return [], [f"{mode} query failed: {type(exc).__name__}"]
