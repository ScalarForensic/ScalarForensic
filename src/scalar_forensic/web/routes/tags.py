"""Tag Triage routes: tag CRUD, discovery, exploration, and classification."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, cast

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings
from scalar_forensic.discovery import MAX_CONTEXT_PAIRS, DiscoveryHit, run_discovery, run_explore
from scalar_forensic.indexer import qdrant_scroll_all
from scalar_forensic.query_eval import QueryEvalHit, score_query_entries
from scalar_forensic.tags import Tag, TagStore
from scalar_forensic.web.session import get_session

_log = logging.getLogger(__name__)

router = APIRouter()

_DEFAULT_COSINE_THRESHOLD = 0.5

_MAX_TAG_NAME_LEN = 200
_MAX_TAG_NOTES_LEN = 4000


# ---------------------------------------------------------------------------
# Tag Triage — investigator-in-the-loop Discovery
# ---------------------------------------------------------------------------


def _tag_client_and_store() -> tuple[QdrantClient, TagStore]:
    """Construct a fresh QdrantClient and TagStore for this request.

    This is intentionally per-request rather than a shared singleton.  TagStore
    construction calls ``_ensure_collection`` which issues one ``get_collections``
    call to Qdrant.  For a single-investigator deployment the overhead is
    acceptable; a long-running multi-user service would want a startup-time check
    and a cached client instead.
    """
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return client, TagStore(client, settings.tags_collection)


def _tag_store() -> TagStore:
    """Construct a fresh :class:`TagStore` per request."""
    return _tag_client_and_store()[1]


def _triplet_threshold(pos_dino: list, neg_dino: list) -> int:
    """Return the triplet-count threshold (75 % of defined pairs, min 1).

    When *neg_dino* is empty the tag is in Recommend mode and triplet scoring
    does not apply — callers must gate on ``triplet_score is None`` separately.
    """
    if not neg_dino:
        return 0  # unused in recommend mode; cosine threshold gates that path
    n_pairs = min(len(pos_dino) * len(neg_dino), MAX_CONTEXT_PAIRS)
    return max(1, n_pairs * 3 // 4)


def _tag_to_json(tag: Tag) -> dict:
    return {
        "tag_id": tag.tag_id,
        "name": tag.name,
        "positive_ids": list(tag.positive_ids),
        "negative_ids": list(tag.negative_ids),
        "target_id": tag.target_id,
        "notes": tag.notes,
        "created_at": tag.created_at,
        "updated_at": tag.updated_at,
    }


def _hit_passes_classify_threshold(
    triplet_score: int | None,
    raw_score: float,
    triplet_threshold: int,
    cosine_threshold: float = _DEFAULT_COSINE_THRESHOLD,
) -> bool:
    """Return True if a hit meets the classification bar.

    Discovery mode (triplet_score is not None): require ≥ triplet_threshold
    pairs satisfied.  Recommend mode (triplet_score is None, tag has no
    negatives): require the raw cosine score ≥ cosine_threshold.
    """
    if triplet_score is not None:
        return triplet_score >= triplet_threshold
    return raw_score >= cosine_threshold


def _fetch_tag_ref_records(
    client: QdrantClient,
    settings: Settings,
    all_ref_ids: list[str],
) -> list:
    """Retrieve tag reference records from case and (if needed) reference collection.

    Tags built with source='reference' store point IDs from the reference
    collection; those IDs won't exist in the case collection. Checking both
    ensures classification works regardless of which collection the tag came from.
    """
    records: list = list(
        client.retrieve(
            collection_name=settings.collection,
            ids=all_ref_ids,
            with_vectors=True,
            with_payload=False,
        )
    )
    if settings.reference_collection:
        found = {str(r.id) for r in records}
        missing = [i for i in all_ref_ids if i not in found]
        if missing:
            records += client.retrieve(
                collection_name=settings.reference_collection,
                ids=missing,
                with_vectors=True,
                with_payload=False,
            )
    return records


# In-process cache of
# (tag_id, updated_at, vector_name, case_collection, reference_collection)
# → {point_id: vector}.  Keyed on updated_at so any tag edit invalidates
# the entry automatically.  Both collection names participate in the key
# because :func:`_fetch_tag_ref_records` falls back to the reference
# collection when an ID isn't in the case collection — changing either
# at runtime must invalidate the cache.  Caches the raw per-ID vector
# map rather than a positive/negative split so callers keep their own
# positive-set policy (different endpoints treat the optional target
# anchor differently).  FIFO-evicted at a size cap to keep worst-case
# memory bounded.
_TAG_REF_VECS_CACHE: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = {}
_TAG_REF_VECS_CACHE_MAX = 256


def _get_cached_tag_ref_vecs(
    client: QdrantClient,
    settings: Settings,
    tag: Tag,
    *,
    vector_name: str = "dino",
) -> dict[str, list[float]]:
    """Return ``{str(point_id): vec}`` for every referenced point on *tag*.

    Includes positives, negatives, and the target anchor (deduplicated).  The
    result is cached per ``(tag_id, updated_at, vector_name, collection)``, so
    repeated classification passes reuse a single Qdrant retrieval — typical
    badge refreshes see near-zero Qdrant traffic once the cache is warm.
    """
    cache_key = (
        str(tag.tag_id),
        tag.updated_at,
        vector_name,
        settings.collection,
        settings.reference_collection or "",
    )
    cached = _TAG_REF_VECS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    all_ref_ids = [str(i) for i in tag.positive_ids + tag.negative_ids]
    if tag.target_id is not None and str(tag.target_id) not in all_ref_ids:
        all_ref_ids.append(str(tag.target_id))

    vecs_by_id: dict[str, list[float]] = {}
    if all_ref_ids:
        for rec in _fetch_tag_ref_records(client, settings, all_ref_ids):
            vecs = rec.vector or {}
            if not isinstance(vecs, dict):
                continue
            v = vecs.get(vector_name)
            if v is None:
                continue
            vecs_by_id[str(rec.id)] = list(v)

    if len(_TAG_REF_VECS_CACHE) >= _TAG_REF_VECS_CACHE_MAX:
        _TAG_REF_VECS_CACHE.pop(next(iter(_TAG_REF_VECS_CACHE)))
    _TAG_REF_VECS_CACHE[cache_key] = vecs_by_id
    return vecs_by_id


def _split_cached_vecs(
    vecs_by_id: dict[str, list[float]],
    pos_ids: list[str],
    neg_ids: list[str] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Return ``(pos_vecs, neg_vecs)`` from a cached id→vec map.

    IDs listed in *pos_ids* present in *vecs_by_id* go to the positive
    bucket; IDs listed in *neg_ids* go to the negative bucket.  Entries
    that match neither are silently ignored — guarding against phantom
    negatives if the cache ever contained an ID the tag no longer
    references (e.g. anchor stripped before splitting).

    For backward compatibility, when *neg_ids* is None every non-positive
    entry in *vecs_by_id* is treated as a negative.
    """
    pos_set = set(pos_ids)
    neg_set = set(neg_ids) if neg_ids is not None else None
    pos_vecs: list[list[float]] = []
    neg_vecs: list[list[float]] = []
    for pid, v in vecs_by_id.items():
        if pid in pos_set:
            pos_vecs.append(v)
        elif neg_set is None or pid in neg_set:
            neg_vecs.append(v)
    return pos_vecs, neg_vecs


def _hit_to_json(hit: DiscoveryHit) -> dict:
    payload = hit.payload or {}
    return {
        "point_id": hit.point_id,
        "triplet_score": hit.triplet_score,
        "raw_score": hit.raw_score,
        "path": payload.get("image_path", ""),
        "image_hash": payload.get("image_hash"),
        "exif": payload.get("exif"),
        "exif_geo_data": payload.get("exif_geo_data"),
        "is_video_frame": bool(payload.get("is_video_frame")),
        "video_path": payload.get("video_path"),
        "video_hash": payload.get("video_hash"),
        "frame_timecode_ms": payload.get("frame_timecode_ms"),
        "is_reference": bool(payload.get("is_reference")),
    }


@router.get("/api/tags")
async def list_tags() -> JSONResponse:
    try:

        def _run() -> list:
            return _tag_store().list()

        tags = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return JSONResponse({"tags": [_tag_to_json(t) for t in tags]})


@router.post("/api/tag")
async def create_tag(
    name: str = Form(...),
    positive_ids: str = Form(default=""),
    negative_ids: str = Form(default=""),
    target_id: str = Form(default=""),
    notes: str = Form(default=""),
) -> JSONResponse:
    """Create or replace a tag by *name*.

    ``positive_ids`` / ``negative_ids`` are comma-separated Qdrant point IDs.
    ``target_id`` is optional; when omitted the first positive is used as an
    implicit anchor at query time.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    if len(name) > _MAX_TAG_NAME_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"name must be <= {_MAX_TAG_NAME_LEN} characters",
        )
    if len(notes) > _MAX_TAG_NOTES_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"notes must be <= {_MAX_TAG_NOTES_LEN} characters",
        )
    pos = [x.strip() for x in positive_ids.split(",") if x.strip()]
    neg = [x.strip() for x in negative_ids.split(",") if x.strip()]
    tgt: str | None = target_id.strip() or None
    try:

        def _run():
            return _tag_store().create(
                name, positive_ids=pos, negative_ids=neg, target_id=tgt, notes=notes
            )

        tag = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return JSONResponse(_tag_to_json(tag))


@router.get("/api/tag/{tag_id}")
async def get_tag(tag_id: str) -> JSONResponse:
    try:

        def _run():
            return _tag_store().get(tag_id)

        tag = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    return JSONResponse(_tag_to_json(tag))


@router.post("/api/tag/{tag_id}/mark")
async def mark_tag(
    tag_id: str,
    point_id: str = Form(...),
    role: str = Form(...),
) -> JSONResponse:
    """Append a single ``(point_id, "positive"|"negative")`` to the tag.

    A point already present in the other role is moved — the latest mark wins.
    """
    if role not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="role must be 'positive' or 'negative'")
    try:

        def _run():
            role_lit = cast(Literal["positive", "negative"], role)
            return _tag_store().mark(tag_id, point_id, role_lit)

        tag = await asyncio.to_thread(_run)
    except LookupError:
        raise HTTPException(status_code=404, detail="Tag not found") from None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return JSONResponse(_tag_to_json(tag))


@router.post("/api/tag/{tag_id}/unmark")
async def unmark_tag(
    tag_id: str,
    point_id: str = Form(...),
) -> JSONResponse:
    try:

        def _run():
            return _tag_store().unmark(tag_id, point_id)

        tag = await asyncio.to_thread(_run)
    except LookupError:
        raise HTTPException(status_code=404, detail="Tag not found") from None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return JSONResponse(_tag_to_json(tag))


@router.post("/api/tag/{tag_id}/set-target")
async def set_tag_target(
    tag_id: str,
    target_id: str = Form(default=""),
) -> JSONResponse:
    """Set or clear the explicit Discovery anchor for a tag.

    The anchor is the reference point Qdrant uses as the starting direction of
    the similarity search.  Candidates must be similar to the anchor *and*
    satisfy the tag's positive/negative triplet constraints.

    When ``target_id`` is empty the explicit anchor is cleared and the engine
    falls back to using ``positives[0]`` automatically at query time.
    """
    tgt: str | None = target_id.strip() or None
    try:

        def _run():
            return _tag_store().set_target(tag_id, tgt)

        tag = await asyncio.to_thread(_run)
    except LookupError:
        raise HTTPException(status_code=404, detail="Tag not found") from None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return JSONResponse(_tag_to_json(tag))


@router.delete("/api/tag/{tag_id}")
async def delete_tag(tag_id: str) -> JSONResponse:
    try:

        def _run() -> bool:
            return _tag_store().delete(tag_id)

        existed = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    if not existed:
        raise HTTPException(status_code=404, detail="Tag not found")
    return JSONResponse({"tag_id": tag_id, "deleted": True})


@router.post("/api/triage")
async def triage(
    tag_id: str = Form(...),
    limit: int = Form(default=50, ge=1, le=500),
    reverse: bool = Form(default=False),
    cosine_threshold: float = Form(default=_DEFAULT_COSINE_THRESHOLD, ge=0.0, le=1.0),
) -> JSONResponse:
    """Run a Tag-Triage query against the indexed dataset using DINOv2.

    Always searches the case collection; tag IDs may reference the reference
    collection via lookup_from when SFN_REFERENCE_COLLECTION is set. Triage
    against the reference collection itself is unsupported because tag point
    IDs live in the case collection and would not resolve there — use
    /api/explore with collection='reference' for reference-side browsing.

    cosine_threshold filters Recommend-mode hits (tag has no negatives).
    Discovery-mode triage ignores it because the score is an integer triplet count.
    """
    settings = Settings()
    triage_collection = settings.collection
    try:
        client, store = _tag_client_and_store()

        def _get_tag() -> Tag | None:
            return store.get(tag_id)

        tag = await asyncio.to_thread(_get_tag)
        if tag is None:
            raise HTTPException(status_code=404, detail="Tag not found")
        hits = await asyncio.to_thread(
            run_discovery,
            client,
            triage_collection,
            tag,
            vector_name="dino",
            limit=limit,
            reverse=reverse,
            reference_collection=None,
            cosine_threshold=cosine_threshold,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    # pair_count reflects exactly what run_discovery sends to Qdrant: the
    # cartesian product of positive_ids × negative_ids (capped).  target_id
    # is the Discovery anchor, not a pair member, so it is excluded here.
    pair_count = min(len(tag.positive_ids) * len(tag.negative_ids), MAX_CONTEXT_PAIRS)
    return JSONResponse(
        {
            "tag": _tag_to_json(tag),
            "reverse": reverse,
            "limit": limit,
            "pair_count": pair_count,
            "hits": [_hit_to_json(h) for h in hits],
        }
    )


@router.post("/api/explore")
async def explore(
    tag_id: str = Form(...),
    limit: int = Form(default=50, ge=1, le=200),
    collection: str = Form(default="dataset"),
) -> JSONResponse:
    """Surface exploration candidates for tag bootstrapping.

    Automatically selects strategy based on tag state:
    - Both positives and negatives present → ContextQuery (boundary-seeking)
    - Otherwise → random sample (cold-start / diversity injection)

    Already-labelled points are excluded so successive runs surface fresh
    candidates.  Returns the same hit envelope as /api/triage so the UI
    can reuse the existing mark/unmark flow.

    ``collection`` controls which collection is explored: ``"dataset"`` (default)
    uses SFN_COLLECTION; ``"reference"`` uses SFN_REFERENCE_COLLECTION.
    """
    settings = Settings()
    if collection not in {"dataset", "reference"}:
        raise HTTPException(status_code=400, detail="collection must be 'dataset' or 'reference'")
    if collection == "reference":
        if not settings.reference_collection:
            raise HTTPException(
                status_code=400,
                detail="SFN_REFERENCE_COLLECTION is not configured.",
            )
        explore_collection = settings.reference_collection
    else:
        explore_collection = settings.collection
    try:
        client, store = _tag_client_and_store()
        tag = await asyncio.to_thread(store.get, tag_id)
        if tag is None:
            raise HTTPException(status_code=404, detail="Tag not found")
        pos_ids = list(tag.positive_ids)
        if tag.target_id is not None and tag.target_id not in {str(i) for i in pos_ids}:
            pos_ids = pos_ids + [tag.target_id]
        neg_ids = list(tag.negative_ids)
        all_ids = pos_ids + neg_ids
        if all_ids:

            def _filter_to_collection() -> tuple[list, list]:
                try:
                    found = client.retrieve(
                        explore_collection,
                        ids=all_ids,
                        with_payload=False,
                        with_vectors=False,
                    )
                    found_set = {str(p.id) for p in found}
                    return (
                        [i for i in pos_ids if str(i) in found_set],
                        [i for i in neg_ids if str(i) in found_set],
                    )
                except Exception:
                    return [], []

            pos_ids, neg_ids = await asyncio.to_thread(_filter_to_collection)
        hits, strategy = await asyncio.to_thread(
            run_explore,
            client,
            explore_collection,
            pos_ids,
            neg_ids,
            vector_name="dino",
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    pair_count = min(len(tag.positive_ids) * len(tag.negative_ids), MAX_CONTEXT_PAIRS)
    return JSONResponse(
        {
            "tag": _tag_to_json(tag),
            "strategy": strategy,
            "limit": limit,
            "pair_count": pair_count,
            "hits": [_hit_to_json(h) for h in hits],
        }
    )


@router.post("/api/tags/classify")
async def classify_tags_for_hashes(request: Request) -> JSONResponse:
    """Evaluate tag membership for a list of image hashes.

    Request body: ``{"image_hashes": ["sha256-1", "sha256-2", ...]}``.
    Response: ``{"by_hash": {"sha256-1": ["weapons", "drugs"], "sha256-2": [], ...}}``.

    Uses the same NumPy triplet scoring as classify-session so that tag badges
    on search hits and on uploaded query images are computed identically.
    """
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    raw_hashes = body.get("image_hashes")
    if raw_hashes is None:
        return JSONResponse({"by_hash": {}, "skipped_tags": []})
    if not isinstance(raw_hashes, list) or not all(isinstance(h, str) for h in raw_hashes):
        raise HTTPException(status_code=400, detail="'image_hashes' must be a list of strings")
    if len(raw_hashes) > 256:
        raise HTTPException(status_code=400, detail="'image_hashes' must contain at most 256 items")
    image_hashes: list[str] = list(dict.fromkeys(raw_hashes))
    if not image_hashes:
        return JSONResponse({"by_hash": {}, "skipped_tags": []})
    raw_ct = body.get("cosine_threshold", _DEFAULT_COSINE_THRESHOLD)
    if not isinstance(raw_ct, (int, float)) or not (0.0 <= raw_ct <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="'cosine_threshold' must be a number in [0.0, 1.0]",
        )
    cosine_threshold: float = float(raw_ct)

    settings = Settings()

    def _classify():
        client, store = _tag_client_and_store()
        tags = store.list()

        # Resolve all hashes → point_ids in one scroll using a should-filter.
        # ``image_hashes`` is already deduplicated by the request handler.
        hash_set = set(image_hashes)
        hash_to_pid: dict[str, str] = {}
        for record in qdrant_scroll_all(
            client,
            settings.collection,
            scroll_filter=Filter(
                should=[
                    FieldCondition(key="image_hash", match=MatchValue(value=h))
                    for h in image_hashes
                ]
            ),
            limit=256,
            with_payload=["image_hash"],
            with_vectors=False,
        ):
            h = (record.payload or {}).get("image_hash")
            if isinstance(h, str) and h in hash_set and h not in hash_to_pid:
                hash_to_pid[h] = str(record.id)
            if len(hash_to_pid) == len(image_hashes):
                break

        if not hash_to_pid:
            return {h: [] for h in image_hashes}, []

        candidate_records = client.retrieve(
            collection_name=settings.collection,
            ids=list(hash_to_pid.values()),
            with_vectors=True,
            with_payload=False,
        )
        # Build (file_id, filename, dino_vec) entries reusing score_query_entries
        pid_to_hash = {v: k for k, v in hash_to_pid.items()}
        entries: list[tuple[str, str, list[float] | None]] = []
        for rec in candidate_records:
            pid = str(rec.id)
            image_hash = pid_to_hash.get(pid)
            if image_hash is None:
                continue
            vecs = rec.vector or {}
            dino_vec = (
                list(vecs.get("dino")) if isinstance(vecs, dict) and vecs.get("dino") else None
            )
            entries.append((pid, image_hash, dino_vec))

        by_hash: dict[str, list[str]] = {h: [] for h in image_hashes}
        skipped_tags: list[str] = []

        for tag in tags:
            # classify_tags_for_hashes folds target into positives for scoring.
            effective_pos = [str(i) for i in tag.positive_ids]
            if tag.target_id is not None and str(tag.target_id) not in effective_pos:
                effective_pos.append(str(tag.target_id))
            if not effective_pos:
                continue
            try:
                vecs_by_id = _get_cached_tag_ref_vecs(client, settings, tag, vector_name="dino")
            except Exception:  # noqa: BLE001
                skipped_tags.append(tag.name)
                continue
            pos_dino, neg_dino = _split_cached_vecs(
                vecs_by_id,
                effective_pos,
                [str(i) for i in tag.negative_ids],
            )

            # Fail safe: the tag declares negatives but none could be retrieved
            # (stale point IDs, re-indexed collection, wrong collection).
            # Do not fall back to permissive cosine classification — skip.
            if tag.negative_ids and not neg_dino:
                _log.warning(
                    "classify: tag %r has %d negative ID(s) but none resolved — skipping",
                    tag.name,
                    len(tag.negative_ids),
                )
                skipped_tags.append(tag.name)
                continue

            threshold = _triplet_threshold(pos_dino, neg_dino)

            hits = score_query_entries(entries, pos_dino, neg_dino, limit=len(entries))
            for hit in hits:
                if not _hit_passes_classify_threshold(
                    hit.triplet_score, hit.raw_score, threshold, cosine_threshold
                ):
                    continue
                # hit.filename holds the image_hash (we used it as the filename field)
                by_hash[hit.filename].append(tag.name)

        return by_hash, skipped_tags

    try:
        by_hash, skipped_tags = await asyncio.to_thread(_classify)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return JSONResponse({"by_hash": by_hash, "skipped_tags": skipped_tags})


@router.post("/api/tags/classify-session")
async def classify_tags_for_session(request: Request) -> JSONResponse:
    """Evaluate tag membership for all session files via their in-memory embeddings.

    Used to badge uploaded query images that are not indexed in the dataset
    collection.  Request body: ``{"session_id": "..."}``.
    Response: ``{"by_hash": {"sha256-hash": ["tag-name", ...], ...}}``.

    Uses the same NumPy triplet scoring as ``/api/triage/query-images`` but
    iterates over every tag and returns results keyed by file SHA-256 so the
    UI can merge them directly into ``hitTags``.
    """
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    session_id: str = body.get("session_id") or ""
    if not session_id:
        return JSONResponse({"by_hash": {}, "skipped_tags": []})

    session = get_session(session_id)
    if session is None:
        return JSONResponse({"by_hash": {}, "skipped_tags": []})

    raw_ct = body.get("cosine_threshold", _DEFAULT_COSINE_THRESHOLD)
    if not isinstance(raw_ct, (int, float)) or not (0.0 <= raw_ct <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="'cosine_threshold' must be a number in [0.0, 1.0]",
        )
    cosine_threshold: float = float(raw_ct)

    settings = Settings()

    def _classify_session():
        cs_client, store = _tag_client_and_store()
        tags = store.list()

        entries = [
            (f.file_id, f.filename, f.dino_embedding)
            for f in session.files
            if not f.is_video and f.dino_embedding
        ]
        file_id_to_hash = {f.file_id: f.file_hash for f in session.files if f.file_hash}

        by_hash: dict[str, list[str]] = {}
        skipped_tags: list[str] = []
        if not entries or not tags:
            return by_hash, skipped_tags

        for tag in tags:
            if not tag.positive_ids:
                continue
            try:
                vecs_by_id = _get_cached_tag_ref_vecs(cs_client, settings, tag, vector_name="dino")
            except Exception:  # noqa: BLE001
                skipped_tags.append(tag.name)
                continue
            # classify_tags_for_session intentionally ignores target: only the
            # explicit positive/negative lists contribute to scoring here.
            pos_dino, neg_dino = _split_cached_vecs(
                vecs_by_id,
                [str(i) for i in tag.positive_ids],
                [str(i) for i in tag.negative_ids],
            )

            if tag.negative_ids and not neg_dino:
                _log.warning(
                    "classify-session: tag %r has %d negative ID(s) but none resolved — skipping",
                    tag.name,
                    len(tag.negative_ids),
                )
                skipped_tags.append(tag.name)
                continue

            threshold = _triplet_threshold(pos_dino, neg_dino)

            hits = score_query_entries(
                entries,
                pos_dino,
                neg_dino,
                limit=len(entries),
            )
            for hit in hits:
                if not _hit_passes_classify_threshold(
                    hit.triplet_score, hit.raw_score, threshold, cosine_threshold
                ):
                    continue
                image_hash = file_id_to_hash.get(hit.file_id)
                if image_hash:
                    by_hash.setdefault(image_hash, []).append(tag.name)

        return by_hash, skipped_tags

    try:
        by_hash, skipped_tags = await asyncio.to_thread(_classify_session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return JSONResponse({"by_hash": by_hash, "skipped_tags": skipped_tags})


@router.post("/api/triage/query-images")
async def triage_query_images(
    tag_id: str = Form(...),
    session_id: str = Form(...),
    limit: int = Form(default=50, ge=1, le=500),
    cosine_threshold: float = Form(default=_DEFAULT_COSINE_THRESHOLD, ge=0.0, le=1.0),
) -> JSONResponse:
    """Run tag triage against uploaded query images using DINOv2 embeddings."""

    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    settings = Settings()

    def _run():
        _client, store = _tag_client_and_store()
        tag = store.get(tag_id)
        if tag is None:
            return None, None, False

        if not tag.positive_ids and tag.target_id is None and not tag.negative_ids:
            return tag, [], False

        vecs_by_id = _get_cached_tag_ref_vecs(_client, settings, tag, vector_name="dino")
        # Mirror classify_tags_for_hashes: fold target into effective_pos so
        # it does not fall through to neg_vecs in _split_cached_vecs.
        effective_pos = [str(i) for i in tag.positive_ids]
        if tag.target_id is not None and str(tag.target_id) not in set(effective_pos):
            effective_pos.append(str(tag.target_id))
        pos_dino, neg_dino = _split_cached_vecs(
            vecs_by_id,
            effective_pos,
            [str(i) for i in tag.negative_ids],
        )

        if tag.negative_ids and not neg_dino:
            _log.warning(
                "triage-query-images: tag %r has %d negative ID(s) but none resolved — skipping",
                tag.name,
                len(tag.negative_ids),
            )
            return tag, [], True

        threshold = _triplet_threshold(pos_dino, neg_dino)

        entries = [
            (f.file_id, f.filename, f.dino_embedding)
            for f in session.files
            if not f.is_video and f.dino_embedding
        ]
        all_hits = score_query_entries(entries, pos_dino, neg_dino, limit=limit)
        hits = [
            h
            for h in all_hits
            if _hit_passes_classify_threshold(
                h.triplet_score, h.raw_score, threshold, cosine_threshold
            )
        ]
        return tag, hits, False

    try:
        tag, hits, skipped = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc

    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")

    file_hash_by_id = {f.file_id: f.file_hash for f in session.files}

    def _qhit_to_json(h: QueryEvalHit) -> dict:
        return {
            "file_id": h.file_id,
            "filename": h.filename,
            "image_url": f"/api/query-image/{session_id}/{h.file_id}",
            "image_hash": file_hash_by_id.get(h.file_id),
            "triplet_score": h.triplet_score,
            "raw_score": h.raw_score,
        }

    eff_pos_count = len(tag.positive_ids)
    if tag.target_id is not None and tag.target_id not in {str(i) for i in tag.positive_ids}:
        eff_pos_count += 1
    pair_count = min(eff_pos_count * len(tag.negative_ids), MAX_CONTEXT_PAIRS)
    return JSONResponse(
        {
            "tag": _tag_to_json(tag),
            "limit": limit,
            "pair_count": pair_count,
            "hits": [_qhit_to_json(h) for h in (hits or [])],
            "skipped_tags": [tag.name] if skipped else [],
        }
    )


# ---------------------------------------------------------------------------
# Point-ID and payload lookup (bridge from image hashes to Qdrant IDs)
# ---------------------------------------------------------------------------


@router.get("/api/point-id")
async def lookup_point_id(image_hash: str) -> JSONResponse:
    """Return the Qdrant point ID for an image identified by its SHA-256 hash.

    Used by the Tag Triage UI to translate search-result image hashes
    (which the operator can see) into Qdrant point IDs (which tags need).

    Checks the case collection first, then falls back to the reference
    collection (if configured).  The response includes which collection
    the point was found in so the caller can preserve that context.
    """
    settings = Settings()

    def _scroll(collection_name: str):
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        records, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="image_hash", match=MatchValue(value=image_hash))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return records

    try:
        records = await asyncio.to_thread(_scroll, settings.collection)
        found_in = settings.collection
        if not records and settings.reference_collection:
            records = await asyncio.to_thread(_scroll, settings.reference_collection)
            found_in = settings.reference_collection
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    if not records:
        raise HTTPException(status_code=404, detail="No indexed point found for that hash")
    return JSONResponse(
        {
            "point_id": str(records[0].id),
            "collection": found_in,
            "is_reference": found_in == settings.reference_collection,
        }
    )


@router.get("/api/point-payload")
async def get_point_payload(point_id: str) -> JSONResponse:
    """Return image metadata (path + hash) for a Qdrant point by its ID.

    Used by the Tag Triage UI to render thumbnails for tag example
    IDs that were not seen in the current triage run (e.g. IDs that were
    added via the CLI or pasted into the create form).

    Checks the case collection first, then falls back to the reference
    collection (if configured) so tags built from reference material can
    still resolve their thumbnails.
    """
    settings = Settings()

    def _retrieve(collection_name: str):
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        return client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=["image_hash", "image_path"],
            with_vectors=False,
        )

    try:
        records = await asyncio.to_thread(_retrieve, settings.collection)
        found_in = settings.collection
        if not records and settings.reference_collection:
            records = await asyncio.to_thread(_retrieve, settings.reference_collection)
            found_in = settings.reference_collection
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    if not records:
        raise HTTPException(status_code=404, detail="Point not found")
    payload = records[0].payload or {}
    return JSONResponse(
        {
            "point_id": point_id,
            "image_hash": payload.get("image_hash"),
            "image_path": payload.get("image_path"),
            "collection": found_in,
            "is_reference": found_in == settings.reference_collection,
        }
    )
