"""Face-modality routes: availability, per-image observations, chips, query faces.

Phase 1 was browse-only.  Phase 1b adds the *query* side: faces detected in an
uploaded image (session-scoped, never persisted) and a cross-file search over
them.  The search ships uncalibrated by the maintainer's ruling of 2026-08-12 —
the raw cosine is displayed and labelled as such (spec §10 divergence).
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from qdrant_client import QdrantClient

from scalar_forensic.config import Settings
from scalar_forensic.faces.audit import AuditLog
from scalar_forensic.faces.chips import chip_paths, write_thumbnail
from scalar_forensic.faces.store import _HARD_FIELDS, FACE_VECTOR_NAME, FaceStore
from scalar_forensic.video import extract_frame_at
from scalar_forensic.web.pipeline import (
    MODEL_REFERENCE_NOTE,
    MODEL_REFERENCE_THRESHOLD,
    calibration_block,
    detect_query_faces,
    face_audit,
    face_score_stats,
    query_embedder_block,
    search_query_faces,
)
from scalar_forensic.web.session import get_session

_log = logging.getLogger(__name__)

router = APIRouter()

_HASH_RE = r"[0-9a-f]{64}"


def _require_hash(value: str) -> str:
    """Validate a hash used as a path component before any filesystem access."""
    if not re.fullmatch(_HASH_RE, value):
        raise HTTPException(status_code=400, detail="Invalid hash")
    return value


def _entry_for(session, file_id: str):
    for e in session.files:
        if e.file_id == file_id:
            return e
    return None


def _require_faces(settings: Settings) -> None:
    if not settings.faces_enabled:
        raise HTTPException(503, "Face modality is disabled (set SFN_FACES_ENABLED=true).")
    err = settings.face_startup_error()
    if err:
        raise HTTPException(503, err)


def _store(settings: Settings) -> FaceStore:
    return FaceStore(
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        settings.face_collection,
        settings.collection,
        0,  # dim unused on read paths — this store never creates the collection
    )


@router.get("/api/faces/availability")
async def faces_availability() -> JSONResponse:
    """Capability flag for the UI.

    This is deliberately not an entry in get_available_modes: those are
    case-collection query modes wired into mode priority and hit merging,
    while faces are a separate flow over a separate collection (spec §4.3).
    """
    settings = Settings()
    if not settings.faces_enabled:
        return JSONResponse(
            {
                "faces_available": False,
                "reason": "Face modality is disabled (set SFN_FACES_ENABLED=true to enable).",
            }
        )
    err = settings.face_startup_error()
    if err:
        return JSONResponse({"faces_available": False, "reason": err})
    try:
        if _store(settings).collection_is_new():
            return JSONResponse(
                {
                    "faces_available": False,
                    "reason": (
                        f"No face collection yet ({settings.face_collection}). "
                        "Run an index pass with --faces to create it."
                    ),
                }
            )
    except Exception as exc:  # Qdrant unreachable — degrade, never 500
        _log.warning("face availability check failed: %s", exc)
        return JSONResponse(
            {"faces_available": False, "reason": f"Face collection unreachable: {exc}"}
        )
    body: dict = {"faces_available": True, "reason": None}
    if settings.face_store_dir is None:
        body["note"] = "degraded-evidence mode: face store disabled, review chips are unavailable"
    return JSONResponse(body)


def _normalized(face: dict) -> dict:
    """Guarantee the split fields on every entry the API hands out.

    Observations written before the split carry neither field.  They were all
    embedded — that was the only outcome that got stored — so defaulting to
    "embedded" states what actually happened rather than inventing a status.
    """
    return {
        **face,
        "embedding_status": face.get("embedding_status") or "embedded",
        "embedding_exclusion_reason": face.get("embedding_exclusion_reason"),
    }


def _browse_order(face: dict) -> tuple[int, float]:
    """Embedded observations first, each group by its own score, descending.

    The two populations cannot share a sort key: review-only faces have no
    composite quality (it is None, because most of its subscores were never
    measured), so they are ordered by detector confidence among themselves.
    """
    embedded = face["embedding_status"] == "embedded"
    score = face.get("quality") if embedded else face.get("det_conf")
    return (0 if embedded else 1, -(score if score is not None else 0.0))


@router.get("/api/faces/by-image/{image_hash}")
async def faces_by_image(image_hash: str) -> JSONResponse:
    _require_hash(image_hash)
    settings = Settings()
    try:
        faces = _store(settings).list_faces(image_hash)
    except Exception as exc:
        _log.warning("face lookup failed for %s: %s", image_hash, exc)
        raise HTTPException(status_code=503, detail=f"Face collection unreachable: {exc}") from exc
    return JSONResponse({"faces": sorted((_normalized(f) for f in faces), key=_browse_order)})


@router.get("/api/faces/explain/{point_id}")
async def face_explain(point_id: str) -> JSONResponse:
    """Step-by-step account of how one stored face observation was produced.

    Assembled *entirely from persisted data* — no pipeline step is re-run, and
    every threshold comes from the observation's stored provenance rather than
    the current environment: the view describes what happened at index time.
    """
    try:
        uuid.UUID(point_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid point id") from None

    settings = Settings()
    try:
        face = _store(settings).get_face(point_id)
    except Exception as exc:
        _log.warning("face explain lookup failed for %s: %s", point_id, exc)
        raise HTTPException(status_code=503, detail=f"Face collection unreachable: {exc}") from exc
    if face is None:
        raise HTTPException(status_code=404, detail="Face observation not found")

    marker = None
    if face.get("image_hash"):
        try:
            marker = _store(settings).get_marker(face["image_hash"])
        except Exception:  # the totals are contextual, not load-bearing
            marker = None

    face = _normalized(face)
    status = face["embedding_status"]
    excluded_at = face["embedding_exclusion_reason"]
    # Which gate the exclusion reason belongs to.  Everything from that gate
    # onwards did not happen: alignment and the post-align checks are skipped
    # for a face the pre-align gate excluded, so reporting them as "passed"
    # would claim measurements the record does not contain.
    _STEP_FOR_REASON = {
        "confidence": "pre_align_gate",
        "size": "pre_align_gate",
        "pose": "pre_align_gate",
        "sharpness": "post_align_gate",
        "exposure": "post_align_gate",
    }
    _ORDER = ["detection", "pre_align_gate", "alignment", "post_align_gate", "embedding"]
    if status == "review_only":
        # An unrecognised or absent reason must not read as "everything before
        # embedding passed": a review-only observation stores null for pose,
        # sharpness, exposure and the aligned hash, so claiming it was aligned
        # and passed the blur check would be a false statement about an exhibit.
        # The earliest gate is the safe default — it claims the least.
        failing_step = _STEP_FOR_REASON.get(excluded_at, "pre_align_gate")
    else:
        failing_step = None
    _from = _ORDER.index(failing_step) if failing_step else len(_ORDER)

    def _passed(step_name: str) -> bool:
        if step_name == "embedding":
            # Never inferred from the reason: a review-only observation has no
            # vector whether or not its exclusion reason was recorded.
            return status == "embedded"
        return _ORDER.index(step_name) < _from

    aligned_chip = face.get("aligned_chip_hash")
    review_chip = face.get("review_chip_hash")
    steps = [
        {
            "step": "detection",
            "sentence": (
                f"A face detector ({face.get('detector_id')}) found a face here and scored its "
                "own confidence in that detection."
            ),
            "scores": {"det_conf": face.get("det_conf")},
            "thresholds_in_force": {
                "min_conf": face.get("min_conf"),
                "detector_score_threshold": face.get("detector_score_threshold"),
                "detect_max_size": face.get("detect_max_size"),
            },
            "passed": _passed("detection"),
        },
        {
            "step": "pre_align_gate",
            "sentence": (
                "Before any further processing the face was checked for confidence, size and "
                "how far it is turned away from the camera."
            ),
            "scores": {
                "confidence": face.get("quality_confidence"),
                "size": face.get("quality_size"),
                "pose": face.get("quality_pose"),
            },
            "thresholds_in_force": {
                "min_conf": face.get("min_conf"),
                "min_size": face.get("min_size"),
                "max_pose": face.get("max_pose"),
            },
            "passed": _passed("pre_align_gate"),
        },
        {
            "step": "alignment",
            "sentence": (
                "The face was rotated and scaled so that its five landmarks land on fixed "
                "reference positions, producing a 112 by 112 pixel crop."
            ),
            "scores": {},
            "thresholds_in_force": {"alignment_version": face.get("alignment_version")},
            "passed": _passed("alignment"),
        },
        {
            "step": "post_align_gate",
            "sentence": (
                "The original-resolution crop was then checked for blur and for over- or "
                "under-exposed pixels."
            ),
            "scores": {
                "sharpness": face.get("quality_sharpness"),
                "exposure": face.get("quality_exposure"),
            },
            "thresholds_in_force": {
                "min_sharpness": face.get("min_sharpness"),
                "max_clipped": face.get("max_clipped"),
            },
            "passed": _passed("post_align_gate"),
        },
        {
            "step": "embedding",
            "sentence": (
                "The aligned crop was converted into a list of numbers by the configured model. "
                "Similar faces produce similar lists; this is what similar faces are found by."
            ),
            "scores": {"embedding_norm": face.get("embedding_norm")},
            "thresholds_in_force": {"normalization_id": face.get("normalization_id")},
            "passed": _passed("embedding"),
        }
        if status == "embedded"
        else {
            "step": "embedding",
            "sentence": (
                f"This face was NOT embedded, because its {excluded_at} check did not meet "
                "the threshold in force when the file was processed. It is kept for "
                "examination by eye and is never compared with other faces."
                if excluded_at
                else "This face was NOT embedded. It is kept for examination by eye and is "
                "never compared with other faces."
            ),
            "scores": {},
            "thresholds_in_force": {},
            "passed": False,
        },
    ]

    return JSONResponse(
        {
            "point_id": face.get("id", point_id),
            "observation_key": face.get("observation_key"),
            "source": {
                "image_hash": face.get("image_hash"),
                "image_path": face.get("image_path"),
                "video_hash": face.get("video_hash"),
                "video_path": face.get("video_path"),
                "frame_timecode_ms": face.get("frame_timecode_ms"),
                "indexed_at": face.get("indexed_at"),
            },
            "geometry": {
                "bbox": face.get("bbox"),
                "landmarks": face.get("landmarks"),
                "detect_scale": face.get("detect_scale"),
                "crop_dilation": face.get("crop_dilation"),
            },
            "steps": steps,
            "embedder": {
                "embedder_model_name": face.get("embedder_model_name"),
                "embedder_model_hash": face.get("embedder_model_hash"),
                "manifest_hash": face.get("manifest_hash"),
                "embedder_dim": face.get("embedder_dim"),
                "normalization_id": face.get("normalization_id"),
                "pipeline_config_hash": face.get("pipeline_config_hash"),
            },
            # Gated on the hash of its own domain: the aligned PNG is addressed
            # in the aligned domain, every review artefact in the review domain.
            # A review-only observation has no aligned hash, and emitting a URL
            # that is a guaranteed 404 would read as a lost file rather than as
            # a chip that was never produced.
            "chips": {
                "aligned": f"/api/faces/chip/{aligned_chip}" if aligned_chip else None,
                "review": f"/api/faces/chip/{review_chip}/review" if review_chip else None,
                "thumb": f"/api/faces/chip/{review_chip}/thumb" if review_chip else None,
            },
            "embedding_status": status,
            "embedding_exclusion_reason": excluded_at,
            "file_totals": {
                "n_detected": (marker or {}).get("n_detected"),
                "n_kept": (marker or {}).get("n_kept"),
                "n_review_only": (marker or {}).get("n_review_only"),
                "review_only_reasons": (marker or {}).get("review_only_reasons") or {},
                "n_rejected": (marker or {}).get("n_rejected") or {},
            },
            "caveat": (
                "This describes how the machine produced a face observation. It is an "
                "investigative lead, not an identification."
            ),
        }
    )


def _chip_file(chip_hash: str, index: int) -> Path:
    settings = Settings()
    _require_hash(chip_hash)
    if settings.face_store_dir is None:
        raise HTTPException(status_code=404, detail="Face chip store is disabled")
    return chip_paths(Path(settings.face_store_dir), chip_hash)[index]


@router.get("/api/faces/chip/{chip_hash}")
async def face_chip(chip_hash: str) -> FileResponse:
    """The aligned 112x112 PNG — the exact tensor the embedder saw."""
    path = _chip_file(chip_hash, 0)
    if not path.exists():
        # The endpoint is given a hash, not an observation, so it cannot tell
        # which case applies — it names both rather than asserting either.
        raise HTTPException(
            status_code=404,
            detail=(
                "No aligned chip for this hash. Either the observation is review-only "
                "and was never aligned or embedded, or the chip file is missing from "
                "the store."
            ),
        )
    return FileResponse(path, media_type="image/png")


@router.get("/api/faces/chip/{chip_hash}/review")
async def face_chip_review(chip_hash: str) -> FileResponse:
    """The source-resolution review crop the examiner actually looks at."""
    path = _chip_file(chip_hash, 1)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Review chip not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/api/faces/chip/{chip_hash}/thumb")
async def face_chip_thumb(chip_hash: str) -> FileResponse:
    """Browse thumbnail; regenerated on demand — it is a derived artefact."""
    thumb = _chip_file(chip_hash, 2)
    if not thumb.exists():
        review = _chip_file(chip_hash, 1)
        if not review.exists():
            raise HTTPException(status_code=404, detail="Chip not found")
        write_thumbnail(review, thumb, Settings().face_thumb_size)
    return FileResponse(thumb, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Query-side faces (Phase 1b).  Session-scoped: nothing here reaches Qdrant or
# SFN_FACE_STORE_DIR, and no vector is ever serialised into a response — the
# client selects probes by *index* and the vectors stay in this process.
# ---------------------------------------------------------------------------


def _resolve_entry(session_id: str, file_id: str):
    """Session/file resolution, before any capability gate.

    Deliberately ordered ahead of ``_require_faces``: an unknown session is a
    malformed request whatever the face modality's state, and answering 503 for
    it would tell the client to go enable faces over a request that would still
    fail afterwards.  Nothing expensive happens here — no model is loaded.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    entry = _entry_for(session, file_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="file not found in session")
    return entry


def _query_image_bytes(entry, timecode_ms: int | None) -> bytes:
    """The bytes to detect on: the upload itself, or one frame of a video."""
    if getattr(entry, "is_video", False):
        if timecode_ms is None:
            raise HTTPException(
                status_code=400,
                detail="timecode_ms is required to detect faces in a video upload",
            )
        img = extract_frame_at(Path(entry.temp_path), timecode_ms)
        if img is None:
            raise HTTPException(status_code=404, detail="Frame not found at given timecode")
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    try:
        return Path(entry.temp_path).read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="file not found in session") from exc


def _detection_token(faces, cfg) -> str:
    """Fingerprint of one detection generation.

    Face *indices* are positions in whatever detection is currently cached on
    the session entry, so a chip URL that carries only an index survives a
    re-detection and starts addressing a different face — a 404 when the new
    detection found fewer faces (the reported defect), and worse, the wrong
    person's crop when it found at least as many.

    Deriving the token from the detection's own content rather than from a
    counter keeps it stateless: re-detecting the identical frame under the
    identical config reproduces the token, so an unchanged view does not
    invalidate URLs the examiner is still looking at.
    """
    digest = hashlib.sha256()
    digest.update(str(getattr(cfg, "config_hash", "")).encode())
    for face in faces:
        digest.update(
            f"|{face.index}|{tuple(face.bbox)}|{face.det_conf:.6f}|{face.embedding_status}".encode()
        )
    return digest.hexdigest()[:16]


def _face_json(face, session_id: str, file_id: str, token: str) -> dict:
    """Serialise one QueryFace for the wire.

    Built field by field on purpose.  ``dataclasses.asdict`` would carry
    ``vector`` and ``review_jpeg`` into the response, which is exactly the leak
    the session-scope rule exists to prevent — ``searchable`` is the only thing
    the client learns about the vector.
    """
    return {
        "index": face.index,
        "bbox": list(face.bbox),
        "landmarks": face.landmarks,
        "det_conf": face.det_conf,
        "detect_scale": face.detect_scale,
        "searchable": face.vector is not None,
        "embedding_status": face.embedding_status,
        "embedding_exclusion_reason": face.embedding_exclusion_reason,
        "quality": face.quality,
        "chip_url": f"/api/faces/query-chip/{session_id}/{file_id}/{token}/{face.index}",
    }


@router.post("/api/faces/query-faces")
def query_faces(
    session_id: str = Form(...),
    file_id: str = Form(...),
    timecode_ms: int | None = Form(None),
) -> JSONResponse:
    """Detect the faces in the uploaded query image.

    Sync ``def`` on purpose: detection and ONNX embedding are CPU-bound, so
    Starlette runs this in its threadpool instead of blocking the event loop.

    Re-detects on every call — the examiner may have moved to a different video
    frame — and caches the result on the session entry so the chip endpoint and
    the search endpoint can address faces by index.
    """
    entry = _resolve_entry(session_id, file_id)
    settings = Settings()
    _require_faces(settings)

    data = _query_image_bytes(entry, timecode_ms)
    try:
        result = detect_query_faces(data, settings, max_faces=settings.face_query_max_faces)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"not an image: {exc}") from exc

    entry.query_faces = result.faces
    entry.query_faces_cfg = result.cfg
    token = _detection_token(result.faces, result.cfg)
    return JSONResponse(
        {
            "faces": [_face_json(f, session_id, file_id, token) for f in result.faces],
            "detection_token": token,
            "n_detected": result.n_detected,
            "n_searchable": result.n_searchable,
            "n_review_only": result.n_review_only,
            "rejected": result.rejected,
            "pipeline_config_hash": result.cfg.config_hash,
            "embedder": query_embedder_block(result.cfg),
            "truncated": result.truncated,
        }
    )


@router.get("/api/faces/query-chip/{session_id}/{file_id}/{token}/{face_index}")
def query_chip(session_id: str, file_id: str, token: str, face_index: int) -> Response:
    """The session review crop for one query face.

    Served from memory with ``no-store``: this crop is never written to
    SFN_FACE_STORE_DIR, and it must not survive in a browser cache either.

    *token* pins the detection generation the index was issued against.  A URL
    from a superseded detection is refused with 409 rather than answered from
    the current one — serving it would put another face's crop under the
    identity the examiner is looking at, which is the failure that matters here.
    """
    entry = _resolve_entry(session_id, file_id)
    faces = entry.query_faces or []
    current = _detection_token(faces, getattr(entry, "query_faces_cfg", None))
    if token != current:
        raise HTTPException(
            status_code=409,
            detail=(
                "stale face detection: this chip belongs to a superseded detection "
                "for this file — re-run query-faces"
            ),
        )
    if face_index < 0 or face_index >= len(faces):
        raise HTTPException(status_code=404, detail="unknown face index")
    jpeg = faces[face_index].review_jpeg
    if jpeg is None:
        raise HTTPException(status_code=404, detail="no review crop for this face")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Cross-file face search (Phase 1b).  Uncalibrated by the maintainer's ruling of
# 2026-08-12: the raw cosine is returned and labelled, the score floor defaults
# to 0.0, and 0.363 travels only as the model authors' reference figure.
# ---------------------------------------------------------------------------


def _audit_log(settings: Settings) -> AuditLog:
    """The face audit log (spec §11).

    Mirrors faces/indexing.py and cli.py: the log sits beside the chip store,
    not inside it.  This is the first *web* route that writes it — every face
    query is a logged act.
    """
    audit_dir = Path(settings.face_store_dir).parent if settings.face_store_dir else Path("data")
    return AuditLog(audit_dir / "face_audit.log")


def _parse_face_indices(raw: str, faces: list) -> list[int]:
    """Indices into the query-faces response, validated against the session.

    A review-only face is refused here rather than filtered out silently: it has
    no vector, so there is nothing to search with, and quietly dropping it would
    leave the examiner believing that face was searched and found nothing.
    """
    idxs: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idxs.append(int(part))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown face index {part}") from None
    if not idxs:
        raise HTTPException(status_code=400, detail="no face indices given")
    for i in idxs:
        if i < 0 or i >= len(faces):
            raise HTTPException(status_code=400, detail=f"unknown face index {i}")
        if faces[i].vector is None:
            raise HTTPException(
                status_code=400,
                detail=f"face {i} is review-only and has no vector; it cannot be searched",
            )
    return idxs


def _parse_point_ids(raw: str) -> list[str]:
    """Stored point ids used as probes, validated before any store access."""
    pids: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            uuid.UUID(part)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid point id {part}") from None
        pids.append(part)
    return pids


def _stored_vectors(store: FaceStore, point_ids: list[str]) -> dict[str, list[float]]:
    """The stored face vector of each point that has one.

    A review-only observation is a point *without* an entry here — the vector
    is structurally absent, not filtered out (see FaceStore.search_faces).
    """
    if not point_ids:
        return {}
    records = store.client.retrieve(
        collection_name=store.collection,
        ids=point_ids,
        with_payload=False,
        with_vectors=[FACE_VECTOR_NAME],
    )
    out: dict[str, list[float]] = {}
    for rec in records:
        vec = rec.vector if isinstance(rec.vector, dict) else {}
        v = (vec or {}).get(FACE_VECTOR_NAME)
        if v is not None:
            out[str(rec.id)] = list(v)
    return out


def _point_probes(store: FaceStore, point_ids: list[str]) -> list[tuple[str, list[float]]]:
    """Stored points as probes.  A vectorless point is refused, not skipped:
    quietly dropping it would leave the examiner believing that face was
    searched and found nothing — the same rule as _parse_face_indices."""
    records = store.client.retrieve(
        collection_name=store.collection,
        ids=point_ids,
        with_payload=False,
        with_vectors=[FACE_VECTOR_NAME],
    )
    found = {str(rec.id): rec for rec in records}
    probes: list[tuple[str, list[float]]] = []
    for pid in point_ids:
        rec = found.get(pid)
        if rec is None:
            raise HTTPException(status_code=400, detail=f"unknown point id {pid}")
        vec = rec.vector if isinstance(rec.vector, dict) else {}
        v = (vec or {}).get(FACE_VECTOR_NAME)
        if v is None:
            raise HTTPException(
                status_code=400,
                detail=f"point {pid} is review-only and has no vector; it cannot be searched",
            )
        probes.append((f"pt:{pid}", list(v)))
    return probes


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def _compat_block(store: FaceStore, cfg) -> dict:
    """Hard mismatch → 409; soft mismatches ride along as warnings (spec §7.2).

    FaceStore.check_compat *raises* on a hard mismatch, so the ValueError is the
    normal path.  The returned list is still scanned for hard field names: a
    store that reports rather than raises must not slip an incomparable search
    through on the strength of that difference alone.
    """
    try:
        msgs = store.check_compat(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    hard = [m for m in msgs if any(f in m for f in _HARD_FIELDS)]
    if hard:
        raise HTTPException(status_code=409, detail="; ".join(hard))
    return {"ok": True, "warnings": list(msgs)}


@router.post("/api/faces/search")
def face_search(
    session_id: str = Form(...),
    file_id: str = Form(...),
    face_indices: str = Form(""),
    point_ids: str = Form(""),
    limit: int = Form(10, ge=1, le=50),
    threshold: float = Form(0.0, ge=0.0, le=1.0),
    exact: bool = Form(True),
    collapse: bool = Form(True),
) -> JSONResponse:
    """Search the face collection with the examiner's selected probe faces.

    Sync ``def``: the kNN is a blocking client call.

    Probes come from two origins (change-set 2026-08-13): ``face_indices`` into
    the session's query faces, and ``point_ids`` of stored observations whose
    vectors are fetched from the collection.  Both refuse a review-only face
    with 400 — the vectorless design must look the same at this surface no
    matter where the probe came from.

    ``threshold`` is a *display floor* on the raw cosine and defaults to 0.0.
    It is never seeded from the model authors' 0.363 — this deployment has no
    calibrated threshold, and putting one in the default would manufacture it.
    """
    entry = _resolve_entry(session_id, file_id)
    settings = Settings()
    _require_faces(settings)

    faces = entry.query_faces or []
    idxs = _parse_face_indices(face_indices, faces) if face_indices.strip() else []
    pids = _parse_point_ids(point_ids)
    if not idxs and not pids:
        raise HTTPException(status_code=400, detail="no face indices given")

    store = _store(settings)
    probes: list[tuple[int | str, list[float]]] = [(i, faces[i].vector) for i in idxs]
    probes += _point_probes(store, pids)

    cfg = entry.query_faces_cfg
    compat = _compat_block(store, cfg)

    try:
        hits = search_query_faces(
            store,
            probes,
            limit=limit,
            threshold=threshold,
            exact=exact,
            collapse=collapse,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.warning("face search failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Face collection unreachable: {exc}") from exc

    search_mode = "exact" if exact else "ann"
    _audit_log(settings).append(
        "query",
        settings.examiner_id,
        probe_hash=entry.file_hash,
        face_indices=idxs,
        probe_point_ids=pids,
        collection=settings.face_collection,
        pipeline_config_hash=getattr(cfg, "config_hash", None),
        # No calibration record exists (spec §10); recorded as null rather than
        # omitted, so the log states that the search ran uncalibrated.
        face_calibration_id=None,
        search_mode=search_mode,
        threshold=threshold,
        limit=limit,
        n_results=len(hits),
        top_scores=[h["score"] for h in hits[:5]],
    )

    return JSONResponse(
        {
            "hits": hits,
            "n_probes": len(probes),
            "search_mode": search_mode,
            "threshold": threshold,
            "limit": limit,
            "calibration": calibration_block(),
            "embedder": query_embedder_block(cfg),
            "compat": compat,
        }
    )


@router.post("/api/faces/compare")
def face_compare(
    session_id: str = Form(...),
    file_id: str = Form(...),
    image_hash: str = Form(...),
) -> JSONResponse:
    """Pairwise raw cosine: query image's faces × one indexed medium's faces.

    Change-set 2026-08-13: drives the both-sides highlight when a match is
    selected.  The matrix covers *comparable* faces only — a review-only
    observation is vectorless on either side and never enters it; the counts
    report how many were left out so their absence is visible.

    No threshold is applied here (spec §10): every pair ships with its raw
    cosine, and any floor is an operator-set display control in the UI.
    """
    entry = _resolve_entry(session_id, file_id)
    settings = Settings()
    _require_faces(settings)
    image_hash = _require_hash(image_hash)

    faces = entry.query_faces or []
    query_probes = [(f.index, f.vector) for f in faces if f.vector is not None]

    store = _store(settings)
    cfg = entry.query_faces_cfg
    compat = _compat_block(store, cfg)

    try:
        rows = store.list_faces(image_hash)
        vectors = _stored_vectors(store, [str(r["id"]) for r in rows])
    except Exception as exc:
        _log.warning("face compare failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Face collection unreachable: {exc}") from exc

    keys = {str(r["id"]): r.get("observation_key") for r in rows}
    pairs = [
        {
            "query_face_index": qi,
            "point_id": pid,
            "observation_key": keys.get(pid),
            "score": _cosine(qv, v),
        }
        for qi, qv in query_probes
        for pid, v in vectors.items()
    ]
    pairs.sort(key=lambda p: -p["score"])

    _audit_log(settings).append(
        "compare",
        settings.examiner_id,
        probe_hash=entry.file_hash,
        target_image_hash=image_hash,
        collection=settings.face_collection,
        pipeline_config_hash=getattr(cfg, "config_hash", None),
        face_calibration_id=None,
        n_pairs=len(pairs),
        top_scores=[p["score"] for p in pairs[:5]],
    )

    return JSONResponse(
        {
            "pairs": pairs,
            "n_query_comparable": len(query_probes),
            "n_query_review_only": len(faces) - len(query_probes),
            "n_match_comparable": len(vectors),
            "n_match_review_only": len(rows) - len(vectors),
            "calibration": calibration_block(),
            "embedder": query_embedder_block(cfg),
            "compat": compat,
        }
    )


# ---------------------------------------------------------------------------
# Per-model explainer surfaces: what the face pipeline did to an indexed medium,
# and what the score distribution behind one probe looks like.  Both are reports
# — neither re-runs a pipeline step, and neither applies a threshold.
# ---------------------------------------------------------------------------


@router.get("/api/faces/audit")
def faces_audit(image_hash: str) -> JSONResponse:
    """How the machine produced the face observations of one indexed medium.

    Sync ``def``: the Qdrant reads are blocking client calls.

    Every threshold in the response comes from the observation's own persisted
    provenance, so this reports the gates **in force at index time** rather than
    the ones the web process happens to be configured with today.
    """
    _require_hash(image_hash)
    settings = Settings()
    _require_faces(settings)

    try:
        body = face_audit(_store(settings), image_hash)
    except Exception as exc:
        _log.warning("face audit lookup failed for %s: %s", image_hash, exc)
        raise HTTPException(status_code=503, detail=f"Face collection unreachable: {exc}") from exc
    if body is None:
        raise HTTPException(status_code=404, detail="No face observations for this image")
    return JSONResponse(body)


@router.post("/api/faces/dist-stats")
def faces_dist_stats(
    session_id: str = Form(...),
    file_id: str = Form(...),
    face_index: int = Form(...),
    sample_size: int = Form(10000, ge=1, le=50000),
) -> JSONResponse:
    """Score distribution one query face draws out of the face collection.

    Sync ``def``: the kNN is a blocking client call.

    Deliberately the same field set as the DINOv2 distribution so the two
    modalities read alike.  The 0.363 rides along as an annotation only — it is
    the model authors' figure, applied nowhere in this path.
    """
    entry = _resolve_entry(session_id, file_id)
    settings = Settings()
    _require_faces(settings)

    faces = entry.query_faces or []
    if face_index < 0 or face_index >= len(faces):
        raise HTTPException(status_code=404, detail="unknown face index")
    vector = faces[face_index].vector
    if vector is None:
        raise HTTPException(
            status_code=400,
            detail=f"face {face_index} is review-only and has no vector",
        )

    store = _store(settings)
    try:
        stats = face_score_stats(store, vector, sample_size=sample_size)
    except Exception as exc:
        _log.warning("face dist-stats query failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Face collection unreachable: {exc}") from exc

    return JSONResponse(
        {
            **vars(stats),
            # States the guarantee rather than a filter: review-only observations
            # carry no vector, so they cannot appear in this population at all.
            "population": (
                f"embedded face observations in {settings.face_collection}; review-only "
                "observations carry no vector and are structurally absent from this "
                "distribution"
            ),
            "model_reference_threshold": MODEL_REFERENCE_THRESHOLD,
            "model_reference_note": MODEL_REFERENCE_NOTE,
        }
    )
