"""Face-modality browse routes: availability, per-image observations, chips.

Browse only — there is deliberately no face *search* endpoint in Phase 1;
cross-file search is gated on a calibration record (spec §10).
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from qdrant_client import QdrantClient

from scalar_forensic.config import Settings
from scalar_forensic.faces.chips import chip_paths, write_thumbnail
from scalar_forensic.faces.store import FaceStore

_log = logging.getLogger(__name__)

router = APIRouter()

_HASH_RE = r"[0-9a-f]{64}"


def _require_hash(value: str) -> str:
    """Validate a hash used as a path component before any filesystem access."""
    if not re.fullmatch(_HASH_RE, value):
        raise HTTPException(status_code=400, detail="Invalid hash")
    return value


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


@router.get("/api/faces/by-image/{image_hash}")
async def faces_by_image(image_hash: str) -> JSONResponse:
    _require_hash(image_hash)
    settings = Settings()
    try:
        faces = _store(settings).list_faces(image_hash)
    except Exception as exc:
        _log.warning("face lookup failed for %s: %s", image_hash, exc)
        raise HTTPException(status_code=503, detail=f"Face collection unreachable: {exc}") from exc
    return JSONResponse({"faces": faces})


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

    chip = face.get("chip_hash")
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
            "passed": True,
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
            "passed": True,
        },
        {
            "step": "alignment",
            "sentence": (
                "The face was rotated and scaled so that its five landmarks land on fixed "
                "reference positions, producing a 112 by 112 pixel crop."
            ),
            "scores": {},
            "thresholds_in_force": {"alignment_version": face.get("alignment_version")},
            "passed": True,
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
            "passed": True,
        },
        {
            "step": "embedding",
            "sentence": (
                "The aligned crop was converted into a list of numbers by the configured model. "
                "Similar faces produce similar lists; this is what similar faces are found by."
            ),
            "scores": {"embedding_norm": face.get("embedding_norm")},
            "thresholds_in_force": {"normalization_id": face.get("normalization_id")},
            "passed": True,
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
            "chips": {
                "aligned": f"/api/faces/chip/{chip}" if chip else None,
                "review": f"/api/faces/chip/{chip}/review" if chip else None,
                "thumb": f"/api/faces/chip/{chip}/thumb" if chip else None,
            },
            "file_totals": {
                "n_detected": (marker or {}).get("n_detected"),
                "n_kept": (marker or {}).get("n_kept"),
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
        raise HTTPException(status_code=404, detail="Chip not found")
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
