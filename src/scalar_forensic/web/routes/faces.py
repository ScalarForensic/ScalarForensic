"""Face-modality browse routes: availability, per-image observations, chips.

Browse only — there is deliberately no face *search* endpoint in Phase 1;
cross-file search is gated on a calibration record (spec §10).
"""

from __future__ import annotations

import logging
import re
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
