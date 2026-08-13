"""Session lifecycle routes: analyze (Phase 1), query (Phase 2), audit."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from scalar_forensic.config import Settings
from scalar_forensic.embedder import get_library_versions
from scalar_forensic.web.pipeline import (
    ProgressEvent,
    QueryProvenance,
    analyze_session,
    get_available_modes,
    get_hit_qdrant_provenance,
    query_semantic_stats,
    query_session,
)
from scalar_forensic.web.session import (
    FileEntry,
    create_session,
    delete_session,
    get_session,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Collections availability
# ---------------------------------------------------------------------------


@router.get("/api/collections")
async def collections() -> JSONResponse:
    settings = Settings()
    modes, has_reference, error = await get_available_modes(settings)
    payload: dict = {"modes": modes, "has_reference": has_reference}
    if error:
        payload["error"] = f"Qdrant unavailable: {error}"
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Phase 1: Analyze (SSE stream)
# ---------------------------------------------------------------------------


@router.post("/api/analyze")
async def analyze(
    files: list[UploadFile],
    modes: str = Form(default="exact,altered,semantic"),
) -> StreamingResponse:
    settings = Settings()
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]

    try:
        session = await create_session(max_active=settings.max_active_sessions)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    tmp_dir = Path(tempfile.mkdtemp(prefix="sfn_"))
    session.temp_dir = tmp_dir

    total_bytes = 0
    try:
        for upload in files:
            file_id = str(uuid.uuid4())
            filename = upload.filename or "unknown"
            # Preserve the original extension so container-detection libraries
            # (PyAV, Pillow) can identify the format from the filename.
            suffix = Path(filename).suffix
            dest = tmp_dir / (file_id + suffix)
            with dest.open("wb") as fout:
                while True:
                    chunk = await upload.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if settings.max_upload_bytes > 0 and total_bytes > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Upload exceeds the configured limit "
                                f"({settings.max_upload_bytes} bytes). "
                                "Set SFN_MAX_UPLOAD_BYTES to change."
                            ),
                        )
                    fout.write(chunk)
            session.files.append(FileEntry(file_id=file_id, filename=filename, temp_path=dest))
    except Exception:
        await delete_session(session.session_id)
        raise

    # Touch last_access so the reaper cannot evict an in-flight session that
    # has not yet received a get_session() call (e.g. long-running analysis).
    session.last_access = time.monotonic()

    async def event_stream():
        # Run the analysis (CPU-intensive) in a thread pool so the event loop
        # stays free to flush SSE chunks to the client in real time.  Without
        # this the loop is blocked between yields and the browser sees no
        # progress until an entire batch finishes.
        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_analysis() -> None:
            try:
                for event in analyze_session(session, mode_list, settings):
                    session.last_access = time.monotonic()
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        task = asyncio.create_task(asyncio.to_thread(_run_analysis))

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event.__dict__)}\n\n"

            await task  # re-raise any unexpected exception from the thread
        finally:
            # Cancel the task if the client disconnects so the asyncio Future
            # is not orphaned.  The underlying thread pool thread cannot be
            # forcibly interrupted, but it will be ignored once the task is
            # cancelled and its result will be discarded.
            task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Phase 2: Query (instant, re-callable on slider change)
# ---------------------------------------------------------------------------


@router.post("/api/query")
async def query(
    session_id: str = Form(...),
    modes: str = Form(default="exact,altered,semantic"),
    threshold_altered: float = Form(default=0.75, ge=0.0, le=1.0),
    threshold_semantic: float = Form(default=0.55, ge=0.0, le=1.0),
    limit: int = Form(default=10, ge=1, le=50),
    unify: bool = Form(default=True),
    include_reference: bool = Form(default=False),
) -> JSONResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    settings = Settings()
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    # Sync Qdrant I/O (potentially one query per video frame) — run in a
    # thread so a slider drag never stalls other examiners' requests.
    results, embedding_models = await asyncio.to_thread(
        query_session,
        session,
        mode_list,
        threshold_altered,
        threshold_semantic,
        limit,
        settings,
        unify=unify,
        include_reference=include_reference,
    )
    provenance = QueryProvenance(
        modes=mode_list,
        threshold_altered=threshold_altered,
        threshold_semantic=threshold_semantic,
        limit=limit,
        timestamp=datetime.now(UTC).isoformat(),
    )

    return JSONResponse(
        {
            "provenance": provenance.__dict__,
            "embedding_models": embedding_models,
            "results": [
                {
                    "file_id": r.file_id,
                    "filename": r.filename,
                    "errors": r.errors,
                    "hits": [
                        {
                            "path": h.path,
                            "scores": h.scores,
                            "exif": h.exif,
                            "exif_geo_data": h.exif_geo_data,
                            "image_hash": h.image_hash,
                            "model_provenance": h.model_provenance,
                            "is_video_frame": h.is_video_frame,
                            "video_path": h.video_path,
                            "video_hash": h.video_hash,
                            "frame_timecode_ms": h.frame_timecode_ms,
                            "matched_frames": [
                                {
                                    "timecode_ms": mf.timecode_ms,
                                    "frame_hash": mf.frame_hash,
                                    "scores": mf.scores,
                                    "path": mf.path,
                                }
                                for mf in h.matched_frames
                            ]
                            if h.matched_frames
                            else None,
                            "query_timecodes": h.query_timecodes,
                            "best_query_timecode_ms": h.best_query_timecode_ms,
                            "is_reference": h.is_reference,
                        }
                        for h in r.hits
                    ],
                }
                for r in results
            ],
        }
    )


# ---------------------------------------------------------------------------
# Semantic score distribution stats (on-demand, per file)
# ---------------------------------------------------------------------------


@router.get("/api/semantic-stats/{session_id}/{file_id}")
async def semantic_stats(session_id: str, file_id: str) -> JSONResponse:
    """Return DINOv2 score-distribution stats for one uploaded file.

    Queries the top-10 000 most-similar collection points at threshold 0 and
    computes min / percentiles / max / mean / stdev / histogram.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    settings = Settings()
    stats, error = query_semantic_stats(session, file_id, settings)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return JSONResponse(stats.__dict__)


# ---------------------------------------------------------------------------
# Forensic audit endpoints (on-demand, for the Audit modal)
# ---------------------------------------------------------------------------


@router.get("/api/library-versions")
async def library_versions() -> JSONResponse:
    """Return the library versions of the current web server process."""
    return JSONResponse(get_library_versions())


@router.get("/api/hit-provenance")
async def hit_provenance(image_hash: str) -> JSONResponse:
    """Return full Qdrant indexing provenance for a given image SHA-256."""
    if not re.fullmatch(r"[0-9a-f]{64}", image_hash):
        raise HTTPException(status_code=400, detail="Invalid hash")
    settings = Settings()
    return JSONResponse(get_hit_qdrant_provenance(image_hash, settings))
