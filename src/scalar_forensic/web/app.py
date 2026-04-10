"""FastAPI web application for ScalarForensic Phase 2 query interface."""

from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from scalar_forensic.config import Settings
from scalar_forensic.embedder import extract_exif_detailed
from scalar_forensic.web.pipeline import (
    QueryProvenance,
    analyze_session,
    get_available_modes,
    query_session,
)
from scalar_forensic.web.session import FileEntry, create_session, get_session

_STATIC_DIR = Path(__file__).parent / "static"
_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif", ".jp2", ".ico", ".psd"}
)

app = FastAPI(title="ScalarForensic", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Collections availability
# ---------------------------------------------------------------------------


@app.get("/api/collections")
async def collections() -> JSONResponse:
    settings = Settings()
    modes, error = await get_available_modes(settings)
    payload: dict = {"modes": modes}
    if error:
        payload["error"] = f"Qdrant unavailable: {error}"
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Phase 1: Analyze (SSE stream)
# ---------------------------------------------------------------------------


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile],
    modes: str = Form(default="exact,altered,semantic"),
) -> StreamingResponse:
    settings = Settings()
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]

    session = await create_session()
    tmp_dir = Path(tempfile.mkdtemp(prefix="sfn_"))

    for upload in files:
        file_id = str(uuid.uuid4())
        filename = upload.filename or "unknown"
        dest = tmp_dir / file_id
        data = await upload.read()
        dest.write_bytes(data)
        session.files.append(FileEntry(file_id=file_id, filename=filename, temp_path=dest))

    async def event_stream():
        for event in analyze_session(session, mode_list, settings):
            yield f"data: {json.dumps(event.__dict__)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Phase 2: Query (instant, re-callable on slider change)
# ---------------------------------------------------------------------------


@app.post("/api/query")
async def query(
    session_id: str = Form(...),
    modes: str = Form(default="exact,altered,semantic"),
    threshold_altered: float = Form(default=0.75),
    threshold_semantic: float = Form(default=0.80),
    limit: int = Form(default=10),
) -> JSONResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    settings = Settings()
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    results = query_session(
        session, mode_list, threshold_altered, threshold_semantic, limit, settings
    )
    provenance = QueryProvenance(
        modes=mode_list,
        threshold_altered=threshold_altered,
        threshold_semantic=threshold_semantic,
        limit=limit,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return JSONResponse(
        {
            "provenance": provenance.__dict__,
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
                        }
                        for h in r.hits
                    ],
                }
                for r in results
            ]
        }
    )


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------


@app.get("/api/query-image/{session_id}/{file_id}")
async def query_image(session_id: str, file_id: str) -> FileResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None or not entry.temp_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(entry.temp_path, filename=Path(entry.filename).name)


@app.get("/api/hit-image")
async def hit_image(path: str) -> FileResponse:
    """Serve a hit image from the server filesystem.

    Security: only image extensions, resolved absolute path.
    """
    p = Path(path).resolve()
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    if p.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not an image file")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p, filename=p.name)


# ---------------------------------------------------------------------------
# Metadata (on-demand, for the detail panels)
# ---------------------------------------------------------------------------


@app.get("/api/query-metadata/{session_id}/{file_id}")
async def query_metadata(session_id: str, file_id: str) -> JSONResponse:
    """Detailed metadata for an uploaded query image."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None or not entry.temp_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    data = entry.temp_path.read_bytes()
    meta = extract_exif_detailed(data)
    meta["filename"] = entry.filename
    meta["hash"] = entry.file_hash or hashlib.sha256(data).hexdigest()
    return JSONResponse(meta)


@app.get("/api/metadata")
async def hit_metadata(path: str) -> JSONResponse:
    """Detailed metadata for a hit image (filesystem path)."""
    p = Path(path).resolve()
    if not p.is_absolute() or p.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    data = p.read_bytes()
    meta = extract_exif_detailed(data)
    meta["filename"] = p.name
    meta["path"] = str(p)
    meta["hash"] = hashlib.sha256(data).hexdigest()
    return JSONResponse(meta)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def start() -> None:
    uvicorn.run("scalar_forensic.web.app:app", host="0.0.0.0", port=8080, reload=False)
