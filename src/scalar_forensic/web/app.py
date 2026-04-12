"""FastAPI web application for ScalarForensic Phase 2 query interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from scalar_forensic.config import Settings
from scalar_forensic.embedder import extract_exif_detailed, get_library_versions
from scalar_forensic.extractor import CONTAINER_EXTENSIONS, extract_container
from scalar_forensic.web.pipeline import (
    QueryProvenance,
    analyze_session,
    get_available_modes,
    get_hit_qdrant_provenance,
    query_semantic_stats,
    query_session,
)
from scalar_forensic.web.session import FileEntry, create_session, get_session

_STATIC_DIR = Path(__file__).parent / "static"
_VIZ_JS_SRC = (_STATIC_DIR / "viz.js").read_text(encoding="utf-8")


def _render_viz_html(data: dict) -> str:
    """Return a self-contained HTML page with the point-cloud data and viz
    code inlined.  No server connection is needed to display the result."""
    viz_js = _VIZ_JS_SRC
    data_json = json.dumps(data, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ScalarForensic — vector visualization</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; overflow: hidden; background: #000; }}
    #vec-canvas {{ position: fixed; inset: 0; width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <canvas id="vec-canvas"></canvas>
  <script>{viz_js}</script>
  <script>initVectorViz({data_json});</script>
</body>
</html>"""


_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif", ".jp2", ".ico", ".psd"}
)

# Cached PCA-projected point cloud, computed once at startup.
_points3d_cache: dict | None = None


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    global _points3d_cache
    settings = Settings()
    if settings.viz_max_points > 0:
        sscd_pts, dino_pts = await asyncio.gather(
            asyncio.to_thread(
                _fetch_and_project, settings.collection_sscd, settings.viz_max_points, settings
            ),
            asyncio.to_thread(
                _fetch_and_project, settings.collection_dino, settings.viz_max_points, settings
            ),
        )
        _points3d_cache = {"sscd": sscd_pts, "dino": dino_pts}
    else:
        _points3d_cache = {"sscd": [], "dino": []}
    if settings.viz_export_path and _points3d_cache:
        _write_viz_export(settings.viz_export_path, _points3d_cache)
    yield


app = FastAPI(title="ScalarForensic", docs_url=None, redoc_url=None, lifespan=lifespan)
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
    threshold_altered: float = Form(default=0.75, ge=0.0, le=1.0),
    threshold_semantic: float = Form(default=0.55, ge=0.0, le=1.0),
    limit: int = Form(default=10, ge=1, le=50),
    unify: bool = Form(default=True),
) -> JSONResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    settings = Settings()
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    results, embedding_models = query_session(
        session,
        mode_list,
        threshold_altered,
        threshold_semantic,
        limit,
        settings,
        unify=unify,
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
                            "container_hash": h.container_hash,
                            "container_path": h.container_path,
                            "container_type": h.container_type,
                            "container_item_name": h.container_item_name,
                            "extraction_kind": h.extraction_kind,
                        }
                        for h in r.hits
                    ],
                }
                for r in results
            ],
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


@app.get("/api/thumbnail/{sha256}")
async def thumbnail(sha256: str) -> FileResponse:
    """Serve a pre-generated thumbnail by SHA-256 hash.

    Thumbnails are written during `sfn index` when SFN_THUMBNAIL_DIR is configured.
    Returns 404 when thumbnail dir is not configured or the file is not yet generated.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise HTTPException(status_code=400, detail="Invalid hash")
    settings = Settings()
    if settings.thumbnail_dir is None:
        raise HTTPException(status_code=404, detail="Thumbnail directory not configured")
    thumb_path = settings.thumbnail_dir / f"{sha256}.jpg"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/api/hit-image")
async def hit_image(path: str) -> FileResponse:
    """Serve a hit image from the server filesystem.

    For direct images: resolves the path and returns the file.
    For container-extracted images (path contains ``::``): re-extracts the
    specific item from the container on demand.

    Security: only absolute resolved paths; image-only extensions for direct files.
    """
    # Container virtual path: "/abs/root.zip::inner/photo.jpg"
    if "::" in path:
        sep_idx = path.index("::")
        container_path_str = path[:sep_idx]
        item_name = path[sep_idx + 2:]
        cp = Path(container_path_str).resolve()
        if not cp.is_absolute() or not cp.exists() or not cp.is_file():
            raise HTTPException(status_code=404, detail="Container file not found")
        if cp.suffix.lower() not in CONTAINER_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Not a container file")
        settings = Settings()
        try:
            extracted = extract_container(
                cp,
                max_depth=settings.max_container_depth,
                pdf_render_dpi=settings.pdf_render_dpi,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
        match = next((img for img in extracted if img.item_name == item_name), None)
        if match is None:
            raise HTTPException(status_code=404, detail="Item not found in container")
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(item_name.split("::")[-1]).suffix or ".png"
        )
        try:
            tmp.write(match.data)
        finally:
            tmp.close()
        return FileResponse(
            tmp.name,
            filename=Path(item_name.split("::")[-1]).name,
            background=_cleanup_background(tmp.name),
        )

    p = Path(path).resolve()
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    if p.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not an image file")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p, filename=p.name)


@app.get("/api/container-download")
async def container_download(path: str) -> FileResponse:
    """Serve a container file (ZIP / PDF / DOCX / ODF) from the server filesystem.

    Security: only absolute resolved paths; container-only extensions.
    """
    p = Path(path).resolve()
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    if p.suffix.lower() not in CONTAINER_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a container file")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p, filename=p.name)


@app.get("/api/container-siblings")
async def container_siblings(container_hash: str, collection: str = "sscd") -> JSONResponse:
    """Return all images in Qdrant that share the same immediate parent container.

    :param container_hash: SHA-256 of the immediate parent container.
    :param collection: Which Qdrant collection to query (``"sscd"`` or ``"dino"``).
    """
    if not re.fullmatch(r"[0-9a-f]{64}", container_hash):
        raise HTTPException(status_code=400, detail="Invalid container hash")
    settings = Settings()
    coll = settings.collection_sscd if collection == "sscd" else settings.collection_dino
    from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    try:
        records, _ = client.scroll(
            collection_name=coll,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="container_hash", match=MatchValue(value=container_hash))
                ]
            ),
            limit=100,
            with_payload=[
                "image_hash",
                "image_path",
                "container_item_name",
                "extraction_kind",
            ],
            with_vectors=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qdrant query failed: {exc}") from exc
    return JSONResponse(
        [
            {
                "image_hash": r.payload.get("image_hash"),
                "image_path": r.payload.get("image_path"),
                "container_item_name": r.payload.get("container_item_name"),
                "extraction_kind": r.payload.get("extraction_kind"),
            }
            for r in records
        ]
    )


def _cleanup_background(path_str: str):
    """Return a BackgroundTask that deletes a temp file after the response is sent."""
    from starlette.background import BackgroundTask  # noqa: PLC0415

    def _delete():
        try:
            Path(path_str).unlink(missing_ok=True)
        except OSError:
            pass

    return BackgroundTask(_delete)


# ---------------------------------------------------------------------------
# Semantic score distribution stats (on-demand, per file)
# ---------------------------------------------------------------------------


@app.get("/api/semantic-stats/{session_id}/{file_id}")
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


@app.get("/api/library-versions")
async def library_versions() -> JSONResponse:
    """Return the library versions of the current web server process."""
    return JSONResponse(get_library_versions())


@app.get("/api/hit-provenance")
async def hit_provenance(image_hash: str) -> JSONResponse:
    """Return full Qdrant indexing provenance for a given image SHA-256."""
    if not re.fullmatch(r"[0-9a-f]{64}", image_hash):
        raise HTTPException(status_code=400, detail="Invalid hash")
    settings = Settings()
    return JSONResponse(get_hit_qdrant_provenance(image_hash, settings))


# ---------------------------------------------------------------------------
# Metadata (on-demand, for the detail panels)
# ---------------------------------------------------------------------------


@app.get("/api/query-metadata/{session_id}/{file_id}")
async def query_metadata(session_id: str, file_id: str) -> JSONResponse:
    """Detailed metadata for an uploaded query file.

    For regular images: EXIF + hash + dimension metadata.
    For container-root entries: container-level metadata (no EXIF).
    For container-extracted images: image metadata plus container provenance fields.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None or not entry.temp_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    data = entry.temp_path.read_bytes()

    # Container-root entries are not images — return minimal metadata.
    if entry.container_type is not None and entry.parent_file_id is None:
        meta: dict = {
            "filename": entry.filename,
            "is_container": True,
            "container_type": entry.container_type,
            "size_bytes": len(data),
            "hash_sha256": entry.file_hash or hashlib.sha256(data).hexdigest(),
            "hash_md5": entry.file_hash_md5 or hashlib.md5(data).hexdigest(),  # noqa: S324
        }
        return JSONResponse(meta)

    meta = extract_exif_detailed(data)
    meta["filename"] = entry.filename
    meta["hash_sha256"] = entry.file_hash or hashlib.sha256(data).hexdigest()
    meta["hash_md5"] = entry.file_hash_md5 or hashlib.md5(data).hexdigest()  # noqa: S324
    # Container provenance for extracted images.
    if entry.parent_file_id is not None:
        meta["parent_file_id"] = entry.parent_file_id
        meta["container_type"] = entry.container_type
        meta["container_item_name"] = entry.container_item_name
        meta["extraction_kind"] = entry.extraction_kind
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
    meta["hash_sha256"] = hashlib.sha256(data).hexdigest()
    meta["hash_md5"] = hashlib.md5(data).hexdigest()  # noqa: S324
    return JSONResponse(meta)


# ---------------------------------------------------------------------------
# 3-D vector visualization
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


def _pca3(vectors: list[list[float]]) -> list[list[float]]:
    """Project high-dimensional vectors down to 3 principal components.

    Returns coordinates normalised to [-1, 1].  Falls back to zero-vectors
    when fewer than 3 points are supplied (degenerate case).
    """
    if len(vectors) < 3:
        return [[0.0, 0.0, 0.0]] * len(vectors)
    X = torch.tensor(vectors, dtype=torch.float32)
    X -= X.mean(dim=0)
    _, _, V = torch.pca_lowrank(X, q=3)
    proj = X @ V
    mx = proj.abs().max()
    if mx > 0:
        proj = proj / mx
    return proj.tolist()


def _fetch_and_project(collection: str, max_points: int, settings: Settings) -> list[list[float]]:
    """Scroll *collection* for up to *max_points* vectors, then PCA-project to 3-D."""
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    vectors: list[list[float]] = []
    offset = None
    try:
        while len(vectors) < max_points:
            batch_size = min(256, max_points - len(vectors))
            records, offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                with_vectors=True,
                offset=offset,
            )
            for r in records:
                v = r.vector
                if isinstance(v, dict):
                    v = next(iter(v.values()), None)
                if v is not None:
                    vectors.append(v)
            if offset is None:
                break
    except Exception as exc:  # collection missing or Qdrant unreachable
        _log.warning("points3d: could not scroll %r: %s", collection, exc)
        return []
    return _pca3(vectors)


def _write_viz_export(path: Path, data: dict) -> None:
    """Write the standalone viz HTML to *path*, creating parent dirs as needed."""
    try:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_viz_html(data), encoding="utf-8")
        _log.info("viz export written to %s", path)
    except Exception as exc:
        _log.warning("could not write viz export to %s: %s", path, exc)


@app.get("/viz")
async def viz_standalone() -> HTMLResponse:
    """Self-contained visualization page — no server calls after load.

    Point-cloud data is embedded as JSON; ``viz.js`` is inlined.
    Suitable for use as a KDE web-page wallpaper or browser-based screensaver
    by pointing the client at ``http://localhost:8080/viz``.
    """
    return HTMLResponse(_render_viz_html(_points3d_cache or {"sscd": [], "dino": []}))


@app.get("/api/points3d")
async def points3d() -> JSONResponse:
    """Return cached PCA-projected 3-D coordinates for both vector collections.

    The point cloud is computed once at startup and served from memory.
    Set ``SFN_VIZ_MAX_POINTS=0`` to disable the visualization entirely.
    """
    return JSONResponse(_points3d_cache or {"sscd": [], "dino": []})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def start() -> None:
    parser = argparse.ArgumentParser(
        prog="sfn-web",
        description="ScalarForensic web UI",
    )
    parser.add_argument(
        "--allow-online",
        action="store_true",
        default=False,
        help=(
            "Allow outward internet connections (e.g. to HuggingFace Hub for first-time "
            "model downloads). Offline by default — see SFN_ALLOW_ONLINE in .env."
        ),
    )
    args = parser.parse_args()

    # Write back to os.environ before constructing Settings so that every
    # per-request Settings() instance created by FastAPI handlers also sees
    # allow_online=True — mutating the object here would have no effect on them.
    if args.allow_online:
        os.environ["SFN_ALLOW_ONLINE"] = "true"

    settings = Settings()

    # Apply HuggingFace offline guard before any model loading occurs.
    # Qdrant / remote-embedder connections are unaffected.
    settings.apply_network_policy()

    # Pre-flight: always check DINOv2 — available modes aren't known until Qdrant
    # is queried, so we conservatively validate all potentially-used model configs.
    err = settings.offline_model_error(need_dino=True)
    if err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    uvicorn.run("scalar_forensic.web.app:app", host="0.0.0.0", port=8080, reload=False)
