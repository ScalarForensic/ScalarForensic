"""FastAPI web application for ScalarForensic Phase 2 query interface."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, UnidentifiedImageError
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import ENV_ALLOW_ONLINE, Settings
from scalar_forensic.discovery import MAX_CONTEXT_PAIRS, DiscoveryHit, run_discovery, run_explore
from scalar_forensic.embedder import (
    _SSCD_INPUT_SIZE,
    _open_rgb,
    _sscd_resize,
    extract_exif_detailed,
    get_library_versions,
    write_thumbnail,
)
from scalar_forensic.indexer import qdrant_scroll_all
from scalar_forensic.query_eval import QueryEvalHit, score_query_entries
from scalar_forensic.tags import Tag, TagStore
from scalar_forensic.video import (
    VIDEO_EXTENSIONS,
    extract_frame_at,
    frame_disk_path,
    get_video_info,
    parse_frame_path,
)
from scalar_forensic.web.pipeline import (
    ProgressEvent,
    QueryProvenance,
    analyze_session,
    get_available_modes,
    get_hit_qdrant_provenance,
    query_semantic_stats,
    query_session,
)
from scalar_forensic.web.session import FileEntry, create_session, get_session

_DEFAULT_COSINE_THRESHOLD = 0.5

_MAX_TAG_NAME_LEN = 200
_MAX_TAG_NOTES_LEN = 4000

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


def _check_allowed_path(p: Path) -> None:
    """Raise 403 unless *p* is under an allowed root (input_dir or frame_store_dir).

    File-serving endpoints require at least one configured root.  When neither
    is set we fail closed rather than serving arbitrary host paths.
    """
    settings = Settings()
    allowed_roots: list[Path] = []
    if settings.input_dir is not None:
        allowed_roots.append(settings.input_dir.resolve())
    if settings.frame_store_dir is not None:
        allowed_roots.append(settings.frame_store_dir.resolve())
    if not allowed_roots:
        raise HTTPException(
            status_code=403,
            detail=(
                "File serving is disabled: neither SFN_INPUT_DIR nor "
                "SFN_FRAME_STORE_DIR is configured"
            ),
        )
    for root in allowed_roots:
        try:
            p.relative_to(root)
            return
        except ValueError:
            continue
    raise HTTPException(status_code=403, detail="Path is outside the allowed directories")


# Cached PCA-projected point cloud, computed once at startup.
_points3d_cache: dict | None = None


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    global _points3d_cache
    settings = Settings()

    # Log effective batch size so operators know which value is in use.
    if settings.batch_size is not None:
        logging.getLogger(__name__).info("Batch size: %d (SFN_BATCH_SIZE)", settings.batch_size)
    else:
        from scalar_forensic.calibration import load_cached_batch_size

        cached = load_cached_batch_size()
        if cached is not None:
            logging.getLogger(__name__).info("Batch size: %d (calibration cache)", cached)
        else:
            logging.getLogger(__name__).info(
                "Batch size: 32 (default — run `sfn` once to auto-calibrate)"
            )

    if settings.viz_max_points > 0:
        sscd_pts, dino_pts = await asyncio.gather(
            asyncio.to_thread(
                _fetch_and_project,
                settings.collection,
                "sscd",
                settings.viz_max_points,
                settings,
            ),
            asyncio.to_thread(
                _fetch_and_project,
                settings.collection,
                "dino",
                settings.viz_max_points,
                settings,
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
    modes, has_reference, error = await get_available_modes(settings)
    payload: dict = {"modes": modes, "has_reference": has_reference}
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
    session.temp_dir = tmp_dir

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
                fout.write(chunk)
        session.files.append(FileEntry(file_id=file_id, filename=filename, temp_path=dest))

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


@app.post("/api/query")
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
    results, embedding_models = query_session(
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


@app.get("/api/query-frames/{session_id}/{file_id}")
async def query_video_frames(session_id: str, file_id: str) -> JSONResponse:
    """Return the list of frames extracted from an uploaded query video.

    Each entry has ``frame_index``, ``timecode_ms``, and ``frame_hash``.
    Used by the frontend to drive the frame slideshow in the query panel.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="File not found")
    if not entry.is_video or not entry.video_frames:
        raise HTTPException(status_code=400, detail="Not a video upload or no frames extracted")
    return JSONResponse(
        {
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "timecode_ms": f.timecode_ms,
                    "frame_hash": f.frame_hash,
                }
                for f in entry.video_frames
            ]
        }
    )


@app.get("/api/query-frame/{session_id}/{file_id}")
async def query_video_frame(session_id: str, file_id: str, timecode_ms: int) -> StreamingResponse:
    """Re-extract and serve a single frame from an uploaded query video as JPEG.

    Called by the slideshow on every navigation step; no frames are cached on
    disk — PyAV re-seeks and decodes on each request.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None or not entry.temp_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not entry.is_video:
        raise HTTPException(status_code=400, detail="Not a video file")
    if timecode_ms < 0:
        raise HTTPException(status_code=400, detail="timecode_ms must be >= 0")
    img = await asyncio.to_thread(extract_frame_at, entry.temp_path, timecode_ms)
    if img is None:
        raise HTTPException(status_code=404, detail="Frame not found at given timecode")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/api/thumbnail/{sha256}")
async def thumbnail(sha256: str) -> FileResponse:
    """Serve a pre-generated thumbnail by SHA-256 hash.

    Thumbnails are written during `sfn index` when SFN_THUMBNAIL_DIR is configured.
    If the thumbnail file is missing but the raw file path is known in Qdrant,
    attempts to regenerate and cache it before serving.
    Returns 404 when the thumbnail dir is not configured or regeneration fails.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise HTTPException(status_code=400, detail="Invalid hash")
    settings = Settings()
    if settings.thumbnail_dir is None:
        raise HTTPException(status_code=404, detail="Thumbnail directory not configured")
    thumb_path = settings.thumbnail_dir / f"{sha256}.jpg"
    if not thumb_path.exists():
        await _try_regenerate_thumbnail(sha256, thumb_path, settings)
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumb_path, media_type="image/jpeg")


async def _try_regenerate_thumbnail(sha256: str, dest: Path, settings: Settings) -> None:
    """Look up *sha256* in Qdrant and regenerate the missing thumbnail from the raw file.

    Resolution order:
      1. data/frames/{sha256}.jpg  — frame store written during indexing
      2. Re-extract from the source video / reopen the source image (path from Qdrant)

    Silently returns when regeneration is not possible (missing record, missing
    file, extraction failure).  On success the JPEG is written to *dest* exactly
    like thumbnails produced during ``sfn index``.
    """

    def _write(img: Image.Image, source_label: str) -> None:
        write_thumbnail(img, dest, settings.thumbnail_size)
        _log.info("thumbnail regen: saved %s from %s", sha256[:12], source_label)

    try:
        # ── Look up source path in Qdrant, then load the file directly ────────
        def _scroll_qdrant() -> list[dict]:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            payloads: list[dict] = []
            try:
                records, _ = client.scroll(
                    collection_name=settings.collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="image_hash", match=MatchValue(value=sha256))]
                    ),
                    limit=4,
                    with_payload=True,
                    with_vectors=False,
                )
                payloads.extend(r.payload for r in records if r.payload)
            except Exception as exc:  # noqa: BLE001
                _log.debug("thumbnail regen: could not scroll %r: %s", settings.collection, exc)
            return payloads

        payloads = await asyncio.to_thread(_scroll_qdrant)

        if not payloads:
            _log.debug("thumbnail regen: no Qdrant record found for hash %s", sha256[:12])
            return

        # Build allowed roots: images live under input_dir, frame JPEGs under frame_store_dir.
        allowed_roots: list[Path] = []
        if settings.input_dir is not None:
            allowed_roots.append(settings.input_dir.resolve())
        if settings.frame_store_dir is not None:
            allowed_roots.append(settings.frame_store_dir.resolve())
        if not allowed_roots:
            _log.debug("thumbnail regen: no allowed roots configured, skipping")
            return

        def _allowed(p: Path) -> bool:
            for root in allowed_roots:
                try:
                    p.relative_to(root)
                    return True
                except ValueError:
                    continue
            _log.warning("thumbnail regen: path outside allowed dirs: %s", p)
            return False

        def _load(image_path: str) -> Image.Image | None:
            raw = Path(image_path)
            if not raw.is_absolute():
                _log.warning("thumbnail regen: image path not absolute: %s", raw)
                return None
            raw = raw.resolve()
            if not _allowed(raw):
                return None
            if not raw.is_file():
                _log.warning("thumbnail regen: file not found: %s", raw)
                return None
            try:
                with Image.open(raw) as img:
                    img.load()
                    return img.copy()
            except Exception as exc:  # noqa: BLE001
                _log.warning("thumbnail regen: could not open %r: %s", raw, exc)
                return None

        # Try each candidate payload until one successfully produces an image.
        for payload in payloads:
            image_path = payload.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                continue
            img = await asyncio.to_thread(_load, image_path)
            if img is not None:
                await asyncio.to_thread(_write, img, image_path)
                return
        _log.debug("thumbnail regen: no usable source found for hash %s", sha256[:12])

    except Exception as exc:  # noqa: BLE001
        _log.warning("thumbnail regen: unexpected error for %s: %s", sha256[:12], exc)


@app.get("/api/hit-image")
async def hit_image(path: str) -> Response:
    """Serve a hit image or stored video frame JPEG from the server filesystem."""
    raw = Path(path)
    if not raw.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    p = raw.resolve()
    _check_allowed_path(p)
    if p.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not an image file")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p, filename=p.name)


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
# Preprocessing preview helpers (audit modal)
# ---------------------------------------------------------------------------


def _sscd_annotated(img: Image.Image, n_crops: int) -> Image.Image:
    resized = _sscd_resize(img)
    out = resized.convert("RGB")
    draw = ImageDraw.Draw(out)
    s = _SSCD_INPUT_SIZE
    w, h = resized.size
    cx, cy = (w - s) // 2, (h - s) // 2
    boxes: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]] = [
        ((cx, cy, cx + s, cy + s), (220, 40, 40)),  # center: red
    ]
    if n_crops == 5:
        boxes += [
            ((0, 0, s, s), (230, 190, 0)),  # TL: yellow
            ((w - s, 0, w, s), (0, 200, 230)),  # TR: cyan
            ((0, h - s, s, h), (40, 200, 40)),  # BL: green
            ((w - s, h - s, w, h), (230, 110, 0)),  # BR: orange
        ]
    for (x1, y1, x2, y2), color in boxes:
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color, width=3)
    return out


def _dino_annotated(img: Image.Image, normalize_size: int) -> Image.Image:
    """Resize shortest edge to normalize_size, draw rectangle showing the center-crop area."""
    w, h = img.size
    scale = normalize_size / min(w, h)
    nw = max(normalize_size, round(w * scale))
    nh = max(normalize_size, round(h * scale))
    out = img.resize((nw, nh), Image.Resampling.BICUBIC).convert("RGB")
    draw = ImageDraw.Draw(out)
    cx, cy = (nw - normalize_size) // 2, (nh - normalize_size) // 2
    draw.rectangle(
        [cx, cy, cx + normalize_size - 1, cy + normalize_size - 1], outline=(220, 40, 40), width=3
    )
    return out


def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_preproc_payload(
    img: Image.Image,
    sscd_n_crops: int,
    dino_normalize_size: int,
) -> dict:
    result: dict = {}
    if sscd_n_crops in (1, 5):
        result["sscd"] = {
            "annotated_url": _to_data_url(_sscd_annotated(img, sscd_n_crops)),
            "resize_size": 331,
            "crop_size": _SSCD_INPUT_SIZE,
            "n_crops": sscd_n_crops,
        }
    if dino_normalize_size > 0:
        result["dino"] = {
            "annotated_url": _to_data_url(_dino_annotated(img, dino_normalize_size)),
            "normalize_size": dino_normalize_size,
        }
    return result


@app.get("/api/query-preprocessed/{session_id}/{file_id}")
async def query_preprocessed(
    session_id: str, file_id: str, timecode_ms: int | None = None
) -> JSONResponse:
    """Return SSCD-annotated and DINOv2-cropped preview images for an uploaded query file."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None or not entry.temp_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if entry.is_video and timecode_ms is None:
        raise HTTPException(status_code=400, detail="timecode_ms is required for video queries")

    settings = Settings()

    def _compute() -> dict:
        if entry.is_video:
            pil = extract_frame_at(entry.temp_path, timecode_ms)
            if pil is None:
                raise ValueError("Frame not found at given timecode")
        else:
            pil = _open_rgb(entry.temp_path.read_bytes())
        return _build_preproc_payload(pil, settings.sscd_n_crops, settings.normalize_size)

    try:
        return JSONResponse(await asyncio.to_thread(_compute))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Unable to read uploaded image") from exc


@app.get("/api/hit-preprocessed")
async def hit_preprocessed(
    path: str,
    sscd_n_crops: int = 1,
    dino_normalize_size: int = 224,
) -> JSONResponse:
    """Return SSCD-annotated and DINOv2-cropped preview images for a dataset hit file."""
    if sscd_n_crops not in (0, 1, 5):
        raise HTTPException(status_code=400, detail="sscd_n_crops must be 0, 1, or 5")
    if dino_normalize_size != 0 and not (32 <= dino_normalize_size <= 4096):
        raise HTTPException(
            status_code=400, detail="dino_normalize_size must be 0 or between 32 and 4096"
        )
    raw = Path(path)
    if not raw.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    p = raw.resolve()
    _check_allowed_path(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    def _compute() -> dict:
        try:
            return _build_preproc_payload(
                _open_rgb(p.read_bytes()), sscd_n_crops, dino_normalize_size
            )
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Cannot decode image: {exc}") from exc

    try:
        return JSONResponse(await asyncio.to_thread(_compute))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Metadata (on-demand, for the detail panels)
# ---------------------------------------------------------------------------


@app.get("/api/query-metadata/{session_id}/{file_id}")
async def query_metadata(session_id: str, file_id: str) -> JSONResponse:
    """Detailed metadata for an uploaded query file (image or video)."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None or not entry.temp_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if entry.is_video:
        # For videos: use PyAV container metadata — never read the full file into
        # memory.  Hashes were already computed during analysis via streaming I/O.
        meta = get_video_info(entry.temp_path)
        meta["filename"] = entry.filename
        meta["size_bytes"] = entry.temp_path.stat().st_size
        if entry.file_hash:
            meta["hash_sha256"] = entry.file_hash
        if entry.file_hash_md5:
            meta["hash_md5"] = entry.file_hash_md5
    else:
        data = entry.temp_path.read_bytes()
        meta = extract_exif_detailed(data)
        meta["filename"] = entry.filename
        meta["hash_sha256"] = entry.file_hash or hashlib.sha256(data).hexdigest()
        meta["hash_md5"] = entry.file_hash_md5 or hashlib.md5(data).hexdigest()  # noqa: S324
    return JSONResponse(meta)


@app.get("/api/metadata")
async def hit_metadata(path: str) -> JSONResponse:
    """Detailed metadata for a hit image or stored video frame (filesystem path)."""
    raw = Path(path)
    if not raw.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    p = raw.resolve()
    _check_allowed_path(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    settings = Settings()

    # Detect frame files by their canonical path structure under frame_store_dir.
    frame_parsed = (
        parse_frame_path(p, settings.frame_store_dir)
        if settings.frame_store_dir is not None
        else None
    )
    if frame_parsed is not None:
        video_hash, timecode_ms = frame_parsed

        # Look up the source video path and frame image_hash from Qdrant.
        frame_sha256: str | None = None
        video_path_str: str | None = None
        try:
            _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            _records, _ = _client.scroll(
                collection_name=settings.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="video_hash", match=MatchValue(value=video_hash)),
                        FieldCondition(
                            key="frame_timecode_ms", match=MatchValue(value=timecode_ms)
                        ),
                        FieldCondition(key="is_video_frame", match=MatchValue(value=True)),
                    ]
                ),
                limit=1,
                with_payload=["image_hash", "video_path"],
                with_vectors=False,
            )
            if _records:
                _payload = _records[0].payload or {}
                frame_sha256 = _payload.get("image_hash")
                video_path_str = _payload.get("video_path")
        except Exception:  # noqa: BLE001
            pass

        meta: dict = {
            "filename": p.name,
            "path": str(p),
            "is_video_frame": True,
            "frame_timecode_ms": timecode_ms,
            "video_hash": video_hash,
        }
        if frame_sha256:
            meta["hash_sha256"] = frame_sha256
        if video_path_str:
            meta["video_path"] = video_path_str
            _vp = Path(video_path_str)
            if _vp.is_file():
                _info = get_video_info(_vp)
                meta.update({f"video_{k}": v for k, v in _info.items()})
        return JSONResponse(meta)

    # Regular image path
    if p.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid path")
    data = p.read_bytes()
    meta = extract_exif_detailed(data)
    meta["filename"] = p.name
    meta["path"] = str(p)
    meta["hash_sha256"] = hashlib.sha256(data).hexdigest()
    meta["hash_md5"] = hashlib.md5(data).hexdigest()  # noqa: S324
    return JSONResponse(meta)


@app.get("/api/frame-metadata")
async def frame_metadata(video_hash: str, timecode_ms: int) -> JSONResponse:
    """Metadata for an indexed video frame identified by video_hash + timecode_ms.

    Constructs the canonical on-disk frame path and delegates to the same logic
    used by /api/metadata for frame files.  Returns 404 when frame_store_dir is
    not configured or the frame file does not exist on disk.
    """
    settings = Settings()
    if settings.frame_store_dir is None:
        raise HTTPException(status_code=404, detail="Frame store not configured")
    p = frame_disk_path(settings.frame_store_dir, video_hash, timecode_ms).resolve()
    _check_allowed_path(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Frame file not found")

    # parse_frame_path will succeed because p is under frame_store_dir
    frame_parsed = parse_frame_path(p, settings.frame_store_dir)
    if frame_parsed is None:
        raise HTTPException(status_code=500, detail="Could not parse frame path")
    parsed_video_hash, parsed_timecode_ms = frame_parsed

    frame_sha256: str | None = None
    video_path_str: str | None = None
    try:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        _records, _ = _client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="video_hash", match=MatchValue(value=parsed_video_hash)),
                    FieldCondition(
                        key="frame_timecode_ms", match=MatchValue(value=parsed_timecode_ms)
                    ),
                    FieldCondition(key="is_video_frame", match=MatchValue(value=True)),
                ]
            ),
            limit=1,
            with_payload=["image_hash", "video_path"],
            with_vectors=False,
        )
        if _records:
            frame_sha256 = _records[0].payload.get("image_hash")
            video_path_str = _records[0].payload.get("video_path")
    except Exception:  # noqa: BLE001
        pass

    meta: dict = {
        "filename": p.name,
        "path": str(p),
        "is_video_frame": True,
        "frame_timecode_ms": parsed_timecode_ms,
        "video_hash": parsed_video_hash,
    }
    if frame_sha256:
        meta["hash_sha256"] = frame_sha256
    if video_path_str:
        meta["video_path"] = video_path_str
        _vp = Path(video_path_str)
        if _vp.is_file():
            _info = get_video_info(_vp)
            meta.update({f"video_{k}": v for k, v in _info.items()})
    return JSONResponse(meta)


# ---------------------------------------------------------------------------
# Video frame serving
# ---------------------------------------------------------------------------


@app.get("/api/video-frame")
async def video_frame(path: str, timecode_ms: int) -> StreamingResponse:
    """Re-extract and serve a single video frame as JPEG.

    ``path`` must be an absolute filesystem path to a video file.
    ``timecode_ms`` is the target timecode in milliseconds.
    """
    if timecode_ms < 0:
        raise HTTPException(status_code=400, detail="timecode_ms must be >= 0")
    raw = Path(path)
    if not raw.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    p = raw.resolve()
    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a video file")
    _check_allowed_path(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")

    img = await asyncio.to_thread(extract_frame_at, p, timecode_ms)
    if img is None:
        raise HTTPException(status_code=404, detail="Frame not found at given timecode")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/api/video-timeline")
async def video_timeline(video_hash: str) -> JSONResponse:
    """Return all indexed frame timecodes for a given video hash.

    Scrolls the unified Qdrant collection for points with a matching
    ``video_hash`` payload field.  Returns timecodes, frame hashes, and virtual
    paths so the frontend can render the timeline bar.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", video_hash):
        raise HTTPException(status_code=400, detail="Invalid video hash")

    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    frames: dict[int, dict] = {}  # timecode_ms → frame info
    try:
        for r in qdrant_scroll_all(
            client,
            settings.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="video_hash", match=MatchValue(value=video_hash)),
                    FieldCondition(key="is_video_frame", match=MatchValue(value=True)),
                ]
            ),
            limit=256,
            with_payload=[
                "image_path",
                "image_hash",
                "frame_timecode_ms",
                "frame_index",
                "video_path",
            ],
        ):
            tc = r.payload.get("frame_timecode_ms")
            if tc is not None and tc not in frames:
                frames[tc] = {
                    "timecode_ms": tc,
                    "frame_hash": r.payload.get("image_hash"),
                    "frame_index": r.payload.get("frame_index"),
                    "virtual_path": r.payload.get("image_path"),
                    "video_path": r.payload.get("video_path"),
                }
    except Exception as exc:  # noqa: BLE001
        _log.debug("video-timeline: could not scroll %r: %s", settings.collection, exc)

    return JSONResponse(
        {
            "video_hash": video_hash,
            "frames": sorted(frames.values(), key=lambda f: f["timecode_ms"]),
        }
    )


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


def _fetch_and_project(
    collection: str,
    vector_name: str,
    max_points: int,
    settings: Settings,
) -> list[list[float]]:
    """Scroll *collection* for up to *max_points* named vectors, then PCA-project to 3-D.

    Only points that carry *vector_name* are considered; the scroll filter uses
    ``HasVectorCondition`` so points indexed by only the other model are skipped.
    """
    from qdrant_client.models import HasVectorCondition

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    vectors: list[list[float]] = []
    offset = None
    try:
        while len(vectors) < max_points:
            batch_size = min(256, max_points - len(vectors))
            records, offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=[HasVectorCondition(has_vector=vector_name)]),
                limit=batch_size,
                with_vectors=[vector_name],
                offset=offset,
            )
            for r in records:
                v = r.vector
                if isinstance(v, dict):
                    v = v.get(vector_name)
                if v is not None:
                    vectors.append(v)
            if offset is None:
                break
    except Exception as exc:  # collection missing or Qdrant unreachable
        _log.warning("points3d: could not scroll %r/%r: %s", collection, vector_name, exc)
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


@app.get("/api/tags")
async def list_tags() -> JSONResponse:
    try:

        def _run() -> list:
            return _tag_store().list()

        tags = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return JSONResponse({"tags": [_tag_to_json(t) for t in tags]})


@app.post("/api/tag")
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


@app.get("/api/tag/{tag_id}")
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


@app.post("/api/tag/{tag_id}/mark")
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


@app.post("/api/tag/{tag_id}/unmark")
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


@app.post("/api/tag/{tag_id}/set-target")
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


@app.delete("/api/tag/{tag_id}")
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


@app.post("/api/triage")
async def triage(
    tag_id: str = Form(...),
    limit: int = Form(default=50, ge=1, le=500),
    reverse: bool = Form(default=False),
    source: str = Form(default="indexed"),
    cosine_threshold: float = Form(default=_DEFAULT_COSINE_THRESHOLD, ge=0.0, le=1.0),
) -> JSONResponse:
    """Run a Tag-Triage query against the indexed dataset using DINOv2.

    source='indexed' (default): search the case collection; tag IDs may reference
    the reference collection via lookup_from when SFN_REFERENCE_COLLECTION is set.
    source='reference': search the reference collection directly for high-quality
    tagging using known material.

    cosine_threshold filters Recommend-mode hits (tag has no negatives).
    Discovery-mode triage ignores it because the score is an integer triplet count.
    """
    settings = Settings()
    if source not in {"indexed", "reference"}:
        raise HTTPException(status_code=400, detail="source must be 'indexed' or 'reference'")
    if source == "reference":
        if not settings.reference_collection:
            raise HTTPException(
                status_code=400,
                detail="SFN_REFERENCE_COLLECTION is not configured",
            )
        triage_collection = settings.reference_collection
        # Tag IDs may live in the case collection (the typical workflow:
        # mark hits found via search → those IDs come from SFN_COLLECTION).
        # Pass the case collection as lookup_from so Qdrant can resolve
        # the reference vectors regardless of which collection the tag
        # was built from.  When the tag's IDs already live in the reference
        # collection, lookup_from is harmless (Qdrant prefers in-collection
        # IDs).
        ref_coll_for_lookup = settings.collection
    else:
        triage_collection = settings.collection
        ref_coll_for_lookup = settings.reference_collection
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
            reference_collection=ref_coll_for_lookup,
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


@app.post("/api/explore")
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
        hits, strategy = await asyncio.to_thread(
            run_explore,
            client,
            explore_collection,
            pos_ids,
            list(tag.negative_ids),
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


@app.post("/api/tags/classify")
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


@app.post("/api/tags/classify-session")
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


@app.post("/api/triage/query-images")
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


@app.get("/api/point-id")
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


@app.get("/api/point-payload")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _check_collection_compat(settings: Settings) -> None:
    """Hard-fail if embedding-affecting settings differ from indexed collection points.

    Checks sscd_n_crops and normalize_size against one existing point per vector type.
    Skips silently when Qdrant is unreachable, the collection does not exist, or the
    collection has no points yet (fresh install).  Payload fields absent in older indexes
    are also skipped so as not to break existing deployments retroactively.
    """
    from qdrant_client.models import Filter, HasVectorCondition

    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        return  # Qdrant unreachable — let requests fail naturally

    if settings.collection not in collections:
        return  # fresh install

    info = client.get_collection(settings.collection)
    vectors_cfg = info.config.params.vectors
    if not isinstance(vectors_cfg, dict):
        return

    errors: list[str] = []
    for vn in ("sscd", "dino"):
        if vn not in vectors_cfg:
            continue
        points, _ = client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(must=[HasVectorCondition(has_vector=vn)]),
            with_payload=[f"{vn}_normalize_size", "sscd_n_crops"],
            with_vectors=False,
            limit=1,
        )
        if not points:
            continue
        payload = points[0].payload

        stored_norm = payload.get(f"{vn}_normalize_size")
        if stored_norm is not None and stored_norm != settings.normalize_size:
            errors.append(
                f"[{vn}] normalize_size: collection has {stored_norm}, "
                f"SFN_NORMALIZE_SIZE={settings.normalize_size}"
            )

        if vn == "sscd":
            stored_crops = payload.get("sscd_n_crops")
            if stored_crops is not None and stored_crops != settings.sscd_n_crops:
                errors.append(
                    f"[sscd] sscd_n_crops: collection has {stored_crops}, "
                    f"SFN_SSCD_N_CROPS={settings.sscd_n_crops}"
                )

    if errors:
        detail = "\n  ".join(errors)
        print(
            f"\n[ERROR] Embedding configuration mismatch — server cannot start safely.\n"
            f"\n  {detail}\n"
            f"\nAnalysis results would be silently wrong if the server started.\n"
            f"Options:\n"
            f"  • Restore the original settings in .env to match the indexed collection, OR\n"
            f"  • Re-index the collection: sfn <input_dir> --sscd / --dino\n",
            file=sys.stderr,
        )
        sys.exit(1)


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
        os.environ[ENV_ALLOW_ONLINE] = "true"

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

    # Pre-flight: reject mismatched embedding config before accepting any requests.
    _check_collection_compat(settings)

    uvicorn.run("scalar_forensic.web.app:app", host="0.0.0.0", port=8080, reload=False)
