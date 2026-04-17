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

import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, UnidentifiedImageError
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings
from scalar_forensic.embedder import (
    _SSCD_INPUT_SIZE,
    _open_rgb,
    _sscd_resize,
    extract_exif_detailed,
    get_library_versions,
)
from scalar_forensic.video import (
    VIDEO_EXTENSIONS,
    extract_frame_at,
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
        thumb = img.copy()
        thumb.thumbnail(
            (settings.thumbnail_size, settings.thumbnail_size), Image.Resampling.LANCZOS
        )
        if thumb.mode not in ("RGB", "L"):
            thumb = thumb.convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically so concurrent requests never read a half-written file.
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, dir=dest.parent, suffix=dest.suffix or ".jpg"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
            thumb.save(tmp_path, format="JPEG", quality=85, optimize=True)
            os.replace(tmp_path, dest)
        except Exception:
            if tmp_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
            raise
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
    out = resized.copy().convert("RGB")
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
                frame_sha256 = _records[0].payload.get("image_hash")
                video_path_str = _records[0].payload.get("video_path")
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
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=settings.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="video_hash", match=MatchValue(value=video_hash)),
                        FieldCondition(key="is_video_frame", match=MatchValue(value=True)),
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=[
                    "image_path",
                    "image_hash",
                    "frame_timecode_ms",
                    "frame_index",
                    "video_path",
                ],
                with_vectors=False,
            )
            for r in records:
                tc = r.payload.get("frame_timecode_ms")
                if tc is not None and tc not in frames:
                    frames[tc] = {
                        "timecode_ms": tc,
                        "frame_hash": r.payload.get("image_hash"),
                        "frame_index": r.payload.get("frame_index"),
                        "virtual_path": r.payload.get("image_path"),
                        "video_path": r.payload.get("video_path"),
                    }
            if offset is None:
                break
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
