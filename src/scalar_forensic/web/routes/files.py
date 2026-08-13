"""File-serving routes: query images, thumbnails, previews, and metadata."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings
from scalar_forensic.embedder import (
    _SSCD_INPUT_SIZE,
    _open_rgb,
    _sscd_resize,
    extract_exif_detailed,
    write_thumbnail,
)
from scalar_forensic.video import (
    extract_frame_at,
    frame_disk_path,
    get_video_info,
    parse_frame_path,
)
from scalar_forensic.web.routes._shared import _check_allowed_path
from scalar_forensic.web.session import get_session

_log = logging.getLogger(__name__)

router = APIRouter()


_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif", ".jp2", ".ico", ".psd"}
)

# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------


@router.get("/api/query-image/{session_id}/{file_id}")
async def query_image(session_id: str, file_id: str) -> FileResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry = next((e for e in session.files if e.file_id == file_id), None)
    if entry is None or not entry.temp_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(entry.temp_path, filename=Path(entry.filename).name)


@router.get("/api/query-frames/{session_id}/{file_id}")
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


@router.get("/api/query-frame/{session_id}/{file_id}")
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


@router.get("/api/thumbnail/{sha256}")
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


@router.get("/api/hit-image")
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


@router.get("/api/query-preprocessed/{session_id}/{file_id}")
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


@router.get("/api/hit-preprocessed")
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


@router.get("/api/query-metadata/{session_id}/{file_id}")
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


def _stored_frame_metadata(p: Path, video_hash: str, timecode_ms: int, settings: Settings) -> dict:
    """Metadata for a stored video frame: Qdrant lookup + assembly.

    Shared by /api/metadata (frame-path branch) and /api/frame-metadata.  The
    Qdrant lookup is best-effort — on failure the frame identity from the path
    is still served, just without hash/source-video enrichment.
    """
    frame_sha256: str | None = None
    video_path_str: str | None = None
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        records, _ = client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="video_hash", match=MatchValue(value=video_hash)),
                    FieldCondition(key="frame_timecode_ms", match=MatchValue(value=timecode_ms)),
                    FieldCondition(key="is_video_frame", match=MatchValue(value=True)),
                ]
            ),
            limit=1,
            with_payload=["image_hash", "video_path"],
            with_vectors=False,
        )
        if records:
            payload = records[0].payload or {}
            frame_sha256 = payload.get("image_hash")
            video_path_str = payload.get("video_path")
    except Exception as exc:  # noqa: BLE001
        _log.debug("frame metadata: could not scroll %r: %s", settings.collection, exc)

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
    return meta


@router.get("/api/metadata")
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
        return JSONResponse(_stored_frame_metadata(p, video_hash, timecode_ms, settings))

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


@router.get("/api/frame-metadata")
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
    return JSONResponse(_stored_frame_metadata(p, parsed_video_hash, parsed_timecode_ms, settings))
