"""Video routes: on-demand frame extraction, indexed-frame timelines, playback."""

from __future__ import annotations

import asyncio
import functools
import io
import logging
import mimetypes
import os
import re
from pathlib import Path

import av
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings
from scalar_forensic.embedder import hash_file
from scalar_forensic.indexer import qdrant_scroll_all
from scalar_forensic.video import VIDEO_EXTENSIONS, extract_frame_at
from scalar_forensic.web.routes._shared import _check_allowed_path

_log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Video frame serving
# ---------------------------------------------------------------------------


def _resolve_video_path(path: str) -> Path:
    """Validate *path* as a servable video file and return it resolved.

    Same allowed-root check as /api/hit-image (``_check_allowed_path``); the set
    of acceptable extensions is the scanner's own ``VIDEO_EXTENSIONS``, never a
    hand-copied list, so what can be played can never drift from what can be
    indexed.
    """
    raw = Path(path)
    if not raw.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    p = raw.resolve()
    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a video file")
    _check_allowed_path(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return p


@router.get("/api/video-frame")
async def video_frame(path: str, timecode_ms: int) -> StreamingResponse:
    """Re-extract and serve a single video frame as JPEG.

    ``path`` must be an absolute filesystem path to a video file.
    ``timecode_ms`` is the target timecode in milliseconds.
    """
    if timecode_ms < 0:
        raise HTTPException(status_code=400, detail="timecode_ms must be >= 0")
    p = _resolve_video_path(path)

    img = await asyncio.to_thread(extract_frame_at, p, timecode_ms)
    if img is None:
        raise HTTPException(status_code=404, detail="Frame not found at given timecode")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@router.get("/api/video-timeline")
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
# Playback viewing copies
# ---------------------------------------------------------------------------
#
# Browsers refuse the QuickTime container, not the bitstreams inside it: the
# operator's Chrome decodes HEVC happily once the same coded frames arrive in an
# MP4 box structure.  So playback never transcodes.  A source whose container the
# browser cannot open is *rewrapped* — every video and audio packet copied byte
# for byte into an MP4 with the moov atom up front — and the rewrap is parked in
# a bounded cache keyed by the SHA-256 of the source file.
#
# The rewrapped file is a VIEWING COPY.  It is not evidence: the original file
# and the frame JPEGs written during indexing remain the authoritative artifacts,
# and every UI surface that offers playback says so.

# Codecs that are legal in an MP4 box structure.  A stream outside this set
# cannot be rewrapped (and must not be re-encoded), so it is left behind and
# named in the playback report rather than silently dropped.
_MP4_LEGAL_CODECS = frozenset(
    {
        "h264",
        "hevc",
        "av1",
        "vp9",
        "mpeg4",
        "mjpeg",
        "aac",
        "mp3",
        "ac3",
        "eac3",
        "alac",
        "opus",
        "flac",
    }
)

# ISO base-media brands are all playable in Chrome; "qt  " is the QuickTime
# brand that is not.  WebM is listed separately by extension because it carries
# no ftyp box at all — .mkv is deliberately absent, since Chrome opens WebM but
# not Matroska at large, so an .mkv takes the rewrap path.
_QUICKTIME_BRAND = "qt  "
_NATIVE_EXTENSIONS = frozenset({".webm"})

# One lock per source digest: two viewers opening the same clip must not race to
# write the same cache entry.
_remux_locks: dict[str, asyncio.Lock] = {}


def _ftyp_brand(p: Path) -> str | None:
    """Return the ISO base-media major brand of *p*, or None if it has no ftyp box."""
    try:
        with p.open("rb") as fh:
            head = fh.read(12)
    except OSError:
        return None
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    return head[8:12].decode("latin-1")


def _needs_remux(p: Path) -> bool:
    """True when the browser cannot open *p*'s container as it stands.

    Container-level judgement only — no codec is inspected, because nothing here
    can fix a codec the browser lacks: a rewrap moves the same bitstream.
    """
    if p.suffix.lower() in _NATIVE_EXTENSIONS:
        return False
    brand = _ftyp_brand(p)
    return brand is None or brand == _QUICKTIME_BRAND


@functools.lru_cache(maxsize=256)
def _source_digest(path_str: str, mtime_ns: int, size: int) -> str:
    """SHA-256 of the source file, memoised on (path, mtime, size).

    This is the same digest the indexer records as ``video_hash``, so a viewing
    copy can always be tied back to the point that produced the frame hit.
    """
    return hash_file(Path(path_str))


def _stream_report(p: Path) -> dict:
    """Container/codec summary used by the playback label and the rewrap filter.

    ``skipped_streams`` names the streams a rewrap would have to leave behind —
    Apple's Live-Photo .MOV files carry LPCM audio, which has no MP4 mapping and
    cannot be re-encoded here, so the viewing copy of one is silent.  The label
    says so rather than letting the operator infer a silent original.
    """
    info: dict = {"container_brand": _ftyp_brand(p), "video_codec": None, "audio_codec": None}
    skipped: list[str] = []
    try:
        with av.open(str(p)) as container:
            info["format"] = container.format.name
            if container.duration is not None:
                info["duration_ms"] = int(container.duration / av.time_base * 1000)
            for s in container.streams:
                if s.type == "video" and info["video_codec"] is None:
                    info["video_codec"] = s.codec_context.name
                    info["video_codec_tag"] = s.codec_context.codec_tag
                elif s.type == "audio" and info["audio_codec"] is None:
                    info["audio_codec"] = s.codec_context.name
                if s.type in ("video", "audio") and s.codec_context.name not in _MP4_LEGAL_CODECS:
                    skipped.append(f"{s.type}:{s.codec_context.name}")
    except (av.FFmpegError, OSError) as exc:
        _log.debug("playback probe failed for %s: %s", p, exc)
        info["probe_error"] = str(exc)
    info["skipped_streams"] = skipped
    return info


def _repair_timestamps(packet: av.Packet, last_dts: int | None) -> bool:
    """Make *packet*'s timestamps muxable in place; True when something moved.

    Real iPhone .MOV files carry the occasional frame whose stored composition
    time lands *before* its decode time (measured on the corpus: 1 packet in 18
    on IMG_3743.MOV).  libavformat's muxer refuses such a packet outright with
    EINVAL, and refuses a decode time that fails to advance.  Both are repaired
    the minimal way — pull the decode stamp back to the composition stamp, or
    nudge it one tick past its predecessor.

    Timing metadata only.  The coded payload never changes, which is the whole
    point of a rewrap; the count of repairs is reported so the adjustment is
    never silent.
    """
    moved = False
    if packet.pts is not None and packet.dts is not None and packet.pts < packet.dts:
        packet.dts = packet.pts
        moved = True
    if last_dts is not None and packet.dts is not None and packet.dts <= last_dts:
        packet.dts = last_dts + 1
        if packet.pts is not None and packet.pts < packet.dts:
            packet.pts = packet.dts
        moved = True
    return moved


def _remux_to_mp4(src: Path, dst: Path) -> dict:
    """Stream-copy *src* into a faststart MP4 at *dst*; return a rewrap report.

    Not a re-encode: packets are demuxed and remuxed with their payloads
    untouched, so the video and audio bitstreams in *dst* are bit-identical to
    those in *src*.  Streams whose codec has no MP4 mapping are left behind and
    named in the report rather than silently dropped, and any timestamp repair
    (:func:`_repair_timestamps`) is counted there too.

    Writes to a sibling ``.part`` file and renames on success, so a reader can
    never observe a half-written viewing copy.
    """
    skipped: list[str] = []
    repaired = 0
    part = dst.with_name(f"{dst.name}.{os.getpid()}.part")
    part.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            av.open(str(src)) as inp,
            av.open(str(part), "w", format="mp4", options={"movflags": "+faststart"}) as out,
        ):
            mapping = {}
            for s in inp.streams:
                if s.type not in ("video", "audio"):
                    continue
                codec = s.codec_context.name
                if codec not in _MP4_LEGAL_CODECS:
                    skipped.append(f"{s.type}:{codec}")
                    continue
                mapping[s.index] = out.add_stream_from_template(s)
            if not mapping:
                raise ValueError("no MP4-compatible stream in source")
            last_dts: dict[int, int] = {}
            for packet in inp.demux([s for s in inp.streams if s.index in mapping]):
                if packet.dts is None:  # flush packet from the demuxer
                    continue
                index = packet.stream.index
                if _repair_timestamps(packet, last_dts.get(index)):
                    repaired += 1
                last_dts[index] = packet.dts
                # Timestamps stay in the source time base; the muxer rescales
                # them onto its own when the packet is written.
                packet.stream = mapping[index]
                out.mux(packet)
        os.replace(part, dst)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return {"skipped_streams": skipped, "timestamp_repairs": repaired}


def _evict_cache(cache_dir: Path, max_bytes: int, keep: Path) -> int:
    """Delete least-recently-served viewing copies until the cache fits *max_bytes*.

    Returns the number of files deleted.  *keep* is never evicted — it is the
    copy the current request is about to serve.  Recency is the file mtime,
    which :func:`_touch` refreshes on every cache hit.
    """
    if max_bytes <= 0:
        return 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for f in cache_dir.glob("*.mp4"):
        try:
            st = f.stat()
        except OSError:
            continue
        total += st.st_size
        entries.append((st.st_mtime, st.st_size, f))
    if total <= max_bytes:
        return 0
    deleted = 0
    for _mtime, size, f in sorted(entries):
        if total <= max_bytes:
            break
        if f == keep:
            continue
        try:
            f.unlink()
        except OSError:  # pragma: no cover - racing eviction
            continue
        total -= size
        deleted += 1
        _log.info("viewing-copy cache: evicted %s (%d bytes)", f.name, size)
    return deleted


def _touch(p: Path) -> None:
    try:
        os.utime(p)
    except OSError:  # pragma: no cover - best effort recency bump
        pass


def _cache_dir_or_503(settings: Settings) -> Path:
    if settings.video_cache_dir is None:
        raise HTTPException(
            status_code=503,
            detail=("Playback of rewrapped containers is disabled: SFN_VIDEO_CACHE_DIR is unset"),
        )
    return settings.video_cache_dir


async def _prepare_viewing_copy(p: Path, settings: Settings) -> tuple[Path, dict]:
    """Return the file to serve for source *p*, plus a report about how it was made.

    ``mode`` is ``"original"`` when the source container is already one the
    browser opens — the source file itself is served, untouched — and
    ``"rewrap"`` when a cached MP4 rewrap is served instead.
    """
    st = p.stat()
    report: dict = {"source_path": str(p), "source_size_bytes": st.st_size}
    if not _needs_remux(p):
        report["mode"] = "original"
        return p, report

    report["mode"] = "rewrap"
    cache_dir = _cache_dir_or_503(settings)
    digest = await asyncio.to_thread(_source_digest, str(p), st.st_mtime_ns, st.st_size)
    report["video_sha256"] = digest
    dst = cache_dir / f"{digest}.mp4"

    lock = _remux_locks.setdefault(digest, asyncio.Lock())
    async with lock:
        if dst.exists():
            report["cached"] = True
            await asyncio.to_thread(_touch, dst)
            return dst, report
        report["cached"] = False
        try:
            report.update(await asyncio.to_thread(_remux_to_mp4, p, dst))
        except (av.FFmpegError, ValueError, OSError) as exc:
            _log.warning("viewing copy: rewrap failed for %s: %s", p, exc)
            raise HTTPException(
                status_code=422,
                detail=f"Cannot rewrap this container for playback: {exc}",
            ) from exc
        await asyncio.to_thread(_evict_cache, cache_dir, settings.video_cache_max_bytes, dst)
    return dst, report


@router.get("/api/video-playback")
async def video_playback(path: str) -> FileResponse:
    """Serve a source video for in-browser playback.

    MP4-family and WebM sources are served as they are on disk.  Everything else
    — QuickTime .MOV above all — is served as a losslessly rewrapped MP4 viewing
    copy from the bounded cache (see ``_prepare_viewing_copy``).  Range requests
    are handled by Starlette's ``FileResponse``, which is what lets the operator
    seek straight to a hit's timecode without downloading the whole clip.
    """
    p = _resolve_video_path(path)
    settings = Settings()
    served, report = await _prepare_viewing_copy(p, settings)
    media_type = (
        "video/mp4"
        if report["mode"] == "rewrap"
        else (mimetypes.guess_type(p.name)[0] or "video/mp4")
    )
    return FileResponse(
        served,
        media_type=media_type,
        headers={"X-SFN-Playback-Mode": report["mode"]},
    )


@router.get("/api/video-playback-info")
async def video_playback_info(path: str) -> JSONResponse:
    """Describe what playback of *path* would serve, without serving it.

    Feeds the viewing-copy label in the UI: how the file reaches the player,
    which codecs it carries, and — for a rewrap — the SHA-256 of the source,
    which is the same value the indexer stored as ``video_hash``.
    """
    p = _resolve_video_path(path)
    settings = Settings()
    st = p.stat()
    info: dict = {
        "source_path": str(p),
        "filename": p.name,
        "source_size_bytes": st.st_size,
        "mode": "original" if not _needs_remux(p) else "rewrap",
        "playback_url": f"/api/video-playback?path={p}",
    }
    info.update(await asyncio.to_thread(_stream_report, p))
    if info["mode"] == "original":
        # Nothing is left behind when the file is served as it lies on disk.
        info["skipped_streams"] = []
    else:
        info["video_sha256"] = await asyncio.to_thread(
            _source_digest, str(p), st.st_mtime_ns, st.st_size
        )
        cache_dir = settings.video_cache_dir
        info["cached"] = (
            cache_dir is not None and (cache_dir / f"{info['video_sha256']}.mp4").exists()
        )
        info["cache_enabled"] = cache_dir is not None
    return JSONResponse(info)
