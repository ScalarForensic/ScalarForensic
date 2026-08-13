"""Video routes: on-demand frame extraction, indexed-frame timelines, playback."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import mimetypes
import os
import re
import sqlite3
import threading
from pathlib import Path
from urllib.parse import quote

import av
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings
from scalar_forensic.embedder import HashCache, hash_file
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


# The persistent hash cache the indexer already fills, opened lazily and shared
# by every request thread.  Keyed by the configured DB path so a settings change
# (tests, a relocated cache) opens a new one instead of serving the old.
_hash_cache_lock = threading.Lock()
_hash_cache: tuple[str, HashCache | None] | None = None


def _reset_hash_cache() -> None:
    """Drop the process-wide HashCache handle (tests, and a settings change)."""
    global _hash_cache
    with _hash_cache_lock:
        if _hash_cache is not None and _hash_cache[1] is not None:
            with contextlib.suppress(Exception):
                _hash_cache[1].close()
        _hash_cache = None


def _hash_cache_for(settings: Settings) -> HashCache | None:
    """Return the shared HashCache, or None when it is disabled or unusable.

    An unwritable or corrupt DB is not a request failure: the digest is simply
    computed the slow way.  The failure is remembered (a None entry) so every
    later request does not retry a broken SQLite file on the request path.
    """
    global _hash_cache
    db_path = settings.hash_cache_path
    if db_path is None:
        return None
    key = str(db_path)
    with _hash_cache_lock:
        if _hash_cache is not None and _hash_cache[0] == key:
            return _hash_cache[1]
        try:
            cache: HashCache | None = HashCache(db_path)
        except (sqlite3.Error, OSError) as exc:
            _log.warning("hash cache unavailable at %s (%s); hashing directly", db_path, exc)
            cache = None
        _hash_cache = (key, cache)
        return cache


def _source_digest(p: Path, settings: Settings | None = None) -> str:
    """SHA-256 of the source file *as it is on disk right now*.

    Backed by the same persistent :class:`HashCache` the indexer fills — keyed on
    ``(resolved path, mtime_ns, size)``, so a touched or rewritten file is
    re-hashed rather than remembered.  The value is never looked up from the
    indexed ``video_hash``: a label beside a rendering must describe the file as
    it is, not as it was indexed (spec §7.1).

    Blocking (a cache miss reads the whole file), so callers must offload it off
    the event loop.
    """
    cache = _hash_cache_for(settings if settings is not None else Settings())
    if cache is None:
        return hash_file(p)
    try:
        digest, was_cached = cache.get_or_hash(p)
    except (sqlite3.Error, OSError) as exc:
        _log.warning("hash cache lookup failed for %s (%s); hashing directly", p, exc)
        return hash_file(p)
    if not was_cached:
        # Persist immediately: unlike an indexing run there is no later flush(),
        # and the point of the cache is to survive the process.  A write failure
        # costs a re-hash next time, never the response.
        try:
            cache.flush()
        except (sqlite3.Error, OSError) as exc:
            _log.warning("hash cache write failed for %s: %s", p, exc)
    return digest


def _cached_source_digest(p: Path, settings: Settings | None = None) -> str | None:
    """The digest of *p* if the HashCache already holds a current one, else None.

    Never hashes.  For handlers that must not pay a whole-file read — None is
    "not computed", not a verdict about the file.
    """
    cache = _hash_cache_for(settings if settings is not None else Settings())
    if cache is None:
        return None
    try:
        return cache.peek(p)
    except (sqlite3.Error, OSError) as exc:  # pragma: no cover - defensive
        _log.debug("hash cache peek failed for %s: %s", p, exc)
        return None


def _stale_evidence_report(computed: str, indexed: str | None) -> dict:
    """Compare the digest computed now with the one the index recorded.

    ``indexed`` is optional — a caller that does not know the indexed hash gets
    ``stale_evidence: None``, which is "not checked", not "checked and fine".
    Only a real comparison can clear a file.
    """
    if not indexed or not re.fullmatch(r"[0-9a-f]{64}", indexed):
        return {"indexed_video_hash": None, "stale_evidence": None, "stale_reason": None}
    if indexed == computed:
        return {"indexed_video_hash": indexed, "stale_evidence": False, "stale_reason": None}
    return {
        "indexed_video_hash": indexed,
        "stale_evidence": True,
        "stale_reason": (
            "The file on disk no longer matches the file that was indexed: "
            f"indexed SHA-256 {indexed[:12]}…, file on disk now {computed[:12]}…. "
            "Frame hits, timecodes and thumbnails describe the indexed file, "
            "not what plays here. Re-index before relying on this playback."
        ),
    }


def _stream_report(p: Path) -> dict:
    """Container/codec summary used by the playback label and the rewrap filter.

    ``skipped_streams`` names the streams a rewrap would have to leave behind —
    Apple's Live-Photo .MOV files carry LPCM audio, which has no MP4 mapping and
    cannot be re-encoded here, so the viewing copy of one is silent.  The label
    says so rather than letting the operator infer a silent original.
    """
    info: dict = {
        "container_brand": _ftyp_brand(p),
        "video_codec": None,
        "audio_codec": None,
        "video_pix_fmt": None,
        "video_profile": None,
    }
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
                    # Read from the container's codec parameters — no frame is
                    # decoded, which is what keeps opening a hit cheap (§5).
                    info["video_pix_fmt"] = s.codec_context.pix_fmt
                    info["video_profile"] = s.codec_context.profile
                elif s.type == "audio" and info["audio_codec"] is None:
                    info["audio_codec"] = s.codec_context.name
                if s.type in ("video", "audio") and s.codec_context.name not in _MP4_LEGAL_CODECS:
                    skipped.append(f"{s.type}:{s.codec_context.name}")
    except (av.FFmpegError, OSError) as exc:
        _log.debug("playback probe failed for %s: %s", p, exc)
        info["probe_error"] = str(exc)
    info["skipped_streams"] = skipped
    return info


# ---------------------------------------------------------------------------
# Codec allowlist (spec §5, §15.3, ruling §16)
# ---------------------------------------------------------------------------
#
# What the browser can decode is decided HERE, from the stream, and never from
# what the browser advertises: the operator's Chrome reports HEVC support and
# then fails to decode an iPhone HEVC file (ruling 2026-08-13).  The allowlist is
# therefore deliberately narrow — a codec is playable only in the pixel formats
# every target browser actually decodes.
#
# Phase 3 detects and reports.  It never encodes: a stream outside the allowlist
# is named as needing a transcode, and the analyst is pointed at Download
# original until the encode path lands (phase 4).

# codec name → maximum bit depth the browsers decode.  4:2:0 chroma is required
# for all of them; 4:2:2 and 4:4:4 have no browser decoder at any depth.
_PLAYABLE_CODECS: dict[str, int] = {
    "h264": 8,  # 8-bit 4:2:0 only — High 10 and 4:4:4 are not decodable
    "vp8": 8,
    "vp9": 10,  # profile 0 and profile 2
    "av1": 10,  # Main profile, 8- and 10-bit
}

# Display names for the reason string.  A label an analyst reads must name the
# codec the way the world names it, not the way libavcodec does.
_CODEC_DISPLAY_NAMES = {
    "h264": "H.264",
    "hevc": "HEVC",
    "vp8": "VP8",
    "vp9": "VP9",
    "av1": "AV1",
    "mpeg4": "MPEG-4 Part 2",
    "mpeg2video": "MPEG-2",
    "mjpeg": "Motion JPEG",
    "prores": "Apple ProRes",
    "dnxhd": "DNxHD",
    "vc1": "VC-1",
    "theora": "Theora",
}


def _codec_label(codec: str | None, bits: int | None, chroma: str | None) -> str:
    """Human name for a stream: ``"HEVC 10-bit"``, ``"H.264 4:4:4"``."""
    name = _CODEC_DISPLAY_NAMES.get(codec or "", codec or "unknown codec")
    parts = [name]
    if bits is not None and bits != 8:
        parts.append(f"{bits}-bit")
    if chroma is not None and chroma != "420":
        parts.append(f"{chroma[0]}:{chroma[1]}:{chroma[2]}")
    return " ".join(parts)


def _pixel_profile(pix_fmt: str | None, profile: str | None) -> tuple[int | None, str | None]:
    """Return ``(bit depth, chroma)`` for a stream, without decoding a frame.

    ``pix_fmt`` is what libavformat parsed out of the container's codec
    parameters, which is enough for every codec in the allowlist.  Where it is
    absent the profile name still settles the question for the two cases that
    matter (``"Main 10"``, ``"High 4:4:4 Predictive"``).  ``(None, None)`` means
    undetermined — reported as such, never guessed.
    """
    if pix_fmt:
        try:
            fmt = av.VideoFormat(pix_fmt)
            bits = max(c.bits for c in fmt.components if c.bits)
        except (ValueError, AttributeError):  # pragma: no cover - unknown format name
            bits = None
        for chroma in ("444", "422", "420", "411", "410"):
            if chroma in pix_fmt:
                return bits, chroma
        # Formats with no subsampling digits (gray, rgb…) are not 4:2:0 and no
        # browser decodes them in these codecs.
        return bits, None
    if profile:
        low = profile.lower()
        bits = 12 if "12" in low else 10 if "10" in low else None
        chroma = "444" if "4:4:4" in low else "422" if "4:2:2" in low else None
        if bits is not None or chroma is not None:
            return bits, chroma
    return None, None


def _decode_verdict(info: dict) -> tuple[bool | None, str]:
    """Can a browser decode this video stream?  ``(verdict, human reason)``.

    ``None`` is "cannot tell" — a stream that could not be probed, or a pixel
    format libavformat did not report.  It is a third answer on purpose: guessing
    "playable" hides a failure, guessing "transcode" claims work is needed on no
    evidence, and §5 forbids inventing a state that cannot be observed.
    """
    if info.get("probe_error"):
        return None, f"The container could not be probed ({info['probe_error']})."
    codec = info.get("video_codec")
    if not codec:
        return None, "No video stream was found in this container."
    bits, chroma = _pixel_profile(info.get("video_pix_fmt"), info.get("video_profile"))
    label = _codec_label(codec, bits, chroma)
    max_bits = _PLAYABLE_CODECS.get(codec)
    if max_bits is None:
        return False, f"{label}: no browser decoder for this codec."
    if bits is None and chroma is None:
        return None, f"{label}: the pixel format could not be read from the container."
    if chroma is not None and chroma != "420":
        return False, f"{label}: browsers decode 4:2:0 chroma only."
    if bits is not None and bits > max_bits:
        return False, f"{label}: this browser cannot decode it."
    return True, f"{label}: decodes natively."


def _playback_mode(info: dict, needs_remux: bool) -> tuple[str, str]:
    """Decide ``mode`` and the reason shown to the analyst.

    Codec first, container second: a rewrap moves the same bitstream, so it
    cannot rescue a stream the browser has no decoder for.
    """
    decodable, reason = _decode_verdict(info)
    if decodable is None:
        return "unknown", f"{reason} Playback cannot be judged from here — download the original."
    if decodable is False:
        return "transcode", (
            f"{reason} A transcoded viewing copy is required; "
            "encoding is not available yet — download the original to view it."
        )
    if needs_remux:
        return "rewrap", (
            f"{reason} The QuickTime container is not one browsers open, so the same "
            "packets are rewrapped into MP4 — no re-encode."
        )
    return "original", f"{reason} The file is streamed as it is stored on disk."


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
                out_stream = out.add_stream_from_template(s)
                # Carry the source's four-character codec tag across.  Left to
                # itself the muxer relabels Apple's hvc1 as hev1; both decode in
                # Chrome, but hvc1 is what the source says and the only one
                # QuickTime and Safari will open.
                if s.codec_tag:
                    out_stream.codec_tag = s.codec_tag
                mapping[s.index] = out_stream
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

    Scope is deliberately narrow: only top-level ``{sha256}.mp4`` rewraps, the
    artifacts this function created.  A bare ``*.mp4`` glob would match — and
    delete — the ``full.mp4`` and chunk files later phases put in the same store,
    including one mid-play.  The whole-tree accounting, per-video eviction and
    playback leases that store needs are spec §6.2, phase 5; this is containment
    until then, not that rewrite.
    """
    if max_bytes <= 0:
        return 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for f in cache_dir.glob("*.mp4"):
        if not re.fullmatch(r"[0-9a-f]{64}", f.stem):
            continue
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
    digest = await asyncio.to_thread(_source_digest, p, settings)
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


@router.get("/api/video-download")
async def video_download(path: str) -> FileResponse:
    """Serve the untouched source file as a download.

    The escape route every playback failure points at: when the browser cannot
    decode a stream, the analyst can still take the original bytes and open them
    in a tool that can.  No viewing copy, no rewrap, no cache — the file as it
    lies on disk (spec §7.5).

    The same resolution flow as every other path-bearing route: absolute path,
    ``resolve()``, extension check, allowed-root containment, regular file.  A
    cache key is never accepted as the identity of a source file (spec §9).

    ``X-SFN-Source-SHA256`` carries the digest **only when the HashCache already
    holds it**.  This is the escape route a failing playback points at, so it is
    the worst place to stall: hashing a cold multi-GB source would read the whole
    file before the first byte moved, and then read it again to serve.  An absent
    header means "not computed here", never "unverified" — ``playback-info`` has
    already computed and displayed the verified digest by the time the analyst
    clicks, and re-hashing the downloaded file reproduces it.

    §1 tension, stated rather than hidden: this places a copy of evidence on the
    analyst's machine.  It is a deliberate act and is audited like any other; the
    deployment's handling policy governs what happens to that copy.
    """
    p = _resolve_video_path(path)
    settings = Settings()
    digest = await asyncio.to_thread(_cached_source_digest, p, settings)
    headers = {"X-SFN-Source-SHA256": digest} if digest else {}
    _log.info("video-download: serving %s (sha256=%s)", p, digest or "not cached")
    return FileResponse(
        p,
        media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream",
        filename=p.name,  # Content-Disposition: attachment, original filename
        headers=headers,
    )


@router.get("/api/video-playback-info")
async def video_playback_info(path: str, video_hash: str | None = None) -> JSONResponse:
    """Describe what playback of *path* would serve, without serving it.

    Feeds the viewing-copy label in the UI: how the file reaches the player,
    which codecs it carries, and the SHA-256 of the source *as it is on disk
    now* — computed, never read back from the index.

    When the caller passes the ``video_hash`` the index recorded for this file,
    the two are compared.  A difference is a **stale-evidence condition**: the
    file on disk is no longer the one that was indexed, so the frame hits, the
    timeline and any label drawn from them describe a different file.  It is
    reported explicitly (``stale_evidence``) rather than served in silence.

    ``mode`` is decided here, server-side, from the stream itself: ``original``,
    ``rewrap``, ``transcode`` (a codec outside the allowlist — detected in this
    phase, encoded in a later one) or ``unknown`` when the container cannot be
    probed.  ``mode_reason`` is the sentence the analyst reads.
    """
    p = _resolve_video_path(path)
    settings = Settings()
    st = p.stat()
    info: dict = {
        "source_path": str(p),
        "filename": p.name,
        "source_size_bytes": st.st_size,
        # Both URLs escape the path the same way.  A filename carrying '#', '?',
        # '&' or a space — routine in iPhone media — silently truncates an
        # unescaped query string.
        "playback_url": f"/api/video-playback?path={quote(str(p))}",
        "download_url": f"/api/video-download?path={quote(str(p))}",
    }
    info.update(await asyncio.to_thread(_stream_report, p))
    info["mode"], info["mode_reason"] = _playback_mode(info, _needs_remux(p))
    info["video_sha256"] = await asyncio.to_thread(_source_digest, p, settings)
    if info["mode"] == "original":
        # Nothing is left behind when the file is served as it lies on disk.
        info["skipped_streams"] = []
    elif info["mode"] == "rewrap":
        cache_dir = settings.video_cache_dir
        info["cached"] = (
            cache_dir is not None and (cache_dir / f"{info['video_sha256']}.mp4").exists()
        )
        info["cache_enabled"] = cache_dir is not None
    info.update(_stale_evidence_report(info["video_sha256"], video_hash))
    return JSONResponse(info)
