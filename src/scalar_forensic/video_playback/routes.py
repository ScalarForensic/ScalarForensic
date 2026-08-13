"""Playback routes: viewing copies, the original download, and the label.

Browsers refuse the QuickTime container, not the bitstreams inside it: the
operator's Chrome decodes HEVC happily once the same coded frames arrive in an
MP4 box structure.  So playback never transcodes.  A source whose container the
browser cannot open is *rewrapped* — every video and audio packet copied byte
for byte into an MP4 with the moov atom up front — and the rewrap is parked in
a bounded cache keyed by the SHA-256 of the source file.

The rewrapped file is a VIEWING COPY.  It is not evidence: the original file
and the frame JPEGs written during indexing remain the authoritative artifacts,
and every UI surface that offers playback says so.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
from pathlib import Path
from urllib.parse import quote

import av
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from scalar_forensic.config import Settings
from scalar_forensic.video_playback.cache import (
    _cache_dir_or_503,
    _touch,
    artifact_locks,
    check_ceiling,
    evict,
    pin,
    release_lease,
    renew_lease,
    rewrap_path,
)
from scalar_forensic.video_playback.codecs import _needs_remux, _playback_mode, _stream_report
from scalar_forensic.video_playback.digest import _cached_source_digest, _source_digest
from scalar_forensic.video_playback.rewrap import _remux_to_mp4
from scalar_forensic.web.routes._shared import _resolve_video_path

_log = logging.getLogger(__name__)

router = APIRouter()


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
    dst = rewrap_path(cache_dir, digest)

    # Register the lease *before* the work: a `FileResponse` streams its body
    # after this handler has returned, so nothing else can tell eviction that
    # this video is being read (§6.2).  The player refreshes it from there.
    renew_lease(digest, settings.video_lease_seconds)
    async with artifact_locks.hold(digest):
        if dst.exists():
            report["cached"] = True
            await asyncio.to_thread(_touch, dst)
            return dst, report
        report["cached"] = False
        # The pin covers the write itself: the lease can expire under a rewrap
        # slower than its ttl, and the .part lives inside the directory LRU
        # would then remove.
        with pin(digest):
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                report.update(await asyncio.to_thread(_remux_to_mp4, p, dst))
            except (av.FFmpegError, ValueError, OSError) as exc:
                _log.warning("viewing copy: rewrap failed for %s: %s", p, exc)
                raise HTTPException(
                    status_code=422,
                    detail=f"Cannot rewrap this container for playback: {exc}",
                ) from exc
            await asyncio.to_thread(evict, cache_dir, settings.video_cache_max_bytes)
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


@router.post("/api/video-lease")
async def video_lease(path: str, release: bool = False) -> JSONResponse:
    """Register, refresh or drop the playback lease on a video (spec §6.2).

    HTTP is stateless: between two of a video's own chunk requests the server
    cannot otherwise tell that the video is still on screen, and eviction that
    guesses will drop a video mid-play.  So the player says so explicitly and
    keeps saying so — one call when playback starts, one per heartbeat, one with
    ``release=true`` when the analyst closes the video.

    ``state`` is three-valued and stays that way: ``held``, ``expired`` (the
    heartbeat stopped) and ``none`` (never registered in this process).  The
    third is not a synonym for the second — a fresh worker process says ``none``
    about a video another worker is serving, and a boolean would report that as
    "not being watched".

    The lease is advisory protection for the *cache*, never an access control:
    it decides what eviction may delete, and nothing about who may read what.
    """
    p = _resolve_video_path(path)
    settings = Settings()
    digest = await asyncio.to_thread(_source_digest, p, settings)
    state = release_lease(digest) if release else renew_lease(digest, settings.video_lease_seconds)
    return JSONResponse(
        {
            "video_sha256": digest,
            "state": state.state,
            "seconds_remaining": state.seconds_remaining,
            "lease_seconds": settings.video_lease_seconds,
        }
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
            cache_dir is not None and rewrap_path(cache_dir, info["video_sha256"]).exists()
        )
        info["cache_enabled"] = cache_dir is not None
    elif info["mode"] == "transcode":
        # §6.3: the estimate is shown whether or not it refuses, so the analyst
        # reads a number rather than only a verdict.  Three-valued, because
        # "this file would not say how big it is" is not "this video is too big".
        verdict = check_ceiling(settings, info)
        info["full_copy"] = {
            "state": verdict.state,
            "estimate_bytes": verdict.estimate_bytes,
            "limit_bytes": verdict.limit_bytes,
            "reason": verdict.reason,
        }
    info.update(_stale_evidence_report(info["video_sha256"], video_hash))
    return JSONResponse(info)
