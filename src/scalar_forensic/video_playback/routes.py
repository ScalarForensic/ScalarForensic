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
import contextlib
import logging
import mimetypes
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import av
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from scalar_forensic.config import Settings
from scalar_forensic.video_playback import states
from scalar_forensic.video_playback.cache import (
    _cache_dir_or_503,
    _touch,
    artifact_dir,
    artifact_locks,
    check_ceiling,
    chunk_name,
    evict,
    pin,
    release_lease,
    renew_lease,
    rewrap_path,
)
from scalar_forensic.video_playback.capability import Pipeline, capability, is_hdr, select
from scalar_forensic.video_playback.codecs import _needs_remux, _playback_mode, _stream_report
from scalar_forensic.video_playback.digest import _cached_source_digest, _source_digest
from scalar_forensic.video_playback.encode import encode_chunk
from scalar_forensic.video_playback.rewrap import _remux_to_mp4
from scalar_forensic.web.routes._shared import _resolve_video_path

_log = logging.getLogger(__name__)

router = APIRouter()

_HEX64 = re.compile(r"[0-9a-f]{64}")


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
        info["cache_enabled"] = settings.video_cache_dir is not None
        info["chunk_seconds"] = settings.video_chunk_seconds
    # The §5 state is decided here, server-side, from the same `mode` the label
    # is drawn from — so "what does the analyst see" has one implementation and
    # cannot drift from "what did the server conclude".  Four modes map onto four
    # states, `unknown` included: a container that would not open has said
    # nothing about whether it plays, and reporting that as `needs-transcode`
    # would be a claim about a stream nobody read (`#147`'s defect class).
    info["player_state"] = states.MODE_TO_STATE[info["mode"]]
    if info["player_state"] == "needs-transcode" and not info.get("cache_enabled"):
        # There is a transcode to do and nowhere to put it (§5 `cache-disabled`).
        info["player_state"] = "cache-disabled"
        info["player_state_reason"] = states.CACHE_UNSET.reason
    info.update(_stale_evidence_report(info["video_sha256"], video_hash))
    return JSONResponse(info)


# ---------------------------------------------------------------------------
# Chunk playback (spec §4.2, §5, §9, §10.1)
# ---------------------------------------------------------------------------


class _Admission:
    """The bound on concurrent chunk encodes, and on how many may wait (§10.4).

    Two numbers, not one.  ``SFN_VIDEO_MAX_WORKERS`` caps how many encoders run
    — §3.5 measured aggregate throughput flat from k=1 to k=8, so more is only
    latency.  ``SFN_VIDEO_QUEUE_MAX`` caps *admitted* requests, running plus
    waiting, because an unbounded wait queue is the same unbounded resource with
    a slower fuse: a LAN host that opens forty videos would otherwise hold forty
    requests until each one times out.  Over the cap the request is refused with
    §10.1's ``queue-full`` row, which says how long to wait, rather than joining
    a line nobody is told the length of.

    Deliberately not a job runner.  Phase 7's ``jobs.py`` owns the worker pool,
    cancellation and refcounts; this is the admission gate the synchronous chunk
    path needs in the meantime, and it is replaced wholesale rather than grown.
    """

    def __init__(self) -> None:
        self._sem: asyncio.Semaphore | None = None
        self._limit = 0
        self.admitted = 0

    def _semaphore(self, workers: int) -> asyncio.Semaphore:
        if self._sem is None or self._limit != workers:
            self._sem = asyncio.Semaphore(workers)
            self._limit = workers
        return self._sem

    @asynccontextmanager
    async def enter(self, settings: Settings) -> AsyncIterator[None]:
        if self.admitted >= settings.video_queue_max:
            raise states.QUEUE_FULL.as_http()
        self.admitted += 1
        try:
            async with self._semaphore(settings.video_max_workers):
                yield
        finally:
            self.admitted -= 1

    def reset(self) -> None:
        """Test hook: the counter is process-wide, like every other bound here."""
        self._sem = None
        self._limit = 0
        self.admitted = 0


admission = _Admission()


def chunk_start_for(t: float, chunk_seconds: int) -> float:
    """The start of the chunk containing timecode *t*.

    Snapping on the server is what makes a seek idempotent: two analysts who
    scrub to 41.2 s and 47.9 s of a 30 s-chunked video both get the chunk at
    30.000 and share one encode, instead of producing two nearly identical
    artifacts under two keys.
    """
    return float(int(t // chunk_seconds) * chunk_seconds)


@dataclass(frozen=True)
class _CachedChunk:
    path: Path
    fingerprint: str
    describe: dict


# Which pipeline last *actually produced* a chunk when a given pipeline was
# selected.  Normally the same fingerprint on both sides; they differ only after
# a §8 GPU fallback.
#
# Without this, a host whose GPU probes clean and then fails at job time never
# hits its own cache: `select()` keeps returning the GPU pipeline, the lookup
# keeps missing the CPU key the artifact actually landed under, and every chunk
# is re-encoded forever — the cache silently stops being one.  (Found by
# `test_the_second_request_is_a_cache_hit_and_does_not_re_encode` on a host
# where the probe passes and `h264_nvenc` then writes no packets.)
#
# It is a **lookup hint and nothing else**: selection is unchanged, so a GPU that
# recovers encodes on the GPU at the next genuine miss, and what is reported is
# always the pipeline that made the file being served (§7.2) — which is why the
# table holds whole `Pipeline` objects and not just their fingerprints. A label
# carrying the GPU's fields under the CPU's key would be the §7.2 defect in
# miniature. One entry per selected pipeline — a handful per process, not per
# video.
_substitutions: dict[str, Pipeline] = {}


def _cached_chunk(
    cache_dir: Path, digest: str, selected: Pipeline, name: str
) -> _CachedChunk | None:
    """The chunk on disk for this window, under the selected key or its substitute."""
    fp = selected.fingerprint()
    direct = artifact_dir(cache_dir, digest, fp) / name
    if direct.is_file():
        return _CachedChunk(direct, fp, selected.describe())
    alt = _substitutions.get(fp)
    if alt is not None and alt.fingerprint() != fp:
        candidate = artifact_dir(cache_dir, digest, alt.fingerprint()) / name
        if candidate.is_file():
            return _CachedChunk(candidate, alt.fingerprint(), alt.describe())
    return None


def reset_substitutions() -> None:
    """Test hook — the table is process-wide, like every other bound here."""
    _substitutions.clear()


def _duration_seconds(info: dict) -> float | None:
    ms = info.get("duration_ms")
    return ms / 1000.0 if ms else None


async def _prepare_chunk(p: Path, settings: Settings, t: float, *, video_hash: str | None) -> dict:
    """Encode (or find) the chunk containing timecode *t* and describe it.

    Every §10.1 condition that can be decided without ffmpeg is decided here,
    before anything is queued or encoded (§9), and each one raises the
    :class:`~.states.Failure` row that names it — so the analyst is told which
    thing went wrong, not that "playback failed".
    """
    info = await asyncio.to_thread(_stream_report, p)
    if "probe_error" in info:
        raise states.PROBE_FAILED.as_http()
    if info.get("video_codec") is None:
        raise states.NO_VIDEO_TRACK.as_http()

    mode, mode_reason = _playback_mode(info, _needs_remux(p))
    if mode != "transcode":
        raise states.NOT_A_TRANSCODE.as_http()

    duration = _duration_seconds(info)
    if duration is None or duration <= 0:
        raise states.BAD_DURATION.as_http()
    if not (0 <= t < duration):
        raise states.TIMECODE_OUT_OF_RANGE.as_http()

    cache_dir = _cache_dir_or_503(settings)
    digest = await asyncio.to_thread(_source_digest, p, settings)
    if video_hash and _HEX64.fullmatch(video_hash) and video_hash != digest:
        # §7.1: the file changed under the session.  Encoding a chunk from it
        # would hand the analyst pixels that no timecode in the UI describes.
        raise states.SOURCE_CHANGED.as_http()

    start = chunk_start_for(t, settings.video_chunk_seconds)
    hdr = is_hdr(info)
    cap = await asyncio.to_thread(capability, settings)
    try:
        selected = select(settings, cap, hdr=hdr)
    except RuntimeError as exc:
        raise states.classify(exc).as_http() from exc

    # The lease goes in before the work: a `FileResponse` streams after the
    # handler returns, so nothing else can tell eviction this video is on
    # screen (§6.2).  The player refreshes it on a heartbeat from here.
    renew_lease(digest, settings.video_lease_seconds)

    report: dict = {
        "video_sha256": digest,
        "chunk_start": start,
        "chunk_seconds": settings.video_chunk_seconds,
        "duration_ms": info["duration_ms"],
        "mode_reason": mode_reason,
        "hdr": hdr,
    }
    next_start = start + settings.video_chunk_seconds
    report["next_chunk_start"] = next_start if next_start < duration else None
    report["final_chunk"] = report["next_chunk_start"] is None

    name = chunk_name(start)
    hit = _cached_chunk(cache_dir, digest, selected, name)
    if hit is not None:
        await asyncio.to_thread(_touch, hit.path)
        report.update(
            cached=True,
            pipeline_fingerprint=hit.fingerprint,
            pipeline=hit.describe,
            fell_back=hit.fingerprint != selected.fingerprint(),
            fallback_reason=None,
            encode_seconds=None,
        )
        return report

    dst = artifact_dir(cache_dir, digest, selected.fingerprint()) / name
    async with admission.enter(settings):
        # Dedup on the artifact, not on the video: two analysts on the same
        # chunk share one encode, two analysts on different chunks of the same
        # video do not queue behind each other (§10.4).
        async with artifact_locks.hold(f"{digest}:{selected.fingerprint()}:{name}"):
            again = _cached_chunk(cache_dir, digest, selected, name)
            if again is not None:  # published while we waited for the lock
                await asyncio.to_thread(_touch, again.path)
                report.update(
                    cached=True,
                    pipeline_fingerprint=again.fingerprint,
                    pipeline=again.describe,
                    fell_back=again.fingerprint != selected.fingerprint(),
                    fallback_reason=None,
                    encode_seconds=None,
                )
                return report
            with pin(digest):
                try:
                    result = await asyncio.to_thread(
                        encode_chunk,
                        settings,
                        cap,
                        p,
                        dst,
                        hdr=hdr,
                        start=start,
                        has_audio=info.get("audio_codec") is not None,
                    )
                except Exception as exc:
                    failure = states.classify(exc)
                    _log.warning("chunk %s of %s failed (%s): %s", start, p, failure.kind, exc)
                    raise failure.as_http() from exc
                published = _relocate_on_fallback(
                    result.path, cache_dir, digest, result.pipeline.fingerprint(), name
                )
                _substitutions[selected.fingerprint()] = result.pipeline
            await asyncio.to_thread(evict, cache_dir, settings.video_cache_max_bytes)

    report.update(
        cached=False,
        pipeline_fingerprint=result.pipeline.fingerprint(),
        pipeline=result.pipeline.describe(),
        fell_back=result.fell_back,
        fallback_reason=result.fallback_reason,
        encode_seconds=round(result.wall_seconds, 3),
        artifact_path=str(published),
    )
    return report


def _relocate_on_fallback(
    published: Path, cache_dir: Path, digest: str, fingerprint: str, name: str
) -> Path:
    """Move a chunk under the key of the pipeline that **ran** (§6.1).

    The destination has to be chosen before the encode, and a §8 GPU fallback
    changes the pipeline mid-encode — so a chunk can land under the key of a
    pipeline that did not produce it, which is precisely "one key holding two
    pictures".  The move is a same-directory ``os.replace``, so it is atomic and
    a reader either sees the old name or the new one.
    """
    correct = artifact_dir(cache_dir, digest, fingerprint) / name
    if correct == published:
        return published
    correct.parent.mkdir(parents=True, exist_ok=True)
    os.replace(published, correct)
    with contextlib.suppress(OSError):
        published.parent.rmdir()  # empty unless something else published there
    return correct


@router.post("/api/video-chunk")
async def video_chunk_prepare(
    path: str, t: float = 0.0, video_hash: str | None = None
) -> JSONResponse:
    """Produce the chunk containing timecode *t* and say where to fetch it (§4.2).

    **Preparing and serving are two requests on purpose.**  A ``<video>`` element
    issues a ``GET`` with ``Range`` headers and nothing else; it cannot POST, and
    it cannot wait ten seconds for an encode without the browser's own media
    stack deciding the source is broken.  So the POST does the work and returns
    JSON, and the GET below serves bytes out of the cache and never encodes.
    That also keeps §9's rule intact — the GET is idempotent and cacheable, and
    a request that changes the machine's state is not disguised as a fetch.

    ``player_state`` is the §5 state this response puts the player in.  It is
    stated by the server rather than inferred by the client, so the failure
    matrix (§10.1) has exactly one implementation.
    """
    p = _resolve_video_path(path)
    settings = Settings()
    report = await _prepare_chunk(p, settings, t, video_hash=video_hash)
    report["player_state"] = "chunk-ready"
    report["chunk_url"] = (
        f"/api/video-chunk?path={quote(str(p))}"
        f"&start={report['chunk_start']:.3f}&fp={report['pipeline_fingerprint']}"
    )
    return JSONResponse(report)


@router.get("/api/video-chunk")
async def video_chunk_get(path: str, start: float, fp: str) -> FileResponse:
    """Serve an already-encoded chunk.  Never encodes; 404 when it is not there.

    ``fp`` selects *which rendering* of this video to serve — a video encoded on
    two hosts has two pipelines and two pictures (§6.1) — and it is never the
    identity of the file: ``path`` is, through the same resolution flow as every
    other route (§9).  A cache key that named a file on its own would let a
    caller read anything the store happens to hold.
    """
    p = _resolve_video_path(path)
    if not _HEX64.fullmatch(fp or ""):
        raise HTTPException(status_code=422, detail="fp is not a pipeline fingerprint")
    settings = Settings()
    cache_dir = _cache_dir_or_503(settings)
    digest = await asyncio.to_thread(_source_digest, p, settings)
    chunk = artifact_dir(cache_dir, digest, fp) / chunk_name(start)
    if not chunk.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "That chunk is not in the cache. It was never encoded, or it was "
                "evicted; POST to this endpoint to produce it."
            ),
        )
    renew_lease(digest, settings.video_lease_seconds)
    await asyncio.to_thread(_touch, chunk)
    return FileResponse(
        chunk,
        media_type="video/mp4",
        headers={
            "X-SFN-Playback-Mode": "transcode",
            "X-SFN-Chunk-Start": f"{start:.3f}",
        },
    )
