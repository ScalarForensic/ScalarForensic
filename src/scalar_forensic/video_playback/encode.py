"""The ffmpeg re-encode: one command, different ``-ss``/``-t`` (spec §4, §10).

A chunk and a full copy are the *same* encode with a different window, which is
why there is one command builder here and not two.  Everything that decides what
the pixels look like comes in as a :class:`~.capability.Pipeline`; this module
decides only *where* in the source to point it and how to publish the result.

Not a rewrap.  :mod:`.rewrap` is the PyAV stream copy that moves packets
untouched; nothing here is lossless and every output carries the §7.4
disclosure.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scalar_forensic.config import Settings
from scalar_forensic.video_playback.cache import part_path, publish
from scalar_forensic.video_playback.capability import (
    TONEMAP_CHAIN,
    Capability,
    Pipeline,
    select,
)

_log = logging.getLogger(__name__)


class CeilingExceeded(RuntimeError):
    """The growing ``.part`` passed the size a single rendering may occupy (§6.3).

    Raised by the full-video job's own watcher, never by ffmpeg.  §6.3's estimate
    applies **no codec factor** because none is measured, and its error runs *low*
    on exactly the 10-bit HEVC corpus this feature exists for — so the admission
    check is a screen and this is the guarantee behind it.  The job is stopped and
    its ``.part`` removed rather than allowed to fill the cache it was admitted
    against.
    """

    def __init__(self, message: str, *, written_bytes: int, limit_bytes: int) -> None:
        super().__init__(message)
        self.written_bytes = written_bytes
        self.limit_bytes = limit_bytes


class Cancelled(RuntimeError):
    """The job was cancelled by an analyst (§4.3).  Not a failure — a decision."""


class EncodeError(RuntimeError):
    """An encode that did not produce an output file.  Carries ffmpeg's own words.

    ``returncode`` and ``timed_out`` exist so :mod:`.states` can tell §10.1's
    rows apart without parsing the message: a **negative** returncode is a
    signal, and ``-9`` with no stderr is what the OOM killer leaves behind —
    "capacity exhausted, retry later", not "this file cannot be encoded, stop
    asking". Reading that distinction out of a prose message would break the
    first time ffmpeg reworded itself.
    """

    def __init__(
        self,
        message: str,
        *,
        command: list[str],
        stderr: str = "",
        returncode: int | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out


@dataclass(frozen=True)
class EncodeResult:
    """What ran, what it produced, and how long it took.

    ``pipeline`` is the one that *actually* ran — on a §8 GPU fallback it is not
    the one the caller asked for, and since the fingerprint is half the cache key
    (§6.1) the caller must re-key on this value rather than on its request.
    ``command`` is the full argv, so `sfn-video render` can print the exact
    invocation that produced a rendering (§7.2).
    """

    path: Path
    pipeline: Pipeline
    command: list[str]
    wall_seconds: float
    fell_back: bool
    fallback_reason: str | None


def build_command(
    settings: Settings,
    pipeline: Pipeline,
    src: Path,
    dst: Path,
    *,
    start: float | None = None,
    duration: float | None = None,
    has_audio: bool = True,
    threads: int | None = None,
    progress: bool = False,
) -> list[str]:
    """The argv for one encode.  Pure — builds, never runs.

    ``-ss`` goes **before** ``-i``: input seeking uses the container index, which
    §3.2 and §3.5 measured flat out to 3 h 41 m on a dense index.  Output seeking
    would decode and discard everything before the window, making chunk cost
    proportional to offset — the property §1 rejects the whole design over.

    ``threads`` and ``progress`` are the full-video job's two additions (§4.3):
    the thread cap is half of remedy (a) — niceness is applied per *thread*, so
    an export that runs one encoding thread per core still competes on every core
    — and ``-progress pipe:1`` is where the progress and ETA come from.  Both are
    off by default, so a chunk's argv is unchanged by this phase.
    """
    cmd = [settings.ffmpeg_path, "-nostdin", "-hide_banner", "-v", "error"]
    if progress:
        cmd += ["-nostats", "-progress", "pipe:1"]
    if threads is not None:
        cmd += ["-threads", str(threads)]
    if start is not None and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", pipeline.filter_chain, "-c:v", pipeline.encoder, *pipeline.rate_control]
    if TONEMAP_CHAIN in pipeline.filter_chain:
        # The tone-map chain already converts to BT.709; tagging says so in the
        # container as well, which is the half §3.1's naive path got wrong —
        # 8-bit pixels under an HDR label render washed out with lifted blacks.
        # Only on the tone-mapped path: stamping bt709 onto an untouched SDR
        # source would mislabel a BT.601 one.
        cmd += [
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
        ]
    cmd += list(pipeline.audio) if has_audio else ["-an"]
    # faststart puts the index first so the browser can play before the whole
    # file has arrived; +genpts because a window cut out of an arbitrary offset
    # does not necessarily start on a clean presentation stamp.
    cmd += ["-movflags", "+faststart", "-fflags", "+genpts", "-f", "mp4", "-y", str(dst)]
    return cmd


def _run(cmd: list[str], timeout: int) -> str:
    """Run ffmpeg to completion; return stderr.  Raises :class:`EncodeError`.

    On timeout the process is killed and reaped rather than left behind (§10.3):
    an orphaned encoder holds a core and the source file open, and §3.5 says one
    encoder is enough to saturate this box.
    """
    proc = subprocess.Popen(  # noqa: S603 - operator-configured binary, built argv
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise EncodeError(
            f"encode timed out after {timeout}s", command=cmd, timed_out=True
        ) from None
    if proc.returncode != 0:
        detail = (stderr or "").strip().splitlines()
        raise EncodeError(
            detail[-1] if detail else f"ffmpeg exited {proc.returncode}",
            command=cmd,
            stderr=stderr or "",
            returncode=proc.returncode,
        )
    return stderr or ""


@dataclass(frozen=True)
class Progress:
    """One ``-progress`` block from a running encode.

    ``out_seconds`` is read from ``out_time=HH:MM:SS.ffffff`` and not from
    ``out_time_ms``, which ffmpeg has emitted in *microseconds* under a
    millisecond name for years: a unit bug here would show a 51-minute export as
    finishing in three seconds, and §4.3's ETA is meant to be read.
    """

    frames: int
    out_seconds: float
    written_bytes: int


def _lower_priority(pid: int, nice: int) -> None:
    """Renice the encoder's whole process group (§4.3 remedy (a)).

    ``PRIO_PGRP`` and not ``PRIO_PROCESS``: on Linux niceness is a **per-thread**
    attribute, so reniceing the pid would leave every encoding thread ffmpeg
    spawns at the parent's priority — the group form reaches all of them, which
    is why the child is started in its own session.  A host that refuses the call
    is logged and the job runs anyway: an export at the wrong priority is worse
    than an export, not worse than nothing.
    """
    if nice <= 0:
        return
    try:
        os.setpriority(os.PRIO_PGRP, pid, nice)
    except OSError as exc:  # pragma: no cover - platform/permission dependent
        _log.warning("could not renice the full-video job to %s: %s", nice, exc)


def _run_watched(
    cmd: list[str],
    timeout: int,
    *,
    nice: int = 0,
    on_progress: Callable[[Progress], None] | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
    part: Path | None = None,
    limit_bytes: int | None = None,
) -> str:
    """Run ffmpeg with ``-progress`` on stdout, watching size and the clock.

    Three things this does that :func:`_run` does not, all of them §4.3's or
    §6.3's: it reports progress while the encode runs, it hands the ``Popen`` to
    the caller so a cancel can stop it, and it **watches the growing ``.part``
    against the size a single rendering may occupy** — the check §6.3 requires
    because its admission estimate carries no codec factor and errs low.

    The process gets its own session so a kill reaches ffmpeg's children as well
    (§10.3), and stderr is drained by a thread and bounded, so a chatty encoder
    can neither deadlock the pipe nor grow this process without limit.
    """
    proc = subprocess.Popen(  # noqa: S603 - operator-configured binary, built argv
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    _lower_priority(proc.pid, nice)
    if on_start is not None:
        on_start(proc)

    stderr_tail: list[str] = []

    def _drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line)
            del stderr_tail[:-_STDERR_LINES]

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    deadline = time.monotonic() + timeout
    fields: dict[str, str] = {}
    overshoot: CeilingExceeded | None = None
    timed_out = False
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            fields[key] = value
            if key != "progress":
                continue
            written = part.stat().st_size if part is not None and part.exists() else 0
            if on_progress is not None:
                on_progress(
                    Progress(
                        frames=_as_int(fields.get("frame")),
                        out_seconds=_out_seconds(fields.get("out_time")),
                        written_bytes=written,
                    )
                )
            fields.clear()
            if limit_bytes is not None and written > limit_bytes:
                overshoot = CeilingExceeded(
                    f"the full copy passed {limit_bytes} bytes while encoding ({written} written)",
                    written_bytes=written,
                    limit_bytes=limit_bytes,
                )
                break
            if time.monotonic() > deadline:
                timed_out = True
                break
    finally:
        if overshoot is not None or timed_out:
            _kill_group(proc)
        proc.wait()
        drainer.join(timeout=5)
        with contextlib.suppress(OSError):
            proc.stdout.close()
        if proc.stderr is not None:
            with contextlib.suppress(OSError):
                proc.stderr.close()

    stderr = "".join(stderr_tail)
    if overshoot is not None:
        raise overshoot
    if timed_out:
        raise EncodeError(f"encode timed out after {timeout}s", command=cmd, timed_out=True)
    if proc.returncode != 0:
        detail = stderr.strip().splitlines()
        raise EncodeError(
            detail[-1] if detail else f"ffmpeg exited {proc.returncode}",
            command=cmd,
            stderr=stderr,
            returncode=proc.returncode,
        )
    return stderr


#: Kept stderr lines (§10.3, "bounded captured stderr").
_STDERR_LINES = 40


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the encoder and anything it started (§10.3)."""
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()


def _as_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _out_seconds(value: str | None) -> float:
    """``HH:MM:SS.ffffff`` → seconds.  ``N/A`` before the first frame lands."""
    if not value or value.startswith("N/A"):
        return 0.0
    parts = value.split(":")
    try:
        return sum(float(p) * 60**i for i, p in enumerate(reversed(parts)))
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class Watch:
    """What the full-video job asks of an encode that a chunk does not (§4.3, §6.3).

    Its presence is what switches :func:`encode` onto :func:`_run_watched`; a
    chunk passes ``None`` and its argv and its runner are unchanged by phase 7.
    """

    nice: int = 0
    threads: int | None = None
    limit_bytes: int | None = None
    on_progress: Callable[[Progress], None] | None = None
    on_start: Callable[[subprocess.Popen], None] | None = None


def encode(
    settings: Settings,
    cap: Capability,
    src: Path,
    dst: Path,
    *,
    hdr: bool,
    start: float | None = None,
    duration: float | None = None,
    has_audio: bool = True,
    watch: Watch | None = None,
) -> EncodeResult:
    """Encode one window of *src* to *dst*, publishing it atomically.

    Writes a sibling ``.part`` and renames on success (§10.2), so no reader ever
    observes a half-written viewing copy and a crash leaves no file that looks
    finished.  The ``.part`` carries the pid, because two processes may hold the
    same cache directory.

    A GPU failure falls back to CPU rather than failing the request (§8).  The
    fallback is recorded on the result — it is a different pipeline, so it is a
    different fingerprint and a different cache key (§6.1), and the label must
    name the encoder that actually ran (§7.2).
    """
    pipeline = select(settings, cap, hdr=hdr)
    part = part_path(dst)
    part.parent.mkdir(parents=True, exist_ok=True)
    fell_back = False
    reason: str | None = None

    def _attempt(pl: Pipeline) -> list[str]:
        """Build and run one invocation for *pl*; return the argv that ran."""
        argv = build_command(
            settings,
            pl,
            src,
            part,
            start=start,
            duration=duration,
            has_audio=has_audio,
            threads=watch.threads if watch else None,
            progress=watch is not None,
        )
        if watch is None:
            _run(argv, settings.video_job_timeout)
        else:
            _run_watched(
                argv,
                settings.video_job_timeout,
                nice=watch.nice,
                on_progress=watch.on_progress,
                on_start=watch.on_start,
                part=part,
                limit_bytes=watch.limit_bytes,
            )
        return argv

    started = time.monotonic()
    try:
        try:
            cmd = _attempt(pipeline)
        except EncodeError as exc:
            if pipeline.hwaccel == "none":
                raise
            if exc.timed_out or exc.returncode == -9:
                # Not a GPU fault to route around.  A timeout means the encode
                # did not fit in SFN_VIDEO_JOB_TIMEOUT and the CPU path is the
                # slower one (§3.1: 6.1× vs 2.7×), so the retry would spend the
                # timeout again and fail again; SIGKILL is the OOM killer, and a
                # second encoder started under memory pressure is how one refused
                # request becomes two dead ones.  Both surface as themselves
                # (§10.1) instead of being reported as a GPU fallback.
                raise
            # §8: a GPU that probed clean can still fail at job time — a driver
            # reset, or another process holding the encoder session.  Retry on
            # CPU, loudly.
            _log.warning("GPU encode failed, falling back to CPU: %s", exc)
            fell_back, reason = True, str(exc)
            cpu_cap = Capability(
                ffmpeg_path=cap.ffmpeg_path,
                ffmpeg_version=cap.ffmpeg_version,
                encoder="libx264",
                hwaccel="none",
                tonemap_ok=cap.tonemap_ok,
                notes=cap.notes,
            )
            pipeline = select(settings, cpu_cap, hdr=hdr)
            cmd = _attempt(pipeline)
        if not part.exists() or part.stat().st_size == 0:
            raise EncodeError("ffmpeg exited 0 but produced no output", command=cmd)
        publish(part, dst)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    return EncodeResult(
        path=dst,
        pipeline=pipeline,
        command=cmd,
        wall_seconds=time.monotonic() - started,
        fell_back=fell_back,
        fallback_reason=reason,
    )


def encode_chunk(
    settings: Settings,
    cap: Capability,
    src: Path,
    dst: Path,
    *,
    hdr: bool,
    start: float,
    has_audio: bool = True,
) -> EncodeResult:
    """One ``SFN_VIDEO_CHUNK_SECONDS`` window starting at *start*.

    A thin wrapper on purpose: §3.3 established that independently encoded
    windows tile exactly at frame level, so a chunk needs no special handling —
    only the window. The final chunk of a source is short, and ffmpeg ends it at
    the last frame without being told the duration is clipped.
    """
    if start < 0:
        raise ValueError("chunk start must be >= 0")
    return encode(
        settings,
        cap,
        src,
        dst,
        hdr=hdr,
        start=start,
        duration=float(settings.video_chunk_seconds),
        has_audio=has_audio,
    )


def job_threads(settings: Settings) -> int:
    """The full job's ``-threads`` value; ``0`` in the config means half the box.

    Half, at least one: §3.5 measured one encode enough to saturate this host, so
    an export that takes every core is exactly §4.3's contention with extra steps.
    A host that will not report its CPU count gets 1 — the conservative end, since
    the number exists to leave room for chunk work.
    """
    if settings.video_job_threads > 0:
        return settings.video_job_threads
    return max(1, (os.cpu_count() or 2) // 2)


def encode_full(
    settings: Settings,
    cap: Capability,
    src: Path,
    dst: Path,
    *,
    hdr: bool,
    has_audio: bool = True,
    limit_bytes: int | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
) -> EncodeResult:
    """The whole source, encoded once (§4.3).

    The same encode as a chunk with no window — and three differences that are
    the phase-7 ruling: it runs niced and thread-capped so chunk work wins the
    contention (§4.3 remedy (a)), it reports progress so the analyst gets a rate
    and an ETA, and its ``.part`` is watched against *limit_bytes* so §6.3's
    uncalibrated estimate cannot be trusted into filling the cache.
    """
    return encode(
        settings,
        cap,
        src,
        dst,
        hdr=hdr,
        has_audio=has_audio,
        watch=Watch(
            nice=settings.video_job_nice,
            threads=job_threads(settings),
            limit_bytes=limit_bytes,
            on_progress=on_progress,
            on_start=on_start,
        ),
    )
