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

import logging
import subprocess
import time
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
) -> list[str]:
    """The argv for one encode.  Pure — builds, never runs.

    ``-ss`` goes **before** ``-i``: input seeking uses the container index, which
    §3.2 and §3.5 measured flat out to 3 h 41 m on a dense index.  Output seeking
    would decode and discard everything before the window, making chunk cost
    proportional to offset — the property §1 rejects the whole design over.
    """
    cmd = [settings.ffmpeg_path, "-nostdin", "-hide_banner", "-v", "error"]
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

    started = time.monotonic()
    try:
        cmd = build_command(
            settings, pipeline, src, part, start=start, duration=duration, has_audio=has_audio
        )
        try:
            _run(cmd, settings.video_job_timeout)
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
            cmd = build_command(
                settings, pipeline, src, part, start=start, duration=duration, has_audio=has_audio
            )
            _run(cmd, settings.video_job_timeout)
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
