"""Opt-in video routing. Missing worker configuration always fails closed."""

from __future__ import annotations

import os
import threading
from pathlib import Path

_capability_lock = threading.Lock()
_capability_cache = None


def configured() -> bool:
    names = (
        "SFN_VIDEO_CPU_STATE",
        "SFN_VIDEO_CHUNK_STATE",
        "SFN_VIDEO_FULL_STATE",
        "SFN_VIDEO_WORKER_CASE",
        "SFN_VIDEO_WORKER_RECEIPTS",
    )
    values = [os.environ.get(name) for name in names]
    if not any(name in os.environ for name in names):
        return False
    from scalar_forensic.media_worker.client import WorkerUnavailable

    if not all(values) or any(not Path(values[i]).is_absolute() for i in (0, 1, 2, 4)):
        raise WorkerUnavailable("video isolation requires all three states, case and receipts")
    if len(set(values[:3])) != 3:
        raise WorkerUnavailable("video slots require distinct state paths")
    return True


def client(slot="cpu"):
    from scalar_forensic.media_worker.client import WorkerUnavailable
    from scalar_forensic.media_worker.video_client import VideoClient
    from scalar_forensic.media_worker.video_runtime import read_state

    if not configured():
        raise WorkerUnavailable("video isolation is not configured")
    case = os.environ["SFN_VIDEO_WORKER_CASE"]
    paths = {
        role: Path(os.environ[f"SFN_VIDEO_{role.upper()}_STATE"])
        for role in ("cpu", "chunk", "full")
    }
    # Each supervisor owns a separate state directory. Validate that fixed
    # layout without requiring unrelated slots to be online during replacement.
    if len({path.parent.resolve() for path in paths.values()}) != 3:
        raise WorkerUnavailable("video slots must have distinct state directories")
    state = read_state(paths[slot], case)
    expected_lock = paths[slot].parent / "control" / "job.lock"
    if state.lock_path.resolve() != expected_lock.resolve():
        raise WorkerUnavailable("video slot lock does not match its supervisor directory")
    # A source switch can arrive while the old source's prefetch still encodes.
    # Admit that chunk for up to 30 seconds; CPU and full slots stay independent.
    return VideoClient(
        paths[slot],
        case,
        os.environ["SFN_VIDEO_WORKER_RECEIPTS"],
        queue_timeout=30 if slot == "chunk" else 5,
    )


def frames(source, fps, max_frames):
    events = client().stream("frames", source, dict(fps=fps, max_frames=max_frames))
    try:
        for kind, value in events:
            if kind == "frame":
                yield value
    finally:
        events.close()


def frame_at(source, timecode_ms):
    image = None
    for kind, value in client().stream("frame_at", source, dict(timecode_ms=timecode_ms)):
        if kind == "frame":
            image = value.image
    return image


def reset_capability_cache():
    global _capability_cache
    with _capability_lock:
        _capability_cache = None


def capability(settings, *, refresh=False):
    """Reuse one renderer probe, validating its generation before each use."""
    from scalar_forensic.media_worker.client import WorkerUnavailable
    from scalar_forensic.media_worker.video_runtime import read_state
    from scalar_forensic.video_playback.capability import Capability

    global _capability_cache
    worker = client("chunk")
    state = read_state(worker.state_path, worker.case_id)
    if state.policy != "sfn-video-nvenc-v1" or state.failure_notice.exists():
        raise WorkerUnavailable("renderer capability generation unavailable")
    key = (state, settings.video_hwaccel)
    with _capability_lock:
        if not refresh and _capability_cache is not None and _capability_cache[0] == key:
            return _capability_cache[1]
    # Never hold the cache lock across slot admission or the RPC. Concurrent
    # misses remain bounded by the slot's admission deadline.
    result = worker.call("capability", parameters={"hwaccel": settings.video_hwaccel})
    result["notes"] = tuple(result["notes"])
    result = Capability(**result)
    if read_state(worker.state_path, worker.case_id) != state or state.failure_notice.exists():
        raise WorkerUnavailable("renderer generation changed during capability probe")
    with _capability_lock:
        _capability_cache = (key, result)
    return result


def encode(settings, source, destination, *, hdr, start, duration, has_audio, watch):
    from scalar_forensic.media_worker.video_client import Cancellation
    from scalar_forensic.video_playback.encode import EncodeResult

    handle = Cancellation()
    if watch is not None and watch.on_start is not None:
        watch.on_start(handle)
    params = dict(
        output_height=settings.video_output_height,
        chunk_seconds=settings.video_chunk_seconds,
        hwaccel=settings.video_hwaccel,
        timeout=settings.video_job_timeout,
        hdr=hdr,
        start=start,
        duration=duration,
        has_audio=has_audio,
        watched=watch is not None,
        threads=watch.threads if watch else None,
        nice=watch.nice if watch else 0,
        limit_bytes=watch.limit_bytes if watch else None,
    )
    result = client("full" if watch is not None else "chunk").call(
        "encode",
        source,
        params,
        destination=destination,
        cancellation=handle,
        on_progress=watch.on_progress if watch else None,
        settings=settings,
    )
    return EncodeResult(path=Path(destination), **result)
