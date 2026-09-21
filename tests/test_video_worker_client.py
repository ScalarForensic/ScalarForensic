"""Real Unix-stream receiver tests; the fixture supervisor confirms replacement."""

import contextlib
import hashlib
import json
import os
import socket
import stat
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from scalar_forensic.media_worker import video_protocol as wire
from scalar_forensic.media_worker.client import WorkerUnavailable
from scalar_forensic.media_worker.video_client import Cancellation, VideoClient
from scalar_forensic.media_worker.video_runtime import VideoRuntime, publish_state


@pytest.fixture
def slot(monkeypatch):
    from scalar_forensic.video_playback import digest

    monkeypatch.setattr(digest, "_source_digest", lambda *a: "c" * 64)
    monkeypatch.setattr(digest, "_cached_source_digest", lambda *a: None)
    with tempfile.TemporaryDirectory(prefix="vc-") as name:
        root = Path(name)
        (root / "ipc").mkdir()
        (root / "source").write_bytes(b"opaque")
        lock = root / "job.lock"
        lock.touch(mode=0o600)
        state = VideoRuntime(
            "a" * 32,
            root / "ipc/s",
            "case",
            "sha256:" + "b" * 64,
            "sfn-video-cpu-v1",
            root / "failed",
            lock,
            root,
            1000,
            2,
        )
        state_path = root / "state"
        publish_state(state_path, state)
        stop = threading.Event()

        def supervisor():
            while not stop.wait(0.005):
                if state.failure_notice.exists():
                    publish_state(state_path, replace(state, generation="d" * 32))
                    return

        thread = threading.Thread(target=supervisor)
        thread.start()
        client = VideoClient(state_path, "case", root / "receipts", replacement_timeout=1)
        yield client, state, root
        stop.set()
        thread.join()


@contextlib.contextmanager
def peer(slot, reply=None):
    _, state, _ = slot
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(state.socket_path))
    listener.listen(1)
    failures = []

    def run():
        try:
            channel, _ = listener.accept()
            with channel, channel.makefile("rwb", buffering=0) as stream:
                if reply is None:
                    from scalar_forensic.media_worker.video_server import serve_connection

                    serve_connection(
                        stream,
                        generation=state.generation,
                        role="cpu",
                        evidence=state.evidence_root,
                        scratch=state.evidence_root,
                        max_output_bytes=state.max_output_bytes,
                    )
                    return
                request, _ = wire.recv_packet(stream)

                def emit(kind, body, payload=b"", **identity):
                    wire.send_packet(
                        stream,
                        dict(
                            version=1,
                            request_id=request["request_id"],
                            generation=identity.get("generation", state.generation),
                            type=kind,
                            body=body,
                        ),
                        payload,
                    )

                reply(emit, request)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        yield
    finally:
        thread.join(timeout=4)
        listener.close()
        assert not thread.is_alive()
        assert not failures


def test_valid_artifact(slot):
    client, _, root = slot
    data = b"opaque mp4 bytes"

    def reply(emit, request):
        assert request["source"] == "source"
        emit("artifact", dict(size=len(data), sha256=hashlib.sha256(data).hexdigest()))
        emit("data", {}, data)
        emit("result", dict(skipped_streams=[], timestamp_repairs=0))

    with peer(slot, reply):
        assert client.call("rewrap", root / "source", destination=root / "final") == dict(
            skipped_streams=[], timestamp_repairs=0
        )
    assert (root / "final").read_bytes() == data
    assert not (root / "active").exists()
    assert not list(root.glob("*.part"))


@pytest.mark.parametrize(
    "fault", ["hash", "truncated", "oversized", "result", "generation", "trailing"]
)
def test_bad_artifact_never_publishes(slot, fault):
    client, state, root = slot
    destination = root / "final"
    destination.write_bytes(b"existing")

    def reply(emit, request):
        emit(
            "artifact",
            dict(
                size=1001 if fault == "oversized" else 3,
                sha256="0" * 64 if fault == "hash" else hashlib.sha256(b"abc").hexdigest(),
            ),
        )
        if fault == "oversized":
            return
        emit("data", {}, b"abc")
        if fault == "truncated":
            return
        emit(
            "result",
            dict(skipped_streams=[], timestamp_repairs="wrong" if fault == "result" else 0),
            generation="f" * 32 if fault == "generation" else state.generation,
        )
        if fault == "trailing":
            emit("result", {})

    with peer(slot, reply), pytest.raises(WorkerUnavailable):
        client.call("rewrap", root / "source", destination=destination)
    assert destination.read_bytes() == b"existing"
    assert state.failure_notice.exists()
    assert not list(root.glob("*.part"))
    assert not (root / "active").exists()


def test_raw_frame_exact(slot):
    client, _, root = slot
    pixels = bytes(range(18))
    digest = hashlib.sha256(
        (3).to_bytes(4, "little") + (2).to_bytes(4, "little") + pixels
    ).hexdigest()

    def reply(emit, request):
        emit(
            "frame",
            dict(width=3, height=2, timecode_ms=123, frame_index=0, frame_hash=digest),
            pixels,
        )
        emit("result", {})

    with peer(slot, reply):
        events = list(client.stream("frames", root / "source", dict(fps=1, max_frames=1)))
    frame = events[0][1]
    assert frame.image.tobytes() == pixels
    assert (frame.timecode_ms, frame.frame_index, frame.frame_hash) == (123, 0, digest)


def test_precancel_does_not_bind(slot):
    from scalar_forensic.video_playback.encode import Cancelled

    client, state, root = slot
    handle = Cancellation()
    handle.cancel()
    with pytest.raises(Cancelled):
        client.call("info", root / "source", cancellation=handle)
    assert not state.failure_notice.exists()
    assert not (root / "active").exists()


def test_active_cancel_confirms_replacement(slot):
    client, state, root = slot
    handle = Cancellation()

    def reply(emit, request):
        handle.cancel()
        time.sleep(0.03)

    with peer(slot, reply), pytest.raises((WorkerUnavailable, RuntimeError)):
        client.call("info", root / "source", cancellation=handle)
    assert state.failure_notice.exists()
    assert not (root / "active").exists()


def test_unmapped_source(slot):
    client, state, _ = slot
    with pytest.raises(WorkerUnavailable):
        client.call("info", Path(__file__))
    assert not state.failure_notice.exists()


def test_slot_busy(slot):
    import fcntl

    client, state, root = slot
    client.queue_timeout = 0.01
    with state.lock_path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(WorkerUnavailable, match="busy"):
            client.call("info", root / "source")


def test_generator_close_retires_live_worker(slot):
    client, state, root = slot
    pixels = b"abc"
    digest = hashlib.sha256((1).to_bytes(4, "little") * 2 + pixels).hexdigest()

    def reply(emit, request):
        emit(
            "frame",
            dict(width=1, height=1, timecode_ms=0, frame_index=0, frame_hash=digest),
            pixels,
        )
        time.sleep(0.1)

    with peer(slot, reply):
        events = client.stream("frames", root / "source", dict(fps=1, max_frames=2))
        assert next(events)[0] == "frame"
        events.close()
    assert state.failure_notice.exists()
    assert not (root / "active").exists()


def test_unconfirmed_removal_retains_active(slot, monkeypatch):
    client, _, root = slot
    client.replacement_timeout = 0

    def reply(emit, request):
        emit("result", {"unknown": 3})

    with peer(slot, reply), pytest.raises(WorkerUnavailable, match="removal"):
        client.call("info", root / "source")
    assert (root / "active").exists()


def test_real_server_frames_preserve_pixels_and_timecodes(slot):
    import av
    import numpy as np

    from scalar_forensic import video

    client, _, root = slot
    source = root / "clip.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for shade in (20, 60, 100, 140):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), shade, np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    expected = list(video.extract_frames(source, fps=4, max_frames=10))
    with peer(slot):
        frames = [
            value
            for kind, value in client.stream("frames", source, dict(fps=4, max_frames=10))
            if kind == "frame"
        ]
    assert [(f.timecode_ms, f.frame_hash, f.image.tobytes()) for f in frames] == [
        (f.timecode_ms, f.frame_hash, f.image.tobytes()) for f in expected
    ]


def test_queued_request_uses_replacement_generation(slot):
    import fcntl

    client, state, root = slot
    replacement = replace(state, generation="e" * 32)
    client.queue_timeout = 1
    observed = []

    def reply(emit, request):
        observed.append(request["generation"])
        emit("result", {}, generation=replacement.generation)

    with state.lock_path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        errors = []

        def run():
            try:
                client.call("info", root / "source")
            except BaseException as exc:
                errors.append(exc)

        with peer(slot, reply):
            thread = threading.Thread(target=run)
            thread.start()
            time.sleep(0.05)
            publish_state(client.state_path, replacement)
            fcntl.flock(lock, fcntl.LOCK_UN)
            thread.join(timeout=3)
    assert not errors
    assert observed == [replacement.generation]


def test_cancel_before_publication_discards_part(slot, monkeypatch):
    client, state, root = slot
    handle = Cancellation()
    original_receipt = client._receipt

    def receipt(row):
        original_receipt(row)
        if row["outcome"] == "validated":
            handle.cancel()

    monkeypatch.setattr(client, "_receipt", receipt)

    def reply(emit, request):
        emit("artifact", dict(size=3, sha256=hashlib.sha256(b"abc").hexdigest()))
        emit("data", {}, b"abc")
        emit("result", dict(skipped_streams=[], timestamp_repairs=0))

    with peer(slot, reply), pytest.raises(RuntimeError):
        client.call("rewrap", root / "source", destination=root / "final", cancellation=handle)
    assert not (root / "final").exists()
    assert not list(root.glob("*.part"))
    assert state.failure_notice.exists()


@pytest.mark.parametrize("confirmed", [True, False])
def test_cancel_blocked_receive_from_another_thread(slot, monkeypatch, confirmed):
    from scalar_forensic.media_worker.client import DeadlineStream
    from scalar_forensic.video_playback.encode import Cancelled

    client, state, root = slot
    if not confirmed:
        client.replacement_timeout = 0
    handle = Cancellation()
    receiving = threading.Event()
    cancelled = threading.Event()
    original_read = DeadlineStream.read

    def read(stream, size):
        receiving.set()
        return original_read(stream, size)

    monkeypatch.setattr(DeadlineStream, "read", read)

    def cancel():
        assert receiving.wait(2)
        # No peer bytes are available: the RPC thread remains inside recv.
        time.sleep(0.03)
        handle.cancel()
        cancelled.set()

    def reply(emit, request):
        assert cancelled.wait(2)

    thread = threading.Thread(target=cancel)
    thread.start()
    try:
        with peer(slot, reply), pytest.raises(Cancelled if confirmed else WorkerUnavailable):
            client.call("info", root / "source", cancellation=handle)
    finally:
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert state.failure_notice.exists()
    assert (root / "active").exists() is not confirmed
    outcomes = [
        json.loads(line)["outcome"] for line in (root / "receipts").read_text().splitlines()
    ]
    assert outcomes == (["started", "cancelled"] if confirmed else ["started"])


def test_receipt_fsyncs_directory_on_first_append(slot, monkeypatch):
    client, _, _ = slot
    calls = []
    original = os.fsync

    def fsync(fd):
        calls.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        original(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    client._receipt({"outcome": "started"})
    client._receipt({"outcome": "validated"})
    assert calls == ["file", "directory", "file"]


def test_receipt_completes_short_writes(slot, monkeypatch):
    client, _, root = slot
    original = os.write
    writes = []

    def short_write(fd, data):
        writes.append(len(data))
        return original(fd, data[:3])

    monkeypatch.setattr(os, "write", short_write)
    client._receipt({"outcome": "started"})
    assert len(writes) > 1
    assert (root / "receipts").read_bytes() == b'{"outcome":"started"}\n'


def test_receipt_failure_before_publication_discards_part(slot, monkeypatch):
    client, state, root = slot
    original = client._receipt

    def receipt(row):
        if row["outcome"] == "validated":
            raise OSError("receipt storage unavailable")
        original(row)

    monkeypatch.setattr(client, "_receipt", receipt)

    def reply(emit, request):
        emit("artifact", dict(size=3, sha256=hashlib.sha256(b"abc").hexdigest()))
        emit("data", {}, b"abc")
        emit("result", dict(skipped_streams=[], timestamp_repairs=0))

    destination = root / "final"
    destination.write_bytes(b"existing")
    with peer(slot, reply), pytest.raises(WorkerUnavailable):
        client.call("rewrap", root / "source", destination=destination)
    assert destination.read_bytes() == b"existing"
    assert not list(root.glob("*.part"))
    assert state.failure_notice.exists()
    assert not (root / "active").exists()


def test_cached_capability_does_not_queue_behind_prefetch(slot, monkeypatch):
    import fcntl
    from types import SimpleNamespace

    from scalar_forensic import video_access

    client, state, root = slot
    state = replace(state, policy="sfn-video-nvenc-v1")
    publish_state(client.state_path, state)
    client.queue_timeout = 0.05
    monkeypatch.setattr(video_access, "client", lambda slot: client)
    monkeypatch.setattr(video_access, "_capability_cache", None, raising=False)
    settings = SimpleNamespace(video_hwaccel="auto")

    def reply(emit, request):
        assert request["operation"] == "capability"
        emit(
            "result",
            dict(
                ffmpeg_path="ffmpeg",
                ffmpeg_version="6",
                encoder="h264_nvenc",
                hwaccel="cuda",
                tonemap_ok=True,
                notes=[],
            ),
        )

    with peer((client, state, root), reply):
        expected = video_access.capability(settings)
    with state.lock_path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert video_access.capability(settings) == expected


@pytest.mark.parametrize(
    "change", ["generation", "hwaccel", "missing", "failed", "refresh", "reset"]
)
def test_capability_cache_revalidates_runtime_and_settings(slot, monkeypatch, change):
    from types import SimpleNamespace

    from scalar_forensic import video_access

    client, state, root = slot
    state = replace(state, policy="sfn-video-nvenc-v1")
    publish_state(client.state_path, state)
    monkeypatch.setattr(video_access, "client", lambda slot: client)
    monkeypatch.setattr(video_access, "_capability_cache", None, raising=False)
    settings = SimpleNamespace(video_hwaccel="auto")
    calls = []

    def call(operation, *, parameters):
        calls.append(parameters["hwaccel"])
        return dict(
            ffmpeg_path="ffmpeg",
            ffmpeg_version="6",
            encoder="h264_nvenc",
            hwaccel="cuda",
            tonemap_ok=True,
            notes=[],
        )

    monkeypatch.setattr(client, "call", call)
    video_access.capability(settings)
    if change == "generation":
        publish_state(client.state_path, replace(state, generation="e" * 32))
    elif change == "hwaccel":
        settings.video_hwaccel = "none"
    elif change == "missing":
        client.state_path.unlink()
    elif change == "failed":
        state.failure_notice.touch(mode=0o600)
    elif change == "reset":
        from scalar_forensic.video_playback.capability import reset_cache

        reset_cache()
    if change in ("missing", "failed"):
        with pytest.raises(WorkerUnavailable):
            video_access.capability(settings)
        assert calls == ["auto"]
    else:
        video_access.capability(settings, refresh=change == "refresh")
        assert calls == ["auto", settings.video_hwaccel]


def test_chunk_admission_waits_for_prefetch_past_old_five_second_limit(slot):
    import fcntl

    client, _, root = slot
    client.queue_timeout = 30
    errors = []
    results = []

    def run():
        try:
            results.append(client.call("info", root / "source"))
        except BaseException as exc:
            errors.append(exc)

    with client.state_path.parent.joinpath("job.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with peer(slot, lambda emit, request: emit("result", {})):
            thread = threading.Thread(target=run)
            thread.start()
            time.sleep(5.2)
            fcntl.flock(lock, fcntl.LOCK_UN)
            thread.join(timeout=3)
    assert not thread.is_alive()
    assert not errors
    assert results == [{}]
