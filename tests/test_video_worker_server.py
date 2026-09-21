import hashlib
import io
import os
import shlex
import shutil
import socket
import stat
import subprocess
import sys

import av
import numpy as np
import pytest

from scalar_forensic import video
from scalar_forensic.media_worker import video_protocol as p
from scalar_forensic.media_worker.video_server import serve_connection


def test_video_listener_preserves_socket_for_host_cleanup():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from scalar_forensic.media_worker import video_server

    with TemporaryDirectory(prefix="video-uds-") as temporary:
        path = Path(temporary) / "worker.sock"
        with video_server.bound_listener(path) as listener:
            assert stat.S_IMODE(path.stat().st_mode) == 0o660
            with socket.socket(socket.AF_UNIX) as client:
                client.connect(str(path))
                channel, _ = listener.accept()
                with channel:
                    client.sendall(b"hello")
                    assert channel.recv(5) == b"hello"
            path.parent.chmod(0o550)
        assert path.exists()
        path.parent.chmod(0o730)
        with pytest.raises(FileExistsError):
            with video_server.bound_listener(path):
                pass
        path.unlink()


def test_video_ipc_directory_requires_distinct_owner_and_private_group(tmp_path, monkeypatch):
    from scalar_forensic.media_worker import video_server

    tmp_path.chmod(0o730)
    with pytest.raises(RuntimeError):
        video_server.validate_ipc_directory(tmp_path)
    monkeypatch.setattr(video_server.os, "geteuid", lambda: os.stat(tmp_path).st_uid + 1)
    video_server.validate_ipc_directory(tmp_path)
    for mode in (0o770, 0o733, 0o750):
        tmp_path.chmod(mode)
        with pytest.raises(RuntimeError):
            video_server.validate_ipc_directory(tmp_path)
    tmp_path.chmod(0o730)
    monkeypatch.setattr(video_server.os, "getegid", lambda: os.stat(tmp_path).st_gid + 1)
    with pytest.raises(RuntimeError):
        video_server.validate_ipc_directory(tmp_path)


class Duplex:
    def __init__(self, data):
        self.input = io.BytesIO(data)
        self.output = io.BytesIO()

    def read(self, size):
        return self.input.read(size)

    def write(self, data):
        return self.output.write(data)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
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
    return path


def run(
    root,
    operation,
    parameters=None,
    role="cpu",
    generation="b" * 32,
    limit=10**7,
    ffmpeg_path="ffmpeg",
):
    wire = io.BytesIO()
    p.send_packet(
        wire,
        dict(
            version=1,
            request_id="a" * 32,
            generation=generation,
            operation=operation,
            source=None if operation in ("capability", "ping") else "clip.mp4",
            parameters=parameters or {},
        ),
    )
    stream = Duplex(wire.getvalue())
    serve_connection(
        stream,
        generation="b" * 32,
        role=role,
        evidence=root,
        scratch=root,
        max_output_bytes=limit,
        ffmpeg_path=ffmpeg_path,
    )
    stream.output.seek(0)
    events = []
    while stream.output.tell() < len(stream.output.getvalue()):
        header, payload = p.recv_packet(stream.output)
        p.validate_event(header, payload)
        events.append((header, payload))
    return events


def test_info_and_frames_preserve_existing_decode(clip):
    assert run(clip.parent, "info")[-1][0]["body"] == video.get_video_info(clip)
    events = run(clip.parent, "frames", {"fps": 4.0, "max_frames": 10})
    expected = list(video.extract_frames(clip, fps=4, max_frames=10))
    assert len(events) == len(expected) + 1
    for (event, payload), frame in zip(events, expected):
        assert payload == frame.image.tobytes()
        assert event["body"]["frame_hash"] == frame.frame_hash
        assert event["body"]["timecode_ms"] == frame.timecode_ms
    assert events[-1][0]["type"] == "result"


def test_rewrap_sends_complete_hashed_artifact(clip):
    events = run(clip.parent, "rewrap")
    artifact = events[0][0]
    data = b"".join(payload for event, payload in events if event["type"] == "data")
    assert artifact["type"] == "artifact"
    assert artifact["body"] == dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    with av.open(io.BytesIO(data)) as container:
        assert len(list(container.decode(video=0))) == 4
    assert events[-1][0]["type"] == "result"


def test_failed_native_decode_and_ceiling_publish_no_artifact(tmp_path, clip):
    assert [event["type"] for event, _ in run(clip.parent, "rewrap", limit=1)] == ["error"]
    clip.write_bytes(b"not a video")
    events = run(tmp_path, "frames", {"fps": 1, "max_frames": 1})
    assert [event["type"] for event, _ in events] == ["error"]
    assert len(events[0][0]["body"]["message"]) <= 4096


def test_role_and_generation_are_checked_before_parsing(tmp_path):
    for kwargs in [dict(role="renderer"), dict(generation="c" * 32)]:
        with pytest.raises(p.ProtocolError):
            run(tmp_path, "info", **kwargs)


def test_ping_requires_no_evidence(tmp_path):
    wire = io.BytesIO()
    p.send_packet(
        wire,
        dict(
            version=1,
            request_id="a" * 32,
            generation="b" * 32,
            operation="ping",
            source=None,
            parameters={},
        ),
    )
    for role in ("cpu", "renderer"):
        stream = Duplex(wire.getvalue())
        serve_connection(
            stream,
            generation="b" * 32,
            role=role,
            evidence=tmp_path / "absent",
            scratch=tmp_path,
            max_output_bytes=10000,
        )
        stream.output.seek(0)
        assert p.recv_packet(stream.output)[0]["body"] == {"ready": True}


def test_source_symlink_cannot_escape_evidence_root(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"private")
    (root / "clip.mp4").symlink_to(outside)
    with pytest.raises(p.ProtocolError):
        run(root, "info")


def test_real_encode_is_bounded_and_records_fixed_command(clip):
    params = dict(
        output_height=144,
        chunk_seconds=5,
        hwaccel="none",
        timeout=30,
        hdr=False,
        start=0,
        duration=1,
        has_audio=False,
        watched=False,
        threads=1,
        nice=0,
        limit_bytes=None,
    )
    events = run(clip.parent, "encode", params, role="renderer")
    result = events[-1][0]
    assert result["type"] == "result", result
    assert result["body"]["pipeline"]["encoder"] == "libx264"
    assert "-progress" in result["body"]["command"]
    assert result["body"]["command"][result["body"]["command"].index("-t") + 1] == "1.000"
    assert "path" not in result["body"]
    events = run(clip.parent, "encode", params, role="renderer", limit=1)
    assert [event["type"] for event, _ in events] == ["error"]
    assert events[0][0]["body"]["kind"] == "ceiling"


def test_worker_import_does_not_load_web_or_models():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scalar_forensic.media_worker.video_server; "
            "import scalar_forensic.video_playback.encode; "
            "assert 'scalar_forensic.video_playback.routes' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "from scalar_forensic.video_playback import router; "
            "from fastapi import APIRouter; assert isinstance(router, APIRouter)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_capability_probe_is_reused_by_encode_in_same_generation(clip):
    log = clip.parent / "ffmpeg.calls"
    wrapper = clip.parent / "ffmpeg-wrapper"
    binary = shutil.which("ffmpeg")
    assert binary is not None
    wrapper.write_text(
        "#!/bin/sh\nprintf x >> "
        + shlex.quote(str(log))
        + "\nexec "
        + shlex.quote(binary)
        + ' "$@"\n'
    )
    wrapper.chmod(0o700)
    events = run(
        clip.parent, "capability", {"hwaccel": "none"}, role="renderer", ffmpeg_path=str(wrapper)
    )
    assert events[-1][0]["body"]["encoder"] == "libx264"
    probe_calls = len(log.read_bytes())
    assert probe_calls > 1
    params = dict(
        output_height=144,
        chunk_seconds=5,
        hwaccel="none",
        timeout=30,
        hdr=False,
        start=0,
        duration=1,
        has_audio=False,
        watched=False,
        threads=1,
        nice=0,
        limit_bytes=None,
    )
    events = run(clip.parent, "encode", params, role="renderer", ffmpeg_path=str(wrapper))
    assert events[-1][0]["type"] == "result"
    assert len(log.read_bytes()) == probe_calls + 1
