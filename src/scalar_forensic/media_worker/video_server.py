"""Sequential fixed-operation video worker; the host owns isolation and lifetime."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import socket
import stat
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scalar_forensic.media_worker import video_protocol as protocol

CPU_OPERATIONS = frozenset(("info", "stream_report", "frames", "frame_at", "rewrap", "ping"))
RENDERER_OPERATIONS = frozenset(("capability", "encode", "ping"))


def validate_ipc_directory(path: Path) -> None:
    """Require a host-owned bootstrap directory the host can seal after ping."""
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid == os.geteuid()
        or info.st_gid != os.getegid()
        or stat.S_IMODE(info.st_mode) != 0o730
    ):
        raise RuntimeError("video IPC requires a distinct owner, worker group and mode 0730")


@contextmanager
def bound_listener(path: Path):
    """Bind once; host owns removal after sealing the directory and stopping us."""
    if os.path.lexists(path):
        raise FileExistsError("worker socket path already exists")
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        path.chmod(0o660)
        listener.listen(1)
        yield listener


@lru_cache(maxsize=3)
def _capability(ffmpeg_path: str, hwaccel: str):
    """Probe once per hardware mode in this worker process/generation."""
    from scalar_forensic.video_playback.capability import probe

    return probe(SimpleNamespace(ffmpeg_path=ffmpeg_path, video_hwaccel=hwaccel))


def _source(evidence: Path, name: str, uploads: Path | None = None) -> Path:
    if name.startswith("__uploads__/"):
        if uploads is None:
            raise protocol.ProtocolError("upload source root is not configured")
        evidence = uploads
        name = name.removeprefix("__uploads__/")
    root = evidence.resolve(strict=True)
    source = (root / name).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise protocol.ProtocolError("source must be a file beneath evidence root")
    return source


def _artifact(path, limit, emit):
    from scalar_forensic.video_playback.encode import CeilingExceeded

    size = path.stat().st_size
    if size > limit:
        raise CeilingExceeded(
            "artifact exceeds worker limit", written_bytes=size, limit_bytes=limit
        )
    if size == 0:
        raise RuntimeError("empty video artifact")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while data := source.read(protocol.MAX_DATA):
            digest.update(data)
        source.seek(0)
        emit("artifact", dict(size=size, sha256=digest.hexdigest()))
        while data := source.read(protocol.MAX_DATA):
            emit("data", {}, data)


def _frame(image, timecode_ms, frame_index, frame_hash, emit):
    width, height = image.size
    if not (
        0 < width <= 32768 and 0 < height <= 32768 and width * height * 3 <= protocol.MAX_PAYLOAD
    ):
        raise RuntimeError("frame exceeds worker transport limit")
    emit(
        "frame",
        dict(
            width=width,
            height=height,
            timecode_ms=timecode_ms,
            frame_index=frame_index,
            frame_hash=frame_hash,
        ),
        image.tobytes(),
    )


def _dispatch(request, source, directory, max_output_bytes, ffmpeg_path, emit):
    # Imports occur only in the worker. Never construct application Settings,
    # which reads operator configuration and routing environment variables.
    from scalar_forensic import video
    from scalar_forensic.video_playback import codecs, encode, rewrap

    op, params = request["operation"], request["parameters"]
    if op == "info":
        return video.get_video_info(source)
    if op == "stream_report":
        return codecs._stream_report(source)
    if op == "frames":
        for frame in video.extract_frames(source, **params):
            _frame(frame.image, frame.timecode_ms, frame.frame_index, frame.frame_hash, emit)
        return {}
    if op == "frame_at":
        image = video.extract_frame_at(source, params["timecode_ms"])
        if image is not None:
            _frame(image, params["timecode_ms"], 0, video._frame_pixel_hash(image), emit)
        return {"found": image is not None}
    output = directory / "output.mp4"
    if op == "rewrap":
        result = rewrap._remux_to_mp4(source, output)
        _artifact(output, max_output_bytes, emit)
        return result
    settings = SimpleNamespace(ffmpeg_path=ffmpeg_path, video_hwaccel=params["hwaccel"])
    cap = _capability(ffmpeg_path, params["hwaccel"])
    if op == "capability":
        return asdict(cap)
    settings.video_output_height = params["output_height"]
    settings.video_chunk_seconds = params["chunk_seconds"]
    settings.video_job_timeout = params["timeout"]
    limit = min(params["limit_bytes"] or max_output_bytes, max_output_bytes)
    # Watch also adds non-pixel -progress/-nostats flags to chunk commands.
    # The recorded actual command includes those flags; pixel settings stay exact.
    watch = encode.Watch(
        nice=params["nice"],
        threads=params["threads"],
        limit_bytes=limit,
        on_progress=(lambda progress: emit("progress", asdict(progress)))
        if params["watched"]
        else None,
    )
    result = encode.encode(
        settings,
        cap,
        source,
        output,
        hdr=params["hdr"],
        start=params["start"],
        duration=params["duration"],
        has_audio=params["has_audio"],
        watch=watch,
    )
    body = asdict(result)
    body.pop("path")
    # Check metadata before publishing the first artifact byte.
    protocol.validate_event(
        dict(
            version=1,
            request_id=request["request_id"],
            generation=request["generation"],
            type="result",
            body=body,
        ),
        b"",
    )
    _artifact(output, limit, emit)
    return body


def serve_connection(
    stream,
    *,
    generation: str,
    role: str,
    evidence: Path,
    scratch: Path,
    max_output_bytes: int,
    ffmpeg_path: str = "ffmpeg",
    uploads: Path | None = None,
) -> None:
    """Read one request and stream one terminal result or bounded error.

    The caller closes the connection. Protocol errors propagate so the launcher
    can retire the generation; native errors produce an explicit error event.
    Scratch must be a deployment-bounded private mount, including for PyAV remux.
    """
    if (
        role not in ("cpu", "renderer")
        or type(max_output_bytes) is not int
        or not 0 < max_output_bytes <= 2**50
    ):
        raise ValueError("invalid worker policy")
    request, payload = protocol.recv_packet(stream)
    protocol.validate_request(request)
    allowed = CPU_OPERATIONS if role == "cpu" else RENDERER_OPERATIONS
    if payload or request["generation"] != generation or request["operation"] not in allowed:
        raise protocol.ProtocolError("request does not match worker role or generation")

    def emit(kind, body, data=b""):
        event = dict(
            version=1, request_id=request["request_id"], generation=generation, type=kind, body=body
        )
        protocol.validate_event(event, data)
        protocol.send_packet(stream, event, data)

    if request["operation"] == "ping":
        emit("result", {"ready": True})
        return

    from scalar_forensic.video_playback.encode import CeilingExceeded, EncodeError

    try:
        source = (
            _source(evidence, request["source"], uploads) if request["source"] is not None else None
        )
        with TemporaryDirectory(prefix="video-", dir=scratch) as temporary:
            result = _dispatch(
                request, source, Path(temporary), max_output_bytes, ffmpeg_path, emit
            )
        emit("result", result)
    except (protocol.ProtocolError, BrokenPipeError, ConnectionError):
        raise
    except Exception as exc:
        kind = "decode" if request["operation"] in CPU_OPERATIONS else "internal"
        if isinstance(exc, CeilingExceeded):
            kind = "ceiling"
        elif isinstance(exc, EncodeError):
            kind = "encode"
        # Escaping and envelope overhead must still fit the 64 KiB JSON packet.
        emit(
            "error",
            dict(
                kind=kind,
                message=str(exc)[:4096],
                stderr=getattr(exc, "stderr", "")[-6000:],
                returncode=getattr(exc, "returncode", None),
                timed_out=getattr(exc, "timed_out", False),
                written_bytes=getattr(exc, "written_bytes", 0),
                limit_bytes=getattr(exc, "limit_bytes", 0),
            ),
        )


def validate_runtime(role: str):
    """Check basic restrictions; only the renderer may see GPU device nodes."""
    if os.geteuid() == 0:
        raise RuntimeError("worker requires nonroot identity")
    try:
        status = dict(
            line.split(":", 1)
            for line in Path("/proc/self/status").read_text().splitlines()
            if ":" in line
        )
        if (
            int(status["NoNewPrivs"]) != 1
            or int(status["Seccomp"]) != 2
            or int(status["CapEff"], 16) != 0
        ):
            raise RuntimeError("worker requires no-new-privileges, seccomp and no capabilities")
        if any(name != "lo" for _, name in socket.if_nameindex()):
            raise RuntimeError("worker requires network isolation")
        if role == "cpu" and (
            any(Path("/dev").glob("nvidia*"))
            or Path("/dev/dri").exists()
            or Path("/dev/kfd").exists()
        ):
            raise RuntimeError("CPU worker requires no GPU devices")
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError("cannot verify worker restrictions") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--role", choices=("cpu", "renderer"), required=True)
    parser.add_argument("--evidence", type=Path, default=Path("/evidence"))
    parser.add_argument("--uploads", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("/scratch"))
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    args = vars(parser.parse_args(argv))
    path = args.pop("socket")
    try:
        protocol.validate_request(
            dict(
                version=1,
                request_id="0" * 32,
                generation=args["generation"],
                operation="capability",
                source=None,
                parameters={"hwaccel": "none"},
            )
        )
        if not 0 < args["max_output_bytes"] <= 2**50:
            raise ValueError("invalid output ceiling")
        validate_runtime(args["role"])
        validate_ipc_directory(path.parent)
        # Never inherit application routing or credentials into parser imports or
        # ffmpeg children. Device visibility is provisioned by the host runtime.
        os.environ.clear()
        os.environ.update(
            PATH="/usr/local/bin:/usr/bin:/bin",
            HOME="/nonexistent",
            TMPDIR=str(args["scratch"]),
            LANG="C.UTF-8",
        )
        signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
        with bound_listener(path) as listener:
            while True:
                channel, _ = listener.accept()
                with channel, channel.makefile("rwb", buffering=0) as stream:
                    serve_connection(stream, **args)
                    channel.shutdown(socket.SHUT_WR)
    except (OSError, ValueError, RuntimeError):
        parser.exit(1, "video worker startup or runtime failure\n")


if __name__ == "__main__":
    main()
