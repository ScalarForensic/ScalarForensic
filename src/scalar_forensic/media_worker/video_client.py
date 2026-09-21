"""Bounded application receiver for a single supervised video slot."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import socket
import tempfile
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from scalar_forensic.media_worker import video_protocol as wire
from scalar_forensic.media_worker.client import DeadlineStream, WorkerUnavailable
from scalar_forensic.media_worker.supervisor import private_file
from scalar_forensic.media_worker.video_runtime import read_state


class Cancellation:
    """Nonblocking cancellation; the RPC thread confirms sandbox removal."""

    def __init__(self):
        self.event = threading.Event()
        self.lock = threading.RLock()
        self.state = None
        self.channel = None
        self.notified = False

    def cancel(self):
        with self.lock:
            self.event.set()
            if self.state is not None and not self.notified:
                os.close(private_file(self.state.failure_notice, os.O_WRONLY | os.O_CREAT))
                self.notified = True
            if self.channel is not None:
                with contextlib.suppress(OSError):
                    self.channel.shutdown(socket.SHUT_RDWR)

    def check(self):
        from scalar_forensic.video_playback.encode import Cancelled

        if self.event.is_set():
            raise Cancelled("isolated video request cancelled")


def _identity(path):
    st = path.stat()
    return dict(
        device=st.st_dev,
        inode=st.st_ino,
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        ctime_ns=st.st_ctime_ns,
    )


def _strings(value, maximum=256):
    wire._require(type(value) is list and len(value) <= maximum)
    for item in value:
        wire._require(type(item) is str and len(item) <= 4096)


def validate_result(operation, body, params, frames):
    """Check operation-specific metadata before exposing it or publishing bytes."""
    if operation == "frames":
        wire._keys(body, "")
    elif operation == "frame_at":
        wire._keys(body, "found")
        wire._require(type(body["found"]) is bool and body["found"] == (frames == 1))
    elif operation == "rewrap":
        wire._keys(body, "skipped_streams timestamp_repairs")
        _strings(body["skipped_streams"])
        wire._integer(body["timestamp_repairs"], 0)
    elif operation == "capability":
        wire._keys(body, "ffmpeg_path ffmpeg_version encoder hwaccel tonemap_ok notes")
        wire._require(body["encoder"] in (None, "libx264", "h264_nvenc"))
        wire._require(body["hwaccel"] in ("none", "cuda"))
        wire._require(type(body["tonemap_ok"]) is bool)
        _strings(body["notes"])
        for key in ("ffmpeg_path", "ffmpeg_version"):
            wire._require(body[key] is None or type(body[key]) is str and len(body[key]) <= 4096)
        wire._require(type(body["ffmpeg_path"]) is str)
    elif operation == "encode":
        from scalar_forensic.video_playback.capability import (
            _RATE_CONTROL,
            AUDIO_ARGS,
            Pipeline,
            _video_filter,
        )

        wire._keys(body, "pipeline command wall_seconds fell_back fallback_reason runtime")
        wire._require(body["runtime"] is None)
        wire._require(type(body["fell_back"]) is bool)
        wire._number(body["wall_seconds"], 0, 86400)
        wire._require(
            body["fallback_reason"] is None
            or type(body["fallback_reason"]) is str
            and len(body["fallback_reason"]) <= 16384
        )
        _strings(body["command"], 512)
        wire._require(bool(body["command"]))
        p = body["pipeline"]
        wire._keys(
            p,
            "hwaccel decoder filter_chain encoder rate_control output_height "
            "chunk_seconds audio ffmpeg_version",
        )
        wire._require(p["encoder"] in _RATE_CONTROL)
        wire._require(p["hwaccel"] == ("cuda" if p["encoder"] == "h264_nvenc" else "none"))
        wire._require(p["decoder"] == "software")
        wire._integer(p["output_height"], 144, 16384)
        wire._integer(p["chunk_seconds"], 1, 600)
        wire._require(
            p["output_height"] == params["output_height"]
            and p["chunk_seconds"] == params["chunk_seconds"]
        )
        wire._require(
            p["filter_chain"] == _video_filter(params["output_height"], hdr=params["hdr"])
        )
        wire._require(
            p["rate_control"] == list(_RATE_CONTROL[p["encoder"]])
            and p["audio"] == list(AUDIO_ARGS)
        )
        wire._require(type(p["ffmpeg_version"]) is str and len(p["ffmpeg_version"]) <= 4096)
        p = {**p, "audio": tuple(p["audio"]), "rate_control": tuple(p["rate_control"])}
        return {**body, "pipeline": Pipeline(**p)}
    else:
        numeric = (
            {"duration_s", "bit_rate", "width", "height", "fps", "frame_count"}
            if operation == "info"
            else {"duration_ms", "bit_rate", "video_width", "video_height"}
        )
        strings = (
            {"codec", "error"}
            if operation == "info"
            else {
                "container_brand",
                "video_codec",
                "audio_codec",
                "video_pix_fmt",
                "video_profile",
                "video_color_trc",
                "video_color_primaries",
                "format",
                "video_codec_tag",
                "probe_error",
            }
        )
        wire._require(
            set(body)
            <= numeric | strings | ({"skipped_streams"} if operation == "stream_report" else set())
        )
        if operation == "stream_report":
            wire._require(
                {
                    "container_brand",
                    "video_codec",
                    "audio_codec",
                    "video_pix_fmt",
                    "video_profile",
                    "video_width",
                    "video_height",
                    "bit_rate",
                    "video_color_trc",
                    "video_color_primaries",
                    "skipped_streams",
                }
                <= set(body)
            )
        for key, value in body.items():
            if key == "skipped_streams":
                _strings(value)
            elif value is not None and key in numeric:
                wire._number(value, 0, 2**63 - 1)
                if key not in ("duration_s", "fps"):
                    wire._integer(value, 0)
            elif value is not None:
                wire._require(type(value) is str and len(value) <= 4096)
    return body


class VideoClient:
    def __init__(self, state_path, case_id, receipts, *, queue_timeout=5, replacement_timeout=20):
        self.state_path = Path(state_path)
        self.case_id = case_id
        self.receipts = Path(receipts)
        self.queue_timeout = queue_timeout
        self.replacement_timeout = replacement_timeout

    def _receipt(self, row):
        data = (json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if len(data) > 16384:
            raise WorkerUnavailable("video receipt exceeds limit")
        fd = private_file(self.receipts, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            first_append = os.fstat(fd).st_size == 0
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                if not written:
                    raise OSError("short video receipt write")
                remaining = remaining[written:]
            os.fsync(fd)
            if first_append:
                directory = os.open(self.receipts.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            os.close(fd)

    def _retire(self, state):
        os.close(private_file(state.failure_notice, os.O_WRONLY | os.O_CREAT))
        until = time.monotonic() + self.replacement_timeout
        while time.monotonic() < until:
            try:
                if read_state(self.state_path, self.case_id).generation != state.generation:
                    return
            except WorkerUnavailable:
                pass  # Missing state is not confirmation of physical removal.
            time.sleep(0.05)
        raise WorkerUnavailable("video worker removal could not be confirmed")

    def call(self, operation, source=None, parameters=None, **kwargs):
        events = self.stream(operation, source, parameters, **kwargs)
        try:
            for kind, value in events:
                if kind == "result":
                    return value
        finally:
            events.close()
        raise WorkerUnavailable("missing video result")

    def stream(
        self,
        operation,
        source=None,
        parameters=None,
        *,
        destination=None,
        cancellation=None,
        on_progress=None,
        settings=None,
    ):
        params = parameters or {}
        handle = cancellation or Cancellation()
        handle.check()
        state = read_state(self.state_path, self.case_id)
        expected_policy = (
            "sfn-video-nvenc-v1" if operation in ("capability", "encode") else "sfn-video-cpu-v1"
        )
        if state.policy != expected_policy:
            raise WorkerUnavailable("video slot policy mismatch")
        try:
            fd = private_file(state.lock_path, os.O_RDWR)
        except (OSError, ValueError) as exc:
            raise WorkerUnavailable("video slot lock unavailable") from exc
        part = None
        active = None
        bound = False
        terminal = False
        row = None
        try:
            until = time.monotonic() + self.queue_timeout
            while True:
                handle.check()
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= until:
                        raise WorkerUnavailable("video worker slot busy")
                    time.sleep(0.025)
            fresh = read_state(self.state_path, self.case_id)
            if fresh.lock_path != state.lock_path or fresh.policy != expected_policy:
                raise WorkerUnavailable("video slot binding changed")
            locked = os.fstat(fd)
            current = fresh.lock_path.stat(follow_symlinks=False)
            if (locked.st_dev, locked.st_ino) != (current.st_dev, current.st_ino):
                raise WorkerUnavailable("video slot lock replaced")
            state = fresh
            active = state.lock_path.parent / "active"
            if active.exists() or state.failure_notice.exists():
                raise WorkerUnavailable("video generation pending replacement")
            path, relative, identity, digest = None, None, None, None
            if source is not None:
                from scalar_forensic.video_playback.digest import (
                    _cached_source_digest,
                    _source_digest,
                )

                path = Path(source).resolve(strict=True)
                root = state.evidence_root.resolve(strict=True)
                uploads = getattr(state, "uploads_root", None)
                uploads = uploads.resolve(strict=True) if uploads is not None else None
                if not path.is_file():
                    raise WorkerUnavailable("video source is outside the configured evidence mount")
                if uploads is not None and path.is_relative_to(uploads):
                    relative = "__uploads__/" + path.relative_to(uploads).as_posix()
                elif path.is_relative_to(root):
                    relative = path.relative_to(root).as_posix()
                    if relative.split("/", 1)[0] == "__uploads__":
                        raise WorkerUnavailable("reserved video source prefix")
                else:
                    raise WorkerUnavailable("video source is outside the configured evidence mount")
                identity = _identity(path)
                digest = (
                    _cached_source_digest
                    if operation in ("info", "stream_report")
                    else _source_digest
                )(path, settings)
                if _identity(path) != identity:
                    raise WorkerUnavailable("video source changed while hashing")
            request = dict(
                version=1,
                request_id=uuid.uuid4().hex,
                generation=state.generation,
                operation=operation,
                source=relative,
                parameters=params,
            )
            wire.validate_request(request)
            row = dict(
                version=1,
                case_id=self.case_id,
                generation=state.generation,
                image_id=state.image_id,
                policy=state.policy,
                request_id=request["request_id"],
                operation=operation,
                source_sha256=digest,
                source_path=relative,
                source_identity=identity,
                started=time.time(),
                outcome="started",
                result_digest=None,
            )
            self._receipt(row)
            with tempfile.NamedTemporaryFile(dir=active.parent, mode="w", delete=False) as pending:
                pending_path = Path(pending.name)
                try:
                    json.dump(
                        dict(
                            request_id=request["request_id"],
                            generation=state.generation,
                            started=time.monotonic(),
                        ),
                        pending,
                    )
                    pending.flush()
                    os.fsync(pending.fileno())
                    os.replace(pending_path, active)
                finally:
                    pending_path.unlink(missing_ok=True)
            bound = True
            with socket.socket(socket.AF_UNIX) as channel:
                with handle.lock:
                    handle.state, handle.channel = state, channel
                    handle.check()
                timeout = min(params.get("timeout", state.job_timeout), state.job_timeout)
                stream = DeadlineStream(channel, time.monotonic() + timeout)
                channel.settimeout(timeout)
                channel.connect(str(state.socket_path))
                wire.send_packet(stream, request)
                channel.shutdown(socket.SHUT_WR)
                frames, artifact, received = 0, None, 0
                frame_hashes = set()
                hasher = hashlib.sha256()
                output = None
                try:
                    while True:
                        handle.check()
                        header, payload = wire.recv_packet(stream)
                        wire._require(
                            header["generation"] == state.generation
                            and header["request_id"] == request["request_id"],
                            "video event identity mismatch",
                        )
                        kind, body = header["type"], header["body"]
                        if kind == "frame":
                            wire._require(
                                operation in ("frames", "frame_at")
                                and body["frame_index"] == frames
                            )
                            maximum = (
                                (params["max_frames"] or 1000000) if operation == "frames" else 1
                            )
                            wire._require(frames < maximum)
                            from scalar_forensic.video import ExtractedFrame, _frame_pixel_hash

                            image = Image.frombytes("RGB", (body["width"], body["height"]), payload)
                            wire._require(
                                _frame_pixel_hash(image) == body["frame_hash"],
                                "frame digest mismatch",
                            )
                            wire._require(body["frame_hash"] not in frame_hashes, "duplicate frame")
                            frame_hashes.add(body["frame_hash"])
                            hasher.update(bytes.fromhex(body["frame_hash"]))
                            hasher.update(body["timecode_ms"].to_bytes(8, "little", signed=True))
                            if operation == "frame_at":
                                wire._require(body["timecode_ms"] == params["timecode_ms"])
                            frames += 1
                            yield (
                                "frame",
                                ExtractedFrame(
                                    image,
                                    body["timecode_ms"],
                                    body["frame_index"],
                                    body["frame_hash"],
                                ),
                            )
                        elif kind == "progress":
                            wire._require(operation == "encode" and artifact is None)
                            if on_progress is not None:
                                from scalar_forensic.video_playback.encode import Progress

                                on_progress(Progress(**body))
                        elif kind == "artifact":
                            wire._require(
                                operation in ("encode", "rewrap")
                                and artifact is None
                                and destination is not None
                            )
                            wire._require(
                                body["size"]
                                <= min(
                                    params.get("limit_bytes") or state.max_output_bytes,
                                    state.max_output_bytes,
                                )
                            )
                            artifact = body
                            destination = Path(destination)
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            part_fd, name = tempfile.mkstemp(
                                prefix=destination.name + ".",
                                suffix=".part",
                                dir=destination.parent,
                            )
                            part = Path(name)
                            output = os.fdopen(part_fd, "wb")
                        elif kind == "data":
                            wire._require(
                                artifact is not None and received + len(payload) <= artifact["size"]
                            )
                            output.write(payload)
                            hasher.update(payload)
                            received += len(payload)
                        elif kind in ("result", "error"):
                            wire._require(stream.read(1) == b"", "trailing video data")
                            if kind == "error":
                                from scalar_forensic.video_playback.encode import (
                                    CeilingExceeded,
                                    EncodeError,
                                )

                                if body["kind"] == "ceiling":
                                    raise CeilingExceeded(
                                        body["message"],
                                        written_bytes=body["written_bytes"],
                                        limit_bytes=body["limit_bytes"],
                                    )
                                if body["kind"] == "encode":
                                    raise EncodeError(
                                        body["message"],
                                        command=[],
                                        stderr=body["stderr"],
                                        returncode=body["returncode"],
                                        timed_out=body["timed_out"],
                                    )
                                raise WorkerUnavailable(
                                    "video worker reported " + body["kind"] + " failure"
                                )
                            result = validate_result(operation, body, params, frames)
                            if operation in ("encode", "rewrap"):
                                wire._require(
                                    artifact is not None
                                    and received == artifact["size"]
                                    and hasher.hexdigest() == artifact["sha256"],
                                    "artifact digest or length mismatch",
                                )
                            if path is not None and _identity(path) != identity:
                                raise WorkerUnavailable("video source changed during request")
                            self._receipt(
                                {
                                    **row,
                                    "outcome": "validated",
                                    "result_digest": artifact["sha256"]
                                    if artifact
                                    else hasher.hexdigest()
                                    if frames
                                    else None,
                                }
                            )
                            with handle.lock:
                                handle.check()
                                if state.failure_notice.exists():
                                    raise WorkerUnavailable(
                                        "video generation invalidated before publication"
                                    )
                                if output is not None:
                                    from scalar_forensic.video_playback.cache import publish

                                    output.close()
                                    output = None
                                    publish(part, destination)
                                if operation == "encode":
                                    result["runtime"] = {
                                        key: row[key]
                                        for key in (
                                            "generation",
                                            "image_id",
                                            "policy",
                                            "request_id",
                                            "source_sha256",
                                            "source_path",
                                            "source_identity",
                                        )
                                    }
                                terminal = True
                                handle.state = None
                            yield "result", result
                            return
                        else:
                            raise wire.ProtocolError("unexpected video event")
                finally:
                    if output is not None:
                        output.close()
        except BaseException as exc:
            if bound and not terminal:
                self._retire(state)
                bound = False
                active.unlink(missing_ok=True)
                if row is not None:
                    self._receipt(
                        {**row, "outcome": "cancelled" if handle.event.is_set() else "failed"}
                    )
                # Socket shutdown deliberately interrupts recv with EOF or an
                # OSError. Only confirmed removal makes that a cancellation.
                handle.check()
            if isinstance(exc, (OSError, wire.ProtocolError, ValueError, TypeError)):
                raise WorkerUnavailable(
                    "video worker unavailable or returned invalid data"
                ) from exc
            raise
        finally:
            with handle.lock:
                handle.channel = None
                handle.state = None
            if part is not None:
                part.unlink(missing_ok=True)
            if terminal and active is not None:
                active.unlink(missing_ok=True)
            os.close(fd)
