"""Reused CPU parser generation; deployment owns replacement and containment."""

from __future__ import annotations

import argparse
import errno
import math
import os
import queue
import re
import socket
import stat
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from scalar_forensic import media_decode
from scalar_forensic.media_worker import protocol
from scalar_forensic.media_worker.client import DeadlineStream


class ResourceLimit(ValueError):
    """Decoded output cannot fit the bounded transport."""


def dispatch(request: protocol.Request, data: bytes, generation: str) -> protocol.Response:
    """Decode only source bytes; validate all sizes before materializing payloads."""
    if request.max_image_pixels != Image.MAX_IMAGE_PIXELS:
        raise protocol.ProtocolError("application and worker image pixel limits differ")
    operation, cap = request.operation, request.cap
    metadata = {}
    if operation == "preprocess":
        rows = [("embedding", media_decode._cap_short_side(media_decode._open_rgb(data, cap), cap))]
    elif operation == "open_rgb":
        rows = [("embedding", media_decode._open_rgb(data, cap))]
    elif operation == "native":
        rows = [("native", media_decode.load_for_detection(data))]
    elif operation == "shared":
        native, embedding = media_decode.decode_shared(data, cap)
        rows = [("native", native)]
        if embedding is not None:
            rows.append(("embedding", embedding))
    elif operation == "display_rgb":
        rows = [("embedding", media_decode.display_rgb(data))]
    elif operation == "thumbnail":
        rows = [("embedding", media_decode.thumbnail_input(data, cap))]
    elif operation in ("exif", "exif_detailed"):
        helper = (
            media_decode.extract_exif if operation == "exif" else media_decode.extract_exif_detailed
        )
        metadata = helper(data)
        rows = []
    else:
        raise protocol.ProtocolError("unknown operation")
    descriptions = []
    total = 0
    for role, image in rows:
        if isinstance(image, Image.Image):
            width, height = image.size
            mode = image.mode
        else:
            height, width, channels = image.shape
            if channels != 3 or image.dtype.name != "uint8":
                raise RuntimeError("invalid decoder array")
            mode = "RGB"
        if mode != "RGB" and not (operation == "thumbnail" and mode == "L"):
            raise RuntimeError("invalid decoder mode")
        total += width * height * (1 if mode == "L" else 3)
        if (
            not (0 < width <= protocol.MAX_DIMENSION and 0 < height <= protocol.MAX_DIMENSION)
            or total > protocol.MAX_RGB
        ):
            raise ResourceLimit("RGB output exceeds limit")
        descriptions.append((role, width, height, mode, image))
    images = tuple(
        protocol.RGBImage(role, width, height, image.tobytes(), mode=mode)
        for role, width, height, mode, image in descriptions
    )
    return protocol.Response(generation, images, metadata)


def _validate_limits(
    generation, concurrency, max_queued, request_timeout, max_requests, max_seconds
):
    if type(generation) is not str or re.fullmatch(r"[0-9a-f]{32}", generation) is None:
        raise ValueError("generation must be 32 lowercase hex characters")
    for value, minimum, maximum in (
        (concurrency, 1, 64),
        (max_queued, 0, 64),
        (max_requests, 1, 10_000_000),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("invalid worker count limit")
    for value in (request_timeout, max_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 86400:
            raise ValueError("worker deadlines must be finite and bounded")


def serve(
    listener: socket.socket,
    *,
    generation: str,
    concurrency: int = 4,
    max_queued: int = 4,
    request_timeout: float = 30,
    max_requests: int = 10000,
    max_seconds: float = 3600,
) -> int:
    """Serve an already-bound listener; return 1 if this generation is suspect.

    Deadlines include admission queue time. Daemon threads allow process exit even
    when a decoder is stuck; callers must terminate the process after this returns.
    The deployment must independently enforce its process wall-time boundary.
    """
    _validate_limits(
        generation, concurrency, max_queued, request_timeout, max_requests, max_seconds
    )
    media_decode.register_heif()
    jobs = queue.Queue(maxsize=concurrency + max_queued)
    active = {}
    lock = threading.Lock()
    failed = threading.Event()
    stop = threading.Event()

    def handle(channel, deadline):
        with channel:
            stream = DeadlineStream(channel, deadline)
            request, data = protocol.read_request(stream)
            try:
                result = dispatch(request, data, generation)
            except (protocol.ProtocolError, TimeoutError):
                raise
            except (ResourceLimit, MemoryError):
                failed.set()
                protocol.write_response(stream, request, generation, error="resource_limit")
            except OSError as exc:
                # Pillow reports malformed streams as OSError without errno.
                # A system errno means the runtime failed, not merely this image.
                error = "invalid_image"
                if exc.errno is not None:
                    failed.set()
                    error = (
                        "resource_limit"
                        if exc.errno
                        in (
                            errno.ENOMEM,
                            errno.EMFILE,
                            errno.ENFILE,
                            errno.ENOSPC,
                            errno.EDQUOT,
                            errno.ENOBUFS,
                            errno.EAGAIN,
                        )
                        else "worker_error"
                    )
                protocol.write_response(stream, request, generation, error=error)
            except (
                ValueError,
                SyntaxError,
                Image.DecompressionBombError,
            ):
                # Only decode exceptions enter this boundary. Socket/protocol I/O
                # remains outside and must retire the generation on any failure.
                protocol.write_response(stream, request, generation, error="invalid_image")
            else:
                protocol.write_response(stream, request, generation, result.images, result.metadata)
            channel.shutdown(socket.SHUT_WR)

    def worker():
        while not stop.is_set():
            try:
                channel, deadline = jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                handle(channel, deadline)
            except BaseException:
                failed.set()
                channel.close()
            finally:
                with lock:
                    active.pop(channel, None)
                jobs.task_done()

    for _ in range(concurrency):
        threading.Thread(target=worker, daemon=True).start()
    end = time.monotonic() + max_seconds
    admitted = 0
    listener.settimeout(0.02)
    try:
        while True:
            now = time.monotonic()
            with lock:
                expired = [channel for channel, deadline in active.items() if deadline <= now]
                for channel in expired:
                    failed.set()
                    try:
                        channel.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    channel.close()
                    active.pop(channel, None)
                remaining = bool(active)
            retiring = failed.is_set() or admitted >= max_requests or now >= end
            if retiring:
                listener.close()
                if not remaining:
                    return int(failed.is_set())
                stop.wait(0.01)
                continue
            try:
                channel, _ = listener.accept()
            except TimeoutError:
                continue
            with lock:
                if failed.is_set() or len(active) >= concurrency + max_queued:
                    channel.close()
                    continue
                deadline = time.monotonic() + request_timeout
                active[channel] = deadline
                admitted += 1
            jobs.put_nowait((channel, deadline))
    finally:
        stop.set()
        listener.close()


@contextmanager
def bound_listener(path: Path):
    """Never overwrite a path, and unlink only the socket inode we created."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError("worker socket path already exists")
    identity = None
    with socket.socket(socket.AF_UNIX) as listener:
        try:
            listener.bind(str(path))
            info = path.lstat()
            identity = (info.st_dev, info.st_ino)
            path.chmod(0o600)
            listener.listen(8)
            yield listener
        finally:
            if identity is not None:
                try:
                    info = path.lstat()
                    if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == identity:
                        path.unlink()
                except FileNotFoundError:
                    pass


def validate_runtime() -> None:
    """Basic Linux restrictions; these checks are not kernel denial evidence."""
    if os.geteuid() == 0:
        raise RuntimeError("worker requires nonroot identity")
    try:
        status = dict(
            line.split(":", 1)
            for line in Path("/proc/self/status").read_text().splitlines()
            if ":" in line
        )
        if (
            int(status["NoNewPrivs"].strip()) != 1
            or int(status["Seccomp"].strip()) != 2
            or int(status["CapEff"].strip(), 16) != 0
        ):
            raise RuntimeError("worker requires no-new-privileges, seccomp and no capabilities")
        if any(name != "lo" for _, name in socket.if_nameindex()):
            raise RuntimeError("worker requires network isolation")
        if (
            any(Path("/dev").glob("nvidia*"))
            or Path("/dev/dri").exists()
            or Path("/dev/kfd").exists()
        ):
            raise RuntimeError("worker requires no GPU devices")
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError("cannot verify worker restrictions") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-queued", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--max-requests", type=int, default=10000)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = vars(parser.parse_args(argv))
    path = args.pop("socket")
    try:
        _validate_limits(**args)
        validate_runtime()
        with bound_listener(path) as listener:
            return serve(listener, **args)
    except (OSError, ValueError, RuntimeError):
        parser.exit(1, "media worker startup or runtime failure\n")


if __name__ == "__main__":
    # main has already closed/unlinked our listener. Do not run interpreter or
    # third-party atexit handlers while a timed-out native decoder may still run.
    os._exit(main())
