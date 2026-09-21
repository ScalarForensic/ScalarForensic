"""Fail-closed Unix-socket client. It never invokes a local decoder."""

from __future__ import annotations

import hashlib
import io
import math
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scalar_forensic.media_worker import protocol

if TYPE_CHECKING:
    from scalar_forensic.media_worker.receipts import ReceiptLog


class WorkerUnavailable(RuntimeError):
    """Worker identity/transport failure; replace the generation before reuse."""


class DeadlineStream:
    """One total deadline, including slow/trickled payloads and terminal EOF."""

    def __init__(self, channel: socket.socket, deadline: float):
        self.channel = channel
        self.deadline = deadline

    def _timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("worker deadline exceeded")
        self.channel.settimeout(remaining)

    def read(self, size: int) -> bytes:
        self._timeout()
        return self.channel.recv(size)

    def write(self, data: bytes) -> int:
        self._timeout()
        self.channel.sendall(data)
        return len(data)


@dataclass(frozen=True)
class DecodedResult:
    request: protocol.Request
    response: protocol.Response


class WorkerClient:
    def __init__(
        self,
        socket_path: Path,
        generation: str,
        *,
        timeout: float = 30,
        receipt_log: ReceiptLog | None = None,
        failure_notice: Path | None = None,
    ):
        if not isinstance(generation, str) or re.fullmatch(r"[0-9a-f]{32}", generation) is None:
            raise ValueError("worker generation must be a 32-character lowercase hex identity")
        if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("worker timeout must be finite and positive")
        if receipt_log is not None and receipt_log.generation != generation:
            raise ValueError("receipt log has a different generation")
        self.socket_path = Path(socket_path)
        self.generation = generation
        self.timeout = timeout
        self.receipt_log = receipt_log
        self.failure_notice = failure_notice
        self._invalidated = False
        self._state_lock = threading.Lock()

    def _check_live(self):
        with self._state_lock:
            if self.failure_notice is not None:
                try:
                    self.failure_notice.lstat()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    self._invalidated = True
                    raise WorkerUnavailable("cannot verify worker generation status") from exc
                else:
                    self._invalidated = True
            if self._invalidated:
                raise WorkerUnavailable("worker generation invalidated; replace it before reuse")

    def _record_worker_failure(self, request):
        with self._state_lock:
            self._invalidated = True
        if self.failure_notice is not None:
            # This directory is shared only by the trusted app and supervisor.
            self.failure_notice.touch(mode=0o600, exist_ok=True)
        if self.receipt_log is not None:
            self.receipt_log.append(request, "worker_failed")

    def decode(self, data: bytes, operation: str, cap: int = 331) -> DecodedResult:
        self._check_live()
        if type(data) is not bytes or not 0 < len(data) <= protocol.MAX_SOURCE:
            raise protocol.ProtocolError("source size/type is outside worker limits")
        request = protocol.Request(
            uuid.uuid4().hex, hashlib.sha256(data).hexdigest(), operation, cap, len(data)
        )
        # Validate all caller-controlled fields before contacting a worker. Do not
        # buffer the source again just to validate its header.
        header = {
            "version": protocol.VERSION,
            "request_id": request.request_id,
            "source_sha256": request.source_sha256,
            "operation": operation,
            "cap": cap,
            "source_bytes": len(data),
            "max_image_pixels": request.max_image_pixels,
        }
        protocol.validate_request_header(header)
        encoded = io.BytesIO()
        protocol.write_header(encoded, header)
        if self.receipt_log is not None:
            self.receipt_log.append(request, "started")
        deadline = time.monotonic() + self.timeout
        try:
            with socket.socket(socket.AF_UNIX) as channel:
                channel.settimeout(self.timeout)
                channel.connect(str(self.socket_path))
                stream = DeadlineStream(channel, deadline)
                stream.write(encoded.getvalue())
                stream.write(data)
                channel.shutdown(socket.SHUT_WR)
                response = protocol.read_response(stream, request, self.generation)
            self._check_live()
            if self.receipt_log is not None:
                self.receipt_log.append(
                    request,
                    "decoded",
                    pixel_hashes=[
                        hashlib.sha256(image.pixels).hexdigest() for image in response.images
                    ],
                )
            return DecodedResult(request, response)
        except protocol.DecodeError as exc:
            if str(exc) != "invalid_image":
                self._record_worker_failure(request)
                raise WorkerUnavailable(
                    "media worker reported a resource or internal failure"
                ) from exc
            if self.receipt_log is not None:
                self.receipt_log.append(request, "decode_failed")
            raise
        except (OSError, protocol.ProtocolError, WorkerUnavailable) as exc:
            self._record_worker_failure(request)
            if isinstance(exc, WorkerUnavailable):
                raise
            raise WorkerUnavailable("media worker unavailable or returned invalid data") from exc
        except (KeyboardInterrupt, SystemExit) as exc:
            try:
                self._record_worker_failure(request)
            except Exception as receipt_error:
                exc.add_note(f"Could not record worker failure: {receipt_error}")
            raise
