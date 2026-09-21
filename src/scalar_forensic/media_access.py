"""Image access facade: explicit local mode or audited, fail-closed CPU worker.

The worker imports media_decode directly, never this routing module. Installing
the package alone does not change local processing or establish OS isolation.
"""

from __future__ import annotations

import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from scalar_forensic import media_decode
from scalar_forensic.media_worker import protocol
from scalar_forensic.media_worker.client import WorkerClient, WorkerUnavailable
from scalar_forensic.media_worker.receipts import ReceiptLog
from scalar_forensic.media_worker.runtime import read_state

_ENV = (
    "SFN_MEDIA_WORKER_SOCKET",
    "SFN_MEDIA_WORKER_GENERATION",
    "SFN_MEDIA_WORKER_CASE",
    "SFN_MEDIA_WORKER_RECEIPTS",
)


_clients: dict[str, tuple[tuple, WorkerClient]] = {}
_clients_lock = threading.Lock()


def _client(socket_path, generation, case_id, receipt_path, timeout, state=None):
    socket_path, receipt_path = Path(socket_path), Path(receipt_path)
    if not socket_path.is_absolute() or not receipt_path.is_absolute():
        raise ValueError("worker socket and receipt paths must be absolute")
    if receipt_path.resolve().is_relative_to(socket_path.resolve().parent):
        raise ValueError("receipt log must not be in the worker IPC directory")
    configuration = (socket_path, case_id, receipt_path, float(timeout), state)
    # Keep one immutable binding per generation, including failed generations.
    # Eviction or a changed timeout must not resurrect an invalidated client.
    with _clients_lock:
        existing = _clients.get(generation)
        if existing is not None:
            previous, client = existing
            if previous != configuration:
                raise ValueError("worker generation configuration changed; use a new generation")
            return client
        client = WorkerClient(
            socket_path,
            generation,
            timeout=float(timeout),
            receipt_log=ReceiptLog(
                receipt_path,
                case_id,
                generation,
                runtime=None
                if state is None
                else {"image_id": state.image_id, "policy": state.policy},
            ),
            failure_notice=None if state is None else state.failure_notice,
        )
        _clients[generation] = (configuration, client)
        return client


def configured_worker() -> WorkerClient | None:
    values = tuple(os.environ.get(name, "") for name in _ENV)
    state_path = os.environ.get("SFN_MEDIA_WORKER_STATE", "")
    if "SFN_MEDIA_WORKER_STATE" in os.environ:
        socket_path, generation, case_id, receipt_path = values
        if (
            not state_path
            or any(name in os.environ for name in _ENV[:2])
            or not case_id
            or not receipt_path
        ):
            raise ValueError(
                "managed worker needs CASE/RECEIPTS and cannot use static SOCKET/GENERATION"
            )
        if not Path(state_path).is_absolute():
            raise ValueError("managed worker state path must be absolute")
        state = read_state(Path(state_path), case_id)
        return _client(
            state.socket_path,
            state.generation,
            case_id,
            receipt_path,
            os.environ.get("SFN_MEDIA_WORKER_TIMEOUT", "40"),
            state,
        )
    if not any(name in os.environ for name in _ENV):
        return None
    if not all(values):
        raise ValueError("isolated media access requires socket, generation, case and receipt log")
    return _client(*values, os.environ.get("SFN_MEDIA_WORKER_TIMEOUT", "40"))


def _pil(image):
    return Image.frombytes(image.mode, (image.width, image.height), image.pixels)


def read_source(path: Path) -> bytes:
    """Bound trusted-side allocation before sending a file to the worker."""
    if configured_worker() is None:
        return path.read_bytes()
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= protocol.MAX_SOURCE:
            raise protocol.ProtocolError("source must be a regular file within worker size limits")
        data = stream.read(info.st_size)
        if len(data) != info.st_size or stream.read(1):
            raise protocol.ProtocolError("source size changed during read")
    return data


def _open_rgb(data: bytes, cap: int = 331) -> Image.Image:
    worker = configured_worker()
    if worker is None:
        return media_decode._open_rgb(data, cap)
    return _pil(worker.decode(data, "open_rgb", cap).response.images[0])


def preprocess_batch(image_data: list[bytes], cap: int) -> list[Image.Image | Exception]:
    worker = configured_worker()
    if worker is None:
        return media_decode.preprocess_batch(image_data, cap)

    def decode(data):
        try:
            # A queued batch may outlive a normal generation handover.
            current = configured_worker()
            if current is None:
                raise WorkerUnavailable("worker configuration removed during batch")
            return _pil(current.decode(data, "preprocess", cap).response.images[0])
        except Exception as exc:  # Per-file error contract; no local fallback.
            return exc

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(decode, image_data))


def extract_exif(data: bytes) -> media_decode.ExifInfo:
    worker = configured_worker()
    if worker is None:
        return media_decode.extract_exif(data)
    return worker.decode(data, "exif").response.metadata


def extract_exif_detailed(data: bytes) -> dict:
    worker = configured_worker()
    if worker is None:
        return media_decode.extract_exif_detailed(data)
    return worker.decode(data, "exif_detailed").response.metadata


def load_for_detection(data: bytes) -> np.ndarray:
    worker = configured_worker()
    if worker is None:
        return media_decode.load_for_detection(data)
    image = worker.decode(data, "native").response.images[0]
    return np.frombuffer(image.pixels, dtype=np.uint8).reshape(image.height, image.width, 3)


def decode_shared(data: bytes, cap: int) -> tuple[np.ndarray, Image.Image | None]:
    worker = configured_worker()
    if worker is None:
        return media_decode.decode_shared(data, cap)
    images = worker.decode(data, "shared", cap).response.images
    image = images[0]
    native = np.frombuffer(image.pixels, dtype=np.uint8).reshape(image.height, image.width, 3)
    return native, _pil(images[1]) if len(images) == 2 else None


def display_rgb(data: bytes) -> Image.Image:
    worker = configured_worker()
    if worker is None:
        return media_decode.display_rgb(data)
    return _pil(worker.decode(data, "display_rgb").response.images[0])


def thumbnail_input(data: bytes, cap: int) -> Image.Image:
    worker = configured_worker()
    if worker is None:
        return media_decode.thumbnail_input(data, cap)
    return _pil(worker.decode(data, "thumbnail", cap).response.images[0])
