"""In-memory session store for the web query pipeline."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scalar_forensic.faces.provenance import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoFrameEntry:
    """Stores per-frame embeddings for a video uploaded through the web UI."""

    frame_index: int
    timecode_ms: int
    frame_hash: str
    sscd_embedding: list[float] | None = None
    dino_embedding: list[float] | None = None


@dataclass
class QueryFace:
    """A face detected in an *uploaded* image.

    Session-scoped: it is never written to Qdrant and never written to the face
    chip store (spec §8 — the web analyze path is ephemeral by design).  A face
    that fails the embedding gate carries ``vector=None``; the review-only
    exclusion guarantee (spec §6.2) is structural on the query side too, exactly
    as the vectorless point is on the index side.  Never give one a vector and
    never replace that with a flag consulted at search time.
    """

    index: int
    bbox: tuple[float, float, float, float]
    landmarks: list[list[float]]
    det_conf: float
    detect_scale: float
    quality: dict[str, float | None]
    embedding_status: str  # "embedded" | "review_only"
    embedding_exclusion_reason: str | None
    vector: list[float] | None
    review_jpeg: bytes | None


@dataclass
class FileEntry:
    file_id: str
    filename: str  # original name, may include relative path from webkitdirectory
    temp_path: Path
    file_hash: str | None = None
    file_hash_md5: str | None = None
    sscd_embedding: list[float] | None = None
    dino_embedding: list[float] | None = None
    error: str | None = None
    is_video: bool = False
    video_frames: list[VideoFrameEntry] | None = None
    # Populated by POST /api/faces/query-faces; in-process only.
    query_faces: list[QueryFace] | None = None
    # The PipelineConfig those faces were produced under.  Kept so the search
    # endpoint gates comparability against the config that actually made the
    # probe vectors, rather than re-deriving one from the current environment.
    query_faces_cfg: PipelineConfig | None = None
    query_faces_timecode_ms: int | None = None
    _query_faces_generation: int = 0


@dataclass
class Session:
    session_id: str
    files: list[FileEntry] = field(default_factory=list)
    temp_dir: Path | None = None
    last_access: float = field(default_factory=time.monotonic)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    active_workers: int = 0
    _work_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def begin_work(self) -> None:
        with self._work_lock:
            if self.cancelled.is_set():
                raise RuntimeError("Session was cancelled")
            self.active_workers += 1

    def end_work(self) -> None:
        with self._work_lock:
            self.active_workers -= 1
            self.last_access = time.monotonic()
            if self.cancelled.is_set() and self.active_workers == 0:
                _cleanup(self)

    def retire(self) -> None:
        with self._work_lock:
            self.cancelled.set()
            if self.active_workers == 0:
                _cleanup(self)


_store: dict[str, Session] = {}
_session_lock: asyncio.Lock = asyncio.Lock()


async def create_session(max_active: int = 0) -> Session:
    """Create and store a new independent session.

    When *max_active* is > 0 and the store is already at capacity, raises
    RuntimeError — callers should translate this to HTTP 503.
    """
    async with _session_lock:
        if max_active > 0 and len(_store) >= max_active:
            raise RuntimeError(f"Session limit reached ({max_active} active sessions)")
        session = Session(session_id=str(uuid.uuid4()))
        _store[session.session_id] = session

    return session


def get_session(session_id: str) -> Session | None:
    session = _store.get(session_id)
    if session is not None:
        session.last_access = time.monotonic()
    return session


def _cleanup(session: Session) -> None:
    for entry in session.files:
        try:
            entry.temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete temp file %s", entry.temp_path)
    if session.temp_dir is not None:
        shutil.rmtree(session.temp_dir, ignore_errors=True)


async def delete_session(session_id: str) -> None:
    """Cancel work and clean files once the last active worker releases them."""
    async with _session_lock:
        session = _store.pop(session_id, None)
    if session is not None:
        session.retire()


async def reap_idle_sessions(max_idle_seconds: int) -> None:
    """Reap idle sessions; an active inference worker holds its session alive."""
    now = time.monotonic()
    async with _session_lock:
        victims = [
            session
            for session in _store.values()
            if session.active_workers == 0 and now - session.last_access > max_idle_seconds
        ]
        for session in victims:
            del _store[session.session_id]
    for session in victims:
        session.retire()


async def shutdown_sessions() -> None:
    """Signal cooperative shutdown; active workers perform their final cleanup."""
    for session_id in list(_store):
        await delete_session(session_id)


_temp_root = Path(tempfile.gettempdir()) / "scalar-forensic-sessions"
_owned_roots: dict[Path, tuple[Path, int]] = {}


def sweep_temp_root(root: Path) -> None:
    """Remove only recognized owned directories whose process lease is unlocked.

    A live process holds flock for its entire lifetime, avoiding PID reuse and
    cross-worker deletion. Unknown names, symlinks and missing locks are ignored.
    """
    if not root.exists():
        return
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        if not re.fullmatch(r"sfn-process-[0-9a-f]{32}", directory.name):
            continue
        try:
            descriptor = os.open(directory / ".owner", os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue
            shutil.rmtree(directory)
        except OSError:
            logger.warning("Could not recover temporary directory %s", directory)
        finally:
            os.close(descriptor)


def initialize_temp_root(root: Path) -> None:
    """Configure the upload root and recover directories left by exited workers."""
    global _temp_root
    _temp_root = Path(root).resolve()
    mode = _upload_directory_mode()
    _temp_root.mkdir(mode=mode, parents=True, exist_ok=True)
    if mode == 0o750:
        _temp_root.chmod(mode)
    sweep_temp_root(_temp_root)


def _upload_directory_mode() -> int:
    state_path = os.environ.get("SFN_VIDEO_CPU_STATE")
    if not state_path:
        return 0o700
    from scalar_forensic.media_worker.video_runtime import read_state

    state = read_state(Path(state_path), os.environ.get("SFN_VIDEO_WORKER_CASE", ""))
    if state.uploads_root is None or state.uploads_root.resolve() != _temp_root.resolve():
        raise RuntimeError("managed video worker must map the configured upload root")
    return 0o750  # Distinct parser UID may read via the trusted operator's group.


def prepare_upload_permissions(path: Path) -> None:
    if _upload_directory_mode() == 0o750:
        path.chmod(0o640)


def session_temp_dir() -> Path:
    """Allocate within this process's locked namespace (never the generic /tmp)."""
    root = _temp_root.resolve()
    mode = _upload_directory_mode()
    if root not in _owned_roots:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory = root / ("sfn-process-" + uuid.uuid4().hex)
        staging = Path(tempfile.mkdtemp(prefix=".initializing-", dir=root))
        staging.chmod(mode)
        descriptor = os.open(staging / ".owner", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        staging.rename(directory)
        # Intentionally keep the descriptor open for the process lifetime.
        _owned_roots[root] = directory, descriptor
    directory = Path(tempfile.mkdtemp(prefix="session-", dir=_owned_roots[root][0]))
    directory.chmod(mode)
    return directory
