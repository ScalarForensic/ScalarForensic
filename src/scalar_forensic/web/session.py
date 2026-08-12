"""In-memory session store for the web query pipeline."""

import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class Session:
    session_id: str
    files: list[FileEntry] = field(default_factory=list)
    temp_dir: Path | None = None
    last_access: float = field(default_factory=time.monotonic)


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


async def delete_session(session_id: str) -> None:
    """Remove a session from the store and clean up its temp files."""
    stale_paths: list[Path] = []
    stale_temp_dir: Path | None = None

    async with _session_lock:
        session = _store.pop(session_id, None)
        if session is not None:
            stale_paths = [e.temp_path for e in session.files]
            stale_temp_dir = session.temp_dir

    for path in stale_paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete temp file %s", path)

    if stale_temp_dir is not None:
        shutil.rmtree(stale_temp_dir, ignore_errors=True)


async def reap_idle_sessions(max_idle_seconds: int) -> None:
    """Delete sessions that have been idle for longer than *max_idle_seconds*."""
    now = time.monotonic()
    victims: list[str] = []
    stale_data: list[tuple[list[Path], Path | None]] = []

    async with _session_lock:
        for sid, session in list(_store.items()):
            if now - session.last_access > max_idle_seconds:
                victims.append(sid)
                stale_data.append(([e.temp_path for e in session.files], session.temp_dir))
        for sid in victims:
            del _store[sid]

    for sid, (paths, temp_dir) in zip(victims, stale_data):
        logger.info("Reaping idle session %s", sid)
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to delete temp file %s", path)
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
