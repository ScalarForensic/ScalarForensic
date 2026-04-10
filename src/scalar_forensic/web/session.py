"""In-memory session store for the web query pipeline."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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


@dataclass
class Session:
    session_id: str
    files: list[FileEntry] = field(default_factory=list)


_store: dict[str, Session] = {}
_current_session_id: str | None = None
_session_lock: asyncio.Lock = asyncio.Lock()


async def create_session() -> Session:
    """Create a new session, replacing the previous one.

    Thread-safe: uses an asyncio lock so concurrent /api/analyze requests
    cannot observe a partially-cleaned-up session or clobber each other's IDs.
    """
    global _current_session_id

    stale_paths: list[Path] = []

    async with _session_lock:
        # Collect paths while holding the lock, then delete after releasing it
        # so filesystem I/O doesn't block other requests waiting on the lock.
        if _current_session_id and _current_session_id in _store:
            old = _store.pop(_current_session_id)
            stale_paths = [e.temp_path for e in old.files]

        session = Session(session_id=str(uuid.uuid4()))
        _store[session.session_id] = session
        _current_session_id = session.session_id

    for path in stale_paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete temp file %s", path)

    return session


def get_session(session_id: str) -> Session | None:
    return _store.get(session_id)
