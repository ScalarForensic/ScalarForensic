"""In-memory session store for the web query pipeline."""

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileEntry:
    file_id: str
    filename: str  # original name, may include relative path from webkitdirectory
    temp_path: Path
    file_hash: str | None = None
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

    async with _session_lock:
        # Clean up previous session temp files
        if _current_session_id and _current_session_id in _store:
            old = _store.pop(_current_session_id)
            for entry in old.files:
                try:
                    entry.temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        session = Session(session_id=str(uuid.uuid4()))
        _store[session.session_id] = session
        _current_session_id = session.session_id

    return session


def get_session(session_id: str) -> Session | None:
    return _store.get(session_id)
