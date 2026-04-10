"""Tests for the in-memory session store (scalar_forensic.web.session)."""

import asyncio
import tempfile
from pathlib import Path

import pytest

import scalar_forensic.web.session as session_module
from scalar_forensic.web.session import FileEntry, create_session, get_session


def _reset_store() -> None:
    """Reset module-level state between tests."""
    session_module._store.clear()
    session_module._current_session_id = None


@pytest.fixture(autouse=True)
def clean_store():
    _reset_store()
    yield
    _reset_store()


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_create_session_basic():
    session = asyncio.run(create_session())
    assert session.session_id
    assert len(session.session_id) == 36  # UUID4 canonical form
    assert session.files == []
    assert get_session(session.session_id) is session


def test_get_session_unknown_returns_none():
    assert get_session("does-not-exist") is None


def test_create_session_replaces_previous():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "img.jpg"
        tmp_path.write_bytes(b"fake")

        first = asyncio.run(create_session())
        first.files.append(FileEntry(file_id="f1", filename="img.jpg", temp_path=tmp_path))
        first_id = first.session_id

        second = asyncio.run(create_session())

        assert second.session_id != first_id
        # Old session removed from store
        assert get_session(first_id) is None
        assert get_session(second.session_id) is second
        # Current session ID updated
        assert session_module._current_session_id == second.session_id


def test_create_session_cleans_up_temp_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_file = Path(tmp) / "to_delete.jpg"
        tmp_file.write_bytes(b"data")

        first = asyncio.run(create_session())
        first.files.append(FileEntry(file_id="f1", filename="to_delete.jpg", temp_path=tmp_file))
        assert tmp_file.exists()

        asyncio.run(create_session())  # should unlink tmp_file
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# Concurrency safety (issue #19)
# ---------------------------------------------------------------------------


def test_create_session_concurrent_single_winner():
    """Concurrent create_session() calls must leave exactly one live session."""

    async def _run():
        sessions = await asyncio.gather(*[create_session() for _ in range(8)])
        return sessions

    results = asyncio.run(_run())

    # All calls returned a Session object
    assert len(results) == 8
    # Exactly one session ID is recorded as current
    assert session_module._current_session_id is not None
    # The store has exactly one entry (all others were cleaned up)
    assert len(session_module._store) == 1
    # That entry matches _current_session_id
    assert session_module._current_session_id in session_module._store
