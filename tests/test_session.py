"""Tests for the in-memory session store (scalar_forensic.web.session)."""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

import scalar_forensic.web.session as session_module
from scalar_forensic.web.session import (
    FileEntry,
    create_session,
    delete_session,
    get_session,
    reap_idle_sessions,
)


def _reset_store() -> None:
    """Reset module-level state between tests."""
    session_module._store.clear()


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


def test_create_session_independent():
    """Multiple sessions coexist — a new session does not evict earlier ones."""
    first = asyncio.run(create_session())
    second = asyncio.run(create_session())

    assert first.session_id != second.session_id
    assert get_session(first.session_id) is first
    assert get_session(second.session_id) is second
    assert len(session_module._store) == 2


def test_get_session_updates_last_access():
    session = asyncio.run(create_session())
    before = session.last_access
    time.sleep(0.01)
    got = get_session(session.session_id)
    assert got is not None
    assert got.last_access > before


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


def test_delete_session_removes_from_store():
    session = asyncio.run(create_session())
    sid = session.session_id
    asyncio.run(delete_session(sid))
    assert get_session(sid) is None
    assert sid not in session_module._store


def test_delete_session_cleans_up_temp_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_file = Path(tmp) / "to_delete.jpg"
        tmp_file.write_bytes(b"data")

        session = asyncio.run(create_session())
        session.files.append(FileEntry(file_id="f1", filename="to_delete.jpg", temp_path=tmp_file))
        assert tmp_file.exists()

        asyncio.run(delete_session(session.session_id))
        assert not tmp_file.exists()


def test_delete_session_cleans_up_temp_dir():
    with tempfile.TemporaryDirectory() as outer:
        session_tmp = Path(outer) / "session_upload_dir"
        session_tmp.mkdir()
        sentinel = session_tmp / "uploaded.jpg"
        sentinel.write_bytes(b"data")

        session = asyncio.run(create_session())
        session.temp_dir = session_tmp
        assert session_tmp.exists()

        asyncio.run(delete_session(session.session_id))
        assert not session_tmp.exists()


def test_delete_session_unknown_is_noop():
    asyncio.run(delete_session("nonexistent-id"))  # must not raise


# ---------------------------------------------------------------------------
# reap_idle_sessions
# ---------------------------------------------------------------------------


def test_reap_idle_sessions_removes_expired():
    session = asyncio.run(create_session())
    session.last_access = time.monotonic() - 200
    asyncio.run(reap_idle_sessions(100))
    assert get_session(session.session_id) is None


def test_reap_idle_sessions_keeps_active():
    session = asyncio.run(create_session())
    asyncio.run(reap_idle_sessions(3600))
    assert get_session(session.session_id) is session


def test_reap_idle_sessions_partial():
    """Only idle sessions are reaped; active ones survive."""
    idle = asyncio.run(create_session())
    idle.last_access = time.monotonic() - 200
    active = asyncio.run(create_session())

    asyncio.run(reap_idle_sessions(100))

    assert get_session(idle.session_id) is None
    assert get_session(active.session_id) is active


# ---------------------------------------------------------------------------
# max_active_sessions cap
# ---------------------------------------------------------------------------


def test_create_session_max_active_enforced():
    asyncio.run(create_session(max_active=2))
    asyncio.run(create_session(max_active=2))
    with pytest.raises(RuntimeError, match="Session limit reached"):
        asyncio.run(create_session(max_active=2))


def test_create_session_no_cap_by_default():
    for _ in range(5):
        asyncio.run(create_session())
    assert len(session_module._store) == 5


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------


def test_create_session_concurrent_all_survive():
    """All concurrent create_session() calls produce independent live sessions."""

    async def _run():
        return await asyncio.gather(*[create_session() for _ in range(8)])

    results = asyncio.run(_run())

    assert len(results) == 8
    assert len(session_module._store) == 8
    for s in results:
        assert get_session(s.session_id) is s
