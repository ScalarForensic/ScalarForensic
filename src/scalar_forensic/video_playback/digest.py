"""The SHA-256 of a source file as it is on disk right now (spec §7.1).

Deliberately not part of :mod:`scalar_forensic.video_playback.cache`: that is
the *artifact* cache (viewing copies on disk), this is the persistent *hash*
cache the indexer fills.  Conflating the two is the drift §6.1 warns about.

The handle is a process-wide singleton holding an open SQLite connection — it
lives in this module and nowhere else.  Importers take the functions, never the
``_hash_cache`` global.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from pathlib import Path

from scalar_forensic.config import Settings
from scalar_forensic.embedder import HashCache, hash_file

_log = logging.getLogger(__name__)

# The persistent hash cache the indexer already fills, opened lazily and shared
# by every request thread.  Keyed by the configured DB path so a settings change
# (tests, a relocated cache) opens a new one instead of serving the old.
_hash_cache_lock = threading.Lock()
_hash_cache: tuple[str, HashCache | None] | None = None


def _reset_hash_cache() -> None:
    """Drop the process-wide HashCache handle (tests, and a settings change)."""
    global _hash_cache
    with _hash_cache_lock:
        if _hash_cache is not None and _hash_cache[1] is not None:
            with contextlib.suppress(Exception):
                _hash_cache[1].close()
        _hash_cache = None


def _hash_cache_for(settings: Settings) -> HashCache | None:
    """Return the shared HashCache, or None when it is disabled or unusable.

    An unwritable or corrupt DB is not a request failure: the digest is simply
    computed the slow way.  The failure is remembered (a None entry) so every
    later request does not retry a broken SQLite file on the request path.
    """
    global _hash_cache
    db_path = settings.hash_cache_path
    if db_path is None:
        return None
    key = str(db_path)
    with _hash_cache_lock:
        if _hash_cache is not None and _hash_cache[0] == key:
            return _hash_cache[1]
        try:
            cache: HashCache | None = HashCache(db_path)
        except (sqlite3.Error, OSError) as exc:
            _log.warning("hash cache unavailable at %s (%s); hashing directly", db_path, exc)
            cache = None
        _hash_cache = (key, cache)
        return cache


def _source_digest(p: Path, settings: Settings | None = None) -> str:
    """SHA-256 of the source file *as it is on disk right now*.

    Backed by the same persistent :class:`HashCache` the indexer fills — keyed on
    ``(resolved path, mtime_ns, size)``, so a touched or rewritten file is
    re-hashed rather than remembered.  The value is never looked up from the
    indexed ``video_hash``: a label beside a rendering must describe the file as
    it is, not as it was indexed (spec §7.1).

    Blocking (a cache miss reads the whole file), so callers must offload it off
    the event loop.
    """
    cache = _hash_cache_for(settings if settings is not None else Settings())
    if cache is None:
        return hash_file(p)
    try:
        digest, was_cached = cache.get_or_hash(p)
    except (sqlite3.Error, OSError) as exc:
        _log.warning("hash cache lookup failed for %s (%s); hashing directly", p, exc)
        return hash_file(p)
    if not was_cached:
        # Persist immediately: unlike an indexing run there is no later flush(),
        # and the point of the cache is to survive the process.  A write failure
        # costs a re-hash next time, never the response.
        try:
            cache.flush()
        except (sqlite3.Error, OSError) as exc:
            _log.warning("hash cache write failed for %s: %s", p, exc)
    return digest


def _cached_source_digest(p: Path, settings: Settings | None = None) -> str | None:
    """The digest of *p* if the HashCache already holds a current one, else None.

    Never hashes.  For handlers that must not pay a whole-file read — None is
    "not computed", not a verdict about the file.
    """
    cache = _hash_cache_for(settings if settings is not None else Settings())
    if cache is None:
        return None
    try:
        return cache.peek(p)
    except (sqlite3.Error, OSError) as exc:  # pragma: no cover - defensive
        _log.debug("hash cache peek failed for %s: %s", p, exc)
        return None
