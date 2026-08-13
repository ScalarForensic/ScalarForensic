"""The bounded on-disk store of viewing copies (spec §6).

The *artifact* cache — files this subsystem wrote and may delete.  The
persistent hash cache is a different thing and lives in
:mod:`scalar_forensic.video_playback.digest`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

from fastapi import HTTPException

from scalar_forensic.config import Settings

_log = logging.getLogger(__name__)

# One lock per source digest: two viewers opening the same clip must not race to
# write the same cache entry.
_remux_locks: dict[str, asyncio.Lock] = {}


def _evict_cache(cache_dir: Path, max_bytes: int, keep: Path) -> int:
    """Delete least-recently-served viewing copies until the cache fits *max_bytes*.

    Returns the number of files deleted.  *keep* is never evicted — it is the
    copy the current request is about to serve.  Recency is the file mtime,
    which :func:`_touch` refreshes on every cache hit.

    Scope is deliberately narrow: only top-level ``{sha256}.mp4`` rewraps, the
    artifacts this function created.  A bare ``*.mp4`` glob would match — and
    delete — the ``full.mp4`` and chunk files later phases put in the same store,
    including one mid-play.  The whole-tree accounting, per-video eviction and
    playback leases that store needs are spec §6.2, phase 5; this is containment
    until then, not that rewrite.
    """
    if max_bytes <= 0:
        return 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for f in cache_dir.glob("*.mp4"):
        if not re.fullmatch(r"[0-9a-f]{64}", f.stem):
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        total += st.st_size
        entries.append((st.st_mtime, st.st_size, f))
    if total <= max_bytes:
        return 0
    deleted = 0
    for _mtime, size, f in sorted(entries):
        if total <= max_bytes:
            break
        if f == keep:
            continue
        try:
            f.unlink()
        except OSError:  # pragma: no cover - racing eviction
            continue
        total -= size
        deleted += 1
        _log.info("viewing-copy cache: evicted %s (%d bytes)", f.name, size)
    return deleted


def _touch(p: Path) -> None:
    try:
        os.utime(p)
    except OSError:  # pragma: no cover - best effort recency bump
        pass


def _cache_dir_or_503(settings: Settings) -> Path:
    if settings.video_cache_dir is None:
        raise HTTPException(
            status_code=503,
            detail=("Playback of rewrapped containers is disabled: SFN_VIDEO_CACHE_DIR is unset"),
        )
    return settings.video_cache_dir
