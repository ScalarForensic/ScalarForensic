"""The bounded on-disk store of viewing copies (spec §6).

The *artifact* cache — files this subsystem wrote and may delete.  The
persistent hash cache is a different thing and lives in
:mod:`scalar_forensic.video_playback.digest`.

Layout, one directory per source video::

    {cache_dir}/{source_digest}/rewrap.mp4        the §2 lossless stream copy
    {cache_dir}/{source_digest}/{key}/full.mp4    the §4.3 whole-file encode
    {cache_dir}/{source_digest}/{key}/c{start}.mp4    a §4.2 chunk
    {cache_dir}/{source_digest}/{key}/*.part      in flight, counted (§6.3)

``key`` is :func:`cache_key` — ``sha256(source identity ‖ pipeline
fingerprint)`` (§6.1).  The pipeline is not in the *rewrap* path because a
rewrap runs no pipeline: it moves coded packets untouched, so its only identity
is the source's.

**Why the source digest is a directory and not just half of a hash.**  §6.2
requires eviction of *whole videos*, never a single chunk out of a video being
watched.  Grouping on disk makes that the natural operation — one video is one
directory — instead of an inference over opaque key names.  A video encoded
under two pipelines holds two ``{key}/`` subdirectories and is still one
eviction unit, which is right: they are two renderings of one piece of evidence.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import HTTPException

from scalar_forensic.config import Settings

_log = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def cache_key(source_digest: str, fingerprint: str) -> str:
    """The §6.1 artifact key: ``sha256(source identity ‖ pipeline fingerprint)``.

    Both halves are fixed-length lower-case hex, so the concatenation is already
    unambiguous; the field names are hashed with them anyway so the pre-image is
    self-describing when it turns up in a log or a §7.2 label.

    Pass the fingerprint of the pipeline that **ran**, never the one that was
    selected: :attr:`~.encode.EncodeResult.pipeline` differs from the request on
    a §8 GPU fallback, and keying on the request would file a CPU encode under
    the GPU's key — one key holding two pictures, the exact defect §6.1 exists
    to prevent.

    **``chunk_seconds`` stays in the key for both artifact kinds** (the decision
    §6.1 left to phase 5).  It is pixel-affecting for a chunk — it moves every
    encode boundary — and not for the §4.3 ``full.mp4``, so one fingerprint means
    changing ``SFN_VIDEO_CHUNK_SECONDS`` needlessly re-encodes full copies.  That
    cost is paid rarely (§16 forecloses changing chunk length without revisiting
    the resolution cap in the same change) and is recoverable by re-encoding.
    The alternative — a fingerprint that covers different fields per artifact
    kind — makes "is this setting in the key?" a question with two answers
    forever, and replaces ``Pipeline``'s one rule ("every field is hashed") with
    a table someone has to keep correct.  A rare recoverable cost beats a
    permanent second rule.
    """
    if not _SHA256_RE.fullmatch(source_digest):
        raise ValueError(f"source digest is not a sha256 hex digest: {source_digest!r}")
    if not _SHA256_RE.fullmatch(fingerprint):
        raise ValueError(f"pipeline fingerprint is not a sha256 hex digest: {fingerprint!r}")
    pre = f"source={source_digest}\npipeline={fingerprint}"
    return hashlib.sha256(pre.encode("utf-8")).hexdigest()


def video_dir(cache_dir: Path, source_digest: str) -> Path:
    """The one directory holding every artifact derived from one source file."""
    if not _SHA256_RE.fullmatch(source_digest):
        raise ValueError(f"source digest is not a sha256 hex digest: {source_digest!r}")
    return cache_dir / source_digest


def rewrap_path(cache_dir: Path, source_digest: str) -> Path:
    return video_dir(cache_dir, source_digest) / "rewrap.mp4"


def artifact_dir(cache_dir: Path, source_digest: str, fingerprint: str) -> Path:
    return video_dir(cache_dir, source_digest) / cache_key(source_digest, fingerprint)


def chunk_name(start: float) -> str:
    """``c{start}.mp4`` with a fixed rendering, so one window is one filename."""
    return f"c{start:.3f}.mp4"


# ---------------------------------------------------------------------------
# Locks that do not accumulate (§10.4)
# ---------------------------------------------------------------------------


class KeyedLocks:
    """One :class:`asyncio.Lock` per key, held only while somebody wants it.

    The dict this replaces grew one entry per source digest ever seen and was
    never cleared (§10.4: "locks must not accumulate unboundedly").  Counting
    waiters rather than bounding the dict means the size is exactly the number of
    in-flight callers — no cap to tune and no eviction that could hand two
    callers different locks for the same key.

    Single process only.  §10.4 requires that assumption be stated: this
    deduplicates work within one ASGI worker.  Two worker processes can both
    encode the same artifact; both publish atomically (§10.2) to the same path,
    so the result is wasted CPU, never a corrupt file.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._waiters: dict[str, int] = {}

    @asynccontextmanager
    async def hold(self, key: str):
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._waiters[key] - 1
            if remaining:
                self._waiters[key] = remaining
            else:
                del self._waiters[key]
                del self._locks[key]

    def __len__(self) -> int:
        return len(self._locks)

    def reset(self) -> None:
        """Drop every lock (tests only — a held lock would be lost)."""
        self._locks.clear()
        self._waiters.clear()


artifact_locks = KeyedLocks()


# ---------------------------------------------------------------------------
# Playback leases (§6.2)
# ---------------------------------------------------------------------------

# HTTP is stateless: between two chunk requests of one video the server has no
# way to know the video is still on screen.  A lease is that knowledge made
# explicit — registered by the player, refreshed by a heartbeat, and expiring on
# its own so a closed tab or a crashed browser cannot pin the cache forever.
_leases: dict[str, float] = {}
_pins: dict[str, int] = {}
_lease_lock = threading.Lock()


@dataclass(frozen=True)
class LeaseState:
    """Three-state on purpose: an absent lease is not an expired one.

    ``state`` is ``"held"`` (registered and unexpired), ``"expired"`` (was
    registered, the heartbeat stopped) or ``"none"`` (never registered here).
    A boolean would answer "is it protected?" and silently give the same answer
    to "nobody is watching" and "we have no idea" — and the second is what a
    fresh process says about a video another worker is serving.
    """

    video: str
    state: str
    seconds_remaining: float | None


def renew_lease(source_digest: str, ttl_seconds: float) -> LeaseState:
    """Register or refresh the playback lease on one video."""
    if ttl_seconds <= 0:
        raise ValueError("lease ttl must be > 0")
    with _lease_lock:
        _leases[source_digest] = time.monotonic() + ttl_seconds
    return lease_state(source_digest)


def release_lease(source_digest: str) -> LeaseState:
    """Drop the lease — the player closed the video rather than stopped beating."""
    with _lease_lock:
        _leases.pop(source_digest, None)
    return lease_state(source_digest)


def lease_state(source_digest: str) -> LeaseState:
    with _lease_lock:
        expiry = _leases.get(source_digest)
    if expiry is None:
        return LeaseState(source_digest, "none", None)
    remaining = expiry - time.monotonic()
    if remaining <= 0:
        return LeaseState(source_digest, "expired", 0.0)
    return LeaseState(source_digest, "held", remaining)


def _leased_now() -> set[str]:
    now = time.monotonic()
    with _lease_lock:
        expired = [d for d, e in _leases.items() if e <= now]
        for d in expired:
            del _leases[d]
        return set(_leases)


@contextmanager
def pin(source_digest: str) -> Iterator[None]:
    """Protect a video for the duration of an in-flight write (§6.2).

    The lease covers *readers*: a `FileResponse` streams its body after the
    handler has returned, so the request cannot bracket the read.  This covers
    *writers*: an encode or rewrap holding a ``.part`` under a video that LRU
    would otherwise pick, which would delete the directory out from under it.
    """
    with _lease_lock:
        _pins[source_digest] = _pins.get(source_digest, 0) + 1
    try:
        yield
    finally:
        with _lease_lock:
            remaining = _pins[source_digest] - 1
            if remaining:
                _pins[source_digest] = remaining
            else:
                del _pins[source_digest]


def protected_videos() -> set[str]:
    """Every video eviction must not touch: leased (read) or pinned (write)."""
    with _lease_lock:
        pinned = set(_pins)
    return _leased_now() | pinned


def reset_leases() -> None:
    """Tests only.  Module state outlives a test the way the probe cache does."""
    with _lease_lock:
        _leases.clear()
        _pins.clear()


# ---------------------------------------------------------------------------
# Accounting and eviction (§6.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoEntry:
    """One eviction unit: everything derived from one source file."""

    video: str
    path: Path
    size_bytes: int
    last_used: float
    legacy: bool = False


@dataclass(frozen=True)
class EvictionReport:
    videos_removed: int = 0
    bytes_freed: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    ceiling_bytes: int = 0
    protected: tuple[str, ...] = field(default_factory=tuple)
    over_ceiling: bool = False


def _tree_size_and_mtime(root: Path) -> tuple[int, float]:
    """Bytes and newest mtime under *root*, ``.part`` files included (§6.3)."""
    total = 0
    newest = 0.0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                st = os.stat(os.path.join(dirpath, name))
            except OSError:  # pragma: no cover - racing eviction or purge
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return total, newest


def scan(cache_dir: Path) -> list[VideoEntry]:
    """Every video in the store, newest use first.

    Recency is the newest mtime anywhere under the video, so serving one chunk
    makes the whole video recent — which is what "least-recently-*played*" means
    when a play touches one file out of forty.

    Top-level ``{sha256}.mp4`` files are the layout this store used before the
    per-video directories.  They are counted and evictable as their own entry
    rather than deleted on sight: they are still valid rewraps, and letting LRU
    retire them keeps a version change from throwing away a warm cache.
    """
    entries: list[VideoEntry] = []
    try:
        children = sorted(cache_dir.iterdir())
    except OSError:
        return entries
    for child in children:
        name = child.name
        if child.is_dir():
            if not _SHA256_RE.fullmatch(name):
                continue
            size, mtime = _tree_size_and_mtime(child)
            entries.append(VideoEntry(name, child, size, mtime))
        elif child.is_file() and name.endswith(".mp4") and _SHA256_RE.fullmatch(child.stem):
            try:
                st = child.stat()
            except OSError:  # pragma: no cover - racing eviction
                continue
            entries.append(VideoEntry(child.stem, child, st.st_size, st.st_mtime, legacy=True))
    entries.sort(key=lambda e: e.last_used, reverse=True)
    return entries


def _remove(entry: VideoEntry) -> None:
    if entry.legacy:
        entry.path.unlink(missing_ok=True)
    else:
        shutil.rmtree(entry.path, ignore_errors=True)


def evict(cache_dir: Path, max_bytes: int, *, protect: set[str] | None = None) -> EvictionReport:
    """Bring the store under *max_bytes* by removing whole videos, LRU first.

    Replaces the ``*.mp4`` glob `#148` narrowed as containment.  Three things
    that glob could not do and this does (§6.2): it accounts for the **whole
    tree**, including chunk directories and in-flight ``.part`` files; it removes
    a **whole video** rather than picking single files, so a video can never lose
    a chunk mid-play; and it refuses to touch a video that is leased or pinned.

    ``max_bytes <= 0`` disables eviction entirely — the store is unbounded by
    operator choice, and silently deleting everything would be the opposite of
    what that setting says.

    When every unprotected video is gone and the store is still over the
    ceiling, the overshoot is **reported and logged**, not resolved by evicting
    something in use.  §6.3 keeps the ceiling an invariant by refusing jobs that
    would not fit *before* they run; a lease that is already over is the residue
    of a smaller ceiling or of artifacts written before the refusal, and dropping
    a video an analyst is watching to fix it trades a bounded overshoot for a
    broken playback.
    """
    entries = scan(cache_dir)
    total = sum(e.size_bytes for e in entries)
    if max_bytes <= 0:
        return EvictionReport(bytes_before=total, bytes_after=total, ceiling_bytes=max_bytes)
    keep = protected_videos() if protect is None else set(protect)
    if total <= max_bytes:
        return EvictionReport(
            bytes_before=total,
            bytes_after=total,
            ceiling_bytes=max_bytes,
            protected=tuple(sorted(keep)),
        )
    removed = 0
    freed = 0
    for entry in reversed(entries):  # oldest first
        if total <= max_bytes:
            break
        if entry.video in keep:
            continue
        _remove(entry)
        total -= entry.size_bytes
        freed += entry.size_bytes
        removed += 1
        _log.info(
            "viewing-copy cache: evicted video %s (%d bytes, last used %.0f)",
            entry.video,
            entry.size_bytes,
            entry.last_used,
        )
    over = total > max_bytes
    if over:
        _log.warning(
            "viewing-copy cache: %d bytes over the %d-byte ceiling after evicting "
            "everything unprotected; %d video(s) are leased or pinned",
            total - max_bytes,
            max_bytes,
            len(keep),
        )
    return EvictionReport(
        videos_removed=removed,
        bytes_freed=freed,
        bytes_before=total + freed,
        bytes_after=total,
        ceiling_bytes=max_bytes,
        protected=tuple(sorted(keep)),
        over_ceiling=over,
    )


# ---------------------------------------------------------------------------
# The ceiling refusal (§6.3)
# ---------------------------------------------------------------------------

# Operator ruling (§16): a full-video job whose estimated output exceeds half the
# ceiling is refused before it starts.  Half, not all, because the store has to
# hold more than one video to be a cache at all.
CEILING_FRACTION = 0.5


@dataclass(frozen=True)
class CeilingVerdict:
    """Whether a full-video job may start.  Three-state, and the third matters.

    ``fits`` — the estimate is under the limit.  ``refused`` — it is over.
    ``unknown`` — the source did not yield the numbers the estimate needs, so
    there is no verdict to state.

    ``unknown`` **refuses the job too**, and is kept distinct from ``refused``
    anyway, because the two say different things to the analyst: one is "this
    video is too big for the cache", the other is "this file would not tell us
    how big it is". Collapsing them into a boolean would print the first
    sentence for the second condition — a claim about the video derived from a
    failure to read it.
    """

    state: str
    estimate_bytes: int | None
    limit_bytes: int
    reason: str | None

    @property
    def allowed(self) -> bool:
        return self.state == "fits"


def estimate_full_output_bytes(info: dict, output_height: int) -> int | None:
    """Estimate the §4.3 full copy's size from the source's measured bitrate.

    ``info`` is a :func:`~.codecs._stream_report`.  ``None`` when the container
    does not carry the duration, the bitrate or the coded height — the estimate
    is required (§6.3), so a missing input is reported as "cannot estimate" and
    never replaced with a guess.

    The scale is the **area** ratio of the §16 output cap, since the cap is
    ``min(ih, H)`` and never upscales: a 2160p source encoded at 1080p carries a
    quarter of the pixels, a 720p source is passed through untouched at 1.0.

    **The estimate is uncalibrated and says so.** It assumes the re-encode's
    bits-per-pixel matches the source's, which is not measured anywhere: §3.5
    timed the encodes and recorded no output sizes, so there is no ratio in this
    repository to apply and none is invented here (§16 forbids inventing a floor;
    the same rule applies to a factor). The direction of the error is knowable
    even if its size is not — a CRF-23 H.264 encode of a 10-bit HEVC source at
    the same resolution is usually *larger* than the source, so this estimate
    runs low on exactly the HEVC corpus this feature exists for. **A job runner
    must therefore check the growing ``.part`` against this number and abort on
    overshoot rather than trusting it** (phase 7; nothing in phase 5 runs a job).
    """
    duration_ms = info.get("duration_ms")
    bit_rate = info.get("bit_rate")
    height = info.get("video_height")
    if not duration_ms or not bit_rate or not height:
        return None
    area_ratio = (min(height, output_height) / height) ** 2
    return int(bit_rate * area_ratio * (duration_ms / 1000.0) / 8)


def check_ceiling(settings: Settings, info: dict) -> CeilingVerdict:
    """Decide whether a full-video job may start (§6.3, §16).

    The ceiling is the invariant: §6.2 cannot hold "never exceed the ceiling",
    "evict whole videos" and "never evict the video being played" at once for a
    single video that does not fit, so the job that would break it is refused
    *before* it runs rather than resolved afterwards by deleting something in
    use.  Chunk playback is unaffected — chunks are bounded by
    ``SFN_VIDEO_CHUNK_SECONDS`` — and Download original (§7.5) is the answer the
    analyst is given.
    """
    ceiling = settings.video_cache_max_bytes
    if ceiling <= 0:
        # No ceiling configured means no invariant to protect.
        return CeilingVerdict("fits", estimate_full_output_bytes(info, 1080), 0, None)
    limit = int(ceiling * CEILING_FRACTION)
    estimate = estimate_full_output_bytes(info, settings.video_output_height)
    if estimate is None:
        return CeilingVerdict(
            "unknown",
            None,
            limit,
            (
                "The size of a full viewing copy cannot be estimated for this file: "
                "its container does not report a duration, a bitrate and a frame "
                "height. A full copy is not started without an estimate, because "
                "one video larger than the cache would evict everything else and "
                "still not fit. Play it in chunks, or download the original."
            ),
        )
    if estimate > limit:
        return CeilingVerdict(
            "refused",
            estimate,
            limit,
            (
                f"A full viewing copy of this video is estimated at "
                f"{estimate / 1024**3:.1f} GiB, over the {limit / 1024**3:.1f} GiB "
                f"a single rendering may occupy ({int(CEILING_FRACTION * 100)}% of "
                f"SFN_VIDEO_CACHE_MAX_BYTES). It is not started: one video that "
                "fills the cache would evict every other. Play it in chunks, or "
                "download the original."
            ),
        )
    return CeilingVerdict("fits", estimate, limit, None)


# ---------------------------------------------------------------------------
# Purge (§13)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PurgeReport:
    videos: int
    files: int
    bytes_freed: int
    digests: tuple[str, ...]


def purge(cache_dir: Path, *, media: str | None = None, all_: bool = False) -> PurgeReport:
    """Delete derived renderings.  Exactly one of *media* or *all_* (§13).

    The LRU ceiling is the only *automatic* retention mechanism (§6.4); this is
    the explicit one, mirroring ``sfn-faces purge``.  "Derived renderings for
    media X were deleted at time T by examiner Y" is a statement that survives a
    courtroom, which a background TTL sweep is not — which is why v1's TTL was
    dropped in favour of this.

    Nothing here touches a source file: the store holds only viewing copies, and
    a viewing copy is never the evidence (§1, §7).
    """
    if bool(media) == bool(all_):
        raise ValueError("specify exactly one of media=<sha256> or all_=True")
    if media is not None and not _SHA256_RE.fullmatch(media):
        raise ValueError(f"media is not a sha256 hex digest: {media!r}")
    targets = [e for e in scan(cache_dir) if all_ or e.video == media]
    files = 0
    freed = 0
    for entry in targets:
        if entry.legacy:
            files += 1
        else:
            files += sum(len(f) for _d, _sub, f in os.walk(entry.path))
        freed += entry.size_bytes
        _remove(entry)
        _log.info("viewing-copy cache: purged video %s (%d bytes)", entry.video, entry.size_bytes)
    return PurgeReport(
        videos=len(targets),
        files=files,
        bytes_freed=freed,
        digests=tuple(e.video for e in targets),
    )


# ---------------------------------------------------------------------------
# Publication (§10.2)
# ---------------------------------------------------------------------------


def part_path(dst: Path) -> Path:
    """The PID-scoped scratch name for *dst* — two processes share one store."""
    return dst.with_name(f"{dst.name}.{os.getpid()}.part")


def publish(part: Path, dst: Path) -> None:
    """fsync *part*, rename it onto *dst*, fsync the directory (§10.2).

    A truncated artifact must never become a cache hit.  ``os.replace`` is
    atomic, so no reader sees a partial file — but without the fsync the rename
    can reach the disk before the data does, and a host crash in between leaves a
    file that *looks* finished at a name callers treat as verified.  The
    directory fsync is what makes the rename itself durable.
    """
    fd = os.open(part, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(part, dst)
    dir_fd = os.open(dst.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    except OSError as exc:  # pragma: no cover - some filesystems refuse this
        if exc.errno not in (errno.EINVAL, errno.EACCES):
            raise
    finally:
        os.close(dir_fd)


_PART_RE = re.compile(r"\.(\d+)\.part$")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - a live pid owned by someone else
        return True
    except OSError:  # pragma: no cover
        return True
    return True


def sweep_orphaned_parts(cache_dir: Path) -> int:
    """Remove ``.part`` files whose writer is gone (§10.2); return the count.

    SIGKILL and a host crash both skip ``encode``'s cleanup, so the store
    accumulates scratch files that count against the ceiling and are never
    published.  The pid in the name is what separates them from a *live* encode
    in a sibling process, which must not be touched.

    Pid reuse is the known hole: a recycled pid keeps an orphan alive one extra
    sweep.  The consequence is a delayed delete, not a deleted live write, so it
    is documented rather than engineered away.
    """
    swept = 0
    for dirpath, _dirnames, filenames in os.walk(cache_dir):
        for name in filenames:
            m = _PART_RE.search(name)
            if m is None:
                continue
            pid = int(m.group(1))
            if pid == os.getpid() or _pid_alive(pid):
                continue
            try:
                os.unlink(os.path.join(dirpath, name))
            except OSError:  # pragma: no cover - racing sweep
                continue
            swept += 1
            _log.info("viewing-copy cache: swept orphaned part file %s (pid %d)", name, pid)
    return swept


_swept = False
_sweep_lock = threading.Lock()


def ensure_swept(cache_dir: Path) -> None:
    """Run :func:`sweep_orphaned_parts` once per process, on first cache use.

    §10.2 asks for a startup sweep.  It is here and not in ``app.py``'s lifespan
    because this subsystem owns its store (``CLAUDE.md``: ``app.py`` keeps only
    app setup) and because the cache directory is a request-time setting — a
    deployment that never plays a video never has one to sweep.
    """
    global _swept
    with _sweep_lock:
        if _swept:
            return
        _swept = True
    try:
        sweep_orphaned_parts(cache_dir)
    except OSError as exc:  # pragma: no cover - unreadable store is reported later
        _log.warning("viewing-copy cache: part-file sweep failed: %s", exc)


def _reset_sweep() -> None:
    """Tests only."""
    global _swept
    with _sweep_lock:
        _swept = False


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


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
    ensure_swept(settings.video_cache_dir)
    return settings.video_cache_dir
