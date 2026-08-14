"""The full-video job: admission, the runner, refcounts and cancellation (§4.3, §10.4).

One background encode of a whole source, so an analyst who is going to watch the
lot stops paying a chunk boundary every thirty seconds.  It is long — §3.1's
1080p CPU rate makes a 4-hour source ~51 minutes — and it is **not** free to the
rest of the system, which is the part v1 got wrong and §4.3 says out loud:

    It competes with chunk playback for the same two workers.

`SFN_VIDEO_MAX_WORKERS=2` is the shape of exactly one viewer — the chunk being
played plus its single §4.2 prefetch — so a full job holding one of those two
puts chunk encoding at k=2 for its whole run: §3.5's 8.21 s becomes a measured
18.31 s, outside the 6–10 s window the double-buffered swap depends on.

**The ruling (2026-08-14, phase 7) is both of §4.3's remedies, in this order.**

1. *Yield.*  The job runs niced (``SFN_VIDEO_JOB_NICE``) and thread-capped
   (``SFN_VIDEO_JOB_THREADS``) so chunk work wins the contention rather than
   splitting it.  **This bites on the CPU pipeline only.**  Niceness and
   ``-threads`` reprioritise CPU scheduling and libx264's thread pool; on the GPU
   pipeline the contended resource is the encoder block and its driver queue,
   which neither knob touches — a niced NVENC job still takes its slot.
   **Measured 2026-08-14, CPU/libx264, n=6 per arm: 18.31 s unniced vs 16.83 s
   niced — ~9%, and neither inside §4.2's 6–10 s window**
   (``docs/benchmarks/video-codec-factor-2026-08-14.md``).  Keep the knobs; the
   9% is free.  Do not describe the job as yielding as though that settled it.
2. *Disclose anyway.*  While a job runs, the player says the next chunk will take
   longer (``contention_notice`` below, rendered by ``full_job.js``).  Yield buys
   9% on one path and is unmeasured on the other, so disclosure is the
   load-bearing remedy and the only one that covers both — and doing (1) alone would
   re-create the option §4.3 rules out: bounded but invisible.

This module owns the runner ``routes._Admission`` deliberately was not: a
semaphore plus an admitted count, with no cancellation, no refcounts and no
progress.  The gate itself moves here unchanged, because a full job and a chunk
compete for the same two workers and a bound they do not share is not a bound.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

# `_RateTracker` is the project's one progress estimator, and #139's ruling
# lives in it: an α=½ EWMA labelled as an extrapolation, with the uncalibrated
# ±1σ band deleted.  Importing it is the whole point — a second smoother here
# would be a second answer to "how long is left", and the two would disagree in
# front of an analyst.
from scalar_forensic.cli import _RateTracker
from scalar_forensic.config import Settings
from scalar_forensic.video_playback import audit, states
from scalar_forensic.video_playback.cache import (
    FULL_NAME,
    artifact_dir,
    evict,
    full_path,
    pin,
    relocate_to_pipeline_key,
    renew_lease,
)
from scalar_forensic.video_playback.capability import Capability, Pipeline
from scalar_forensic.video_playback.encode import (
    Cancelled,
    EncodeResult,
    Progress,
    _kill_group,
    encode_full,
    job_threads,
)

_log = logging.getLogger(__name__)


class Admission:
    """The bound on concurrent encodes, and on how many may wait (§10.4).

    Two numbers, not one.  ``SFN_VIDEO_MAX_WORKERS`` caps how many encoders run
    — §3.5 measured aggregate throughput flat from k=1 to k=8, so more is only
    latency.  ``SFN_VIDEO_QUEUE_MAX`` caps *admitted* requests, running plus
    waiting, because an unbounded wait queue is the same unbounded resource with
    a slower fuse: a LAN host that opens forty videos would otherwise hold forty
    requests until each one times out.  Over the cap the request is refused with
    §10.1's ``queue-full`` row, which says how long to wait, rather than joining
    a line nobody is told the length of.

    **A full-video job takes a slot from this same gate**, for its whole run.
    That is §4.3's contention made explicit rather than worked around: with the
    default two workers, one remains for chunk playback while an export runs, and
    with ``SFN_VIDEO_MAX_WORKERS=1`` an export blocks chunk encoding outright —
    which is what a single-worker deployment asked for and is visible in this one
    place instead of being distributed across two competing pools.
    """

    def __init__(self) -> None:
        self._sem: asyncio.Semaphore | None = None
        self._limit = 0
        self.admitted = 0

    def _semaphore(self, workers: int) -> asyncio.Semaphore:
        if self._sem is None or self._limit != workers:
            self._sem = asyncio.Semaphore(workers)
            self._limit = workers
        return self._sem

    @asynccontextmanager
    async def enter(self, settings: Settings) -> AsyncIterator[None]:
        if self.admitted >= settings.video_queue_max:
            raise states.QUEUE_FULL.as_http()
        self.admitted += 1
        try:
            async with self._semaphore(settings.video_max_workers):
                yield
        finally:
            self.admitted -= 1

    def reset(self) -> None:
        """Test hook: the counter is process-wide, like every other bound here."""
        self._sem = None
        self._limit = 0
        self.admitted = 0


admission = Admission()


#: What the analyst is told while an export runs, in the player, next to the
#: spinner it slows down.  §4.3 rules out "bounded but invisible", and the yield
#: above is only bounded on the CPU path — so this sentence is not decoration,
#: it is the half of the remedy that covers every pipeline.
CONTENTION_NOTICE = (
    "A full viewing copy is being produced. It shares this host's two encode workers "
    "with playback, so the next chunk will take longer than usual to appear."
)


#: Shown for the whole life of a job that was started over a §6.3 refusal, and
#: after it, next to the copy it produced.  The ruling of 2026-08-14 requires the
#: override to be **disclosed**, so this travels on every ``view()`` rather than
#: being a one-off message at the moment of the click: an analyst who opens the
#: page later must still be able to see that a capacity gate was set aside, and by
#: whom.  It says what the override did *not* do, because that is the part a
#: reader will otherwise assume wrongly.
OVERRIDE_NOTICE = (
    "The capacity estimate for this full viewing copy was refused and set aside by the "
    "examiner named here. The estimate is advisory; the cache ceiling is not. This export "
    "is still stopped if it passes the size a single rendering may occupy."
)


@dataclass(frozen=True)
class JobRequest:
    """Everything the runner needs, already validated by the route (§9).

    The route resolves and validates the path, probes the container, computes the
    digest and asks :func:`~.cache.check_ceiling`; nothing here re-derives any of
    it, because a second implementation of a path check is the defect
    ``_resolve_video_path`` exists in one module to prevent.
    """

    source: Path
    digest: str
    duration_seconds: float
    hdr: bool
    has_audio: bool
    capability: Capability
    pipeline: Pipeline
    cache_dir: Path
    limit_bytes: int | None
    estimate_bytes: int | None
    #: Set only when this job was started over a §6.3 refusal (ruling
    #: 2026-08-14).  ``overridden_by`` is ``SFN_EXAMINER_ID`` — the route refuses
    #: an unattributable override rather than recording ``None`` — and
    #: ``overridden_verdict`` is the verdict that was set aside, ``refused`` or
    #: ``unknown``.  Both default to "no override", so every other caller and
    #: every existing job is unchanged: this is per request and never a mode.
    overridden_by: str | None = None
    overridden_verdict: str | None = None
    #: ``SFN_EXAMINER_ID`` as it was when this encode was started, or ``None``
    #: when the deployment sets none.  One encode, one audit record, attributed
    #: to whoever started it (§7.3): a job is refcounted, so a second analyst
    #: joining it adds a claimant to that record rather than producing a second
    #: one for an encode that only ran once.
    started_by: str | None = None

    @property
    def override(self) -> dict | None:
        """The disclosure the browser renders, or ``None`` for a normal job.

        ``estimate_bytes`` here is the forecast that was overridden — it is
        ``None`` for an ``unknown`` verdict, which is the whole content of that
        verdict and is reported as such rather than as a zero.
        """
        if self.overridden_by is None:
            return None
        return {
            "examiner_id": self.overridden_by,
            "verdict": self.overridden_verdict,
            "estimate_bytes": self.estimate_bytes,
            "limit_bytes": self.limit_bytes,
            "notice": OVERRIDE_NOTICE,
        }


class FullJob:
    """One running (or finished) full-video encode, and what it will admit to.

    Progress is ffmpeg's own ``-progress`` output, not an interpolation: the
    fraction is media seconds written over the probed duration, and the rate is
    media seconds per wall second.  The ETA is labelled *"at current rate"* and
    is nothing more than that (`#139`).
    """

    def __init__(self, request: JobRequest, settings: Settings) -> None:
        self.request = request
        self.settings = settings
        self.state = "full-job-running"
        self.cancelled = False
        self.waiters = 1
        self.started_at = time.monotonic()
        self.finished_at: float | None = None
        self.frames = 0
        self.out_seconds = 0.0
        self.written_bytes = 0
        self.failure: states.Failure | None = None
        self.result: EncodeResult | None = None
        #: The §7.2 record of what actually ran, set when the encode returns.
        #: ``None`` while the job is running — a label that named a pipeline
        #: before an encoder had produced a frame would be describing a
        #: selection, which is the thing §7.2 exists to stop the label doing.
        self.rendering: audit.Rendering | None = None
        self.artifact: Path | None = None
        self.task: asyncio.Task | None = None
        self._proc: subprocess.Popen | None = None
        self._tracker = _RateTracker()
        self._last_tick = self.started_at
        self._last_out = 0.0

    # -- lifecycle -------------------------------------------------------
    def attach(self, proc: subprocess.Popen) -> None:
        """Hold the encoder so a cancel can stop it — including one that raced.

        A DELETE that arrives between ``Popen`` and this callback would otherwise
        set a flag nobody reads and leave a 51-minute encode running; checking
        the flag here is what closes that window.
        """
        self._proc = proc
        if self.cancelled:
            _kill_group(proc)

    def observe(self, progress: Progress) -> None:
        """One ``-progress`` block.  Runs on the encode's thread, not the loop."""
        now = time.monotonic()
        moved_ms = int((progress.out_seconds - self._last_out) * 1000)
        if moved_ms > 0:
            self._tracker.update(moved_ms, now - self._last_tick)
            self._last_out = progress.out_seconds
            self._last_tick = now
        self.frames = progress.frames
        self.out_seconds = progress.out_seconds
        self.written_bytes = progress.written_bytes

    def cancel(self) -> None:
        """Stop the encoder now (§4.3: "cancellable explicitly")."""
        self.cancelled = True
        if self._proc is not None:
            _kill_group(self._proc)

    # -- what it will admit to -------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    @property
    def fraction(self) -> float | None:
        if self.request.duration_seconds <= 0:
            return None
        return min(1.0, self.out_seconds / self.request.duration_seconds)

    @property
    def eta_seconds(self) -> float | None:
        """Seconds left at the current rate, or ``None`` before there is one.

        ``_RateTracker.eta`` withholds an answer until two observations exist: a
        single block is not a rate, and an ETA drawn from one is a number with no
        content that an analyst will nonetheless plan around.
        """
        remaining_ms = int((self.request.duration_seconds - self.out_seconds) * 1000)
        if remaining_ms <= 0:
            return None
        return self._tracker.eta(remaining_ms)

    def view(self) -> dict:
        """The JSON the status endpoint returns.  The server states every state."""
        eta = self.eta_seconds
        rate = self._tracker.rate
        return {
            "video_sha256": self.request.digest,
            "player_state": self.state,
            "cancelled": self.cancelled,
            "waiters": self.waiters,
            "frames": self.frames,
            "out_seconds": round(self.out_seconds, 3),
            "duration_seconds": round(self.request.duration_seconds, 3),
            "fraction": None if self.fraction is None else round(self.fraction, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "written_bytes": self.written_bytes,
            "estimate_bytes": self.request.estimate_bytes,
            "limit_bytes": self.request.limit_bytes,
            "override": self.request.override,
            # §7.2: the label records the pipeline that *ran*, in full — filter
            # chain with parameters, encoder and rate control, output height,
            # ffmpeg version, the thread cap this copy was encoded under and what
            # became of the audio.  `None` until there is an encode to describe.
            "rendering": None if self.rendering is None else self.rendering.describe(),
            # media seconds encoded per wall second — the same number §3.1 quotes
            # as "2.7× realtime", so the label the analyst reads matches the
            # measurement the spec is argued from.
            "rate": None if rate is None else round(rate / 1000.0, 3),
            "eta_seconds": None if eta is None else round(eta, 1),
            "eta_label": eta_label(eta),
            "contention_notice": CONTENTION_NOTICE if self.state == "full-job-running" else None,
            "full_url": self.full_url(),
            "error": None if self.failure is None else self.failure.as_detail(),
        }

    def full_url(self) -> str | None:
        if self.state != "full-job-done" or self.result is None:
            return None
        from urllib.parse import quote

        return (
            f"/api/video-full?path={quote(str(self.request.source))}"
            f"&fp={self.result.pipeline.fingerprint()}"
        )


def eta_label(seconds: float | None) -> str | None:
    """`#139`'s labelling, verbatim in shape: an extrapolation, said to be one.

    Never a confidence interval and never a completion time — "~4 min remaining
    at current rate" is a claim this project can defend; "done at 14:32" is not.
    """
    if seconds is None:
        return None
    if seconds < 90:
        return f"~{max(1, int(round(seconds)))} s remaining at current rate"
    minutes = int(round(seconds / 60))
    if minutes < 90:
        return f"~{minutes} min remaining at current rate"
    hours = seconds / 3600
    return f"~{hours:.1f} h remaining at current rate"


class JobRunner:
    """One full-video job per source video, per worker process (§10.4).

    Refcounted: two analysts watching the same video share one encode, and a
    cancel from one of them **releases their claim rather than killing the job**
    the other is waiting on.  The encoder stops when the last claim goes.

    Per worker process, and no further: like :class:`~.cache.KeyedLocks`, this is
    an in-process table, so two ASGI workers can run the same export twice.  Both
    publish atomically to the same path (§10.2), so the cost is wasted CPU and
    never a corrupt file.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, FullJob] = {}

    def get(self, digest: str) -> FullJob | None:
        return self._jobs.get(digest)

    def start(self, request: JobRequest, settings: Settings) -> FullJob:
        """Start a job, or join the one already running for this video."""
        existing = self._jobs.get(request.digest)
        if existing is not None and existing.state == "full-job-running":
            existing.waiters += 1
            return existing
        job = FullJob(request, settings)
        self._jobs[request.digest] = job
        job.task = asyncio.create_task(self._run(job))
        return job

    def cancel(self, digest: str) -> str:
        """Drop one claim.  ``cancelled`` when that was the last, else ``released``."""
        job = self._jobs.get(digest)
        if job is None or job.state != "full-job-running":
            return "none"
        job.waiters -= 1
        if job.waiters > 0:
            return "released"
        job.cancel()
        return "cancelled"

    def reset(self) -> None:
        """Test hook.  Kills what is running: the table is process-wide state."""
        for job in list(self._jobs.values()):
            if job.state == "full-job-running":
                job.cancel()
        self._jobs.clear()

    async def _run(self, job: FullJob) -> None:
        request, settings = job.request, job.settings
        dst = full_path(request.cache_dir, request.digest, request.pipeline.fingerprint())
        try:
            async with admission.enter(settings):
                if job.cancelled:
                    raise Cancelled("cancelled before the encoder started")
                # The lease and the pin together are what stop eviction removing
                # the directory this job is writing into, for the ~51 minutes it
                # is writing (§6.2): the lease covers readers, the pin writers.
                renew_lease(request.digest, settings.video_lease_seconds)
                with pin(request.digest):
                    result = await asyncio.to_thread(
                        encode_full,
                        settings,
                        request.capability,
                        request.source,
                        dst,
                        hdr=request.hdr,
                        has_audio=request.has_audio,
                        limit_bytes=request.limit_bytes,
                        on_progress=job.observe,
                        on_start=job.attach,
                    )
                    published = await asyncio.to_thread(
                        relocate_to_pipeline_key,
                        result.path,
                        request.cache_dir,
                        request.digest,
                        result.pipeline.fingerprint(),
                        FULL_NAME,
                    )
                await asyncio.to_thread(evict, request.cache_dir, settings.video_cache_max_bytes)
        except BaseException as exc:  # noqa: BLE001 - every ending is a reported state
            job.finished_at = time.monotonic()
            # §7.3: an encode that ran is recorded whatever became of it.  A
            # cancelled export and a failed one are different outcomes, and a log
            # that held only the successes would answer "what did this examiner
            # produce" while saying nothing about what they started.
            audit.record_transcode(
                settings,
                source=request.source,
                video_sha256=request.digest,
                scope=audit.SCOPE_FULL,
                outcome=audit.OUTCOME_CANCELLED if job.cancelled else audit.OUTCOME_FAILED,
                examiner_id=request.started_by,
                pipeline_fingerprint=request.pipeline.fingerprint(),
                error=None if job.cancelled else f"{type(exc).__name__}: {exc}",
                waiters=job.waiters,
                elapsed_seconds=round(job.elapsed_seconds, 1),
                written_bytes=job.written_bytes,
                override=request.override,
            )
            if job.cancelled:
                # A cancelled encode dies of SIGKILL, which `classify` would read
                # as the OOM killer (§10.1).  It is neither a failure nor a §5
                # state: nothing is running, so the video is back to
                # `needs-transcode` and `cancelled` says why.
                job.state = "needs-transcode"
                _log.info("full-video job for %s cancelled", request.source)
                return
            job.failure = states.classify_full_job(exc)
            job.state = job.failure.state
            _log.warning(
                "full-video job for %s failed (%s): %s", request.source, job.failure.kind, exc
            )
            return
        job.finished_at = time.monotonic()
        job.result = result
        # `result.pipeline`, not `request.pipeline`.  They differ after a §8
        # GPU→CPU fallback, and `request` is the object in scope for most of this
        # method — a record built from it would name the encoder that was
        # selected and not the one that produced the file, which is exactly the
        # false label §7.2 exists to prevent.  `threads` is not a Pipeline field
        # (it changes no pixel the cache key must separate) but libx264's output
        # does depend on it, so a full copy that does not record it cannot be
        # reproduced byte-for-byte on a host with a different core count.
        job.rendering = audit.Rendering.from_result(
            result,
            scope=audit.SCOPE_FULL,
            has_audio=request.has_audio,
            threads=job_threads(settings),
            duration_seconds=request.duration_seconds,
        )
        job.artifact = published
        job.state = "full-job-done"
        audit.record_transcode(
            settings,
            source=request.source,
            video_sha256=request.digest,
            scope=audit.SCOPE_FULL,
            outcome=audit.OUTCOME_SUCCESS,
            examiner_id=request.started_by,
            rendering=job.rendering,
            # Claimants, not records: this encode ran once and is recorded once.
            waiters=job.waiters,
            artifact_path=str(published),
            artifact_bytes=published.stat().st_size if published.exists() else 0,
            elapsed_seconds=round(job.elapsed_seconds, 1),
            estimate_bytes=request.estimate_bytes,
            override=request.override,
        )
        _log.info(
            "full-video job for %s done in %.1f s (%s bytes, estimate %s, override %s)",
            request.source,
            job.elapsed_seconds,
            published.stat().st_size if published.exists() else 0,
            request.estimate_bytes,
            # Closes the audit line the route opened: the forecast that was set
            # aside, next to the bytes that were actually written.  An examiner
            # defending the override has both halves in the log, not just the
            # decision.
            request.overridden_by or "none",
        )


runner = JobRunner()


def artifact_for(cache_dir: Path, digest: str, fingerprint: str) -> Path:
    """Where a finished full copy lies, for the route that serves its bytes."""
    return artifact_dir(cache_dir, digest, fingerprint) / FULL_NAME
