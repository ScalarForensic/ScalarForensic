"""The §5 player states and the §10.1 failure matrix, in one place.

Two tables that have to agree, so they live together:

* **What the analyst can be shown** — :data:`PLAYER_STATES`, the §5 list plus
  the three states §5 left implicit because v1 only asked "can this play?".
* **What a failure turns into** — :func:`classify`, which maps every condition
  §10.1 names onto an HTTP status, one of those player states, and whether
  retrying could plausibly help.

Keeping them apart is how a failure ends up rendered as a state the UI has no
branch for, or as no state at all.  :mod:`.routes` is the only caller today;
phase 7's job runner maps the same failures onto ``full-job-failed`` through
:func:`classify` rather than re-deriving the table.

**Retryability is advertised, never acted on here.**  §10.1 ends "nothing may
retry-storm": the server states whether a retry could help and how long to wait,
and the player obeys it.  A client that retries a ``retryable=False`` failure is
looping on a condition that will not change.
"""

from __future__ import annotations

import errno
import signal
from dataclasses import dataclass

from fastapi import HTTPException

from scalar_forensic.video_playback.encode import CeilingExceeded, EncodeError

# ---------------------------------------------------------------------------
# §5 player states
# ---------------------------------------------------------------------------
#
# §5: "Every state below must be represented in the UI ... nothing here may
# invent a state it cannot observe."  Both halves bind.  The list therefore
# carries exactly what the server can observe — which since phase 7 built
# /api/video-full includes the three `full-job-*` states it withheld.

#: The three §5 states the full-video job puts the player in.  Phase 6 named them
#: separately because nothing could enter them; phase 7 built the endpoints that
#: do, so they are ordinary members of :data:`PLAYER_STATES` now and this set is
#: kept only to say which three the job owns.
FULL_JOB_STATES: frozenset[str] = frozenset(
    {"full-job-running", "full-job-done", "full-job-failed"}
)

#: Every state the server may put the player in.
PLAYER_STATES: frozenset[str] = FULL_JOB_STATES | frozenset(
    {
        # --- §5, verbatim -------------------------------------------------
        "playable",  # plays directly, no controls added
        "needs-transcode",  # reason + Play + Request full video + Download original
        "chunk-encoding",  # spinner with elapsed time; no fabricated percentage
        "chunk-ready",  # plays
        "chunk-failed",  # the failure reason, and Download original
        "cache-disabled",  # SFN_VIDEO_CACHE_DIR unset or unwritable
        "capacity-exhausted",  # queue or cache full
        # --- added in phase 6, because §5's list has no answer for "we have
        #     not asked yet" or "we asked and the file would not say" --------
        "idle",  # no video selected; the player is not claiming anything
        "probing",  # playback-info is in flight — NOT a verdict
        "unknown",  # the container could not be probed
    }
)

# `playable`, `needs-transcode` and `unknown` are the three answers to one
# question, and the third is the one this project keeps dropping: `#147` shipped
# an evidence viewer that displayed "unknown" as "mismatch".  A container that
# will not open has told us nothing about whether it would play; saying
# "needs-transcode" would be a claim about a stream nobody read, and saying
# "playable" would be worse.  `codecs._playback_mode` already returns `unknown`
# as a fourth mode for exactly this reason — this is that value reaching the UI
# instead of being flattened on the way.
MODE_TO_STATE: dict[str, str] = {
    "original": "playable",
    "rewrap": "playable",
    "transcode": "needs-transcode",
    "unknown": "unknown",
}


# ---------------------------------------------------------------------------
# §10.1 failure matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    """One row of §10.1: what happened, what the analyst sees, what to do next.

    ``kind`` is the §10.1 condition name and is what a log record and a test
    both key on — the reason strings are written for an analyst and will be
    reworded; the kind is the identity.
    """

    kind: str
    status: int
    state: str
    retryable: bool
    reason: str
    retry_after_seconds: int | None = None

    def as_detail(self) -> dict:
        """The JSON body of the error response."""
        return {
            "error": self.kind,
            "player_state": self.state,
            "reason": self.reason,
            "retryable": self.retryable,
            "retry_after_seconds": self.retry_after_seconds,
        }

    def as_http(self) -> HTTPException:
        headers = (
            {"Retry-After": str(self.retry_after_seconds)}
            if self.retry_after_seconds is not None
            else None
        )
        return HTTPException(status_code=self.status, detail=self.as_detail(), headers=headers)


_DOWNLOAD = "Download the original to view it."

# Every §10.1 condition that is decided *before* an encode starts is raised by
# `routes` as one of these directly; the ones only ffmpeg can report are reached
# through `classify`.

CACHE_UNSET = Failure(
    kind="cache-unset",
    status=503,
    state="cache-disabled",
    retryable=False,
    reason=(
        "Chunk playback needs a viewing-copy cache and SFN_VIDEO_CACHE_DIR is unset, "
        f"so there is nowhere to write one. {_DOWNLOAD}"
    ),
)

NO_PIPELINE = Failure(
    kind="no-encode-pipeline",
    status=503,
    state="chunk-failed",
    retryable=False,
    reason="",  # filled from Capability.unavailable_reason, which is specific
)

SOURCE_GONE = Failure(
    kind="source-disappeared",
    status=404,
    state="chunk-failed",
    retryable=False,
    reason="The source file is no longer at the path the index recorded.",
)

SOURCE_CHANGED = Failure(
    kind="source-changed",
    status=409,
    state="chunk-failed",
    retryable=False,
    reason=(
        "The file on disk changed since this session opened it, so a chunk encoded now "
        "would not be the video the timecodes describe. Re-open the hit to re-probe it."
    ),
)

BAD_DURATION = Failure(
    kind="malformed-duration",
    status=422,
    state="chunk-failed",
    retryable=False,
    reason=(
        "This container does not report a usable duration, so a timecode cannot be "
        f"checked against it and a chunk cannot be bounded. {_DOWNLOAD}"
    ),
)

NO_VIDEO_TRACK = Failure(
    kind="no-video-track",
    status=422,
    state="chunk-failed",
    retryable=False,
    reason=f"No video stream was found in this container. {_DOWNLOAD}",
)

PROBE_FAILED = Failure(
    kind="corrupt-input",
    status=422,
    state="unknown",
    retryable=False,
    reason=(
        "The container could not be opened, so nothing can be said about what it "
        f"holds — including whether it would play. {_DOWNLOAD}"
    ),
)

NOT_A_TRANSCODE = Failure(
    kind="not-a-transcode",
    status=409,
    state="playable",
    retryable=False,
    reason=(
        "This video does not need transcoding — it is served whole by "
        "/api/video-playback. Nothing is encoded for it."
    ),
)

TIMECODE_OUT_OF_RANGE = Failure(
    kind="timecode-out-of-range",
    status=422,
    state="chunk-failed",
    retryable=False,
    reason="The requested timecode lies outside this video's duration.",
)

OVERRIDE_UNATTRIBUTED = Failure(
    kind="override-unattributed",
    status=403,
    state="capacity-exhausted",
    retryable=False,
    # §6.3's override is a forensic act, so it is refused when it cannot be
    # attributed: a record naming nobody is worse than the refusal it replaces,
    # because it reads as authority in a log an examiner may have to defend.
    reason=(
        "Overriding the capacity refusal is recorded against the examiner who does it, "
        "and SFN_EXAMINER_ID is not set on this server, so there is no one to record. "
        "The full copy was not started. Set SFN_EXAMINER_ID and try again, or download "
        "the original."
    ),
)

QUEUE_FULL = Failure(
    kind="queue-full",
    status=503,
    state="capacity-exhausted",
    retryable=True,
    # §3.5: ~8.2 s per 30 s chunk at k=1, 16.35 s at k=2.  A full queue drains
    # in tens of seconds, so this is a wait, not a defeat.
    retry_after_seconds=15,
    reason=(
        "This host is already encoding as many chunks as it will queue. "
        "The request was refused rather than queued without limit; try again shortly."
    ),
)

DISK_FULL = Failure(
    kind="disk-full",
    status=507,
    state="capacity-exhausted",
    retryable=True,
    retry_after_seconds=60,
    reason=(
        "The viewing-copy cache filesystem is full, so the chunk could not be written. "
        f"Freeing space or lowering SFN_VIDEO_CACHE_MAX_BYTES will fix it. {_DOWNLOAD}"
    ),
)

CACHE_UNWRITABLE = Failure(
    kind="cache-unwritable",
    status=503,
    state="cache-disabled",
    retryable=False,
    reason=(
        "The viewing-copy cache directory cannot be written to. Chunk playback is "
        f"unavailable until SFN_VIDEO_CACHE_DIR is writable. {_DOWNLOAD}"
    ),
)

ENCODE_TIMEOUT = Failure(
    kind="job-timeout",
    status=504,
    state="chunk-failed",
    retryable=True,
    retry_after_seconds=30,
    reason=(
        f"Encoding this chunk hit SFN_VIDEO_JOB_TIMEOUT and the encoder was stopped. {_DOWNLOAD}"
    ),
)

ENCODER_KILLED = Failure(
    kind="encoder-killed",
    status=507,
    state="capacity-exhausted",
    retryable=True,
    retry_after_seconds=60,
    reason=(
        "The encoder was killed by the operating system before it finished — on this "
        "host that is almost always the out-of-memory killer. Fewer concurrent encodes "
        f"or a lower SFN_VIDEO_OUTPUT_HEIGHT will help. {_DOWNLOAD}"
    ),
)

FULL_COPY_OVERSHOOT = Failure(
    kind="full-copy-overshoot",
    status=507,
    state="capacity-exhausted",
    retryable=False,
    reason=(
        "The full viewing copy grew past the size a single rendering may occupy and was "
        "stopped before it filled the cache. The estimate that admitted it is uncalibrated "
        "— it applies no codec factor, because none is measured — and it under-reads on "
        "10-bit HEVC sources. Play it in chunks, or download the original."
    ),
)

NO_SUCH_JOB = Failure(
    kind="no-such-job",
    status=404,
    state="needs-transcode",
    retryable=False,
    reason="No full-video job is running for this video in this worker process.",
)

ENCODE_FAILED = Failure(
    kind="encode-failed",
    status=422,
    state="chunk-failed",
    retryable=False,
    reason="",  # filled from ffmpeg's own last line
)

# §10.1's "GPU failure or saturation" and "missing audio track" are deliberately
# absent from this table: neither is a failure by the time it reaches a caller.
# A GPU that fails at job time falls back to CPU (§8, `encode.encode`) and
# produces a chunk under the CPU pipeline's key; a source with no audio track is
# encoded with `-an`.  Both end in `chunk-ready`, and the tests pin that, because
# a mapping table is as wrong when it invents a failure as when it drops one.

# `errno` values that mean "the filesystem is full", not "the code is wrong".
_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EFBIG})
_DENIED_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})


def classify(exc: BaseException) -> Failure:
    """Map an exception raised on the chunk path onto its §10.1 row.

    Anything unrecognised becomes :data:`ENCODE_FAILED` carrying the exception's
    own words — a 422 the analyst can read and act on, rather than a 500 that
    says only that something happened.
    """
    if isinstance(exc, CeilingExceeded):
        return _with_reason(
            FULL_COPY_OVERSHOOT,
            (
                f"The full viewing copy passed {exc.limit_bytes / 1024**3:.1f} GiB while "
                f"encoding ({exc.written_bytes / 1024**3:.1f} GiB written) and was stopped "
                "before it filled the cache. The estimate that admitted it is uncalibrated "
                "— it applies no codec factor, because none is measured — and it under-reads "
                f"on 10-bit HEVC sources. {_DOWNLOAD}"
            ),
        )
    if isinstance(exc, EncodeError):
        if exc.timed_out:
            return ENCODE_TIMEOUT
        if exc.returncode is not None and exc.returncode < 0:
            # A negative returncode is a signal.  SIGKILL with no message is what
            # the OOM killer leaves behind; the other signals are reported by
            # name rather than guessed at.
            sig = -exc.returncode
            name = signal.Signals(sig).name if sig in set(signal.Signals) else f"signal {sig}"
            if sig == signal.SIGKILL:
                return ENCODER_KILLED
            return _with_reason(ENCODE_FAILED, f"The encoder was terminated by {name}. {_DOWNLOAD}")
        return _with_reason(
            ENCODE_FAILED, f"ffmpeg could not encode this chunk: {exc}. {_DOWNLOAD}"
        )
    if isinstance(exc, FileNotFoundError):
        return SOURCE_GONE
    if isinstance(exc, OSError):
        if exc.errno in _FULL_ERRNOS:
            return DISK_FULL
        if exc.errno in _DENIED_ERRNOS:
            return CACHE_UNWRITABLE
        return _with_reason(ENCODE_FAILED, f"The chunk could not be written: {exc}. {_DOWNLOAD}")
    if isinstance(exc, RuntimeError):
        # `capability.select` raises this with an analyst-facing sentence when
        # the host has no usable pipeline at all (no encoder, or an HDR source
        # on a build without libzimg — which is refused, never encoded badly).
        return _with_reason(NO_PIPELINE, str(exc))
    return _with_reason(ENCODE_FAILED, f"This chunk could not be produced: {exc}. {_DOWNLOAD}")


def classify_full_job(exc: BaseException) -> Failure:
    """The same §10.1 row, landing in ``full-job-failed`` (§5, §10.1).

    A full-video job fails for exactly the conditions a chunk fails for — the
    file, the encoder, the disk and the clock do not care which window was asked
    for — so this is :func:`classify` with the *state* rewritten and **not** a
    second table. §10.1 says so in as many words: "phase 7's job runner maps the
    same conditions onto ``full-job-failed`` through the same function rather
    than re-deriving them." The ``kind`` is untouched, because that is the
    identity a log record and a test key on; only what the player renders moves.
    """
    row = classify(exc)
    return Failure(
        kind=row.kind,
        status=row.status,
        state="full-job-failed",
        retryable=row.retryable,
        reason=row.reason,
        retry_after_seconds=row.retry_after_seconds,
    )


def _with_reason(base: Failure, reason: str) -> Failure:
    return Failure(
        kind=base.kind,
        status=base.status,
        state=base.state,
        retryable=base.retryable,
        reason=reason,
        retry_after_seconds=base.retry_after_seconds,
    )
