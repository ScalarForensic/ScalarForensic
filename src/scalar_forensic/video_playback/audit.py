"""What actually ran: on the label, and in the examiner record (§7.2, §7.3).

§7.3 names ``faces/audit.py`` and ``faces/provenance.py`` as the pattern and says
this **must not invent a second one**, so that is taken literally:
:class:`~scalar_forensic.faces.audit.AuditLog` is *imported and used*, not
re-implemented.  One JSONL shape, one appender, one ``iter_events`` for the whole
tool — a forensic tool with two audit formats has two answers to "what did this
program do", and a reviewer would have to be told which file to believe.  What is
new here is the thing being recorded, not the way of recording it.

**Why a :class:`Rendering` and not just a ``Pipeline``.**
:meth:`~.capability.Pipeline.describe` already answers "which pipeline", and it is
half the cache key (§6.1), so it deliberately carries only what changes a pixel.
Three of §7.2's requirements are therefore *not* in it:

* **the audio transformation or omission** — ``-an`` versus the AAC args is
  decided per source, from whether the source carries an audio track at all;
* **``-threads``** — a full copy is thread-capped (§4.3) and libx264's output
  depends on its thread count, so an invocation printed without it does not
  reproduce the bytes on a host with a different core count;
* **the window** — ``-ss``/``-t``, which is chunk arithmetic, not pipeline.

:class:`Rendering` is the pipeline plus exactly those, and ``command`` is the argv
that ran — so §7.2's "a reviewer can reproduce it" is a record, not a claim.

**What is recorded is an encode, never a request.**  A cache hit re-serves an
existing rendering and transcodes nothing, so it writes no record; the encode that
produced those bytes already wrote one.  A full copy has one writer
(``jobs.JobRunner._run``) and many possible claimants — one record per *encode*,
attributed to the examiner who started it, with the claimant count on the record.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from scalar_forensic.config import Settings
from scalar_forensic.faces.audit import AuditLog
from scalar_forensic.video_playback.capability import Pipeline
from scalar_forensic.video_playback.encode import EncodeResult

_log = logging.getLogger(__name__)

#: Event types written by this subsystem.  Kept as constants because they are
#: read back by ``sfn-video render`` and by anyone grepping the log.
EVENT_TRANSCODE = "video_transcode"
EVENT_PURGE = "video_purge"
EVENT_OVERRIDE = "video_full_copy_override"

OUTCOME_SUCCESS = "success"
OUTCOME_FAILED = "failed"
OUTCOME_CANCELLED = "cancelled"

SCOPE_CHUNK = "chunk"
SCOPE_FULL = "full"

#: §7.2's "any audio transformation or omission", as the two sentences the label
#: and the record both use.  Constants, so the test asserts the string the
#: analyst reads rather than a paraphrase of it.
AUDIO_REENCODED = (
    "Audio re-encoded to AAC 128 kb/s. This is not the source's audio bitstream; "
    "judge audio from the original."
)
AUDIO_OMITTED = (
    "Audio omitted (-an): the source carries no audio track for this rendering to carry."
)

#: The audit log's filename, beside the cache directory and never inside it.
AUDIT_FILENAME = "video_audit.log"


@dataclass(frozen=True)
class Rendering:
    """The full §7.2 record of one encode: the pipeline, and what it was pointed at.

    ``command`` is ``None`` for a rendering that was found in the cache rather
    than produced: the argv that made those bytes was recorded when they were
    made, and reconstructing a plausible one to fill the field would put a
    sentence on a label that no process ever ran.  ``sfn-video render`` is where
    an invocation is retrieved (from the log) or rebuilt (clearly labelled as a
    reproduction recipe rather than a record).
    """

    pipeline: Pipeline
    scope: str
    has_audio: bool
    command: tuple[str, ...] | None = None
    threads: int | None = None
    start_seconds: float | None = None
    duration_seconds: float | None = None
    fell_back: bool = False
    fallback_reason: str | None = None

    @classmethod
    def from_result(
        cls,
        result: EncodeResult,
        *,
        scope: str,
        has_audio: bool,
        threads: int | None = None,
        start_seconds: float | None = None,
        duration_seconds: float | None = None,
    ) -> Rendering:
        """Build from what an encode returned.

        ``result.pipeline``, never the caller's request: on a §8 GPU→CPU fallback
        they name different encoders, and a label built from the request would
        name the encoder that was *selected* and not the one that produced the
        bytes — §7.2's exact failure mode, in a record an examiner may have to
        defend.
        """
        return cls(
            pipeline=result.pipeline,
            scope=scope,
            has_audio=has_audio,
            command=tuple(result.command),
            threads=threads,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            fell_back=result.fell_back,
            fallback_reason=result.fallback_reason,
        )

    @property
    def audio_transformation(self) -> str:
        return AUDIO_REENCODED if self.has_audio else AUDIO_OMITTED

    def describe(self) -> dict:
        """The §7.2 label payload: the pipeline that ran, and the rest of §7.2.

        ``command_line`` is the argv as a line a reviewer can paste, and it exists
        because §7.2's requirement is *reproduction*: the browser had been joining
        the argv on spaces, which produces a line that does not survive being
        copied — the filter chain carries ``'min(ih,1080)'``, and a shell hands
        ffmpeg something it rejects with ``No such filter: '1080)'`` (measured
        2026-08-14, `data/reports/c24-task2-label-vs-render.md`).  Quoting is
        ``shlex.join`` here and nowhere else, so the label and ``sfn-video render``
        cannot print one argv two ways; ``command`` stays alongside it because a
        list is what a machine reads and what the audit log stores.
        """
        return {
            **self.pipeline.describe(),
            "scope": self.scope,
            "audio_transformation": self.audio_transformation,
            "threads": self.threads,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "fell_back": self.fell_back,
            "fallback_reason": self.fallback_reason,
            "command": None if self.command is None else list(self.command),
            "command_line": None if self.command is None else shlex.join(self.command),
        }


def audit_dir(settings: Settings) -> Path:
    """Where the log lives: **beside** the cache directory, never inside it.

    ``sfn-video purge --all`` and the §6.2 LRU both empty the cache directory, and
    an audit log that a retention sweep can delete is not an audit log.  Mirrors
    ``faces/``, where the log sits beside the chip store for the same reason —
    and lands in the same ``data/`` directory as ``face_audit.log`` under the
    default configuration.
    """
    return settings.video_cache_dir.parent if settings.video_cache_dir else Path("data")


def audit_log(settings: Settings) -> AuditLog:
    return AuditLog(audit_dir(settings) / AUDIT_FILENAME)


def find_transcode(
    settings: Settings, *, video_sha256: str, scope: str, at: float | None = None
) -> dict | None:
    """The record of the rendering that covers *at*, or ``None`` (§7.2).

    Keyed on the **digest**, not the path: a file that moved is the same evidence
    and its rendering is the same rendering, while a different file at the same
    path is not — matching on the path would be the §7.1 error with extra steps.

    A chunk record is matched by its own recorded window rather than by
    recomputing one from ``SFN_VIDEO_CHUNK_SECONDS``: the setting can have changed
    since, and the question being asked is "which rendering did this analyst see",
    not "which one would be produced now".

    The **last** match wins.  A rendering can be produced more than once — evicted
    and re-encoded, or re-encoded after an ffmpeg upgrade — and the most recent is
    the one whose bytes are in the cache.
    """
    found = None
    for rec in audit_log(settings).iter_events():
        if rec.get("event") != EVENT_TRANSCODE or rec.get("outcome") != OUTCOME_SUCCESS:
            continue
        if rec.get("video_sha256") != video_sha256 or rec.get("scope") != scope:
            continue
        rendering = rec.get("rendering") or {}
        if scope == SCOPE_CHUNK:
            start = rendering.get("start_seconds")
            duration = rendering.get("duration_seconds")
            if start is None or duration is None or at is None:
                continue
            if not start <= at < start + duration:
                continue
        found = rec
    return found


def _append(settings: Settings, event: str, examiner_id: str | None, **fields) -> None:
    """Append one event, or log loudly and carry on.

    A failed audit write must not fail the analyst's request: the rendering it
    describes already exists on disk, so raising here would report an error for
    work that succeeded while un-writing none of it.  It is an ERROR in the
    process log rather than a silence, because a deployment whose audit log
    cannot be written needs fixing before the next act, not after.
    """
    try:
        audit_log(settings).append(event, examiner_id, **fields)
    except OSError as exc:
        _log.error("video audit record could not be written (%s): %s", event, exc)


def record_transcode(
    settings: Settings,
    *,
    source: Path,
    video_sha256: str,
    scope: str,
    outcome: str,
    examiner_id: str | None,
    requested_timecode: float | None = None,
    rendering: Rendering | None = None,
    pipeline_fingerprint: str | None = None,
    error: str | None = None,
    **extra,
) -> None:
    """§7.3's record for one encode — chunk or full, succeeded or not.

    ``video_sha256`` is the **verified** digest: the one ``digest.py`` computed
    from the file as it is on disk, revalidated by the ``HashCache``, never the
    ``video_hash`` the index recorded (§7.1).  Both call sites already hold it
    because they had to have it to find the artifact.

    On a non-success outcome there is no ``rendering``, and
    ``pipeline_fingerprint`` names the pipeline that was **selected** — nothing
    was produced for it to describe.  The two never both appear, so a reader
    cannot mistake a selection for a rendering.

    ``examiner_id`` is passed rather than read from *settings* here: the full job
    is attributed to whoever started it, which is not necessarily whoever is in
    the environment when it ends.  It may be ``None`` — playback is not gated on
    ``SFN_EXAMINER_ID`` the way the faces modality is, and recording that the act
    was unattributed is the honest entry.  (The §6.3 override is the exception:
    that one is *refused* when it cannot be attributed.)
    """
    fingerprint = pipeline_fingerprint
    if fingerprint is None and rendering is not None:
        fingerprint = rendering.pipeline.fingerprint()
    _append(
        settings,
        EVENT_TRANSCODE,
        examiner_id,
        scope=scope,
        outcome=outcome,
        source=str(source),
        video_sha256=video_sha256,
        requested_timecode=requested_timecode,
        pipeline_fingerprint=fingerprint,
        rendering=None if rendering is None else rendering.describe(),
        error=error,
        **extra,
    )


def record_override(
    settings: Settings,
    *,
    source: Path,
    video_sha256: str,
    examiner_id: str,
    verdict: str,
    estimate_bytes: int | None,
    limit_bytes: int | None,
) -> None:
    """The §6.3 capacity refusal an examiner set aside.

    Filed *in addition to* the WARNING the route already logs, not instead of it.
    The two have different readers: the WARNING is an operational alarm an
    operator watching the process log sees as it happens, and this is the durable
    record a reviewer reads months later.  They are emitted from one call site so
    they cannot drift.
    """
    _append(
        settings,
        EVENT_OVERRIDE,
        examiner_id,
        source=str(source),
        video_sha256=video_sha256,
        verdict=verdict,
        estimate_bytes=estimate_bytes,
        limit_bytes=limit_bytes,
    )


#: The three things `sfn-video render` may have to say about what it found.
#: Constants, because they are the sentences a reviewer reads and a test that
#: asserted a paraphrase of one would pass while the tool said something else.
RENDER_NO_RECORD = (
    "NO RECORD: this host's audit log holds no rendering of that window of this file. "
    "The invocation below is what this host would run now — a reproduction recipe, not a "
    "record of what ran. It reproduces the original bytes only if the ffmpeg build, the "
    "settings and the hardware match the host that produced them."
)
RENDER_SOURCE_CHANGED = (
    "STALE EVIDENCE: the file at this path no longer hashes to the digest the recorded "
    "rendering was made from. The record describes a rendering of the bytes that used to "
    "be here, and nothing below reproduces from the file as it is now. Re-index before "
    "relying on either."
)
RENDER_PART_NOTE = (
    "The invocation is verbatim: ffmpeg writes the '.part' file it names, which is renamed "
    "to the published artifact on success (spec §10.2)."
)
RENDER_NO_PIPELINE = (
    "NO PIPELINE: this host cannot encode, so there is no invocation to rebuild. "
    "The reason follows; the recorded invocation, if any, is printed above it."
)


# Display order, §7.2's own order where it states one — the same order the label
# uses (`static/js/video_playback/rendering.js`), so a reviewer reading the screen
# and a reviewer reading this command read the same fields in the same sequence.
# Like that file's list, this one is a *display order* and not a filter: a key
# absent from it is appended under its own name rather than dropped.
_RENDER_ORDER = (
    "scope",
    "hwaccel",
    "decoder",
    "filter_chain",
    "tone_mapped",
    "encoder",
    "rate_control",
    "output_height",
    "chunk_seconds",
    "audio",
    "audio_transformation",
    "threads",
    "start_seconds",
    "duration_seconds",
    "ffmpeg_version",
    "fell_back",
    "fallback_reason",
)
# Names, never glosses: the field's name with its unit where the value carries
# none.  A sentence here would be a second wording of something the record
# already states.
_RENDER_LABELS = {
    "scope": "scope",
    "hwaccel": "hwaccel",
    "decoder": "decoder",
    "filter_chain": "filter chain",
    "tone_mapped": "tone-mapped",
    "encoder": "encoder",
    "rate_control": "rate control",
    "output_height": "output height",
    "chunk_seconds": "chunk length (s)",
    "audio": "audio args",
    "audio_transformation": "audio",
    "threads": "threads",
    "start_seconds": "window start (s)",
    "duration_seconds": "window length (s)",
    "ffmpeg_version": "ffmpeg version",
    "fell_back": "fell back to CPU",
    "fallback_reason": "fallback reason",
}
# Printed elsewhere in the report, never as a field row: the fingerprint heads it
# and the invocation is a line to copy, not a table cell.
_RENDER_NOT_ROWS = frozenset({"fingerprint", "command", "command_line"})


def _render_value(value: object) -> str | None:
    """One value as the report prints it, or ``None`` for "do not print a row".

    ``None`` is "the server did not state one", which is not a value and must not
    print as ``None`` or as an invented zero.  A boolean prints yes/no because
    ``False`` is a statement: dropping it would leave "did this fall back to the
    CPU?" unanswered on every rendering that did not.  Both rules are the label's
    (`rendering.js:_renderingValue`) — the two surfaces describe one record and
    may not disagree about how it reads.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _pipeline_lines(rendering: dict) -> list[str]:
    """Every §7.2 field of one rendering, one per line, in §7.2's order.

    Walks the record rather than naming ten fields, and this is the whole point:
    the hand-written list this replaces omitted seven fields the label showed —
    including ``start_seconds``/``duration_seconds``, so ``--at 5`` answered with
    the chunk at 0 and never said so, and ``fell_back``, so a rendering the GPU
    declined read exactly like one it did not (measured 2026-08-14,
    `data/reports/c24-task2-label-vs-render.md`).  It is the same defect
    ``Pipeline.describe()`` derives itself from ``fields()`` to prevent, one layer
    further out: a field added to ``Rendering`` now reaches both surfaces without
    anyone remembering this list exists.
    """
    width = max(len(label) for label in _RENDER_LABELS.values()) + 2
    lines = []
    seen = set(_RENDER_NOT_ROWS)

    def emit(key: str) -> None:
        seen.add(key)
        value = _render_value(rendering[key])
        if value is None:
            return
        lines.append(f"  {(_RENDER_LABELS.get(key, key) + ':').ljust(width)}{value}")

    for key in _RENDER_ORDER:
        if key in rendering:
            emit(key)
    for key in rendering:
        if key not in seen:
            emit(key)
    return lines


def reproduction_report(
    settings: Settings, source: Path, *, scope: str, at: float | None = None
) -> list[str]:
    """The lines ``sfn-video render`` prints: §7.2's "a reviewer can reproduce it".

    Two answers, and they are labelled apart because they are not the same claim.
    A **record** says what ran: the argv, the pipeline it ran under, when, and on
    whose act.  A **reconstruction** says what this host would run now — useful,
    and not evidence of anything.  Printing the second in the shape of the first
    is the failure this whole section exists to prevent.

    The digest is computed here, from the file as it lies (§7.1), and the record
    is matched on it — so a source that changed since the rendering is reported
    as stale rather than silently answered with a recipe.
    """
    # Deferred: `routes` imports this module at import time, so the arithmetic it
    # owns can only be borrowed at call time.  One implementation of the snap,
    # since two would put a reviewer on a different chunk boundary than the
    # analyst was on.
    from scalar_forensic.video_playback.digest import _source_digest

    digest = _source_digest(source, settings)
    lines = [
        f"Source:             {source}",
        f"SHA-256 (verified): {digest}",
    ]

    record = find_transcode(settings, video_sha256=digest, scope=scope, at=at)
    if record is None:
        stale = _record_for_another_digest(settings, source, scope=scope, digest=digest)
        if stale is not None:
            lines += ["", RENDER_SOURCE_CHANGED, f"  recorded digest:  {stale['video_sha256']}"]
        else:
            lines += ["", RENDER_NO_RECORD]
    else:
        rendering = record["rendering"]
        lines += [
            f"Recorded:           {record['ts']} by examiner {record['examiner_id'] or '(none)'}",
            f"Pipeline:           {rendering['fingerprint']}",
            *_pipeline_lines(rendering),
            "",
            "Invocation that produced this rendering:",
            f"  {shlex.join(rendering['command'])}",
            "",
            RENDER_PART_NOTE,
        ]
        return lines

    rebuilt, reason = _rebuild_command(settings, source, digest, scope=scope, at=at)
    if rebuilt is None:
        lines += ["", RENDER_NO_PIPELINE, f"  {reason}"]
        return lines
    pipeline, command, window = rebuilt
    described = Rendering(
        pipeline=pipeline,
        scope=scope,
        has_audio=window["has_audio"],
        threads=window["threads"],
    ).describe()
    lines += [
        f"Pipeline (as it would be selected now): {pipeline.fingerprint()}",
        *_pipeline_lines(described),
        "",
        "Invocation this host would run now:",
        f"  {shlex.join(command)}",
        "",
        RENDER_PART_NOTE,
    ]
    return lines


def _record_for_another_digest(
    settings: Settings, source: Path, *, scope: str, digest: str
) -> dict | None:
    """A successful record for this *path* whose digest is not the current one."""
    found = None
    for rec in audit_log(settings).iter_events():
        if rec.get("event") != EVENT_TRANSCODE or rec.get("outcome") != OUTCOME_SUCCESS:
            continue
        if rec.get("source") != str(source) or rec.get("scope") != scope:
            continue
        if rec.get("video_sha256") != digest:
            found = rec
    return found


def _rebuild_command(
    settings: Settings, source: Path, digest: str, *, scope: str, at: float | None
) -> tuple[tuple[Pipeline, list[str], dict] | None, str]:
    """Rebuild the invocation this host would run.  ``build_command`` is pure.

    The full copy carries ``-threads``; a chunk carries none, and neither is a
    :class:`Pipeline` field — which is exactly why a reproduction that printed the
    pipeline alone would not reproduce a full copy's bytes on a host with a
    different core count.
    """
    from scalar_forensic.video_playback.cache import (
        FULL_NAME,
        artifact_dir,
        chunk_name,
        part_path,
    )
    from scalar_forensic.video_playback.capability import capability, is_hdr, select
    from scalar_forensic.video_playback.codecs import _stream_report
    from scalar_forensic.video_playback.encode import build_command, job_threads
    from scalar_forensic.video_playback.routes import chunk_start_for

    info = _stream_report(source)
    if "probe_error" in info:
        return None, f"the container could not be probed: {info['probe_error']}"
    try:
        pipeline = select(settings, capability(settings), hdr=is_hdr(info))
    except RuntimeError as exc:
        return None, str(exc)

    has_audio = info.get("audio_codec") is not None
    cache_dir = settings.video_cache_dir or Path("data/video_cache")
    if scope == SCOPE_FULL:
        window = {
            "start": None,
            "duration": None,
            "threads": job_threads(settings),
            "name": FULL_NAME,
        }
    else:
        start = chunk_start_for(at or 0.0, settings.video_chunk_seconds)
        window = {
            "start": start,
            "duration": float(settings.video_chunk_seconds),
            "threads": None,
            "name": chunk_name(start),
        }
    window["has_audio"] = has_audio
    dst = artifact_dir(cache_dir, digest, pipeline.fingerprint()) / str(window["name"])
    command = build_command(
        settings,
        pipeline,
        source,
        part_path(dst),
        start=window["start"],
        duration=window["duration"],
        has_audio=has_audio,
        threads=window["threads"],
        progress=scope == SCOPE_FULL,
    )
    return (pipeline, command, window), ""


def record_purge(
    settings: Settings,
    *,
    examiner_id: str | None,
    scope: str,
    video_sha256: str | None,
    videos: int,
    files: int,
    bytes_freed: int,
) -> None:
    """§13's "when were they destroyed", filed rather than only printed.

    Mirrors ``sfn-faces purge``'s record field for field where the fields mean the
    same thing; a viewing copy is derived and re-derivable, so what matters is
    that the deletion is attributable and dated.
    """
    _append(
        settings,
        EVENT_PURGE,
        examiner_id,
        scope=scope,
        video_sha256=video_sha256,
        videos=videos,
        files=files,
        bytes_freed=bytes_freed,
    )
