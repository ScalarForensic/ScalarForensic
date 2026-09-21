"""Trusted state for one reusable CPU or NVENC video slot."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from scalar_forensic.media_worker.client import WorkerUnavailable
from scalar_forensic.media_worker.supervisor import private_file

POLICIES = {"cpu": "sfn-video-cpu-v1", "renderer": "sfn-video-nvenc-v1"}
PATH_FIELDS = ("socket_path", "failure_notice", "lock_path", "evidence_root")


@dataclass(frozen=True)
class VideoRuntime:
    generation: str
    socket_path: Path
    case_id: str
    image_id: str
    policy: str
    failure_notice: Path
    lock_path: Path
    evidence_root: Path
    max_output_bytes: int
    job_timeout: int
    uploads_root: Path | None = None


def validate(state: VideoRuntime) -> None:
    if (
        type(state.generation) is not str
        or re.fullmatch(r"[0-9a-f]{32}", state.generation) is None
        or type(state.image_id) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", state.image_id) is None
        or type(state.case_id) is not str
        or not 1 <= len(state.case_id) <= 256
        or any(ord(c) < 32 for c in state.case_id)
        or state.policy not in POLICIES.values()
        or type(state.max_output_bytes) is not int
        or not 1 <= state.max_output_bytes <= 2**50
        or type(state.job_timeout) is not int
        or not 1 <= state.job_timeout <= 86400
    ):
        raise ValueError("invalid video runtime identity or limits")
    if any(not getattr(state, field).is_absolute() for field in PATH_FIELDS):
        raise ValueError("video runtime paths must be absolute")
    if state.uploads_root is not None and not state.uploads_root.is_absolute():
        raise ValueError("video upload root must be absolute")
    ipc = state.socket_path.resolve().parent
    if any(
        getattr(state, field).resolve().is_relative_to(ipc)
        for field in ("failure_notice", "lock_path")
    ):
        raise ValueError("video control paths must be outside parser IPC")


def publish_state(path: Path, state: VideoRuntime) -> None:
    validate(state)
    row = {"version": 1, **asdict(state)}
    for field in PATH_FIELDS:
        row[field] = str(row[field])
    row["uploads_root"] = None if state.uploads_root is None else str(state.uploads_root)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(row, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_state(path: Path, case_id: str) -> VideoRuntime:
    try:
        with os.fdopen(private_file(path, os.O_RDONLY), "rb") as stream:
            raw = stream.read(8193)
        if len(raw) > 8192:
            raise ValueError("oversized video state")
        row = json.loads(raw)
        if type(row) is not dict or type(row.get("version")) is not int or row.pop("version") != 1:
            raise ValueError("invalid video state version")
        # Exact schema follows the dataclass constructor; extra/missing fields fail.
        for field in PATH_FIELDS:
            row[field] = Path(row[field])
        if row.get("uploads_root") is not None:
            row["uploads_root"] = Path(row["uploads_root"])
        state = VideoRuntime(**row)
        validate(state)
        if state.case_id != case_id:
            raise ValueError("video worker belongs to another case")
        return state
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise WorkerUnavailable("video runtime unavailable or invalid") from exc
