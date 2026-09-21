"""Trusted supervisor state shared with the application, never with a parser."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scalar_forensic.media_worker.client import WorkerUnavailable

POLICY = "sfn-image-worker-v1"


@dataclass(frozen=True)
class RuntimeState:
    generation: str
    socket_path: Path
    case_id: str
    image_id: str
    failure_notice: Path
    policy: str = POLICY


def _validate(state: RuntimeState) -> None:
    if (
        type(state.generation) is not str
        or re.fullmatch(r"[0-9a-f]{32}", state.generation) is None
        or type(state.case_id) is not str
        or not 1 <= len(state.case_id) <= 256
        or any(ord(char) < 32 for char in state.case_id)
        or type(state.image_id) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", state.image_id) is None
        or state.policy != POLICY
    ):
        raise ValueError("invalid worker runtime identity")
    if not state.socket_path.is_absolute() or not state.failure_notice.is_absolute():
        raise ValueError("worker runtime paths must be absolute")
    if state.failure_notice.resolve().is_relative_to(state.socket_path.resolve().parent):
        raise ValueError("failure notice must not be in parser-writable IPC storage")


def publish_state(path: Path, state: RuntimeState) -> None:
    """Publish a complete replacement; readers never observe half a JSON object."""
    _validate(state)
    row = {
        "version": 1,
        "generation": state.generation,
        "socket_path": str(state.socket_path),
        "case_id": state.case_id,
        "image_id": state.image_id,
        "failure_notice": str(state.failure_notice),
        "policy": state.policy,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(row, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_state(path: Path, case_id: str) -> RuntimeState:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or not 0 < info.st_size <= 8192
            ):
                raise ValueError("runtime state must be a small, private, owned regular file")
            row = json.loads(stream.read(8193))
        if (
            type(row) is not dict
            or set(row)
            != {
                "version",
                "generation",
                "socket_path",
                "case_id",
                "image_id",
                "failure_notice",
                "policy",
            }
            or type(row["version"]) is not int
            or row["version"] != 1
        ):
            raise ValueError("invalid runtime state schema")
        state = RuntimeState(
            generation=row["generation"],
            socket_path=Path(row["socket_path"]),
            case_id=row["case_id"],
            image_id=row["image_id"],
            failure_notice=Path(row["failure_notice"]),
            policy=row["policy"],
        )
        _validate(state)
        if state.case_id != case_id:
            raise ValueError("worker belongs to a different case")
        return state
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise WorkerUnavailable("managed media worker state unavailable or invalid") from exc
