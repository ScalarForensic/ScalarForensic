"""Trusted-side decode receipts and read-only generation recovery manifests.

Decoded does not mean indexed or scientifically verified. A recovery manifest
identifies sources to review/rebuild; it never deletes evidence or database rows.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path

from scalar_forensic.media_worker.protocol import VERSION, Request, validate_request_header


def _identity(case_id: str, generation: str):
    if (
        type(case_id) is not str
        or not 1 <= len(case_id) <= 256
        or any(ord(char) < 32 for char in case_id)
        or type(generation) is not str
        or re.fullmatch(r"[0-9a-f]{32}", generation) is None
    ):
        raise ValueError("invalid case or worker generation identity")


class ReceiptLog:
    def __init__(self, path: Path, case_id: str, generation: str, *, runtime: dict | None = None):
        _identity(case_id, generation)
        self.path = Path(path)
        self.case_id = case_id
        self.generation = generation
        self.runtime = runtime

    def append(self, request: Request, event: str, *, pixel_hashes: list[str] | None = None):
        if event not in ("started", "decoded", "decode_failed", "worker_failed"):
            raise ValueError("invalid receipt event")
        validate_request_header(
            {
                "version": VERSION,
                "request_id": request.request_id,
                "source_sha256": request.source_sha256,
                "operation": request.operation,
                "cap": request.cap,
                "source_bytes": request.source_bytes,
                "max_image_pixels": request.max_image_pixels,
            }
        )
        hashes = [] if pixel_hashes is None else pixel_hashes
        if len(hashes) > 2 or any(
            type(h) is not str or re.fullmatch(r"[0-9a-f]{64}", h) is None for h in hashes
        ):
            raise ValueError("invalid pixel digests")
        row = {
            "version": 1,
            "at": datetime.now(UTC).isoformat(),
            "event": event,
            "case_id": self.case_id,
            "generation": self.generation,
            "request_id": request.request_id,
            "source_sha256": request.source_sha256,
            "operation": request.operation,
            "cap": request.cap,
            "source_bytes": request.source_bytes,
            "max_image_pixels": request.max_image_pixels,
            "protocol_version": VERSION,
            "pixel_hashes": hashes,
        }
        if self.runtime is not None:
            row["runtime"] = self.runtime
        data = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o077:
                raise OSError("receipt target must be a private, singly-linked regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                if not written:
                    raise OSError("short receipt write")
                remaining = remaining[written:]
            os.fsync(fd)
            if info.st_size == 0:
                directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            os.close(fd)


def affected_sources(path: Path, case_id: str, generation: str) -> list[str]:
    """Include submitted-but-unfinished sources; never infer that they were safe."""
    _identity(case_id, generation)
    sources = set()
    with Path(path).open("rb") as stream:
        while line := stream.readline(65537):
            if len(line) > 65536 or not line.endswith(b"\n"):
                raise ValueError("truncated or oversized receipt record")
            try:
                row = json.loads(line)
                if (
                    type(row) is not dict
                    or type(row.get("version")) is not int
                    or row["version"] != 1
                    or row.get("event")
                    not in ("started", "decoded", "decode_failed", "worker_failed")
                    or type(row.get("source_sha256")) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]) is None
                ):
                    raise ValueError("invalid receipt schema")
                _identity(row.get("case_id"), row.get("generation"))
            except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
                raise ValueError("invalid receipt record") from exc
            if row["case_id"] == case_id and row["generation"] == generation:
                sources.add(row["source_sha256"])
    return sorted(sources)


def main():
    parser = argparse.ArgumentParser(
        description="List source hashes affected by a worker generation"
    )
    parser.add_argument("receipt_log", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--generation", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "case_id": args.case,
                "generation": args.generation,
                "source_sha256": affected_sources(args.receipt_log, args.case, args.generation),
                "action": "review_or_rebuild_derived_results; originals_unchanged",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
