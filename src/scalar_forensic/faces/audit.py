"""Append-only JSONL audit log for face-modality acts (spec §7.4).

One line per event, opened append-only per call — no held handle, so a
crashed run cannot lose buffered events and an examiner can tail the file
during an index run.  Phase 1 event types: "enablement", "index_run",
"purge".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, event_type: str, examiner_id: str, **fields) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event_type,
            "examiner_id": examiner_id,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def iter_events(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
