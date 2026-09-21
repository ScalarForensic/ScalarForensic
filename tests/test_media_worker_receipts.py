import hashlib
import json
from dataclasses import replace

import pytest

from scalar_forensic.media_worker.protocol import Request


def receipts():
    from scalar_forensic.media_worker import receipts as module

    return module


def request():
    return Request("a" * 32, hashlib.sha256(b"source").hexdigest(), "preprocess", 331, 6)


def test_started_and_outcome_bind_source_to_generation(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = receipts().ReceiptLog(path, "case-a", "b" * 32)
    submitted = replace(request(), max_image_pixels=100)
    log.append(submitted, "started")
    log.append(submitted, "decoded", pixel_hashes=["c" * 64])
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in lines] == ["started", "decoded"]
    assert all(row["max_image_pixels"] == 100 and row["protocol_version"] == 2 for row in lines)
    assert lines[0]["source_sha256"] == request().source_sha256
    assert lines[1]["request_id"] == "a" * 32
    assert lines[1]["generation"] == "b" * 32
    assert lines[1]["case_id"] == "case-a"
    assert lines[1]["pixel_hashes"] == ["c" * 64]
    assert path.stat().st_mode & 0o777 == 0o600


def test_audit_refuses_symlink_without_touching_original(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"immutable evidence")
    link = tmp_path / "audit.jsonl"
    link.symlink_to(original)
    log = receipts().ReceiptLog(link, "case-a", "b" * 32)
    with pytest.raises(OSError):
        log.append(request(), "started")
    assert original.read_bytes() == b"immutable evidence"


def test_recovery_includes_unfinished_requests_not_just_success(tmp_path):
    path = tmp_path / "audit.jsonl"
    receipts().ReceiptLog(path, "case-a", "b" * 32).append(request(), "started")
    other = Request("c" * 32, hashlib.sha256(b"other").hexdigest(), "native", 331, 5)
    receipts().ReceiptLog(path, "case-a", "d" * 32).append(other, "started")
    result = receipts().affected_sources(path, "case-a", "b" * 32)
    assert result == [request().source_sha256]


def test_recovery_refuses_corrupt_or_oversized_records(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("{" + "a" * 65536)
    with pytest.raises(ValueError):
        receipts().affected_sources(path, "case-a", "b" * 32)


def test_log_refuses_unbounded_case_label(tmp_path):
    with pytest.raises(ValueError):
        receipts().ReceiptLog(tmp_path / "audit.jsonl", "x" * 257, "b" * 32)
