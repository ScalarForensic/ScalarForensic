import json

from scalar_forensic.faces.audit import AuditLog


def test_append_writes_jsonl_with_timestamp(tmp_path):
    log = AuditLog(tmp_path / "sub" / "face_audit.log")
    log.append("purge", examiner_id="ex1", image_hash="h1", n_deleted=3)
    lines = (tmp_path / "sub" / "face_audit.log").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "purge" and rec["examiner_id"] == "ex1" and rec["n_deleted"] == 3
    assert "ts" in rec


def test_append_only_accumulates(tmp_path):
    log = AuditLog(tmp_path / "a.log")
    log.append("enablement", examiner_id="ex1")
    log.append("index_run", examiner_id="ex1", n_kept=5)
    assert [e["event"] for e in log.iter_events()] == ["enablement", "index_run"]


def test_iter_events_on_missing_file_is_empty(tmp_path):
    assert list(AuditLog(tmp_path / "nope.log").iter_events()) == []
