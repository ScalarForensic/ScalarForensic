"""Per-run ingestion manifest + SFN_REPORT_DIR config (runbook work item 2026-08-13).

Unit tests cover the manifest module and the new Settings.report_dir field;
the end-to-end tests reuse the ingest fakes from test_ingest_characterization
to assert that a real cli.index() run writes the manifest next to the CSV
report, with config snapshot, model hashes and the discovered input list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scalar_forensic.config import Settings
from scalar_forensic.manifest import (
    MANIFEST_VERSION,
    input_file_entry,
    manifest_path_for,
    settings_snapshot,
    write_run_manifest,
)
from tests.test_ingest_characterization import (  # noqa: F401 — shared fixtures
    _run_index,
    fake_client_cls,
    ingest_env,
    store,
)

# ---------------------------------------------------------------------------
# Settings.report_dir (SFN_REPORT_DIR)
# ---------------------------------------------------------------------------


def test_report_dir_defaults_to_data_reports():
    assert Settings().report_dir == Path.cwd() / "data/reports"


def test_report_dir_env_override(monkeypatch):
    monkeypatch.setenv("SFN_REPORT_DIR", "/evidence/created_by_scalar/reports")
    assert Settings().report_dir == Path("/evidence/created_by_scalar/reports")


def test_report_dir_empty_rejected(monkeypatch):
    monkeypatch.setenv("SFN_REPORT_DIR", "")
    with pytest.raises(ValueError, match="SFN_REPORT_DIR"):
        Settings()


# ---------------------------------------------------------------------------
# manifest module units
# ---------------------------------------------------------------------------


def test_manifest_path_sits_next_to_csv_report():
    csv_path = Path("/x/reports/sfn_ingestion_20260813_120000.csv")
    assert manifest_path_for(csv_path) == Path(
        "/x/reports/sfn_ingestion_20260813_120000.manifest.json"
    )


def test_settings_snapshot_is_json_safe_and_redacts_secrets(monkeypatch):
    monkeypatch.setenv("SFN_QDRANT_API_KEY", "hunter2")
    monkeypatch.setenv("SFN_EMBEDDING_API_KEY", "hunter3")
    snap = settings_snapshot(Settings())

    dumped = json.dumps(snap)  # must not raise — every value JSON-serializable
    assert "hunter2" not in dumped and "hunter3" not in dumped
    assert snap["qdrant_api_key"] == "***redacted***"
    assert snap["embedding_api_key"] == "***redacted***"
    # Paths become strings; private attributes are excluded
    assert isinstance(snap["thumbnail_dir"], str)
    assert not any(k.startswith("_") for k in snap)
    # a couple of comparability-critical fields must be present
    assert snap["normalize_size"] == 224
    assert snap["sscd_n_crops"] == 5
    assert isinstance(snap["report_dir"], str)


def test_settings_snapshot_unset_secret_is_none():
    snap = settings_snapshot(Settings())
    assert snap["qdrant_api_key"] is None


def test_input_file_entry_with_and_without_hash(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"12345")
    assert input_file_entry(f) == {"path": str(f), "size": 5}
    assert input_file_entry(f, sha256="deadbeef") == {
        "path": str(f),
        "size": 5,
        "sha256": "deadbeef",
    }
    # unreadable / vanished file: size is None, entry still recorded
    assert input_file_entry(tmp_path / "gone.bin") == {
        "path": str(tmp_path / "gone.bin"),
        "size": None,
    }


def test_write_run_manifest_creates_parents_and_round_trips(tmp_path):
    csv_path = tmp_path / "deep" / "reports" / "run.csv"
    files = [{"path": "/in/a.jpg", "size": 3, "sha256": "aa"}]
    models = {"dino": {"model_name": "m", "model_hash": "h", "embedding_dim": 4}}

    out = write_run_manifest(
        csv_path,
        settings=Settings(),
        target_collection="case_x",
        input_root=Path("/in"),
        files=files,
        models=models,
    )

    assert out == csv_path.with_name("run.manifest.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["manifest_version"] == MANIFEST_VERSION
    assert data["created_at"]  # ISO timestamp, non-empty
    assert data["collection"] == "case_x"
    assert data["report_csv"] == str(csv_path)
    assert data["input"] == {"root": "/in", "file_count": 1, "files": files}
    assert data["models"] == models
    assert data["settings"]["collection"] == "sfn"


def test_write_run_manifest_degrades_unexpected_types_to_str(tmp_path):
    """A non-JSON value (e.g. a Path leaking from a models dict) must not
    abort the run at manifest time — it degrades to its string form."""
    out = write_run_manifest(
        tmp_path / "run.csv",
        settings=Settings(),
        target_collection="c",
        input_root=Path("/in"),
        files=[],
        models={"dino": {"model_name": "m", "model_hash": "h", "odd": Path("/p")}},
    )
    assert json.loads(out.read_text(encoding="utf-8"))["models"]["dino"]["odd"] == "/p"


# ---------------------------------------------------------------------------
# End-to-end: cli.index() writes the manifest at run start
# ---------------------------------------------------------------------------


def test_index_writes_manifest_next_to_report(ingest_env, store):  # noqa: F811
    csv_path = ingest_env.tmp / "report.csv"
    _run_index(ingest_env.images, csv_path)

    manifest_file = ingest_env.tmp / "report.manifest.json"
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))

    assert data["collection"] == "sfn"
    assert data["report_csv"] == str(csv_path)
    # full config snapshot
    assert data["settings"]["normalize_size"] == 336  # from ingest_env's env
    assert data["settings"]["collection"] == "sfn"
    # model hashes as computed for the upsert payload
    assert data["models"] == {
        "dino": {"model_name": "fake-dino", "model_hash": "hash-fake-dino", "embedding_dim": 4}
    }
    # discovered input list: all 4 files, each with path+size+sha256 (hash
    # pass has run by manifest time), and no derived artefacts
    files = data["input"]["files"]
    assert data["input"]["root"] == str(ingest_env.images)
    assert data["input"]["file_count"] == len(files) == 4
    names = {Path(f["path"]).name for f in files}
    assert names == {"red.png", "green.png", "blue.png", "red_copy.png"}
    for f in files:
        assert f["size"] > 0
        assert len(f["sha256"]) == 64


def test_index_default_report_path_honours_sfn_report_dir(ingest_env, store, monkeypatch):  # noqa: F811
    report_dir = ingest_env.tmp / "created_by_scalar" / "reports"
    monkeypatch.setenv("SFN_REPORT_DIR", str(report_dir))

    _run_index(ingest_env.images, None)

    csvs = sorted(report_dir.glob("sfn_ingestion_*.csv"))
    manifests = sorted(report_dir.glob("sfn_ingestion_*.manifest.json"))
    assert len(csvs) == 1 and len(manifests) == 1
    assert manifests[0].name == csvs[0].name.replace(".csv", ".manifest.json")
    data = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert data["report_csv"] == str(csvs[0])
