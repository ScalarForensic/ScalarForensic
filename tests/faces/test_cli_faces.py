import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from scalar_forensic.cli import faces_app, index
from scalar_forensic.faces.store import PurgeResult

runner = CliRunner()


def _typer_app(fn):
    import typer

    app = typer.Typer()
    app.command()(fn)
    return app


def test_index_requires_at_least_one_modality(tmp_path, monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    result = runner.invoke(_typer_app(index), [str(tmp_path)])
    assert result.exit_code == 1
    assert "--dino, --sscd or --faces" in result.output


def test_faces_flag_without_env_is_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    result = runner.invoke(_typer_app(index), [str(tmp_path), "--faces"])
    assert result.exit_code == 1
    assert "SFN_FACES_ENABLED" in result.output


def test_faces_flag_enabled_without_models_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.delenv("SFN_FACE_DETECTOR_MODEL", raising=False)
    monkeypatch.delenv("SFN_FACE_EMBEDDER_MODEL", raising=False)
    result = runner.invoke(_typer_app(index), [str(tmp_path), "--faces"])
    assert result.exit_code == 1
    assert "SFN_FACE_DETECTOR_MODEL" in result.output


def _enable_faces(monkeypatch, tmp_path):
    det = tmp_path / "yunet.onnx"
    det.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"
    emb.write_bytes(b"x")
    (tmp_path / "emb.onnx.manifest.json").write_text("{}")
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_FACE_DETECTOR_MODEL", str(det))
    monkeypatch.setenv("SFN_FACE_EMBEDDER_MODEL", str(emb))
    monkeypatch.setenv("SFN_EXAMINER_ID", "ex1")
    monkeypatch.setenv("SFN_FACE_STORE_DIR", str(tmp_path / "faces"))


@pytest.fixture
def run_faces_cli(tmp_path, monkeypatch):
    """Drive ``index --faces`` over one image with a scripted FaceIndexResult.

    The pipeline is a mock except for its AuditLog, which is real and writes
    into *tmp_path* — the audit record is what several of these tests assert on,
    and a mocked appender would assert only that the call was made, not that the
    record on disk reconciles.
    """
    from PIL import Image

    from scalar_forensic.faces.audit import AuditLog
    from scalar_forensic.faces.indexing import FaceIndexResult

    img_dir = tmp_path / "media"
    img_dir.mkdir()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(img_dir / "a.jpg")

    def _run(
        *,
        detected: int,
        kept: int,
        review_only: int = 0,
        rejected: dict[str, int] | None = None,
        review_reasons: dict[str, int] | None = None,
        dropped_noncanonical: int = 0,
    ):
        _enable_faces(monkeypatch, tmp_path)
        pipeline = MagicMock()
        pipeline.audit = AuditLog(tmp_path / "face_audit.log")
        pipeline.cfg.config_hash = "cfg1"
        pipeline.store.collection_is_new.return_value = False
        pipeline.store.check_compat.return_value = []
        pipeline.store.processed_hashes.return_value = set()
        pipeline.process_image.return_value = FaceIndexResult(
            n_detected=detected,
            n_kept=kept,
            n_review_only=review_only,
            rejected=dict(rejected or {}),
            review_only_reasons=dict(review_reasons or {}),
            n_dropped_noncanonical=dropped_noncanonical,
        )
        with patch(
            "scalar_forensic.faces.indexing.FacePipeline.from_settings", return_value=pipeline
        ):
            # --report keeps the ingestion CSV out of the repo's data/reports/.
            result = runner.invoke(
                _typer_app(index),
                [str(img_dir), "--faces", "--report", str(tmp_path / "ingestion.csv")],
            )
        events = [
            json.loads(line)
            for line in (tmp_path / "face_audit.log").read_text().splitlines()
            if line
        ]
        return result, events

    return _run


def test_cli_summary_reconciles_counts(run_faces_cli):
    result, _ = run_faces_cli(
        detected=6,
        kept=1,
        review_only=3,
        review_reasons={"size": 3},
        rejected={"confidence": 2},
    )
    assert result.exit_code == 0, result.output
    assert "6 detected" in result.output
    assert "1 comparable" in result.output
    assert "3 retained for review" in result.output
    assert "2 rejected" in result.output


def test_audit_index_run_records_review_only(run_faces_cli):
    _, events = run_faces_cli(
        detected=6,
        kept=1,
        review_only=3,
        review_reasons={"size": 3},
        rejected={"confidence": 2},
    )
    ev = [e for e in events if e["event"] == "index_run"][-1]
    assert ev["n_review_only"] == 3
    assert ev["review_only_reasons"] == {"size": 3}
    assert ev["n_kept"] + ev["n_review_only"] + sum(ev["n_rejected"].values()) == ev["n_detected"]


def test_audit_index_run_records_noncanonical_drops(run_faces_cli):
    # A non-canonical drop is subtracted from n_detected before any gate sees
    # the face; without it in the record the reconciliation above is unfalsifiable.
    _, events = run_faces_cli(detected=2, kept=2, dropped_noncanonical=4)
    ev = [e for e in events if e["event"] == "index_run"][-1]
    assert ev["n_dropped_noncanonical"] == 4


def test_cli_summary_omits_review_clause_when_none(run_faces_cli):
    result, _ = run_faces_cli(detected=2, kept=2)
    assert "retained for review" not in result.output
    assert "2 comparable" in result.output


def test_purge_requires_exactly_one_scope(monkeypatch, tmp_path):
    _enable_faces(monkeypatch, tmp_path)
    result = runner.invoke(faces_app, ["purge"])
    assert result.exit_code == 1
    assert "exactly one" in result.output


def test_purge_media_deletes_chips_and_audits(monkeypatch, tmp_path):
    _enable_faces(monkeypatch, tmp_path)
    store_dir = tmp_path / "faces"
    chash = "ab" + "0" * 62
    from scalar_forensic.faces.chips import chip_paths

    png, jpg, thumb = chip_paths(store_dir, chash)
    png.parent.mkdir(parents=True, exist_ok=True)
    for p in (png, jpg, thumb):
        p.write_bytes(b"x")

    store = MagicMock()
    store.purge_media.return_value = PurgeResult(n_points=2, chip_hashes=[chash])
    store.unreferenced_chip_hashes.return_value = [chash]
    with (
        patch("scalar_forensic.faces.store.FaceStore", return_value=store),
        patch("qdrant_client.QdrantClient", MagicMock()),
    ):
        result = runner.invoke(faces_app, ["purge", "--media", "h1"])

    assert result.exit_code == 0, result.output
    assert "Purged 2 face point(s) and 3 chip file(s)." in result.output
    assert not png.exists() and not jpg.exists() and not thumb.exists()

    import json

    events = [
        json.loads(line) for line in (tmp_path / "face_audit.log").read_text().splitlines() if line
    ]
    assert events[-1]["event"] == "purge"
    assert events[-1]["examiner_id"] == "ex1" and events[-1]["n_points"] == 2


def test_purge_keeps_chips_still_referenced(monkeypatch, tmp_path):
    # Chips are content-addressed and therefore shared: purging one medium must
    # not unlink a chip a surviving observation still authenticates.
    _enable_faces(monkeypatch, tmp_path)
    store_dir = tmp_path / "faces"
    from scalar_forensic.faces.chips import chip_paths

    gone = "ab" + "0" * 62
    shared = "cd" + "1" * 62
    for chash in (gone, shared):
        png, jpg, thumb = chip_paths(store_dir, chash)
        png.parent.mkdir(parents=True, exist_ok=True)
        for p in (png, jpg, thumb):
            p.write_bytes(b"x")

    store = MagicMock()
    store.purge_media.return_value = PurgeResult(n_points=2, chip_hashes=[gone, shared])
    store.unreferenced_chip_hashes.return_value = [gone]
    with (
        patch("scalar_forensic.faces.store.FaceStore", return_value=store),
        patch("qdrant_client.QdrantClient", MagicMock()),
    ):
        result = runner.invoke(faces_app, ["purge", "--media", "h1"])

    assert result.exit_code == 0, result.output
    store.unreferenced_chip_hashes.assert_called_once_with([gone, shared])
    assert not any(p.exists() for p in chip_paths(store_dir, gone))
    assert all(p.exists() for p in chip_paths(store_dir, shared))
    assert "3 chip file(s)" in result.output


def test_purge_unlinks_review_only_chip_pair(monkeypatch, tmp_path):
    # A review-only observation has no aligned PNG: the unlink loop must reach
    # the review pair through review_chip_paths, and must not count a missing PNG.
    _enable_faces(monkeypatch, tmp_path)
    store_dir = tmp_path / "faces"
    from scalar_forensic.faces.chips import review_chip_paths

    chash = "ef" + "2" * 62
    jpg, thumb = review_chip_paths(store_dir, chash)
    jpg.parent.mkdir(parents=True, exist_ok=True)
    for p in (jpg, thumb):
        p.write_bytes(b"x")

    store = MagicMock()
    store.purge_media.return_value = PurgeResult(n_points=1, chip_hashes=[chash])
    store.unreferenced_chip_hashes.return_value = [chash]
    with (
        patch("scalar_forensic.faces.store.FaceStore", return_value=store),
        patch("qdrant_client.QdrantClient", MagicMock()),
    ):
        result = runner.invoke(faces_app, ["purge", "--media", "h1"])

    assert result.exit_code == 0, result.output
    assert not jpg.exists() and not thumb.exists()
    # Two files, not four: chip_paths and review_chip_paths overlap on the
    # review pair, and the absent PNG must not inflate the audited count.
    assert "2 chip file(s)" in result.output


def test_purge_all_aborts_without_confirmation(monkeypatch, tmp_path):
    _enable_faces(monkeypatch, tmp_path)
    store = MagicMock()
    with (
        patch("scalar_forensic.faces.store.FaceStore", return_value=store),
        patch("qdrant_client.QdrantClient", MagicMock()),
    ):
        result = runner.invoke(faces_app, ["purge", "--all"], input="n\n")
    assert result.exit_code == 1
    assert "Aborted." in result.output
    store.purge_all.assert_not_called()
