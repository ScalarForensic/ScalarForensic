from unittest.mock import MagicMock, patch

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
