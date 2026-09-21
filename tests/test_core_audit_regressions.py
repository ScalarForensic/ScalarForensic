import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from scalar_forensic.calibration import (
    cache_identity,
    load_cached_batch_size,
    save_cached_batch_size,
)
from scalar_forensic.config import Settings
from scalar_forensic.web.pipeline.modes import get_available_modes
from scalar_forensic.web.routes import files, tags
from tests.faces.test_cli_faces import run_faces_cli as run_faces_cli
from tests.test_ingest_characterization import (
    _run_index,
)
from tests.test_ingest_characterization import (
    fake_client_cls as fake_client_cls,
)
from tests.test_ingest_characterization import (
    ingest_env as ingest_env,
)
from tests.test_ingest_characterization import (
    store as store,
)


@pytest.mark.parametrize(
    "field,value", [("device", "cuda"), ("normalize_size", 512), ("sscd_n_crops", 1)]
)
def test_calibration_rejects_changed_runtime(tmp_path, field, value):
    settings = Settings()
    settings.device = "cpu"
    settings.normalize_size = 224
    settings.sscd_n_crops = 5
    identity = cache_identity(settings, vector_names=["sscd"])
    path = tmp_path / "batch.json"
    save_cached_batch_size(64, identity=identity, cache_file=path)
    assert load_cached_batch_size(path, identity=identity) == 64
    setattr(settings, field, value)
    assert (
        load_cached_batch_size(path, identity=cache_identity(settings, vector_names=["sscd"]))
        is None
    )


def test_calibration_cache_rejects_different_model_selection(tmp_path):
    settings = Settings()
    path = tmp_path / "batch.json"
    save_cached_batch_size(
        64, identity=cache_identity(settings, vector_names=["sscd"]), cache_file=path
    )
    assert (
        load_cached_batch_size(path, identity=cache_identity(settings, vector_names=["sscd"])) == 64
    )
    assert (
        load_cached_batch_size(
            path, identity=cache_identity(settings, vector_names=["sscd", "dino"])
        )
        is None
    )
    save_cached_batch_size(
        16, identity=cache_identity(settings, vector_names=["sscd", "dino"]), cache_file=path
    )
    assert (
        load_cached_batch_size(
            path, identity=cache_identity(settings, vector_names=["dino", "sscd"])
        )
        == 16
    )


def test_calibration_cache_without_known_selection_cannot_be_reused(tmp_path):
    path = tmp_path / "batch.json"
    identity = cache_identity(Settings())
    save_cached_batch_size(64, identity=identity, cache_file=path)
    assert load_cached_batch_size(path, identity=identity) is None


@pytest.mark.parametrize("dino,expected", [(False, 64), (True, 32)])
def test_cli_cache_reuse_matches_selected_pipeline(ingest_env, monkeypatch, capsys, dino, expected):
    monkeypatch.delenv("SFN_BATCH_SIZE")
    settings = Settings()
    save_cached_batch_size(64, identity=cache_identity(settings, vector_names=["sscd"]))
    _run_index(ingest_env.images, ingest_env.tmp / "report.csv", dino=dino, sscd=True)
    output = capsys.readouterr()
    if expected == 64:
        assert "Batch size: 64  (calibration cache)" in output.out
    else:
        assert "using batch_size=32" in output.err


def test_startup_defers_cache_resolution_until_request_models_are_known(
    tmp_path, monkeypatch, caplog
):
    from scalar_forensic.web.app import app, lifespan

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SFN_SESSION_TEMP_ROOT", str(tmp_path / "sessions"))
    save_cached_batch_size(64, identity=cache_identity(Settings(), vector_names=["sscd"]))
    with patch("scalar_forensic.web.app._check_collection_compat"), caplog.at_level("INFO"):

        async def start_stop():
            async with lifespan(app):
                pass

        asyncio.run(start_stop())
    assert "resolved per request" in caplog.text
    assert "Batch size: 64" not in caplog.text


@pytest.mark.parametrize("vectors,expected_batches", [(["sscd"], [2, 1]), (["sscd", "dino"], [3])])
def test_video_analysis_uses_cache_for_its_active_models_only(
    tmp_path, monkeypatch, vectors, expected_batches
):
    from scalar_forensic.web.pipeline import analysis

    monkeypatch.chdir(tmp_path)
    settings = Settings()
    settings.batch_size = None
    save_cached_batch_size(2, identity=cache_identity(settings, vector_names=["sscd"]))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"hashable test video")
    batches = {key: [] for key in vectors}

    class Embedder:
        normalize_size = 336

        def __init__(self, key):
            self.key = key

        def embed_images(self, images):
            batches[self.key].append(len(images))
            return [[1.0, 0.0]] * len(images)

    frames = [
        SimpleNamespace(
            image=Image.new("RGB", (16, 16)),
            frame_index=i,
            timecode_ms=i * 1000,
            frame_hash=f"frame-{i}",
        )
        for i in range(3)
    ]
    monkeypatch.setattr(analysis, "extract_frames", lambda *a, **k: iter(frames))
    entry = SimpleNamespace(temp_path=source, filename="source.mp4", file_id="video")
    list(analysis._analyze_video_file(entry, {key: Embedder(key) for key in vectors}, settings))
    assert batches == {key: expected_batches for key in vectors}
    assert len(entry.video_frames) == 3


def test_legacy_calibration_is_not_reused(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text('{"batch_size": 64, "device": "cuda"}')
    assert load_cached_batch_size(path, identity=cache_identity(Settings())) is None


def test_faces_only_skips_embedding_calibration_warning(run_faces_cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SFN_BATCH_SIZE", raising=False)
    (tmp_path / "data" / "sample_images").mkdir(parents=True)
    result, _, _ = run_faces_cli(detected=0, kept=0)
    assert result.exit_code == 0, result.output
    assert "remote embedder" not in result.output
    assert "Auto-calibration" not in result.output


def test_cli_reports_current_heif_availability(run_faces_cli, tmp_path, monkeypatch):
    from scalar_forensic import scanner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scanner, "_HEIF_AVAILABLE", False)
    first, _, _ = run_faces_cli(detected=0, kept=0)
    assert first.exit_code == 0, first.output
    assert "disabled (install pillow-heif)" in first.output
    monkeypatch.setattr(scanner, "_HEIF_AVAILABLE", True)
    second, _, _ = run_faces_cli(detected=0, kept=0)
    assert second.exit_code == 0, second.output
    assert "enabled (pillow-heif)" in second.output


@pytest.mark.parametrize("query", [False, True])
@pytest.mark.parametrize("suffix", ["png", "jpg"])
def test_preview_preserves_configured_resolution(tmp_path, query, suffix):
    path = tmp_path / f"source.{suffix}"
    image = Image.new("RGB", (1536, 1536))
    image.save(path)
    settings = Settings()
    settings.normalize_size = 512
    entry = SimpleNamespace(file_id="f", temp_path=path, is_video=False)
    with (
        patch.object(files, "Settings", return_value=settings),
        patch.object(files, "_check_allowed_path"),
        patch.object(files, "get_session", return_value=SimpleNamespace(files=[entry])),
        patch.object(files, "_build_preproc_payload", return_value={}) as build,
    ):
        if query:
            asyncio.run(files.query_preprocessed("s", "f"))
        else:
            asyncio.run(files.hit_preprocessed(str(path), 5, 512))
    assert build.call_args.args[0].size == (512, 512)


@pytest.mark.parametrize("fail", [False, True])
def test_tag_store_client_closed_after_operation(fail):
    store = MagicMock()
    store.list.return_value = []
    if fail:
        store.list.side_effect = RuntimeError("unavailable")
    with patch.object(tags, "_tag_store", return_value=store):
        if fail:
            with pytest.raises(Exception):
                asyncio.run(tags.list_tags())
        else:
            asyncio.run(tags.list_tags())
    store.client.close.assert_called_once()


def test_tag_constructor_failure_closes_client():
    client = MagicMock()
    with (
        patch.object(tags, "QdrantClient", return_value=client),
        patch.object(tags, "TagStore", side_effect=RuntimeError("schema unavailable")),
        pytest.raises(RuntimeError, match="schema unavailable"),
    ):
        tags._tag_client_and_store()
    client.close.assert_called_once()


def test_triage_missing_tag_closes_client():
    client, store = MagicMock(), MagicMock()
    store.get.return_value = None
    with (
        patch.object(tags, "_tag_client_and_store", return_value=(client, store)),
        pytest.raises(Exception) as caught,
    ):
        asyncio.run(tags.triage("missing"))
    assert caught.value.status_code == 404
    client.close.assert_called_once()


@pytest.mark.parametrize("fail", [False, True])
def test_modes_client_closed_after_vector_inspection(fail):
    settings = Settings()
    client = MagicMock()
    client.get_collections.return_value.collections = [SimpleNamespace(name=settings.collection)]
    client.get_collection.return_value.config.params.vectors = {"dino": {}}
    if fail:
        client.get_collection.side_effect = RuntimeError("inspection failed")
    with patch("scalar_forensic.web.pipeline.modes.QdrantClient", return_value=client):
        modes, _, error = asyncio.run(get_available_modes(settings))
    assert modes == (["exact"] if fail else ["exact", "semantic"])
    assert (error is not None) == fail
    client.close.assert_called_once()


@pytest.mark.parametrize("fail", [False, True])
def test_modes_client_closed_for_empty_or_failed_probe(fail):
    client = MagicMock()
    client.get_collections.return_value.collections = []
    if fail:
        client.get_collections.side_effect = RuntimeError("offline")
    with (
        patch("scalar_forensic.web.pipeline.modes.QdrantClient", return_value=client),
        patch("scalar_forensic.web.pipeline.modes.asyncio.sleep"),
    ):
        asyncio.run(get_available_modes(Settings()))
    assert client.close.call_count == (4 if fail else 1)
