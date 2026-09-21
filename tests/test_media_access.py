import io
import json
import multiprocessing
import uuid
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from scalar_forensic import media_decode


def access():
    from scalar_forensic import media_access

    return media_access


def source(mode="RGB", orientation=1):
    image = Image.new(mode, (24, 16), 127 if mode == "L" else (10, 20, 30))
    encoded = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    image.save(encoded, format="PNG", exif=exif)
    return encoded.getvalue()


def run_worker(listener, generation):
    from scalar_forensic.media_worker.server import serve

    raise SystemExit(serve(listener, generation=generation, max_seconds=2, max_requests=20))


@contextmanager
def worker(tmp_path, monkeypatch):
    from scalar_forensic.media_worker.server import bound_listener

    ipc = tmp_path / "ipc"
    ipc.mkdir()
    path = ipc / "worker.sock"
    log = tmp_path / "receipts.jsonl"
    generation = uuid.uuid4().hex
    with bound_listener(path) as listener:
        process = multiprocessing.get_context("fork").Process(
            target=run_worker, args=(listener, generation)
        )
        process.start()
        monkeypatch.setenv("SFN_MEDIA_WORKER_SOCKET", str(path))
        monkeypatch.setenv("SFN_MEDIA_WORKER_GENERATION", generation)
        monkeypatch.setenv("SFN_MEDIA_WORKER_CASE", "case-a")
        monkeypatch.setenv("SFN_MEDIA_WORKER_RECEIPTS", str(log))
        try:
            yield log
        finally:
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join(3)


def test_local_mode_matches_existing_pixels():
    actual = access().preprocess_batch([source()], 4)[0]
    assert actual.size == (6, 4)
    assert actual.tobytes() == bytes((10, 20, 30)) * 24


def test_partial_worker_configuration_never_selects_local_mode(monkeypatch):
    monkeypatch.setenv("SFN_MEDIA_WORKER_SOCKET", "/missing")
    with pytest.raises(ValueError):
        access().extract_exif(source())


@pytest.mark.parametrize(
    "setting",
    [
        "SFN_MEDIA_WORKER_STATE",
        "SFN_MEDIA_WORKER_SOCKET",
        "SFN_MEDIA_WORKER_GENERATION",
        "SFN_MEDIA_WORKER_CASE",
        "SFN_MEDIA_WORKER_RECEIPTS",
    ],
)
def test_empty_worker_configuration_never_decodes_locally(monkeypatch, setting):
    monkeypatch.setenv(setting, "")
    with pytest.raises(ValueError):
        access().display_rgb(source())


def test_remote_mode_matches_native_and_embedding_orientation(tmp_path, monkeypatch):
    data = source(orientation=6)
    expected_native, expected_derived = media_decode.decode_shared(data, 4)
    assert expected_derived is None
    with worker(tmp_path, monkeypatch) as log:
        native, derived = access().decode_shared(data, 4)
        assert derived is None
        np.testing.assert_array_equal(native, expected_native)
        image = access().preprocess_batch([data], 4)[0]
        assert image.size == (6, 4)
        assert access().extract_exif(data) == {"exif": True, "exif_geo_data": False}
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert [row["event"] for row in rows] == ["started", "decoded"] * 3
        assert {row["generation"] for row in rows} == {access().configured_worker().generation}


def test_remote_wrappers_never_call_local_decoders(tmp_path, monkeypatch):
    with worker(tmp_path, monkeypatch):

        def forbidden(*args, **kwargs):
            pytest.fail("trusted process attempted local encoded-file decoding")

        for name in (
            "_open_rgb",
            "load_for_detection",
            "extract_exif",
            "extract_exif_detailed",
            "decode_shared",
            "display_rgb",
            "thumbnail_input",
            "preprocess_batch",
        ):
            monkeypatch.setattr(media_decode, name, forbidden)
        data = source()
        assert access()._open_rgb(data).size == (24, 16)
        assert access().load_for_detection(data).shape == (16, 24, 3)
        assert access().display_rgb(data).mode == "RGB"
        assert access().thumbnail_input(data, 4).size == (4, 3)
        assert access().extract_exif_detailed(data)["width"] == 24


def test_remote_grayscale_thumbnail_retains_mode(tmp_path, monkeypatch):
    with worker(tmp_path, monkeypatch):
        image = access().thumbnail_input(source(mode="L"), 4)
        assert image.mode == "L"
        assert image.size == (4, 3)


def test_original_source_bytes_unchanged(tmp_path, monkeypatch):
    data = source()
    original = tmp_path / "original.png"
    original.write_bytes(data)
    with worker(tmp_path, monkeypatch):
        assert access().preprocess_batch([original.read_bytes()], 4)[0].size == (6, 4)
    assert original.read_bytes() == data


@pytest.mark.parametrize(
    "module,name",
    [
        ("scalar_forensic.embedder", "_open_rgb"),
        ("scalar_forensic.embedder", "extract_exif"),
        ("scalar_forensic.embedder", "extract_exif_detailed"),
        ("scalar_forensic.faces.decode", "load_for_detection"),
        ("scalar_forensic.faces.indexing", "decode_shared"),
    ],
)
def test_existing_entrypoints_fail_closed_when_worker_missing(tmp_path, monkeypatch, module, name):
    import importlib

    from scalar_forensic.media_worker.client import WorkerUnavailable

    helper = getattr(importlib.import_module(module), name)
    monkeypatch.setenv("SFN_MEDIA_WORKER_SOCKET", str(tmp_path / "ipc" / "absent.sock"))
    monkeypatch.setenv("SFN_MEDIA_WORKER_GENERATION", uuid.uuid4().hex)
    monkeypatch.setenv("SFN_MEDIA_WORKER_CASE", "case-a")
    monkeypatch.setenv("SFN_MEDIA_WORKER_RECEIPTS", str(tmp_path / "receipts.jsonl"))
    with pytest.raises(WorkerUnavailable):
        if name == "decode_shared":
            helper(source(), 4)
        else:
            helper(source())


@pytest.mark.parametrize(
    "setting,value",
    [
        ("SFN_MEDIA_WORKER_TIMEOUT", "41"),
        ("SFN_MEDIA_WORKER_CASE", "case-b"),
        ("SFN_MEDIA_WORKER_RECEIPTS", "/tmp/different-receipts.jsonl"),
    ],
)
def test_generation_configuration_cannot_reset_invalidation(tmp_path, monkeypatch, setting, value):
    from scalar_forensic.media_worker.client import WorkerUnavailable

    monkeypatch.setenv("SFN_MEDIA_WORKER_SOCKET", str(tmp_path / "ipc" / "absent.sock"))
    monkeypatch.setenv("SFN_MEDIA_WORKER_GENERATION", uuid.uuid4().hex)
    monkeypatch.setenv("SFN_MEDIA_WORKER_CASE", "case-a")
    monkeypatch.setenv("SFN_MEDIA_WORKER_RECEIPTS", str(tmp_path / "receipts.jsonl"))
    with pytest.raises(WorkerUnavailable):
        access().extract_exif(source())
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError, match="generation"):
        access().configured_worker()


def test_remote_source_reader_rejects_size_before_reading(tmp_path, monkeypatch):
    from pathlib import Path

    from scalar_forensic.media_worker import protocol

    original = tmp_path / "original"
    original.write_bytes(b"x" * 17)
    monkeypatch.setenv("SFN_MEDIA_WORKER_SOCKET", str(tmp_path / "ipc" / "absent.sock"))
    monkeypatch.setenv("SFN_MEDIA_WORKER_GENERATION", uuid.uuid4().hex)
    monkeypatch.setenv("SFN_MEDIA_WORKER_CASE", "case-a")
    monkeypatch.setenv("SFN_MEDIA_WORKER_RECEIPTS", str(tmp_path / "receipts.jsonl"))
    monkeypatch.setattr(protocol, "MAX_SOURCE", 16)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("unbounded read"))
    with pytest.raises(protocol.ProtocolError, match="source"):
        access().read_source(original)
