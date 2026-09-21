import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scalar_forensic.faces import chips
from scalar_forensic.media_worker.client import WorkerUnavailable
from scalar_forensic.web.app import app
from scalar_forensic.web.routes import files


@pytest.fixture
def configured_missing_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("SFN_MEDIA_WORKER_SOCKET", str(tmp_path / "ipc" / "absent.sock"))
    monkeypatch.setenv("SFN_MEDIA_WORKER_GENERATION", uuid.uuid4().hex)
    monkeypatch.setenv("SFN_MEDIA_WORKER_CASE", "case-a")
    monkeypatch.setenv("SFN_MEDIA_WORKER_RECEIPTS", str(tmp_path / "receipts.jsonl"))
    return tmp_path


def test_face_thumbnail_cannot_fall_back_to_local_decode(configured_missing_worker):
    root = configured_missing_worker
    source = root / "review.jpg"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(source)
    with pytest.raises(WorkerUnavailable):
        chips.write_thumbnail(source, root / "thumb.jpg", 4)
    assert not (root / "thumb.jpg").exists()


def test_heif_view_cannot_fall_back_to_local_decode(configured_missing_worker):
    source = configured_missing_worker / "source.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(source)
    with pytest.raises(WorkerUnavailable):
        files._transcode_heif_to_jpeg(str(source), source.stat().st_mtime_ns)


def test_thumbnail_regeneration_does_not_publish_when_worker_missing(
    configured_missing_worker, monkeypatch
):
    root = configured_missing_worker
    source = root / "source.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(source)
    point = SimpleNamespace(payload={"image_path": str(source)})
    monkeypatch.setattr(
        files,
        "QdrantClient",
        lambda **kwargs: SimpleNamespace(scroll=lambda **kwargs: ([point], None)),
    )
    settings = SimpleNamespace(
        qdrant_url="unused",
        qdrant_api_key=None,
        collection="test",
        input_dir=root,
        frame_store_dir=None,
        thumbnail_size=4,
    )
    dest = root / "thumbnail.jpg"
    with pytest.raises(WorkerUnavailable):
        asyncio.run(files._try_regenerate_thumbnail("a" * 64, dest, settings))
    assert not dest.exists()


def test_worker_failure_is_visible_as_service_unavailable(configured_missing_worker, monkeypatch):
    root = configured_missing_worker
    source = root / "image.heic"
    Image.new("RGB", (8, 8)).save(source, format="PNG")
    monkeypatch.setattr(
        "scalar_forensic.web.routes._shared.Settings",
        lambda: SimpleNamespace(input_dir=root, frame_store_dir=None),
    )
    monkeypatch.setattr(files.scanner, "_HEIF_AVAILABLE", True)
    result = TestClient(app, raise_server_exceptions=False).get(
        "/api/hit-image", params={"path": str(source)}
    )
    assert result.status_code == 503
    assert "worker" in result.json()["detail"].lower()
