"""Regressions for marker-only provenance, modality disablement and bad uploads."""

import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scalar_forensic.faces.chips import chip_paths
from scalar_forensic.faces.store import FaceStore
from scalar_forensic.web.app import app
from scalar_forensic.web.pipeline.faces_stats import face_audit
from scalar_forensic.web.routes import faces as routes
from scalar_forensic.web.session import Session

HASH = "a" * 64
POINT = "11111111-1111-1111-1111-111111111111"
PROVENANCE = {
    "identity_schema_version": 2,
    "embedder_model_hash": "e" * 64,
    "manifest_hash": "f" * 64,
    "embedder_dim": 128,
    "alignment_version": "arcface-112-v1",
    "normalization_id": "affine-127.5-128.0",
}


class RecordedStore(FaceStore):
    """Persisted fixture at the storage boundary; compatibility stays real."""

    def __init__(self, observations):
        super().__init__(None, "case_faces", "case", 128)
        self.observations = observations

    def _meta_payload(self):
        return PROVENANCE.copy()

    def get_marker(self, *args, **kwargs):
        return {"n_detected": 0, "n_kept": 0, "n_rejected": {}}

    def list_faces(self, *args, **kwargs):
        return self.observations

    def get_face(self, point_id):
        return self.observations[0]


def test_marker_only_audit_reports_unknown_compatibility():
    body = face_audit(RecordedStore([]), HASH)
    assert body["n_observations"] == 0
    assert body["compat"]["ok"] is None
    assert "unknown" in " ".join(body["compat"]["warnings"]).lower()


@pytest.mark.parametrize("model_hash, expected", [("e" * 64, True), ("b" * 64, False)])
def test_audit_preserves_real_provenance_comparison(model_hash, expected):
    row = {**PROVENANCE, "embedder_model_hash": model_hash}
    body = face_audit(RecordedStore([row]), HASH)
    assert body["compat"]["ok"] is expected


READ_PATHS = [
    f"/api/faces/by-image/{HASH}",
    f"/api/faces/explain/{POINT}",
    f"/api/faces/chip/{HASH}",
    f"/api/faces/chip/{HASH}/review",
    f"/api/faces/chip/{HASH}/thumb",
    f"/api/faces/audit?image_hash={HASH}",
]


@pytest.mark.parametrize("path", READ_PATHS)
def test_disabled_modality_blocks_evidence_before_storage_access(monkeypatch, tmp_path, path):
    settings = SimpleNamespace(faces_enabled=False, face_store_dir=tmp_path)
    monkeypatch.setattr(routes, "Settings", lambda: settings)

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled face modality accessed storage")

    monkeypatch.setattr(routes, "_store", forbidden)
    monkeypatch.setattr(routes, "chip_paths", forbidden)
    response = TestClient(app).get(path)
    assert response.status_code == 503
    assert "disabled" in response.json()["detail"].lower()


@pytest.mark.parametrize("path", READ_PATHS)
def test_enabled_evidence_remains_viewable_without_inference_weights(monkeypatch, tmp_path, path):
    settings = SimpleNamespace(
        faces_enabled=True,
        face_store_dir=tmp_path,
        face_startup_error=lambda: "Inference model is missing",
    )
    monkeypatch.setattr(routes, "Settings", lambda: settings)
    store = RecordedStore([{**PROVENANCE, "image_hash": HASH, "bbox": [1, 2, 3, 4]}])
    monkeypatch.setattr(routes, "_store", lambda _: store)
    for chip in chip_paths(tmp_path, HASH):
        chip.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red").save(chip)
    response = TestClient(app).get(path)
    assert response.status_code == 200
    if "/chip/" in path:
        assert Image.open(io.BytesIO(response.content)).size == (8, 8)
    elif "/by-image/" in path:
        assert response.json()["faces"][0]["bbox"] == [1, 2, 3, 4]
    elif "/explain/" in path:
        assert response.json()["geometry"]["bbox"] == [1, 2, 3, 4]
    else:
        assert response.json()["compat"]["ok"] is True


def _upload(monkeypatch, tmp_path):
    image = tmp_path / "query.png"
    Image.new("RGB", (20, 20)).save(image)
    entry = SimpleNamespace(is_video=False, temp_path=image, _query_faces_generation=0)
    session = Session("session", [entry])
    monkeypatch.setattr(routes, "_resolve_entry", lambda *args: entry)
    monkeypatch.setattr(routes, "get_session", lambda _: session)
    monkeypatch.setattr(
        routes,
        "Settings",
        lambda: SimpleNamespace(
            faces_enabled=True, face_startup_error=lambda: None, face_query_max_faces=25
        ),
    )


def test_decompression_bomb_upload_returns_bad_image(monkeypatch, tmp_path):
    _upload(monkeypatch, tmp_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/faces/query-faces", data={"session_id": "session", "file_id": "file"}
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_unrelated_detection_runtime_error_is_not_bad_image(monkeypatch, tmp_path):
    _upload(monkeypatch, tmp_path)

    def broken_inference(*args, **kwargs):
        raise RuntimeError("inference engine failed")

    monkeypatch.setattr(routes, "detect_query_faces", broken_inference)
    with pytest.raises(RuntimeError, match="inference engine failed"):
        routes.query_faces("session", "file", None)


def test_disabled_modality_blocks_session_chip(monkeypatch):
    monkeypatch.setattr(routes, "Settings", lambda: SimpleNamespace(faces_enabled=False))

    def forbidden(*args):
        pytest.fail("Disabled face modality read a session crop")

    monkeypatch.setattr(routes, "_resolve_entry", forbidden)
    response = TestClient(app).get("/api/faces/query-chip/session/file/token/0")
    assert response.status_code == 503
