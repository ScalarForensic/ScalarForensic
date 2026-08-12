from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from scalar_forensic.web.app import app

client = TestClient(app)


def test_availability_false_when_disabled(monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    resp = client.get("/api/faces/availability")
    assert resp.status_code == 200
    body = resp.json()
    assert body["faces_available"] is False and "SFN_FACES_ENABLED" in body["reason"]


def _settings(tmp_path, **over):
    s = MagicMock()
    s.faces_enabled = True
    s.face_startup_error.return_value = None
    s.face_collection = "case1_faces"
    s.collection = "case1"
    s.face_store_dir = tmp_path
    s.face_thumb_size = 256
    s.qdrant_url = "http://localhost:6333"
    s.qdrant_api_key = None
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_availability_true_when_configured(tmp_path):
    store = MagicMock()
    store.collection_is_new.return_value = False
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        body = client.get("/api/faces/availability").json()
    assert body["faces_available"] is True and body["reason"] is None


def test_availability_notes_degraded_evidence_without_store_dir(tmp_path):
    store = MagicMock()
    store.collection_is_new.return_value = False
    with (
        patch(
            "scalar_forensic.web.routes.faces.Settings",
            return_value=_settings(tmp_path, face_store_dir=None),
        ),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        body = client.get("/api/faces/availability").json()
    assert body["faces_available"] is True
    assert "degraded-evidence" in body["note"]


def test_by_image_returns_store_payloads(tmp_path):
    store = MagicMock()
    store.collection_is_new.return_value = False
    store.list_faces.return_value = [{"id": "p1", "chip_hash": "c" * 64, "quality": 0.8}]
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        resp = client.get(f"/api/faces/by-image/{'a' * 64}")
    assert resp.status_code == 200
    assert resp.json()["faces"][0]["chip_hash"] == "c" * 64


def test_by_image_rejects_non_hash_path_component(tmp_path):
    with patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)):
        resp = client.get("/api/faces/by-image/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)


def test_chip_404_when_missing(tmp_path):
    with patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)):
        resp = client.get(f"/api/faces/chip/{'b' * 64}")
    assert resp.status_code == 404


def test_chip_rejects_invalid_hash(tmp_path):
    with patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)):
        resp = client.get("/api/faces/chip/not-a-hash")
    assert resp.status_code == 400


def test_chip_serves_png_and_review_jpeg(tmp_path):
    import numpy as np

    from scalar_forensic.faces.chips import write_chips

    rng = np.random.default_rng(0)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    chash = write_chips(
        tmp_path, aligned, source, bbox=(100, 100, 200, 200), dilation=0.15, thumb_size=256
    )
    with patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)):
        png = client.get(f"/api/faces/chip/{chash}")
        review = client.get(f"/api/faces/chip/{chash}/review")
        thumb = client.get(f"/api/faces/chip/{chash}/thumb")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert review.status_code == 200 and review.headers["content-type"] == "image/jpeg"
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


def test_thumb_regenerates_when_missing(tmp_path):
    import numpy as np

    from scalar_forensic.faces.chips import chip_paths, write_chips

    rng = np.random.default_rng(1)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    chash = write_chips(
        tmp_path, aligned, source, bbox=(50, 50, 300, 300), dilation=0.15, thumb_size=128
    )
    _, _, thumb_path = chip_paths(tmp_path, chash)
    thumb_path.unlink()
    with patch(
        "scalar_forensic.web.routes.faces.Settings",
        return_value=_settings(tmp_path, face_thumb_size=128),
    ):
        resp = client.get(f"/api/faces/chip/{chash}/thumb")
    assert resp.status_code == 200
    assert thumb_path.exists()  # self-healed, derived artefact
