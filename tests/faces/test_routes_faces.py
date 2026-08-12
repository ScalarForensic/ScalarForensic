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

    from scalar_forensic.faces.chips import write_aligned_chips

    rng = np.random.default_rng(0)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    # Aligned PNG and review JPEG live under different hashes (domain-separated).
    ahash, rhash = write_aligned_chips(
        tmp_path, aligned, source, bbox=(100, 100, 200, 200), dilation=0.15, thumb_size=256
    )
    with patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)):
        png = client.get(f"/api/faces/chip/{ahash}")
        review = client.get(f"/api/faces/chip/{rhash}/review")
        thumb = client.get(f"/api/faces/chip/{rhash}/thumb")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert review.status_code == 200 and review.headers["content-type"] == "image/jpeg"
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


def test_thumb_regenerates_when_missing(tmp_path):
    import numpy as np

    from scalar_forensic.faces.chips import review_chip_paths, write_aligned_chips

    rng = np.random.default_rng(1)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    _, rhash = write_aligned_chips(
        tmp_path, aligned, source, bbox=(50, 50, 300, 300), dilation=0.15, thumb_size=128
    )
    _, thumb_path = review_chip_paths(tmp_path, rhash)
    thumb_path.unlink()
    with patch(
        "scalar_forensic.web.routes.faces.Settings",
        return_value=_settings(tmp_path, face_thumb_size=128),
    ):
        resp = client.get(f"/api/faces/chip/{rhash}/thumb")
    assert resp.status_code == 200
    assert thumb_path.exists()  # self-healed, derived artefact


def _stored_face():
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "is_face": True,
        "image_hash": "a" * 64,
        "image_path": "/case/x.jpg",
        "video_hash": None,
        "video_path": None,
        "frame_timecode_ms": None,
        "observation_key": f"{'a' * 64}::10:20:30:40",
        "bbox": [10, 20, 30, 40],
        "landmarks": [[12, 25], [30, 25], [21, 32], [14, 38], [28, 38]],
        "det_conf": 0.93,
        "detect_scale": 1.0,
        "quality_confidence": 0.93,
        "quality_size": 30.0,
        "quality_pose": 0.05,
        "quality_sharpness": 140.0,
        "quality_exposure": 0.01,
        "quality": 0.86,
        "embedding_norm": 21.7,
        "chip_hash": "c" * 64,
        "indexed_at": "2026-08-12T06:00:00+00:00",
        # provenance as recorded at index time
        "detector_id": "yunet",
        "detector_model_hash": "d" * 64,
        "detector_score_threshold": 0.5,
        "detect_max_size": 1600,
        "embedder_model_name": "emb.onnx",
        "embedder_model_hash": "e" * 64,
        "manifest_hash": "m" * 64,
        "embedder_dim": 512,
        "alignment_version": "arcface-112-v1",
        "normalization_id": "affine-127.5-128.0",
        "min_conf": 0.8,
        "min_size": 64,
        "min_sharpness": 25.0,
        "max_clipped": 0.6,
        "max_pose": 0.35,
        "crop_dilation": 0.15,
        "pipeline_config_hash": "f" * 64,
    }


def test_explain_returns_bundle_from_persisted_data(tmp_path):
    store = MagicMock()
    store.get_face.return_value = _stored_face()
    store.get_marker.return_value = {"n_detected": 14, "n_kept": 3, "n_rejected": {"size": 6}}
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        body = client.get("/api/faces/explain/11111111-1111-1111-1111-111111111111").json()

    assert body["source"]["image_hash"] == "a" * 64
    assert body["geometry"]["bbox"] == [10, 20, 30, 40]
    assert len(body["geometry"]["landmarks"]) == 5
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["detection"]["passed"] is True
    assert steps["detection"]["thresholds_in_force"]["min_conf"] == 0.8
    assert steps["pre_align_gate"]["scores"]["pose"] == 0.05
    assert steps["post_align_gate"]["scores"]["sharpness"] == 140.0
    assert body["embedder"]["embedder_model_hash"] == "e" * 64
    assert body["file_totals"]["n_detected"] == 14
    assert body["chips"]["aligned"].endswith("c" * 64)


def test_explain_thresholds_come_from_stored_config_not_env(tmp_path, monkeypatch):
    # Env says 0.99; the stored observation was indexed at 0.8.  The explainer
    # describes what happened at index time, so the stored value must win.
    monkeypatch.setenv("SFN_FACE_MIN_CONF", "0.99")
    store = MagicMock()
    store.get_face.return_value = _stored_face()
    store.get_marker.return_value = None
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        body = client.get("/api/faces/explain/11111111-1111-1111-1111-111111111111").json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["detection"]["thresholds_in_force"]["min_conf"] == 0.8


def test_explain_404_on_unknown_point(tmp_path):
    store = MagicMock()
    store.get_face.return_value = None
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        resp = client.get("/api/faces/explain/11111111-1111-1111-1111-111111111111")
    assert resp.status_code == 404


def test_explain_validates_point_id_before_store_access(tmp_path):
    store = MagicMock()
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
    ):
        resp = client.get("/api/faces/explain/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)
    store.get_face.assert_not_called()
