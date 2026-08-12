import pytest

from scalar_forensic.config import Settings


def test_faces_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    s = Settings()
    assert s.faces_enabled is False
    assert s.face_collection == f"{s.collection}_faces"
    assert s.face_detect_max_size == 1600
    assert s.face_crop_dilation == pytest.approx(0.15)


def test_face_collection_follows_case_collection(monkeypatch):
    monkeypatch.setenv("SFN_COLLECTION", "case42")
    assert Settings().face_collection == "case42_faces"


def test_crop_dilation_bounds(monkeypatch):
    monkeypatch.setenv("SFN_FACE_CROP_DILATION", "0.9")
    with pytest.raises(ValueError, match="SFN_FACE_CROP_DILATION"):
        Settings()


def test_startup_error_when_enabled_without_models(monkeypatch, tmp_path):
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_EXAMINER_ID", "ex1")
    monkeypatch.delenv("SFN_FACE_DETECTOR_MODEL", raising=False)
    err = Settings().face_startup_error()
    assert err is not None and "SFN_FACE_DETECTOR_MODEL" in err


def test_startup_error_requires_examiner_id(monkeypatch, tmp_path):
    det = tmp_path / "yunet.onnx"
    det.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"
    emb.write_bytes(b"x")
    (tmp_path / "emb.onnx.manifest.json").write_text("{}")
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_FACE_DETECTOR_MODEL", str(det))
    monkeypatch.setenv("SFN_FACE_EMBEDDER_MODEL", str(emb))
    monkeypatch.delenv("SFN_EXAMINER_ID", raising=False)
    err = Settings().face_startup_error()
    assert err is not None and "SFN_EXAMINER_ID" in err


def test_no_startup_error_when_fully_configured(monkeypatch, tmp_path):
    det = tmp_path / "yunet.onnx"
    det.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"
    emb.write_bytes(b"x")
    (tmp_path / "emb.onnx.manifest.json").write_text("{}")
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_FACE_DETECTOR_MODEL", str(det))
    monkeypatch.setenv("SFN_FACE_EMBEDDER_MODEL", str(emb))
    monkeypatch.setenv("SFN_EXAMINER_ID", "ex1")
    assert Settings().face_startup_error() is None
