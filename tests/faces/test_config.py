import pytest

from scalar_forensic.config import Settings


def test_faces_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    s = Settings()
    assert s.faces_enabled is False
    assert s.face_collection == f"{s.collection}_faces"
    assert s.face_detect_max_size == 1600
    assert s.face_crop_dilation == pytest.approx(0.25)


def test_thumb_size_default_and_override(monkeypatch):
    monkeypatch.delenv("SFN_FACE_THUMB_SIZE", raising=False)
    assert Settings().face_thumb_size == 256
    monkeypatch.setenv("SFN_FACE_THUMB_SIZE", "128")
    assert Settings().face_thumb_size == 128


def test_thumb_size_rejects_non_positive(monkeypatch):
    monkeypatch.setenv("SFN_FACE_THUMB_SIZE", "0")
    with pytest.raises(ValueError, match="SFN_FACE_THUMB_SIZE"):
        Settings()


def test_face_collection_follows_case_collection(monkeypatch):
    monkeypatch.setenv("SFN_COLLECTION", "case42")
    assert Settings().face_collection == "case42_faces"


def test_crop_dilation_default_is_pinned_without_env(monkeypatch, tmp_path):
    # An examiner machine sets this in .env, which load_dotenv folds into the
    # process env — so pinning the *code* default needs both stripped. The
    # env file must exist and be empty: a missing one makes load_dotenv fall
    # back to find_dotenv(), which walks up to the repo's own .env.
    monkeypatch.delenv("SFN_FACE_CROP_DILATION", raising=False)
    empty = tmp_path / "empty.env"
    empty.write_text("")
    s = Settings(env_file=empty)
    assert s.face_crop_dilation == pytest.approx(0.25)


def test_crop_dilation_bounds(monkeypatch):
    monkeypatch.setenv("SFN_FACE_CROP_DILATION", "0.9")
    with pytest.raises(ValueError, match="SFN_FACE_CROP_DILATION"):
        Settings()


def test_review_thresholds_default(monkeypatch):
    monkeypatch.delenv("SFN_FACE_REVIEW_MIN_CONF", raising=False)
    monkeypatch.delenv("SFN_FACE_REVIEW_MIN_SIZE", raising=False)
    s = Settings()
    assert s.face_review_min_conf == 0.6
    assert s.face_review_min_size == 48


def test_review_thresholds_clamped_to_embedding_gate(monkeypatch):
    # An explicit embedding threshold below the review DEFAULT must not raise.
    monkeypatch.setenv("SFN_FACE_MIN_CONF", "0.5")
    monkeypatch.setenv("SFN_FACE_MIN_SIZE", "32")
    s = Settings()
    assert s.face_review_min_conf == 0.5
    assert s.face_review_min_size == 32
    notes = s.face_threshold_notes()
    assert any("SFN_FACE_REVIEW_MIN_CONF" in n for n in notes)
    assert any("SFN_FACE_REVIEW_MIN_SIZE" in n for n in notes)


def test_review_conf_below_detector_floor_is_noted(monkeypatch):
    monkeypatch.setenv("SFN_FACE_REVIEW_MIN_CONF", "0.2")
    s = Settings()
    assert any("0.5" in n for n in s.face_threshold_notes())


def test_review_thresholds_reject_nonsense(monkeypatch):
    monkeypatch.setenv("SFN_FACE_REVIEW_MIN_SIZE", "0")
    with pytest.raises(ValueError, match="SFN_FACE_REVIEW_MIN_SIZE"):
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
