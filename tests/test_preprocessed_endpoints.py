"""Tests for /api/query-preprocessed and /api/hit-preprocessed endpoints."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scalar_forensic.web.app import app
from scalar_forensic.web.session import FileEntry

_PREPROC_PAYLOAD = {"sscd": {}, "dino": {}}


@pytest.fixture(autouse=True)
def _no_qdrant_lifespan(monkeypatch):
    monkeypatch.setenv("SFN_VIZ_MAX_POINTS", "0")


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


def _tiny_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _image_entry(file_id: str = "img1", *, tmp_path: Path) -> FileEntry:
    p = tmp_path / f"{file_id}.jpg"
    p.write_bytes(_tiny_jpeg())
    return FileEntry(file_id=file_id, filename="photo.jpg", temp_path=p, is_video=False)


def _video_entry(file_id: str = "vid1", *, tmp_path: Path) -> FileEntry:
    p = tmp_path / f"{file_id}.mp4"
    p.write_bytes(b"fake-video")
    return FileEntry(file_id=file_id, filename="clip.mp4", temp_path=p, is_video=True)


def _mock_session(entries):
    sess = MagicMock()
    sess.files = entries
    return sess


# ---------------------------------------------------------------------------
# /api/query-preprocessed
# ---------------------------------------------------------------------------


class TestQueryPreprocessed:
    def test_missing_session_returns_404(self, client):
        with patch("scalar_forensic.web.app.get_session", return_value=None):
            r = client.get("/api/query-preprocessed/no-session/img1")
        assert r.status_code == 404

    def test_missing_file_returns_404(self, client, tmp_path):
        sess = _mock_session([_image_entry(tmp_path=tmp_path)])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get("/api/query-preprocessed/s1/no-such-file")
        assert r.status_code == 404

    def test_video_without_timecode_returns_400(self, client, tmp_path):
        entry = _video_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get(f"/api/query-preprocessed/s1/{entry.file_id}")
        assert r.status_code == 400

    def test_video_with_timecode_returns_payload(self, client, tmp_path):
        entry = _video_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        frame = Image.new("RGB", (32, 32))
        with (
            patch("scalar_forensic.web.app.get_session", return_value=sess),
            patch("scalar_forensic.web.app.extract_frame_at", return_value=frame),
            patch("scalar_forensic.web.app._build_preproc_payload", return_value=_PREPROC_PAYLOAD),
            patch("scalar_forensic.web.app.Settings"),
        ):
            r = client.get(f"/api/query-preprocessed/s1/{entry.file_id}?timecode_ms=0")
        assert r.status_code == 200

    def test_video_frame_not_found_returns_404(self, client, tmp_path):
        entry = _video_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with (
            patch("scalar_forensic.web.app.get_session", return_value=sess),
            patch("scalar_forensic.web.app.extract_frame_at", return_value=None),
            patch("scalar_forensic.web.app.Settings"),
        ):
            r = client.get(f"/api/query-preprocessed/s1/{entry.file_id}?timecode_ms=99999")
        assert r.status_code == 404

    def test_image_entry_returns_payload(self, client, tmp_path):
        entry = _image_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with (
            patch("scalar_forensic.web.app.get_session", return_value=sess),
            patch("scalar_forensic.web.app._build_preproc_payload", return_value=_PREPROC_PAYLOAD),
            patch("scalar_forensic.web.app.Settings"),
        ):
            r = client.get(f"/api/query-preprocessed/s1/{entry.file_id}")
        assert r.status_code == 200
        assert r.json() == _PREPROC_PAYLOAD


# ---------------------------------------------------------------------------
# /api/hit-preprocessed
# ---------------------------------------------------------------------------


class TestHitPreprocessed:
    def test_relative_path_returns_400(self, client):
        r = client.get("/api/hit-preprocessed?path=relative/file.jpg")
        assert r.status_code == 400

    def test_invalid_sscd_n_crops_returns_400(self, client, tmp_path):
        p = tmp_path / "img.jpg"
        p.write_bytes(_tiny_jpeg())
        r = client.get(f"/api/hit-preprocessed?path={p}&sscd_n_crops=3")
        assert r.status_code == 400

    def test_dino_normalize_size_zero_returns_400_below_min(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        p = tmp_path / "img.jpg"
        p.write_bytes(_tiny_jpeg())
        r = client.get(f"/api/hit-preprocessed?path={p}&dino_normalize_size=-1")
        assert r.status_code == 400

    def test_dino_normalize_size_too_large_returns_400(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        p = tmp_path / "img.jpg"
        p.write_bytes(_tiny_jpeg())
        r = client.get(f"/api/hit-preprocessed?path={p}&dino_normalize_size=9999")
        assert r.status_code == 400

    def test_path_outside_allowed_roots_returns_403(self, client, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        other = tmp_path / "other" / "img.jpg"
        other.parent.mkdir()
        other.write_bytes(_tiny_jpeg())
        settings_mock = MagicMock()
        settings_mock.input_dir = allowed
        settings_mock.frame_store_dir = None
        with patch("scalar_forensic.web.app.Settings", return_value=settings_mock):
            r = client.get(f"/api/hit-preprocessed?path={other}")
        assert r.status_code == 403

    def test_missing_file_returns_404(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        missing = tmp_path / "gone.jpg"
        r = client.get(f"/api/hit-preprocessed?path={missing}")
        assert r.status_code == 404

    def test_non_image_file_returns_400(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        bad = tmp_path / "video.mp4"
        bad.write_bytes(b"not-an-image")
        r = client.get(f"/api/hit-preprocessed?path={bad}&sscd_n_crops=1&dino_normalize_size=224")
        assert r.status_code == 400

    def test_valid_image_returns_payload(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        p = tmp_path / "img.jpg"
        p.write_bytes(_tiny_jpeg())
        with patch("scalar_forensic.web.app._build_preproc_payload", return_value=_PREPROC_PAYLOAD):
            r = client.get(f"/api/hit-preprocessed?path={p}&sscd_n_crops=1&dino_normalize_size=224")
        assert r.status_code == 200
        assert r.json() == _PREPROC_PAYLOAD

    def test_dino_normalize_size_zero_disables_dino(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        p = tmp_path / "img.jpg"
        p.write_bytes(_tiny_jpeg())
        with patch(
            "scalar_forensic.web.app._build_preproc_payload", return_value={"sscd": {}}
        ) as mock_build:
            r = client.get(f"/api/hit-preprocessed?path={p}&sscd_n_crops=1&dino_normalize_size=0")
        assert r.status_code == 200
        mock_build.assert_called_once()
        _, _, dino_size = mock_build.call_args[0]
        assert dino_size == 0
