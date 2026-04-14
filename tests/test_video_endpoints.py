"""Tests for video-related API endpoints.

Covered endpoints:
  GET /api/query-frames/{session_id}/{file_id}
  GET /api/query-frame/{session_id}/{file_id}?timecode_ms=…
  GET /api/video-frame?path=…&timecode_ms=…
  GET /api/video-timeline?video_hash=…
  GET /api/thumbnail/{sha256}
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scalar_forensic.web.app import app
from scalar_forensic.web.session import FileEntry, VideoFrameEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_qdrant_lifespan(monkeypatch):
    """Skip the lifespan Qdrant prefetch so tests are hermetic."""
    monkeypatch.setenv("SFN_VIZ_MAX_POINTS", "0")


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


def _video_entry(file_id: str = "vid1", *, tmp_path: Path) -> FileEntry:
    """FileEntry that looks like a successfully processed video upload."""
    p = tmp_path / (file_id + ".mp4")
    p.write_bytes(b"fake-video-data")
    return FileEntry(
        file_id=file_id,
        filename="clip.mp4",
        temp_path=p,
        is_video=True,
        video_frames=[
            VideoFrameEntry(frame_index=0, timecode_ms=0, frame_hash="a" * 64),
            VideoFrameEntry(frame_index=1, timecode_ms=1000, frame_hash="b" * 64),
        ],
    )


def _image_entry(file_id: str = "img1", *, tmp_path: Path) -> FileEntry:
    """FileEntry for a non-video upload."""
    p = tmp_path / (file_id + ".jpg")
    p.write_bytes(b"fake-jpeg-data")
    return FileEntry(file_id=file_id, filename="photo.jpg", temp_path=p, is_video=False)


def _mock_session(entries: list[FileEntry]) -> MagicMock:
    sess = MagicMock()
    sess.files = entries
    return sess


# ---------------------------------------------------------------------------
# /api/query-frames — frame list for an uploaded query video
# ---------------------------------------------------------------------------


class TestQueryFramesList:
    def test_missing_session_returns_404(self, client):
        with patch("scalar_forensic.web.app.get_session", return_value=None):
            r = client.get("/api/query-frames/no-such-session/vid1")
        assert r.status_code == 404

    def test_missing_file_returns_404(self, client, tmp_path):
        sess = _mock_session([_image_entry(tmp_path=tmp_path)])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get("/api/query-frames/s1/no-such-file")
        assert r.status_code == 404

    def test_non_video_entry_returns_400(self, client, tmp_path):
        entry = _image_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get(f"/api/query-frames/s1/{entry.file_id}")
        assert r.status_code == 400

    def test_video_entry_no_frames_returns_400(self, client, tmp_path):
        p = tmp_path / "vid.mp4"
        p.write_bytes(b"x")
        entry = FileEntry(
            file_id="v1", filename="v.mp4", temp_path=p, is_video=True, video_frames=None
        )
        sess = _mock_session([entry])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get("/api/query-frames/s1/v1")
        assert r.status_code == 400

    def test_valid_video_returns_frame_list(self, client, tmp_path):
        entry = _video_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get(f"/api/query-frames/s1/{entry.file_id}")
        assert r.status_code == 200
        body = r.json()
        assert "frames" in body
        frames = body["frames"]
        assert len(frames) == 2
        assert frames[0] == {"frame_index": 0, "timecode_ms": 0, "frame_hash": "a" * 64}
        assert frames[1] == {"frame_index": 1, "timecode_ms": 1000, "frame_hash": "b" * 64}


# ---------------------------------------------------------------------------
# /api/query-frame — serve a single frame image from a query video
# ---------------------------------------------------------------------------


class TestQueryFrameImage:
    def test_missing_session_returns_404(self, client):
        with patch("scalar_forensic.web.app.get_session", return_value=None):
            r = client.get("/api/query-frame/no-session/v1?timecode_ms=0")
        assert r.status_code == 404

    def test_missing_file_returns_404(self, client, tmp_path):
        sess = _mock_session([_image_entry(tmp_path=tmp_path)])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get("/api/query-frame/s1/no-file?timecode_ms=0")
        assert r.status_code == 404

    def test_non_video_entry_returns_400(self, client, tmp_path):
        entry = _image_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get(f"/api/query-frame/s1/{entry.file_id}?timecode_ms=0")
        assert r.status_code == 400

    def test_negative_timecode_returns_400(self, client, tmp_path):
        entry = _video_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with patch("scalar_forensic.web.app.get_session", return_value=sess):
            r = client.get(f"/api/query-frame/s1/{entry.file_id}?timecode_ms=-1")
        assert r.status_code == 400

    def test_frame_not_found_at_timecode_returns_404(self, client, tmp_path):
        entry = _video_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with (
            patch("scalar_forensic.web.app.get_session", return_value=sess),
            patch("scalar_forensic.web.app.extract_frame_at", return_value=None),
        ):
            r = client.get(f"/api/query-frame/s1/{entry.file_id}?timecode_ms=9999999")
        assert r.status_code == 404

    def test_valid_frame_returns_jpeg(self, client, tmp_path):
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        entry = _video_entry(tmp_path=tmp_path)
        sess = _mock_session([entry])
        with (
            patch("scalar_forensic.web.app.get_session", return_value=sess),
            patch("scalar_forensic.web.app.extract_frame_at", return_value=img),
        ):
            r = client.get(f"/api/query-frame/s1/{entry.file_id}?timecode_ms=0")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"


# ---------------------------------------------------------------------------
# /api/video-frame — serve a frame from a dataset video on disk
# ---------------------------------------------------------------------------


class TestVideoFrame:
    def test_non_video_extension_returns_400(self, client, tmp_path):
        # Extension check fires before _check_allowed_path — no SFN_INPUT_DIR needed.
        p = tmp_path / "image.jpg"
        p.write_bytes(b"x")
        r = client.get(f"/api/video-frame?path={p}&timecode_ms=0")
        assert r.status_code == 400

    def test_negative_timecode_returns_400(self, client, tmp_path):
        p = tmp_path / "clip.mp4"
        r = client.get(f"/api/video-frame?path={p}&timecode_ms=-1")
        assert r.status_code == 400

    def test_missing_file_returns_404(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        p = tmp_path / "missing.mp4"
        r = client.get(f"/api/video-frame?path={p}&timecode_ms=0")
        assert r.status_code == 404

    def test_frame_not_found_returns_404(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"fake")
        with patch("scalar_forensic.web.app.extract_frame_at", return_value=None):
            r = client.get(f"/api/video-frame?path={p}&timecode_ms=99999")
        assert r.status_code == 404

    def test_valid_frame_returns_jpeg(self, client, tmp_path, monkeypatch):
        from PIL import Image

        monkeypatch.setenv("SFN_INPUT_DIR", str(tmp_path))
        img = Image.new("RGB", (32, 32), color=(0, 128, 255))
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"fake")
        with patch("scalar_forensic.web.app.extract_frame_at", return_value=img):
            r = client.get(f"/api/video-frame?path={p}&timecode_ms=0")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"


# ---------------------------------------------------------------------------
# /api/video-timeline — indexed frame list from Qdrant by video hash
# ---------------------------------------------------------------------------


class TestVideoTimeline:
    _VALID_HASH = "a" * 64

    def test_invalid_hash_format_returns_400(self, client):
        for bad in ("short", "not-hex!!", "x" * 63, "G" * 64):
            r = client.get(f"/api/video-timeline?video_hash={bad}")
            assert r.status_code == 400, f"expected 400 for hash {bad!r}"

    def test_valid_hash_format_accepted(self, client):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient", return_value=mock_client),
        ):
            r = client.get(f"/api/video-timeline?video_hash={self._VALID_HASH}")
        assert r.status_code == 200
        assert "frames" in r.json()

    def test_empty_qdrant_result_returns_empty_frames(self, client):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient", return_value=mock_client),
        ):
            r = client.get(f"/api/video-timeline?video_hash={self._VALID_HASH}")
        assert r.status_code == 200
        assert r.json()["frames"] == []


# ---------------------------------------------------------------------------
# /api/thumbnail/{sha256}
# ---------------------------------------------------------------------------


class TestThumbnail:
    _VALID_HASH = "c" * 64

    def test_invalid_hash_format_returns_400(self, client):
        for bad in ("short", "UPPERCASE" * 8, "z" * 64):
            r = client.get(f"/api/thumbnail/{bad}")
            assert r.status_code == 400, f"expected 400 for {bad!r}"

    def test_no_thumbnail_dir_returns_404(self, client, monkeypatch):
        settings_mock = MagicMock()
        settings_mock.thumbnail_dir = None
        with patch("scalar_forensic.web.app.Settings", return_value=settings_mock):
            r = client.get(f"/api/thumbnail/{self._VALID_HASH}")
        assert r.status_code == 404

    def test_missing_thumbnail_file_returns_404(self, client, tmp_path):
        settings_mock = MagicMock()
        settings_mock.thumbnail_dir = tmp_path
        with patch("scalar_forensic.web.app.Settings", return_value=settings_mock):
            r = client.get(f"/api/thumbnail/{self._VALID_HASH}")
        assert r.status_code == 404

    def test_existing_thumbnail_returns_jpeg(self, client, tmp_path):
        settings_mock = MagicMock()
        settings_mock.thumbnail_dir = tmp_path
        thumb = tmp_path / f"{self._VALID_HASH}.jpg"
        # minimal valid JPEG header
        thumb.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
        )
        with patch("scalar_forensic.web.app.Settings", return_value=settings_mock):
            r = client.get(f"/api/thumbnail/{self._VALID_HASH}")
        assert r.status_code == 200
