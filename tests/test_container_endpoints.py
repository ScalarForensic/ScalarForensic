"""Tests for container-related API endpoints:
- /api/container-download
- /api/container-siblings
- /api/hit-image (container virtual path)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PilImage

import scalar_forensic.web.app as app_module


@pytest.fixture(autouse=True)
def no_qdrant(monkeypatch):
    """Disable lifespan Qdrant fetch so tests are hermetic."""
    monkeypatch.setenv("SFN_VIZ_MAX_POINTS", "0")


def _client() -> TestClient:
    return TestClient(app_module.app)


def _make_png_bytes() -> bytes:
    """Return a minimal valid 1×1 PNG."""
    buf = io.BytesIO()
    PilImage.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _make_zip_with_image(tmp_dir: Path) -> Path:
    """Create a ZIP file containing one PNG and return its path."""
    png = _make_png_bytes()
    zip_path = tmp_dir / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("photo.png", png)
    return zip_path


# ---------------------------------------------------------------------------
# /api/container-download
# ---------------------------------------------------------------------------


class TestContainerDownload:
    def test_valid_zip_returns_file(self, tmp_path):
        zip_path = _make_zip_with_image(tmp_path)
        resp = _client().get("/api/container-download", params={"path": str(zip_path)})
        assert resp.status_code == 200

    def test_rejects_relative_path(self):
        resp = _client().get("/api/container-download", params={"path": "relative/path.zip"})
        assert resp.status_code == 400

    def test_rejects_non_container_extension(self, tmp_path):
        img_path = tmp_path / "image.png"
        img_path.write_bytes(_make_png_bytes())
        resp = _client().get("/api/container-download", params={"path": str(img_path)})
        assert resp.status_code == 400

    def test_rejects_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.zip"
        resp = _client().get("/api/container-download", params={"path": str(missing)})
        assert resp.status_code == 404

    def test_rejects_path_outside_data_root(self, tmp_path, monkeypatch):
        zip_path = _make_zip_with_image(tmp_path)
        data_root = tmp_path / "allowed"
        data_root.mkdir()
        monkeypatch.setenv("SFN_DATA_ROOT", str(data_root))
        resp = _client().get("/api/container-download", params={"path": str(zip_path)})
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# /api/container-siblings
# ---------------------------------------------------------------------------


class TestContainerSiblings:
    _VALID_HASH = "a" * 64  # 64 hex chars

    def test_rejects_invalid_hash_format(self):
        resp = _client().get("/api/container-siblings", params={"container_hash": "not-a-hash"})
        assert resp.status_code == 400

    def test_rejects_invalid_collection(self):
        resp = _client().get(
            "/api/container-siblings",
            params={"container_hash": self._VALID_HASH, "collection": "invalid"},
        )
        assert resp.status_code == 400

    def test_valid_request_returns_list(self):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        with patch("scalar_forensic.web.app.QdrantClient", return_value=mock_client):
            resp = _client().get(
                "/api/container-siblings",
                params={"container_hash": self._VALID_HASH, "collection": "sscd"},
            )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_default_collection_is_sscd(self):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        with patch("scalar_forensic.web.app.QdrantClient", return_value=mock_client):
            resp = _client().get(
                "/api/container-siblings",
                params={"container_hash": self._VALID_HASH},
            )
        assert resp.status_code == 200

    def test_dino_collection_accepted(self):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        with patch("scalar_forensic.web.app.QdrantClient", return_value=mock_client):
            resp = _client().get(
                "/api/container-siblings",
                params={"container_hash": self._VALID_HASH, "collection": "dino"},
            )
        assert resp.status_code == 200

    def test_sibling_payload_shape(self):
        record = MagicMock()
        record.payload = {
            "image_hash": "b" * 64,
            "image_path": "/data/img.jpg",
            "container_item_name": "photo.png",
            "extraction_kind": "embedded",
        }
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([record], None)
        with patch("scalar_forensic.web.app.QdrantClient", return_value=mock_client):
            resp = _client().get(
                "/api/container-siblings",
                params={"container_hash": self._VALID_HASH, "collection": "sscd"},
            )
        data = resp.json()
        assert len(data) == 1
        assert set(data[0].keys()) == {
            "image_hash",
            "image_path",
            "container_item_name",
            "extraction_kind",
        }

    def test_qdrant_error_returns_500(self):
        mock_client = MagicMock()
        mock_client.scroll.side_effect = RuntimeError("connection refused")
        with patch("scalar_forensic.web.app.QdrantClient", return_value=mock_client):
            resp = _client().get(
                "/api/container-siblings",
                params={"container_hash": self._VALID_HASH, "collection": "sscd"},
            )
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /api/hit-image — container virtual path
# ---------------------------------------------------------------------------


class TestHitImageContainer:
    def test_rejects_relative_container_path(self):
        resp = _client().get(
            "/api/hit-image", params={"path": "relative/path.zip::photo.png"}
        )
        assert resp.status_code == 400

    def test_rejects_non_container_extension_in_virtual_path(self, tmp_path):
        img_path = tmp_path / "image.png"
        img_path.write_bytes(_make_png_bytes())
        resp = _client().get(
            "/api/hit-image", params={"path": f"{img_path}::inner.png"}
        )
        assert resp.status_code == 400

    def test_rejects_missing_container(self, tmp_path):
        missing = tmp_path / "nonexistent.zip"
        resp = _client().get(
            "/api/hit-image", params={"path": f"{missing}::photo.png"}
        )
        assert resp.status_code == 404

    def test_returns_image_from_zip(self, tmp_path):
        zip_path = _make_zip_with_image(tmp_path)
        resp = _client().get(
            "/api/hit-image", params={"path": f"{zip_path}::photo.png"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")

    def test_rejects_path_outside_data_root(self, tmp_path, monkeypatch):
        zip_path = _make_zip_with_image(tmp_path)
        data_root = tmp_path / "allowed"
        data_root.mkdir()
        monkeypatch.setenv("SFN_DATA_ROOT", str(data_root))
        resp = _client().get(
            "/api/hit-image", params={"path": f"{zip_path}::photo.png"}
        )
        assert resp.status_code == 403
