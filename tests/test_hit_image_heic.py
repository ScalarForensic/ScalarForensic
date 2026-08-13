"""HEIC/HEIF serving through /api/hit-image and /api/metadata.

The scanner indexes ``.heic``/``.heif`` when pillow-heif is available, so the
web routes must accept the same extensions (the whitelist is derived from
``scanner.py`` — a second literal copy drifted once already).  Chrome cannot
render ``image/heic``, so /api/hit-image transcodes HEIF sources to JPEG.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scalar_forensic.web.app import app

pillow_heif = pytest.importorskip("pillow_heif")


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def heic_file(tmp_path: Path) -> Path:
    """Write a tiny synthetic HEIC file and return its path."""
    pillow_heif.register_heif_opener()
    p = tmp_path / "IMG_0001.HEIC"
    Image.new("RGB", (8, 8), (200, 30, 30)).save(p, format="HEIF")
    return p


def _allow(root: Path):
    settings_mock = MagicMock()
    settings_mock.input_dir = root
    settings_mock.frame_store_dir = None
    return patch("scalar_forensic.web.routes._shared.Settings", return_value=settings_mock)


class TestHitImageHeic:
    def test_heic_is_transcoded_to_jpeg(self, client, heic_file, tmp_path):
        with _allow(tmp_path):
            r = client.get(f"/api/hit-image?path={heic_file}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        with Image.open(io.BytesIO(r.content)) as img:
            assert img.format == "JPEG"
            assert img.size == (8, 8)

    def test_lowercase_heif_suffix_accepted(self, client, heic_file, tmp_path):
        renamed = heic_file.with_suffix(".heif")
        heic_file.rename(renamed)
        with _allow(tmp_path):
            r = client.get(f"/api/hit-image?path={renamed}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"

    def test_jpeg_still_served_raw(self, client, tmp_path):
        p = tmp_path / "plain.jpg"
        Image.new("RGB", (4, 4)).save(p, format="JPEG")
        raw = p.read_bytes()
        with _allow(tmp_path):
            r = client.get(f"/api/hit-image?path={p}")
        assert r.status_code == 200
        assert r.content == raw

    def test_heic_rejected_when_heif_unavailable(self, client, heic_file, tmp_path):
        with (
            _allow(tmp_path),
            patch("scalar_forensic.scanner._HEIF_AVAILABLE", False),
        ):
            r = client.get(f"/api/hit-image?path={heic_file}")
        assert r.status_code == 400

    def test_corrupt_heic_returns_422(self, client, tmp_path):
        p = tmp_path / "broken.heic"
        p.write_bytes(b"not a heif payload")
        with _allow(tmp_path):
            r = client.get(f"/api/hit-image?path={p}")
        assert r.status_code == 422


class TestMetadataHeic:
    def test_heic_metadata_accepted(self, client, heic_file, tmp_path):
        settings_mock = MagicMock()
        settings_mock.input_dir = tmp_path
        settings_mock.frame_store_dir = None
        with (
            _allow(tmp_path),
            patch("scalar_forensic.web.routes.files.Settings", return_value=settings_mock),
        ):
            r = client.get(f"/api/metadata?path={heic_file}")
        assert r.status_code == 200
        meta = r.json()
        assert meta["filename"] == heic_file.name
        assert len(meta["hash_sha256"]) == 64


class TestWhitelistDerivation:
    def test_route_whitelist_matches_scanner(self):
        from scalar_forensic import scanner
        from scalar_forensic.web.routes import files

        expected = scanner.IMAGE_EXTENSIONS | (
            scanner._HEIF_EXTENSIONS if scanner._HEIF_AVAILABLE else frozenset()
        )
        assert files._allowed_image_extensions() == expected
