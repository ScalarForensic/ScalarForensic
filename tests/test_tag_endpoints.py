"""Tests for Tag/Triage REST endpoints added in the qdrant-forensics feature branch.

Covers: GET /api/tags, POST /api/tag, POST /api/tags/classify,
        POST /api/triage, POST /api/explore.
All Qdrant I/O is stubbed so no real server is required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from scalar_forensic.tags import Tag
from scalar_forensic.web.app import app


@pytest.fixture(autouse=True)
def _no_qdrant_lifespan(monkeypatch):
    monkeypatch.setenv("SFN_VIZ_MAX_POINTS", "0")


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


def _tag(name: str = "weapons", *, pos=(), neg=()) -> Tag:
    now = datetime.now(UTC).isoformat()
    return Tag(
        tag_id="tag-1",
        name=name,
        positive_ids=list(pos),
        negative_ids=list(neg),
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# GET /api/tags
# ---------------------------------------------------------------------------


class TestListTags:
    def test_success_returns_tag_list(self, client):
        tag = _tag()
        with patch("scalar_forensic.web.app._tag_store") as mock_store_fn:
            mock_store_fn.return_value.list.return_value = [tag]
            r = client.get("/api/tags")
        assert r.status_code == 200
        data = r.json()
        assert len(data["tags"]) == 1
        assert data["tags"][0]["name"] == "weapons"

    def test_qdrant_error_returns_503(self, client):
        with patch("scalar_forensic.web.app._tag_store") as mock_store_fn:
            mock_store_fn.return_value.list.side_effect = RuntimeError("connection refused")
            r = client.get("/api/tags")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/tag
# ---------------------------------------------------------------------------


class TestCreateTag:
    def test_success_returns_tag_json(self, client):
        tag = _tag()
        with patch("scalar_forensic.web.app._tag_store") as mock_store_fn:
            mock_store_fn.return_value.create.return_value = tag
            r = client.post("/api/tag", data={"name": "weapons"})
        assert r.status_code == 200
        assert r.json()["name"] == "weapons"

    def test_qdrant_error_returns_503(self, client):
        with patch("scalar_forensic.web.app._tag_store") as mock_store_fn:
            mock_store_fn.return_value.create.side_effect = RuntimeError("down")
            r = client.post("/api/tag", data={"name": "weapons"})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/tags/classify
# ---------------------------------------------------------------------------


class TestClassifyTags:
    def test_invalid_json_returns_400(self, client):
        r = client.post(
            "/api/tags/classify",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_array_body_returns_400(self, client):
        r = client.post("/api/tags/classify", json=[1, 2, 3])
        assert r.status_code == 400

    def test_hashes_not_a_list_returns_400(self, client):
        r = client.post("/api/tags/classify", json={"image_hashes": "not-a-list"})
        assert r.status_code == 400

    def test_hashes_contain_non_string_returns_400(self, client):
        r = client.post("/api/tags/classify", json={"image_hashes": [1, 2]})
        assert r.status_code == 400

    def test_too_many_hashes_returns_400(self, client):
        r = client.post("/api/tags/classify", json={"image_hashes": ["h"] * 257})
        assert r.status_code == 400

    def test_exactly_256_hashes_is_accepted(self, client):
        hashes = [f"sha256-{i}" for i in range(256)]
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient"),
            patch("scalar_forensic.web.app.TagStore") as mock_ts,
            patch("scalar_forensic.web.app.qdrant_scroll_all", return_value=iter([])),
        ):
            mock_ts.return_value.list.return_value = []
            r = client.post("/api/tags/classify", json={"image_hashes": hashes})
        assert r.status_code == 200

    def test_empty_hashes_returns_empty(self, client):
        r = client.post("/api/tags/classify", json={"image_hashes": []})
        assert r.status_code == 200
        assert r.json() == {"by_hash": {}}

    def test_missing_key_returns_empty(self, client):
        r = client.post("/api/tags/classify", json={})
        assert r.status_code == 200
        assert r.json() == {"by_hash": {}}

    def test_no_matching_records_returns_empty_per_hash(self, client):
        """Scroll returns nothing → every requested hash maps to an empty tag list."""
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient"),
            patch("scalar_forensic.web.app.TagStore") as mock_ts,
            patch("scalar_forensic.web.app.qdrant_scroll_all", return_value=iter([])),
        ):
            mock_ts.return_value.list.return_value = []
            r = client.post(
                "/api/tags/classify",
                json={"image_hashes": ["sha256-a", "sha256-b"]},
            )
        assert r.status_code == 200
        by_hash = r.json()["by_hash"]
        assert by_hash == {"sha256-a": [], "sha256-b": []}


# ---------------------------------------------------------------------------
# POST /api/triage
# ---------------------------------------------------------------------------


class TestTriage:
    def test_tag_not_found_returns_404(self, client):
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient"),
            patch("scalar_forensic.web.app.TagStore") as mock_ts,
        ):
            mock_ts.return_value.get.return_value = None
            r = client.post("/api/triage", data={"tag_id": "missing"})
        assert r.status_code == 404

    def test_qdrant_error_returns_503(self, client):
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient"),
            patch("scalar_forensic.web.app.TagStore") as mock_ts,
        ):
            mock_ts.return_value.get.side_effect = RuntimeError("down")
            r = client.post("/api/triage", data={"tag_id": "tag-1"})
        assert r.status_code == 503

    def test_success_returns_hits_envelope(self, client):
        tag = _tag(pos=["p1"], neg=["n1"])
        from scalar_forensic.discovery import DiscoveryHit

        hit = DiscoveryHit(
            point_id="pt-1", triplet_score=3, cosine_margin=0.9, payload={}, vector_name="dino"
        )
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient"),
            patch("scalar_forensic.web.app.TagStore") as mock_ts,
            patch("scalar_forensic.web.app.run_discovery", return_value=[hit]),
        ):
            mock_ts.return_value.get.return_value = tag
            r = client.post("/api/triage", data={"tag_id": "tag-1"})
        assert r.status_code == 200
        body = r.json()
        assert body["tag"]["name"] == "weapons"
        assert len(body["hits"]) == 1
        assert body["hits"][0]["point_id"] == "pt-1"


# ---------------------------------------------------------------------------
# POST /api/explore
# ---------------------------------------------------------------------------


class TestExplore:
    def test_tag_not_found_returns_404(self, client):
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient"),
            patch("scalar_forensic.web.app.TagStore") as mock_ts,
        ):
            mock_ts.return_value.get.return_value = None
            r = client.post("/api/explore", data={"tag_id": "missing"})
        assert r.status_code == 404

    def test_qdrant_error_returns_503(self, client):
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient"),
            patch("scalar_forensic.web.app.TagStore") as mock_ts,
        ):
            mock_ts.return_value.get.side_effect = RuntimeError("down")
            r = client.post("/api/explore", data={"tag_id": "tag-1"})
        assert r.status_code == 503

    def test_success_returns_strategy_and_hits(self, client):
        tag = _tag(pos=["p1"], neg=["n1"])
        from scalar_forensic.discovery import DiscoveryHit

        hit = DiscoveryHit(
            point_id="pt-2", triplet_score=None, cosine_margin=0.5, payload={}, vector_name="dino"
        )
        with (
            patch("scalar_forensic.web.app.Settings"),
            patch("scalar_forensic.web.app.QdrantClient"),
            patch("scalar_forensic.web.app.TagStore") as mock_ts,
            patch("scalar_forensic.web.app.run_explore", return_value=([hit], "context")),
        ):
            mock_ts.return_value.get.return_value = tag
            r = client.post("/api/explore", data={"tag_id": "tag-1"})
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "context"
        assert body["hits"][0]["point_id"] == "pt-2"
