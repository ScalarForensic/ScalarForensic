"""Tests for the /viz standalone visualization endpoint."""

import pytest
from fastapi.testclient import TestClient

import scalar_forensic.web.app as app_module


@pytest.fixture(autouse=True)
def no_qdrant(monkeypatch):
    """Disable the lifespan Qdrant fetch so tests are hermetic and fast.

    SFN_VIZ_MAX_POINTS=0 causes the lifespan to skip the threaded scroll
    and set an empty cache, avoiding connection timeouts on CI.
    """
    monkeypatch.setenv("SFN_VIZ_MAX_POINTS", "0")


def _client() -> TestClient:
    return TestClient(app_module.app)


def test_viz_returns_html():
    resp = _client().get("/viz")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_viz_contains_bootstrap_call():
    resp = _client().get("/viz")
    assert "initVectorViz(" in resp.text


def test_viz_empty_cache(monkeypatch):
    """Endpoint must not crash when _points3d_cache is None."""
    monkeypatch.setattr(app_module, "_points3d_cache", None)
    resp = _client().get("/viz")
    assert resp.status_code == 200
    assert "initVectorViz(" in resp.text
