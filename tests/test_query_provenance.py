"""Tests for QueryProvenance dataclass and /api/query provenance field (issues #18, #10)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from scalar_forensic.web.pipeline import QueryProvenance

# ---------------------------------------------------------------------------
# QueryProvenance dataclass
# ---------------------------------------------------------------------------


def test_query_provenance_fields():
    p = QueryProvenance(
        modes=["exact", "altered"],
        threshold_altered=0.75,
        threshold_semantic=0.55,
        limit=10,
        timestamp="2026-04-10T14:32:00+00:00",
    )
    assert p.modes == ["exact", "altered"]
    assert p.threshold_altered == 0.75
    assert p.threshold_semantic == 0.55
    assert p.limit == 10
    assert p.timestamp == "2026-04-10T14:32:00+00:00"


def test_query_provenance_timestamp_format():
    ts = datetime.now(UTC).isoformat()
    p = QueryProvenance(
        modes=["exact"],
        threshold_altered=0.75,
        threshold_semantic=0.55,
        limit=5,
        timestamp=ts,
    )
    parsed = datetime.fromisoformat(p.timestamp)
    assert parsed.tzinfo is not None, "timestamp must be timezone-aware"


def test_query_provenance_dict():
    p = QueryProvenance(
        modes=["semantic"],
        threshold_altered=0.80,
        threshold_semantic=0.60,
        limit=20,
        timestamp="2026-04-10T00:00:00+00:00",
    )
    d = p.__dict__
    expected_keys = {"modes", "threshold_altered", "threshold_semantic", "limit", "timestamp"}
    assert set(d.keys()) == expected_keys
    assert isinstance(d["modes"], list)
    assert isinstance(d["threshold_altered"], float)
    assert isinstance(d["threshold_semantic"], float)
    assert isinstance(d["limit"], int)
    assert isinstance(d["timestamp"], str)


# ---------------------------------------------------------------------------
# /api/query — provenance in response
# ---------------------------------------------------------------------------


def _make_session(file_id: str = "f1") -> MagicMock:
    entry = MagicMock()
    entry.file_id = file_id
    entry.filename = "photo.jpg"
    entry.error = None
    entry.file_hash = "abc123"
    entry.sscd_embedding = None
    entry.dino_embedding = None
    session = MagicMock()
    session.files = [entry]
    return session


def test_api_query_returns_provenance():
    from scalar_forensic.web.app import app

    session = _make_session()

    with (
        patch("scalar_forensic.web.app.get_session", return_value=session),
        patch("scalar_forensic.web.app.query_session", return_value=([], {})),
        patch("scalar_forensic.web.app.Settings"),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/query",
            data={
                "session_id": "sess-1",
                "modes": "exact,altered",
                "threshold_altered": "0.75",
                "threshold_semantic": "0.55",
                "limit": "10",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "provenance" in body
    prov = body["provenance"]
    expected_keys = {"modes", "threshold_altered", "threshold_semantic", "limit", "timestamp"}
    assert set(prov.keys()) == expected_keys


def test_api_query_provenance_modes_match():
    from scalar_forensic.web.app import app

    session = _make_session()

    with (
        patch("scalar_forensic.web.app.get_session", return_value=session),
        patch("scalar_forensic.web.app.query_session", return_value=([], {})),
        patch("scalar_forensic.web.app.Settings"),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/query",
            data={
                "session_id": "sess-1",
                "modes": "exact,semantic",
                "threshold_altered": "0.75",
                "threshold_semantic": "0.55",
                "limit": "5",
            },
        )

    assert resp.status_code == 200
    prov = resp.json()["provenance"]
    assert set(prov["modes"]) == {"exact", "semantic"}
    assert prov["threshold_altered"] == 0.75
    assert prov["threshold_semantic"] == 0.55
    assert prov["limit"] == 5


def test_api_query_provenance_timestamp_is_utc():
    from scalar_forensic.web.app import app

    session = _make_session()

    with (
        patch("scalar_forensic.web.app.get_session", return_value=session),
        patch("scalar_forensic.web.app.query_session", return_value=([], {})),
        patch("scalar_forensic.web.app.Settings"),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/query",
            data={
                "session_id": "sess-1",
                "modes": "exact",
                "threshold_altered": "0.75",
                "threshold_semantic": "0.55",
                "limit": "10",
            },
        )

    ts = resp.json()["provenance"]["timestamp"]
    parsed = datetime.fromisoformat(ts)
    assert parsed.utcoffset() == timedelta(0), "timestamp must be UTC (zero offset)"
