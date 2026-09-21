"""Actual ASGI startup and ingress enforce operator configuration."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from scalar_forensic.web.app import app


def test_direct_asgi_checks_case_reference_and_rejects_body_before_analysis(monkeypatch, tmp_path):
    monkeypatch.setenv("SFN_COLLECTION", "case")
    monkeypatch.setenv("SFN_REFERENCE_COLLECTION", "reference")
    monkeypatch.setenv("SFN_MAX_UPLOAD_BYTES", "128")
    monkeypatch.setenv("SFN_SESSION_TEMP_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(app, "middleware_stack", None)
    with patch("scalar_forensic.web.app._check_collection_compat") as check:
        with TestClient(app) as client:
            response = client.post("/api/analyze", content=b"x" * 129)
            assert response.status_code == 413
            assert client.get("/").status_code == 200
    assert [call.args[0].collection for call in check.call_args_list] == ["case", "reference"]
    assert (tmp_path / "sessions").is_dir()
