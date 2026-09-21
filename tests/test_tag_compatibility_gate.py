"""Similarity routes must preserve a compatibility conflict, never search through it."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scalar_forensic.safeguards import CompatibilityError
from scalar_forensic.web.routes.tags import router


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/tags/classify", {"image_hashes": ["a" * 64]}),
        ("/api/tags/classify-session", {"session_id": "session"}),
    ],
)
def test_classification_preserves_conflict_before_scoring(path, body):
    app = FastAPI()
    app.include_router(router)
    client = MagicMock()
    with (
        patch(
            "scalar_forensic.web.routes.tags._tag_client_and_store",
            return_value=(client, MagicMock()),
        ),
        patch("scalar_forensic.web.routes.tags.get_session", return_value=MagicMock()),
        patch(
            "scalar_forensic.web.routes.tags.require_collection_compatible",
            side_effect=CompatibilityError("rebuild"),
        ),
        patch("scalar_forensic.web.routes.tags.score_query_entries") as score,
    ):
        response = TestClient(app).post(path, json=body)
    assert response.status_code == 409
    assert response.json()["detail"] == "rebuild"
    score.assert_not_called()
    client.retrieve.assert_not_called()


def test_query_triage_preserves_conflict():
    app = FastAPI()
    app.include_router(router)
    with (
        patch(
            "scalar_forensic.web.routes.tags._tag_client_and_store",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch("scalar_forensic.web.routes.tags.get_session", return_value=MagicMock()),
        patch(
            "scalar_forensic.web.routes.tags.require_collection_compatible",
            side_effect=CompatibilityError("rebuild"),
        ),
        patch("scalar_forensic.web.routes.tags.score_query_entries") as score,
    ):
        response = TestClient(app).post(
            "/api/triage/query-images", data={"tag_id": "tag", "session_id": "session"}
        )
    assert response.status_code == 409
    score.assert_not_called()


def test_tag_reference_vectors_refresh_after_collection_rebuild():
    from scalar_forensic.config import Settings
    from scalar_forensic.web.routes.tags import _get_tag_ref_vecs

    client = MagicMock()
    client.retrieve.side_effect = [
        [MagicMock(id=1, vector={"dino": [1.0, 0.0]})],
        [MagicMock(id=1, vector={"dino": [0.0, 1.0]})],
    ]
    settings = Settings()
    tag = MagicMock(
        tag_id="same-tag",
        updated_at="unchanged",
        positive_ids=["1"],
        negative_ids=[],
        target_id=None,
    )
    assert _get_tag_ref_vecs(client, settings, tag)["1"] == [1.0, 0.0]
    assert _get_tag_ref_vecs(client, settings, tag)["1"] == [0.0, 1.0]
