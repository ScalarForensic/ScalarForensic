from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from scalar_forensic.web.routes import faces
from scalar_forensic.web.session import FileEntry


@pytest.mark.parametrize("endpoint", ["search", "compare", "stats"])
@pytest.mark.parametrize("token", ["", "previous-detection"])
def test_uploaded_requests_refuse_missing_or_stale_detection(endpoint, token, tmp_path):
    entry = FileEntry("f", "query.jpg", tmp_path / "query.jpg", query_faces=[])
    with (
        patch.object(faces, "_resolve_entry", return_value=entry),
        patch.object(faces, "_require_faces"),
        patch.object(faces, "_store") as store,
        pytest.raises(HTTPException) as error,
    ):
        if endpoint == "search":
            faces.face_search("s", "f", face_indices="0", point_ids="", detection_token=token)
        elif endpoint == "compare":
            faces.face_compare("s", "f", "a" * 64, None, None, detection_token=token)
        else:
            faces.faces_dist_stats("s", "f", 0, 10, detection_token=token)
    assert error.value.status_code == 409
    store.assert_not_called()


def test_detection_guard_accepts_only_the_snapshot_that_issued_the_token(tmp_path):
    entry = FileEntry("f", "query.jpg", tmp_path / "query.jpg", query_faces=[])
    token = faces._detection_token([], None, None)
    faces._require_detection_token(entry, token)
    entry.query_faces_timecode_ms = 1000
    with pytest.raises(HTTPException) as error:
        faces._require_detection_token(entry, token)
    assert error.value.status_code == 409


def test_explain_reuses_the_observation_store(monkeypatch):
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    store = MagicMock()
    store.get_face.return_value = {"image_hash": "a" * 64}
    store.get_marker.return_value = {}
    with patch.object(faces, "_store", return_value=store) as factory:
        faces.face_explain("11111111-1111-1111-1111-111111111111")
    factory.assert_called_once()
