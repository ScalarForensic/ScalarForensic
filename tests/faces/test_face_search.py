"""Stage 2 (Phase 1b): cross-file face search.

Three things this file exists to hold in place:

* a review-only query face is refused as a probe (400) — it has no vector, and
  the refusal is what that vectorlessness looks like at the API surface;
* a hard comparability mismatch refuses the search with 409 rather than
  returning scores computed across two embedding spaces;
* the response labels the score as uncalibrated and scopes 0.363 as the model
  authors' figure.  0.363 is never the applied threshold — that default is 0.0.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scalar_forensic.web.app import app
from scalar_forensic.web.pipeline.faces_search import (
    MODEL_REFERENCE_THRESHOLD,
    search_query_faces,
)

client = TestClient(app)


def _payload(image_hash, score, review="rev1", **over):
    d = {
        "point_id": f"pid-{image_hash}",
        "score": score,
        "image_hash": image_hash,
        "image_path": f"/evidence/{image_hash}.jpg",
        "observation_key": f"{image_hash}::1:2:3:4",
        "bbox": [1, 2, 3, 4],
        "det_conf": 0.9,
        "quality": 0.7,
        "review_chip_hash": review,
        "is_video_frame": False,
        "video_hash": None,
        "video_path": None,
        "frame_timecode_ms": None,
    }
    d.update(over)
    return d


def test_collapse_keeps_one_hit_per_image_hash_with_the_best_score():
    store = MagicMock()
    store.search_faces.return_value = [
        _payload("aaa", 0.91),
        _payload("aaa", 0.44),
        _payload("bbb", 0.60),
    ]
    hits = search_query_faces(
        store, [(0, [1.0, 0.0])], limit=10, threshold=0.0, exact=True, collapse=True
    )
    assert [h["image_hash"] for h in hits] == ["aaa", "bbb"]
    assert hits[0]["score"] == pytest.approx(0.91)
    assert hits[0]["n_collapsed"] == 2


def test_uncollapsed_search_returns_every_observation():
    store = MagicMock()
    store.search_faces.return_value = [
        _payload("aaa", 0.91),
        _payload("aaa", 0.44),
        _payload("bbb", 0.60),
    ]
    hits = search_query_faces(
        store, [(0, [1.0, 0.0])], limit=10, threshold=0.0, exact=True, collapse=False
    )
    assert [h["score"] for h in hits] == [0.91, 0.60, 0.44]


def test_hits_carry_the_query_face_index_that_produced_them():
    store = MagicMock()
    store.search_faces.side_effect = [[_payload("aaa", 0.5)], [_payload("bbb", 0.8)]]
    hits = search_query_faces(
        store,
        [(0, [1.0, 0.0]), (3, [0.0, 1.0])],
        limit=10,
        threshold=0.0,
        exact=True,
        collapse=True,
    )
    by_hash = {h["image_hash"]: h for h in hits}
    assert by_hash["aaa"]["query_face_index"] == 0
    assert by_hash["bbb"]["query_face_index"] == 3


def test_search_is_exact_by_default_and_records_the_mode():
    store = MagicMock()
    store.search_faces.return_value = []
    search_query_faces(store, [(0, [1.0])], limit=5, threshold=0.2, exact=True, collapse=True)
    kwargs = store.search_faces.call_args.kwargs
    assert kwargs["exact"] is True
    assert kwargs["threshold"] == 0.2
    assert kwargs["limit"] == 5


def test_video_frame_hits_are_flagged_from_the_timecode():
    store = MagicMock()
    store.search_faces.return_value = [
        _payload("aaa", 0.5, frame_timecode_ms=1200, video_hash="vh", video_path="/v.mp4")
    ]
    hits = search_query_faces(
        store, [(0, [1.0])], limit=5, threshold=0.0, exact=True, collapse=True
    )
    assert hits[0]["is_video_frame"] is True
    assert hits[0]["frame_timecode_ms"] == 1200
    assert hits[0]["face"]["review_url"] == "/api/faces/chip/rev1/review"


def _review_only_face():
    from scalar_forensic.web.session import QueryFace

    return QueryFace(
        index=0,
        bbox=(1.0, 2.0, 3.0, 4.0),
        landmarks=[[1.0, 2.0]] * 5,
        det_conf=0.7,
        detect_scale=1.0,
        quality={
            "confidence": 0.7,
            "size": 40.0,
            "pose": None,
            "sharpness": None,
            "exposure": None,
        },
        embedding_status="review_only",
        embedding_exclusion_reason="size",
        vector=None,
        review_jpeg=b"\xff\xd8",
    )


def test_review_only_probe_is_refused_with_400():
    entry = MagicMock(query_faces=[_review_only_face()])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
    ):
        resp = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    assert resp.status_code == 400
    assert "review-only" in resp.json()["detail"]


def test_unknown_and_empty_face_indices_are_refused_with_400():
    entry = MagicMock(query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
    ):
        unknown = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "9"},
        )
        empty = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": " "},
        )
    assert unknown.status_code == 400
    assert "unknown face index 9" in unknown.json()["detail"]
    assert empty.status_code == 400
    assert "no face indices given" in empty.json()["detail"]


def test_hard_compat_mismatch_refuses_the_search():
    store = MagicMock()
    store.check_compat.return_value = ["embedder_model_hash mismatch: abc != def"]
    entry = MagicMock(query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        resp = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    assert resp.status_code == 409


def test_check_compat_raising_also_refuses_with_409():
    """The real FaceStore raises on a hard mismatch rather than returning it."""
    store = MagicMock()
    store.check_compat.side_effect = ValueError("not comparable: embedder_model_hash")
    entry = MagicMock(query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        resp = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    assert resp.status_code == 409
    assert "not comparable" in resp.json()["detail"]


def test_soft_compat_mismatch_is_a_warning_not_a_refusal():
    store = MagicMock()
    store.check_compat.return_value = ["min_size: collection has 64, this run has 48"]
    store.search_faces.return_value = []
    entry = MagicMock(query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        patch("scalar_forensic.web.routes.faces._audit_log"),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    ):
        resp = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    assert resp.status_code == 200
    assert resp.json()["compat"] == {
        "ok": True,
        "warnings": ["min_size: collection has 64, this run has 48"],
    }


def test_response_labels_the_score_as_uncalibrated_and_scopes_0363():
    """The ruling: show the number, label it honestly, never claim 0.363 as ours."""
    store = MagicMock()
    store.check_compat.return_value = []
    store.search_faces.return_value = []
    entry = MagicMock(query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        patch("scalar_forensic.web.routes.faces._audit_log"),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    ):
        body = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        ).json()

    cal = body["calibration"]
    assert cal["status"] == "uncalibrated"
    assert cal["active_record"] is None
    assert "not evidential" in cal["banner"]
    assert cal["model_reference_threshold"] == MODEL_REFERENCE_THRESHOLD
    assert "not a threshold calibrated on this deployment" in cal["model_reference_note"]
    # 0.363 must never be the applied threshold
    assert body["threshold"] == 0.0
    assert body["search_mode"] == "exact"
    assert body["n_probes"] == 1


def test_the_default_score_floor_is_zero_not_the_reference_figure():
    store = MagicMock()
    store.check_compat.return_value = []
    store.search_faces.return_value = []
    entry = MagicMock(query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        patch("scalar_forensic.web.routes.faces._audit_log"),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    ):
        client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    assert store.search_faces.call_args.kwargs["threshold"] == 0.0


def test_every_search_is_written_to_the_face_audit_log():
    store = MagicMock()
    store.check_compat.return_value = []
    store.search_faces.return_value = []
    entry = MagicMock(
        file_hash="deadbeef",
        query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")],
    )
    log = MagicMock()
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        patch("scalar_forensic.web.routes.faces._audit_log", return_value=log),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    ):
        client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    log.append.assert_called_once()
    assert log.append.call_args.args[0] == "query"
    fields = log.append.call_args.kwargs
    assert fields["search_mode"] == "exact"
    assert fields["face_calibration_id"] is None
    assert "n_results" in fields and "probe_hash" in fields


def test_ann_mode_is_recorded_when_exact_is_opted_out():
    store = MagicMock()
    store.check_compat.return_value = []
    store.search_faces.return_value = []
    entry = MagicMock(query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")])
    log = MagicMock()
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        patch("scalar_forensic.web.routes.faces._audit_log", return_value=log),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    ):
        body = client.post(
            "/api/faces/search",
            data={
                "session_id": "s",
                "file_id": "f",
                "face_indices": "0",
                "exact": "false",
            },
        ).json()
    assert body["search_mode"] == "ann"
    assert log.append.call_args.kwargs["search_mode"] == "ann"
    assert store.search_faces.call_args.kwargs["exact"] is False


def test_search_404s_on_unknown_session():
    with patch("scalar_forensic.web.routes.faces.get_session", return_value=None):
        resp = client.post(
            "/api/faces/search",
            data={"session_id": "nope", "file_id": "f", "face_indices": "0"},
        )
    assert resp.status_code == 404
