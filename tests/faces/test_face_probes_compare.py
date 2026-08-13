"""Change-set 2026-08-13: pairwise compare endpoint and stored-point probes.

Two backend additions the basket UI stands on:

* ``POST /api/faces/compare`` — pairwise raw cosine between the query image's
  comparable faces and one indexed medium's comparable faces.  Review-only
  observations are vectorless on both sides and never enter the matrix; the
  counts report them so their absence is visible, not silent.
* ``POST /api/faces/search`` additionally accepts stored ``point_ids`` as
  probes.  A review-only point has no stored vector and is refused with the
  same 400 the session-side review-only probe gets — the exclusion guarantee
  holds no matter where a probe comes from.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scalar_forensic.faces.store import FACE_VECTOR_NAME
from scalar_forensic.web.app import app

client = TestClient(app)

PID_EMB = "11111111-1111-1111-1111-111111111111"
PID_REV = "22222222-2222-2222-2222-222222222222"


def _entry(query_faces):
    return MagicMock(
        query_faces=query_faces,
        query_faces_cfg=MagicMock(config_hash="9f2c"),
        file_hash="a" * 64,
    )


def _session_face(index=0, vector=(1.0, 0.0)):
    return MagicMock(index=index, vector=None if vector is None else list(vector))


def _record(pid, vector):
    rec = MagicMock()
    rec.id = pid
    rec.vector = {} if vector is None else {FACE_VECTOR_NAME: list(vector)}
    return rec


def _store(rows=(), records=()):
    store = MagicMock()
    store.collection = "case1_faces"
    store.check_compat.return_value = []
    store.list_faces.return_value = list(rows)
    store.client.retrieve.return_value = list(records)
    return store


def _patched(store):
    return (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        patch("scalar_forensic.web.routes.faces._audit_log"),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    )


def _post_compare(store, entry, image_hash="b" * 64):
    patches = _patched(store) + (
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
    )
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        return client.post(
            "/api/faces/compare",
            data={"session_id": "s", "file_id": "f", "image_hash": image_hash},
        )


# ── POST /api/faces/compare ────────────────────────────────────────────────


def test_compare_pairs_cover_comparable_faces_only():
    # One comparable and one review-only face on each side: the matrix is 1×1.
    entry = _entry([_session_face(0, (1.0, 0.0)), _session_face(1, None)])
    store = _store(
        rows=[
            {"id": PID_EMB, "observation_key": "obs-emb"},
            {"id": PID_REV, "observation_key": "obs-rev"},
        ],
        records=[_record(PID_EMB, (1.0, 0.0)), _record(PID_REV, None)],
    )
    resp = _post_compare(store, entry)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["pairs"]) == 1
    pair = body["pairs"][0]
    assert pair["query_face_index"] == 0
    assert pair["point_id"] == PID_EMB
    assert pair["score"] == pytest.approx(1.0)
    assert body["n_query_comparable"] == 1
    assert body["n_query_review_only"] == 1
    assert body["n_match_comparable"] == 1
    assert body["n_match_review_only"] == 1


def test_compare_returns_the_full_matrix_score_descending():
    entry = _entry([_session_face(0, (1.0, 0.0)), _session_face(1, (0.0, 1.0))])
    store = _store(
        rows=[{"id": PID_EMB, "observation_key": "k1"}],
        records=[_record(PID_EMB, (1.0, 0.0))],
    )
    body = _post_compare(store, entry).json()
    assert [(p["query_face_index"], pytest.approx(p["score"])) for p in body["pairs"]] == [
        (0, pytest.approx(1.0)),
        (1, pytest.approx(0.0)),
    ]


def test_compare_is_labelled_uncalibrated_and_applies_no_threshold():
    entry = _entry([_session_face(0, (1.0, 0.0))])
    store = _store(
        rows=[{"id": PID_EMB, "observation_key": "k1"}],
        records=[_record(PID_EMB, (-1.0, 0.0))],
    )
    body = _post_compare(store, entry).json()
    # A negative cosine survives: no floor is applied server-side.
    assert body["pairs"][0]["score"] == pytest.approx(-1.0)
    assert body["calibration"]["status"] == "uncalibrated"
    assert "threshold" not in body


def test_compare_refuses_a_hard_compat_mismatch_with_409():
    entry = _entry([_session_face(0, (1.0, 0.0))])
    store = _store()
    store.check_compat.side_effect = ValueError("embedder_model_hash differs")
    resp = _post_compare(store, entry)
    assert resp.status_code == 409


def test_compare_validates_the_image_hash_before_store_access():
    entry = _entry([_session_face(0, (1.0, 0.0))])
    store = _store()
    resp = _post_compare(store, entry, image_hash="../etc/passwd")
    assert resp.status_code == 400
    store.list_faces.assert_not_called()


# ── POST /api/faces/search with stored point_ids ───────────────────────────


def _post_search(store, entry, data):
    patches = _patched(store) + (
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.search_query_faces", return_value=[]),
    )
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8] as search,
    ):
        resp = client.post("/api/faces/search", data={"session_id": "s", "file_id": "f", **data})
    return resp, search


def test_search_accepts_stored_point_ids_as_probes():
    entry = _entry([_session_face(0, (1.0, 0.0))])
    store = _store(records=[_record(PID_EMB, (0.5, 0.5))])
    resp, search = _post_search(store, entry, {"face_indices": "0", "point_ids": PID_EMB})
    assert resp.status_code == 200
    probes = search.call_args.args[1]
    assert (0, [1.0, 0.0]) in probes
    assert (f"pt:{PID_EMB}", [0.5, 0.5]) in probes


def test_search_point_probes_alone_are_enough():
    entry = _entry([])
    store = _store(records=[_record(PID_EMB, (0.5, 0.5))])
    resp, search = _post_search(store, entry, {"point_ids": PID_EMB})
    assert resp.status_code == 200
    assert search.call_args.args[1] == [(f"pt:{PID_EMB}", [0.5, 0.5])]


def test_search_refuses_a_review_only_point_probe_with_400():
    # The stored point exists but carries no face vector: demoted to
    # review-only.  Same guarantee, same status as the session-side refusal.
    entry = _entry([])
    store = _store(records=[_record(PID_REV, None)])
    resp, _ = _post_search(store, entry, {"point_ids": PID_REV})
    assert resp.status_code == 400
    assert "review-only" in resp.json()["detail"]


def test_search_refuses_an_unknown_point_id_with_400():
    entry = _entry([])
    store = _store(records=[])
    resp, _ = _post_search(store, entry, {"point_ids": PID_EMB})
    assert resp.status_code == 400
    assert "unknown point id" in resp.json()["detail"]


def test_search_validates_point_ids_before_store_access():
    entry = _entry([])
    store = _store()
    resp, _ = _post_search(store, entry, {"point_ids": "../etc/passwd"})
    assert resp.status_code == 400
    store.client.retrieve.assert_not_called()


def test_search_with_neither_probe_kind_keeps_the_existing_400():
    entry = _entry([])
    store = _store()
    resp, _ = _post_search(store, entry, {"face_indices": " ", "point_ids": ""})
    assert resp.status_code == 400
    assert "no face indices given" in resp.json()["detail"]
