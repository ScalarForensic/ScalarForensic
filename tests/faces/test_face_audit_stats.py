"""Face audit and face score-distribution stats (Stage 7 backend).

The stats mirror web/pipeline/stats.py field for field on purpose: the two
modalities must read alike in the UI.  The audit reports the gates that were in
force *at index time*, read from the persisted payload — no pipeline step is
re-run and today's environment is never consulted.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scalar_forensic.web.app import app
from scalar_forensic.web.pipeline.faces_stats import face_score_stats

client = TestClient(app)


def test_face_score_stats_has_the_same_field_set_as_the_dino_stats():
    from scalar_forensic.web.pipeline.stats import SemanticStats

    store = MagicMock()
    store.search_faces.return_value = [
        {"point_id": f"p{i}", "score": s} for i, s in enumerate([0.1, 0.2, 0.6, 0.9])
    ]
    stats = face_score_stats(store, [1.0, 0.0], sample_size=100)
    assert set(vars(stats)) == set(SemanticStats.__dataclass_fields__)
    assert len(stats.histogram) == 20
    assert stats.max_score == pytest.approx(0.9)
    assert stats.count == 4


def test_face_score_stats_are_computed_with_exact_search():
    store = MagicMock()
    store.search_faces.return_value = [{"point_id": "p", "score": 0.5}]
    face_score_stats(store, [1.0], sample_size=50)
    assert store.search_faces.call_args.kwargs["exact"] is True
    assert store.search_faces.call_args.kwargs["threshold"] == 0.0


def test_face_score_stats_match_the_dino_percentiles_line_for_line():
    """Same inputs through both implementations must give the same numbers."""
    import statistics as _statistics

    scores = [0.02, 0.11, 0.19, 0.31, 0.44, 0.52, 0.63, 0.71, 0.88]
    store = MagicMock()
    store.search_faces.return_value = [
        {"point_id": f"p{i}", "score": s} for i, s in enumerate(scores)
    ]
    stats = face_score_stats(store, [1.0], sample_size=100)

    cuts = _statistics.quantiles(scores, n=100, method="inclusive")
    assert stats.p10 == pytest.approx(cuts[9])
    assert stats.p25 == pytest.approx(cuts[24])
    assert stats.median == pytest.approx(cuts[49])
    assert stats.p75 == pytest.approx(cuts[74])
    assert stats.p90 == pytest.approx(cuts[89])
    assert stats.stdev == pytest.approx(_statistics.stdev(scores))
    assert stats.mean == pytest.approx(sum(scores) / len(scores))
    # Histogram is on the normalised [-1,1] -> [0,1] scale, 20 buckets.
    assert sum(stats.histogram) == len(scores)
    assert stats.histogram[int(((0.02 + 1.0) / 2.0) * 20)] >= 1


def test_face_score_stats_of_a_single_hit_report_zero_stdev():
    store = MagicMock()
    store.search_faces.return_value = [{"point_id": "p", "score": 0.5}]
    stats = face_score_stats(store, [1.0], sample_size=50)
    assert stats.stdev == 0.0
    assert stats.median == pytest.approx(0.5)


def _audit_store() -> MagicMock:
    store = MagicMock()
    store.get_marker.return_value = {
        "n_detected": 6,
        "n_kept": 2,
        "n_review_only": 3,
        "review_only_reasons": {"size": 3},
        "rejected": {"confidence": 1},
    }
    store.list_faces.return_value = [
        {
            "id": "p1",
            "min_conf": 0.8,
            "min_size": 64,
            "review_min_size": 36,
            "detector_id": "yunet",
            "detector_model_hash": "d1",
            "embedder_model_hash": "e1",
            "manifest_hash": "m1",
            "embedder_dim": 128,
            "alignment_version": "arcface-112-v1",
            "normalization_id": "affine-0.0-1.0",
            "pipeline_config_hash": "c1",
            "cv2_version": "4.10",
            "ort_version": "1.19",
            "sfn_version": "0.1",
            "crop_dilation": 0.25,
            "max_pose": 0.35,
            "min_sharpness": 25.0,
            "max_clipped": 0.6,
            "review_min_conf": 0.6,
            "detector_score_threshold": 0.5,
            "detect_max_size": 1600,
            "embedder_model_name": "sface",
        }
    ]
    store.check_compat.return_value = []
    return store


def test_face_audit_reports_index_time_thresholds_not_current_env():
    store = _audit_store()
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        body = client.get("/api/faces/audit?image_hash=" + "a" * 64).json()

    assert body["gates_in_force_at_index_time"]["review_min_size"] == 36
    assert body["file_totals"]["n_review_only"] == 3
    assert body["detector"]["detector_id"] == "yunet"
    assert "caveat" in body


def test_face_audit_rejects_a_malformed_hash():
    with patch("scalar_forensic.web.routes.faces.Settings"):
        assert client.get("/api/faces/audit?image_hash=zz").status_code == 400


def test_face_audit_404s_when_the_image_has_no_face_marker():
    store = _audit_store()
    store.get_marker.return_value = None
    store.list_faces.return_value = []
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        resp = client.get("/api/faces/audit?image_hash=" + "b" * 64)
    assert resp.status_code == 404


def _query_face(vector):
    from scalar_forensic.web.session import QueryFace

    return QueryFace(
        index=0,
        bbox=(1.0, 2.0, 3.0, 4.0),
        landmarks=[[1.0, 2.0]] * 5,
        det_conf=0.9,
        detect_scale=1.0,
        quality={"confidence": 0.9, "size": 90.0, "pose": 0.1, "sharpness": 80.0, "exposure": 0.1},
        embedding_status="embedded" if vector else "review_only",
        embedding_exclusion_reason=None if vector else "size",
        vector=vector,
        review_jpeg=b"\xff\xd8",
    )


def _post_dist_stats(entry, store=None):
    patches = [
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
    ]
    if store is not None:
        patches += [
            patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
            patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return client.post(
            "/api/faces/dist-stats",
            data={"session_id": "s", "file_id": "f", "face_index": 0},
        )


def test_dist_stats_names_its_population_and_quotes_the_reference_figure():
    store = MagicMock()
    store.search_faces.return_value = [
        {"point_id": f"p{i}", "score": s} for i, s in enumerate([0.1, 0.4, 0.7])
    ]
    entry = MagicMock(query_faces=[_query_face([1.0, 0.0])])
    body = _post_dist_stats(entry, store).json()

    assert body["count"] == 3
    assert len(body["histogram"]) == 20
    assert "review-only" in body["population"]
    assert "structurally absent" in body["population"]
    assert body["model_reference_threshold"] == 0.363
    assert "not a threshold calibrated" in body["model_reference_note"]
    # The reference figure is quoted, never applied.
    assert store.search_faces.call_args.kwargs["threshold"] == 0.0


def test_dist_stats_refuses_a_review_only_face():
    entry = MagicMock(query_faces=[_query_face(None)])
    resp = _post_dist_stats(entry)
    assert resp.status_code == 400
    assert "review-only" in resp.json()["detail"]


def test_dist_stats_404s_on_an_unknown_face_index():
    entry = MagicMock(query_faces=[])
    resp = _post_dist_stats(entry)
    assert resp.status_code == 404


def test_face_audit_never_re_runs_a_pipeline_step():
    """Only persisted reads: get_marker, list_faces, check_compat."""
    store = _audit_store()
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        body = client.get("/api/faces/audit?image_hash=" + "a" * 64).json()

    called = {c[0] for c in store.method_calls}
    assert called <= {"get_marker", "list_faces", "check_compat", "_meta_payload"}
    assert body["embedder"]["embedder_dim"] == 128
    assert body["pipeline_config_hash"] == "c1"
