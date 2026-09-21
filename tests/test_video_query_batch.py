"""Batched RPCs preserve per-frame scores and bound request size."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scalar_forensic.config import Settings
from scalar_forensic.web.pipeline.query import query_session
from scalar_forensic.web.session import FileEntry, Session


def _session(tmp_path, count):
    entry = FileEntry("f", "query.mp4", tmp_path / "query.mp4", is_video=True)
    entry.video_frames = [
        SimpleNamespace(
            timecode_ms=i * 1000,
            dino_embedding=[float(i + 1), 0.0],
            sscd_embedding=[float(i + 1), 0.0],
        )
        for i in range(count)
    ]
    return Session("session", [entry])


@pytest.mark.parametrize("unify", [True, False])
def test_video_batches_bound_rpcs_preserve_modes_frames_and_reference(tmp_path, unify):
    settings = Settings()
    settings.collection = "case"
    settings.reference_collection = "reference"
    client = MagicMock()

    def batch(collection_name, requests):
        return [
            SimpleNamespace(
                points=[
                    SimpleNamespace(
                        score=request.query[0] / 200,
                        payload={"image_path": "/same.jpg", "image_hash": "digest"},
                    )
                ]
            )
            for request in requests
        ]

    client.query_batch_points.side_effect = batch
    with (
        patch("scalar_forensic.web.pipeline.query.QdrantClient", return_value=client),
        patch("scalar_forensic.web.pipeline.query.require_collection_compatible"),
    ):
        results, _ = query_session(
            _session(tmp_path, 130),
            ["altered", "semantic"],
            0.0,
            0.0,
            10,
            settings,
            unify=unify,
            include_reference=True,
        )
    client.query_points.assert_not_called()
    assert client.query_batch_points.call_count == 10
    assert max(len(c.kwargs["requests"]) for c in client.query_batch_points.call_args_list) <= 64
    hits = results[0].hits
    assert not results[0].errors
    if unify:
        assert len(hits) == 2
        for hit in hits:
            assert hit.scores == {"altered": 0.65, "semantic": 0.65}
            assert set(hit.query_timecodes) == set(range(0, 130000, 1000))
            assert hit.best_query_timecode_ms == 129000
    else:
        assert len(hits) == 520
        assert all(len(h.scores) == 1 and len(h.query_timecodes) == 1 for h in hits)
    assert {h.is_reference for h in hits} == {False, True}
    client.close.assert_called_once()


def test_failed_batch_retains_later_frames_and_reports_errors(tmp_path):
    settings = Settings()
    client = MagicMock()
    client.query_batch_points.side_effect = [
        RuntimeError("unavailable"),
        [SimpleNamespace(points=[SimpleNamespace(score=0.8, payload={"image_path": "/later"})])],
    ]
    with (
        patch("scalar_forensic.web.pipeline.query.QdrantClient", return_value=client),
        patch("scalar_forensic.web.pipeline.query.require_collection_compatible"),
    ):
        results, _ = query_session(_session(tmp_path, 33), ["semantic"], 0, 0, 10, settings)
    assert results[0].hits[0].query_timecodes == [32000]
    assert results[0].errors
    assert "RuntimeError" in results[0].errors[0]
    client.close.assert_called_once()


def test_real_qdrant_batch_matches_individual_cosine_results(tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(":memory:")
    settings = Settings()
    settings.collection = "case"
    client.create_collection(
        "case",
        vectors_config={
            "dino": VectorParams(size=2, distance=Distance.COSINE),
            "sscd": VectorParams(size=2, distance=Distance.COSINE),
        },
    )
    client.upsert(
        "case",
        [
            PointStruct(id=i, vector={"dino": v, "sscd": v}, payload={"image_path": f"/image-{i}"})
            for i, v in enumerate(([1.0, 0.0], [1.0, 1.0], [0.0, 1.0]))
        ],
    )
    expected = client.query_points("case", query=[1.0, 0.0], using="dino", limit=10).points
    with (
        patch("scalar_forensic.web.pipeline.query.QdrantClient", return_value=client),
        patch("scalar_forensic.web.pipeline.query.require_collection_compatible"),
    ):
        results, _ = query_session(_session(tmp_path, 3), ["semantic"], 0, 0, 10, settings)
    assert [(h.path, h.scores["semantic"]) for h in results[0].hits] == [
        (p.payload["image_path"], p.score) for p in expected
    ]
    assert all(h.query_timecodes == [0, 1000, 2000] for h in results[0].hits)
