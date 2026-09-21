"""Reindex regressions checked against Qdrant's actual stored payload/vectors."""

import warnings
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from scalar_forensic.indexer import Indexer

_META = {
    "model_name": "dino-test",
    "model_hash": "dino-hash",
    "embedding_dim": 2,
    "normalize_size": 336,
    "inference_dtype": "float32",
    "library_versions": {"torch": "test"},
}


@pytest.fixture
def indexers():
    with closing(QdrantClient(":memory:")) as client:
        with (
            patch("scalar_forensic.indexer.QdrantClient", return_value=client),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            dino = Indexer(
                "http://unused",
                "case",
                "dino",
                2,
                initial_vectors_config={
                    name: VectorParams(size=2, distance=Distance.COSINE)
                    for name in ("dino", "sscd")
                },
            )
            sscd = Indexer("http://unused", "case", "sscd", 2, is_reference=True)
        yield dino, sscd


def _points(indexer):
    records, _ = indexer.client.scroll("case", with_vectors=True, limit=100)
    return {record.payload["image_hash"]: record for record in records}


def test_reindex_refreshes_each_payload_without_erasing_other_model(indexers, tmp_path):
    """Dropping the core update or replacing the whole point breaks this test."""
    dino, sscd = indexers
    old = [tmp_path / "old-a.jpg", tmp_path / "old-b.jpg"]
    sscd.upsert_batch(
        old,
        ["sha-a", "sha-b"],
        [[1.0, 0.0], [1.0, 0.0]],
        {**_META, "model_name": "sscd-test", "model_hash": "sscd-hash", "sscd_n_crops": 5},
        exif_payloads={old[0]: {"exif_gps": True}, old[1]: {"exif_gps": False}},
        image_hashes_md5=["md5-a", "md5-b"],
    )
    before = _points(sscd)
    moved = [tmp_path / "new-a.jpg", tmp_path / "new-b.jpg", tmp_path / "new-c.jpg"]
    dino.upsert_batch(
        moved,
        ["sha-a", "sha-b", "sha-c"],
        [[0.0, 1.0]] * 3,
        _META,
        exif_payloads={moved[0]: {"exif_gps": False}, moved[1]: {"exif_gps": True}},
    )
    points = _points(dino)
    assert len(points) == 3
    for digest, path, gps, md5 in (
        ("sha-a", moved[0], False, "md5-a"),
        ("sha-b", moved[1], True, "md5-b"),
    ):
        point = points[digest]
        assert point.id == before[digest].id
        assert point.payload["image_path"] == str(path.resolve())
        assert point.payload["exif_gps"] is gps
        assert point.payload["image_hash_md5"] == md5
        assert point.payload["is_reference"] is True
        assert point.vector == {"dino": [0.0, 1.0], "sscd": [1.0, 0.0]}
        assert point.payload["dino_model_hash"] == "dino-hash"
        assert {k: v for k, v in point.payload.items() if k.startswith("sscd_")} == {
            k: v for k, v in before[digest].payload.items() if k.startswith("sscd_")
        }
    assert points["sha-c"].payload["image_path"] == str(moved[2].resolve())
    assert points["sha-c"].vector == {"dino": [0.0, 1.0]}

    # A later pass without optional extraction must retain the known flags.
    dino.upsert_batch(moved[:2], ["sha-a", "sha-b"], [[1.0, 0.0]] * 2, _META)
    assert _points(dino)["sha-a"].payload["exif_gps"] is False
    assert _points(dino)["sha-b"].payload["exif_gps"] is True


def test_reindex_refreshes_video_slot_metadata(indexers, tmp_path):
    dino, sscd = indexers
    frame = tmp_path / "old-frame.jpg"
    metadata = {
        "video_hash": "video-sha",
        "video_path": "/old/source.mp4",
        "frame_timecode_ms": 40,
        "frame_index": 1,
        "extraction_fps": 1.0,
        "max_frames_cap": 100,
        "pyav_version": "12.0",
    }
    sscd.upsert_batch([frame], ["frame-sha"], [[1.0, 0.0]], _META, video_metadata=[metadata])
    sscd.mark_video_complete("video-sha", 1)
    original = _points(sscd)["frame-sha"]
    moved = tmp_path / "new-frame.jpg"
    dino.upsert_batch(
        [moved],
        ["frame-sha"],
        [[0.0, 1.0]],
        _META,
        video_metadata=[{**metadata, "video_path": "/new/source.mp4", "pyav_version": "13.0"}],
    )
    point = _points(dino)["frame-sha"]
    assert point.id == original.id
    assert point.payload["image_path"] == str(moved.resolve())
    assert point.payload["video_path"] == "/new/source.mp4"
    assert point.payload["pyav_version"] == "13.0"
    assert point.payload["sscd_video_frames_total"] == 1
    assert point.payload["frame_timecode_ms"] == 40
    assert point.vector == {"sscd": [1.0, 0.0], "dino": [0.0, 1.0]}


@pytest.mark.parametrize("operation", ["update_vectors", "set_payload"])
def test_failed_existing_update_does_not_acknowledge_or_publish_new_payload(indexers, operation):
    dino, _ = indexers
    dino.upsert_batch([Path("/old.jpg")], ["sha"], [[1.0, 0.0]], _META)
    before = _points(dino)["sha"].payload
    with (
        patch.object(dino.client, operation, side_effect=RuntimeError("write failed")),
        pytest.raises(RuntimeError, match="write failed"),
    ):
        dino.upsert_batch([Path("/new.jpg")], ["sha"], [[0.0, 1.0]], _META)
    point = _points(dino)["sha"]
    assert point.payload == before
    # The existing two-step write is not a transaction: payload errors can
    # leave the vector updated, but must propagate to prevent acknowledgement.
    assert point.vector["dino"] == ([1.0, 0.0] if operation == "update_vectors" else [0.0, 1.0])
