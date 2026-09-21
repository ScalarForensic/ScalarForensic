"""Source identity and verified purge against an isolated real Qdrant client."""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from scalar_forensic.faces.store import FaceStore
from tests.faces.test_store import _cfg


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    store = FaceStore(client, "faces", "case", 512)
    store.ensure_collection(_cfg(), "test", None)
    yield store
    client.close()


def seed(store, source, timecode, chip="shared"):
    image_hash = "f" * 64
    unit = store.unit_key(image_hash, source, timecode)
    pid = store.face_point_id(image_hash, timecode, (1, 2, 3, 4), video_hash=source)
    store.upsert_faces(
        [
            PointStruct(
                id=pid,
                vector={},
                payload={
                    "is_face": True,
                    "image_hash": image_hash,
                    "video_hash": source,
                    "source_unit": unit,
                    "frame_timecode_ms": timecode,
                    "review_chip_hash": chip,
                },
            ),
            store.marker_point(image_hash, source, "cfg", 1, 0, {}, frame_timecode_ms=timecode),
        ]
    )
    return pid


def test_repeated_frame_observations_markers_and_stale_scope(store):
    a = seed(store, "a" * 64, 0)
    b = seed(store, "a" * 64, 1000)
    c = seed(store, "b" * 64, 0)
    assert len({a, b, c}) == 3
    assert len(store.processed_units("cfg")) == 3
    assert store.stale_face_points("f" * 64, {a}, video_hash="a" * 64, frame_timecode_ms=0) == []
    stale = store.stale_face_points("f" * 64, set(), video_hash="a" * 64, frame_timecode_ms=1000)
    assert [p["id"] for p in stale] == [b]
    seed(store, "a" * 64, 0)
    assert len(store.processed_units("cfg")) == 3


def test_video_purge_removes_rollup_and_markers_preserves_shared_chips(store):
    seed(store, "a" * 64, 0)
    seed(store, "a" * 64, 1000)
    survivor = seed(store, "b" * 64, 0)
    store.upsert_faces([store.video_rollup_point("a" * 64, "cfg", 2, 0, {}, 2)])
    result = store.purge_media("a" * 64)
    assert result.n_points == 5
    assert store.unreferenced_chip_hashes(result.chip_hashes) == []
    assert store.get_face(survivor)
    assert len(store.processed_units("cfg")) == 1
    assert store.purge_media("a" * 64).n_points == 0
    store.check_compat(_cfg())


def test_legacy_schema_requires_explicit_rebuild(store):
    meta_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "sfn:face-collection-meta"))
    store.client.set_payload("faces", {"identity_schema_version": 1}, [meta_id], wait=True)
    with pytest.raises(ValueError, match="rebuild"):
        store.check_compat(_cfg())


def test_purge_refuses_invalid_digest(store):
    with pytest.raises(ValueError, match="SHA-256"):
        store.purge_media("../not-a-digest")


def test_rollup_recounts_previously_committed_frames(store):
    seed(store, "a" * 64, 0)
    store.refresh_video_rollup("a" * 64, "cfg")
    seed(store, "a" * 64, 1000)
    store.refresh_video_rollup("a" * 64, "cfg")
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, "face-marker-video:" + "a" * 64))
    assert store.get_face(pid)["n_frames"] == 2
    assert store.get_face(pid)["n_detected"] == 2


def test_purge_failure_is_not_reported_as_completed(store, monkeypatch):
    seed(store, "a" * 64, 0)
    monkeypatch.setattr(store.client, "delete", lambda **kw: None)
    with pytest.raises(RuntimeError, match="verification failed"):
        store.purge_media("a" * 64)


def test_legacy_image_purge_does_not_leave_markers_or_delete_video_frames(store):
    digest = "c" * 64
    legacy_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    shared_video = str(uuid.uuid4())
    store.upsert_faces(
        [
            PointStruct(
                id=legacy_ids[0],
                vector={},
                payload={
                    "is_face": True,
                    "image_hash": digest,
                    "video_hash": None,
                    "review_chip_hash": "shared",
                },
            ),
            PointStruct(
                id=legacy_ids[1],
                vector={},
                payload={
                    "is_face_marker": True,
                    "image_hash": digest,
                    "video_hash": None,
                },
            ),
            PointStruct(
                id=shared_video,
                vector={},
                payload={
                    "is_face": True,
                    "image_hash": digest,
                    "video_hash": "a" * 64,
                    "frame_timecode_ms": 0,
                    "review_chip_hash": "shared",
                },
            ),
        ]
    )
    result = store.purge_media(digest)
    assert result.n_points == 2
    assert all(store.get_face(pid) is None for pid in legacy_ids)
    assert store.get_face(shared_video)
    assert store.unreferenced_chip_hashes(result.chip_hashes) == []
