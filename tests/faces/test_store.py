from unittest.mock import MagicMock, patch

import pytest

from scalar_forensic.faces.provenance import PipelineConfig
from scalar_forensic.faces.store import FaceStore


def _cfg(**over):
    base = dict(
        detector_id="yunet",
        detector_model_hash="d" * 64,
        detector_score_threshold=0.5,
        detect_max_size=1600,
        embedder_model_name="emb.onnx",
        embedder_model_hash="e" * 64,
        manifest_hash="m" * 64,
        embedder_dim=512,
        alignment_version="arcface-112-v1",
        normalization_id="affine-127.5-128.0",
        min_conf=0.8,
        min_size=64,
        min_sharpness=25.0,
        max_clipped=0.6,
        max_pose=0.35,
        crop_dilation=0.15,
        sfn_version="1.0",
        cv2_version="4.10",
        ort_version="1.18",
    )
    base.update(over)
    return PipelineConfig(**base)


@pytest.fixture()
def store():
    client = MagicMock()  # injected, TagStore-style — no class patching needed
    client.get_collections.return_value.collections = []
    yield FaceStore(client, "case1_faces", "case1", 512), client


def test_point_id_is_deterministic_and_bbox_rounded(store):
    s, _ = store
    a = s.face_point_id("h1", 1234, (10.4, 20.6, 30.0, 40.0))
    b = s.face_point_id("h1", 1234, (10.0, 21.0, 30.0, 40.0))
    assert a == b  # rounds to same ints
    assert a != s.face_point_id("h1", 5678, (10, 21, 30, 40))
    assert a == s.face_point_id("h1", 1234, (10.4, 20.6, 30.0, 40.0))  # stable


def test_point_id_includes_alignment_version(store):
    s, _ = store
    with patch("scalar_forensic.faces.store.ALIGNMENT_VERSION", "arcface-112-v2"):
        other = s.face_point_id("h1", 1234, (10, 21, 30, 40))
    assert other != s.face_point_id("h1", 1234, (10, 21, 30, 40))


def test_observation_key_matches_point_id_inputs(store):
    s, _ = store
    assert s.observation_key("h1", None, (1.0, 2.0, 3.0, 4.0)) == "h1::1:2:3:4"


def test_ensure_collection_creates_with_quantization_and_indexes(store):
    s, client = store
    s.ensure_collection(_cfg(), examiner_id="ex1", authorization_ref="case order 7")
    create = client.create_collection.call_args
    assert "face" in create.kwargs["vectors_config"]
    assert create.kwargs["quantization_config"] is not None
    assert create.kwargs["on_disk_payload"] is True
    indexed = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
    assert {
        "image_hash",
        "image_path",
        "video_hash",
        "video_path",
        "is_face",
        "group_id",
        "quality",
        "frame_timecode_ms",
    } <= indexed
    # Meta point recorded the enablement + case binding.
    upserted = client.upsert.call_args.kwargs["points"]
    meta = upserted[0].payload
    assert meta["case_collection"] == "case1"
    assert meta["enablement"]["examiner_id"] == "ex1"


def test_ensure_collection_is_noop_when_collection_exists(store):
    s, client = store
    existing = MagicMock()
    existing.name = "case1_faces"
    client.get_collections.return_value.collections = [existing]
    s.ensure_collection(_cfg(), examiner_id="ex1", authorization_ref=None)
    client.create_collection.assert_not_called()
    client.upsert.assert_not_called()


def test_collection_is_new_reflects_server_state(store):
    s, client = store
    assert s.collection_is_new() is True
    existing = MagicMock()
    existing.name = "case1_faces"
    client.get_collections.return_value.collections = [existing]
    assert s.collection_is_new() is False


def _existing_meta(client, payload):
    rec = MagicMock()
    rec.payload = payload
    client.retrieve.return_value = [rec]


def test_check_compat_hard_mismatch_raises(store):
    s, client = store
    _existing_meta(
        client,
        {
            "case_collection": "case1",
            "embedder_model_hash": "x" * 64,
            "manifest_hash": "m" * 64,
            "embedder_dim": 512,
            "alignment_version": "arcface-112-v1",
            "normalization_id": "affine-127.5-128.0",
        },
    )
    with pytest.raises(ValueError, match="embedder_model_hash"):
        s.check_compat(_cfg())


def test_check_compat_soft_mismatch_warns(store):
    s, client = store
    _existing_meta(
        client,
        {
            "case_collection": "case1",
            "embedder_model_hash": "e" * 64,
            "manifest_hash": "m" * 64,
            "embedder_dim": 512,
            "alignment_version": "arcface-112-v1",
            "normalization_id": "affine-127.5-128.0",
            "detector_model_hash": "old" * 21 + "x",
        },
    )
    warnings = s.check_compat(_cfg())
    assert any("detector_model_hash" in w for w in warnings)


def test_check_compat_absent_fields_are_unknown_not_mismatch(store):
    s, client = store
    _existing_meta(client, {"case_collection": "case1"})
    assert s.check_compat(_cfg()) == []


def test_check_compat_wrong_case_collection_raises(store):
    s, client = store
    _existing_meta(client, {"case_collection": "other_case"})
    with pytest.raises(ValueError, match="case"):
        s.check_compat(_cfg())


def test_marker_point_is_per_processed_unit(store):
    s, _ = store
    frame = s.marker_point("frame_hash", "vid1", "cfg", 3, 2, {"pose": 1})
    video_frame2 = s.marker_point("frame2_hash", "vid1", "cfg", 1, 1, {})
    assert frame.id != video_frame2.id  # per frame, never per video
    assert frame.payload["is_face_marker"] is True
    assert frame.payload["n_detected"] == 3 and frame.payload["n_kept"] == 2
    assert frame.payload["n_rejected"] == {"pose": 1}
    assert frame.payload["faces_pipeline_config_hash"] == "cfg"


def test_processed_hashes_filters_on_config_hash(store):
    s, _ = store
    rec = MagicMock()
    rec.payload = {"image_hash": "h1"}
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", return_value=iter([rec])):
        assert s.processed_hashes("cfg") == {"h1"}


def test_purge_media_deletes_points_and_returns_chip_hashes(store):
    s, client = store
    rec1 = MagicMock()
    rec1.id = "p1"
    rec1.payload = {"chip_hash": "c" * 64}
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", return_value=iter([rec1])):
        result = s.purge_media("h1")
    assert result.n_points == 1
    assert result.chip_hashes == ["c" * 64]  # caller unlinks chip files (spec §7.5)
    client.delete.assert_called_once()


def test_purge_all_preserves_meta_point(store):
    s, client = store
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", return_value=iter([])):
        s.purge_all()
    # Points deleted by filter (is_face / is_face_marker), never delete_collection:
    # the meta point carries the enablement record and must survive routine purges.
    client.delete_collection.assert_not_called()
    client.delete.assert_called()


def test_list_faces_returns_payloads_for_media(store):
    s, _ = store
    rec = MagicMock()
    rec.id = "p1"
    rec.payload = {"chip_hash": "c" * 64, "quality": 0.9}
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", return_value=iter([rec])):
        faces = s.list_faces("h1")
    assert faces == [{"id": "p1", "chip_hash": "c" * 64, "quality": 0.9}]
