from unittest.mock import MagicMock, patch

import pytest

from scalar_forensic.faces.provenance import PipelineConfig
from scalar_forensic.faces.store import FACE_VECTOR_NAME, FaceStore


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
        review_min_conf=0.6,
        review_min_size=48,
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


def test_video_rollup_never_collides_with_frame_marker(store):
    s, _ = store
    # A rollup keyed on the video hash must not share an ID with the frame
    # marker of a frame that happens to hash to the same value.
    rollup = s.video_rollup_point("v1", "cfg", 10, 6, {"pose": 4}, n_frames=5)
    assert rollup.id != s.marker_point("v1", "v1", "cfg", 10, 6, {}).id
    assert rollup.payload["is_face_video_rollup"] is True
    assert rollup.payload["n_frames"] == 5


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
    rec1.payload = {"aligned_chip_hash": "a" * 64, "review_chip_hash": "r" * 64}
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", return_value=iter([rec1])):
        result = s.purge_media("h1")
    assert result.n_points == 1
    # Both artefact domains are collected; the caller unlinks chip files (spec §7.5).
    assert set(result.chip_hashes) == {"a" * 64, "r" * 64}
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


def test_review_thresholds_are_soft_not_hard(store):
    # Changing them must not raise — they cannot change what a vector means.
    s, client = store
    _existing_meta(client, {"case_collection": "case1", **_cfg().to_payload()})
    notes = s.check_compat(_cfg(review_min_size=24))
    assert any("review_min_size" in n for n in notes)


def _face_rec(pid, payload):
    rec = MagicMock()
    rec.id = pid
    rec.payload = payload
    return rec


def test_clear_face_vector_calls_delete_vectors(store):
    s, client = store
    s.clear_face_vector(["id-a", "id-b"])
    client.delete_vectors.assert_called_once()
    kwargs = client.delete_vectors.call_args.kwargs
    assert kwargs["vectors"] == [FACE_VECTOR_NAME]
    assert set(kwargs["points"]) == {"id-a", "id-b"}


def test_clear_face_vector_noop_on_empty(store):
    s, client = store
    s.clear_face_vector([])
    client.delete_vectors.assert_not_called()


def _scroll_spy(recs):
    """Patch qdrant_scroll_all, recording the kwargs it was called with.

    Asserting on the yielded records alone would mock past the payload
    projection: if the projection omitted a hash field, real Qdrant would
    return payloads without that key, every such chip would look
    unreferenced, and purge would unlink files surviving observations still
    need.  A mock that supplies the key regardless cannot catch that.
    """
    calls: list[dict] = []

    def fake(client, collection, **kwargs):
        calls.append({"collection": collection, **kwargs})
        return iter(recs)

    return fake, calls


def test_unreferenced_chip_hashes_keeps_shared_chips(store):
    # A chip still referenced by a surviving observation must not be unlinked.
    s, _ = store
    fake, calls = _scroll_spy([_face_rec("p1", {"is_face": True, "review_chip_hash": "shared"})])
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", fake):
        assert s.unreferenced_chip_hashes(["shared", "orphan"]) == ["orphan"]
    assert calls[0]["collection"] == "case1_faces"


def test_unreferenced_chip_hashes_checks_both_domains(store):
    # An embedded observation's aligned PNG counts as a reference too.
    s, _ = store
    fake, calls = _scroll_spy([_face_rec("p1", {"is_face": True, "aligned_chip_hash": "a1"})])
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", fake):
        assert s.unreferenced_chip_hashes(["a1", "a2"]) == ["a2"]
    # Both fields must be fetched, or the unfetched one always looks orphaned.
    assert calls[0]["with_payload"] == ["aligned_chip_hash", "review_chip_hash"]


def test_unreferenced_chip_hashes_scrolls_faces_not_markers(store):
    # Marker and rollup points carry no chip hashes; the filter must key on
    # is_face so a payload-index change cannot silently widen the scan.
    s, _ = store
    fake, calls = _scroll_spy([])
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", fake):
        s.unreferenced_chip_hashes(["x"])
    conditions = calls[0]["scroll_filter"].must
    assert [c.key for c in conditions] == ["is_face"]
    assert conditions[0].match.value is True


def test_unreferenced_chip_hashes_deduplicates(store):
    s, _ = store
    fake, _ = _scroll_spy([])
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", fake):
        assert s.unreferenced_chip_hashes(["a", "a", "b"]) == ["a", "b"]


def test_unreferenced_chip_hashes_noop_on_empty(store):
    s, client = store
    assert s.unreferenced_chip_hashes([]) == []
    client.scroll.assert_not_called()


def test_review_only_points_are_purged_by_purge_all(store):
    s, _ = store
    recs = [
        _face_rec(
            "p1",
            {"is_face": True, "embedding_status": "review_only", "review_chip_hash": "r1"},
        )
    ]
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", return_value=iter(recs)):
        result = s.purge_all()
    assert result.n_points == 1
    assert "r1" in result.chip_hashes


def test_marker_records_review_only_counts(store):
    s, _ = store
    p = s.marker_point(
        "img",
        None,
        "cfg",
        n_detected=6,
        n_kept=1,
        rejected={"confidence": 2},
        n_review_only=3,
        review_only_reasons={"size": 2, "pose": 1},
        n_dropped_noncanonical=0,
    )
    assert p.payload["n_review_only"] == 3
    assert p.payload["review_only_reasons"] == {"size": 2, "pose": 1}
    assert p.payload["n_dropped_noncanonical"] == 0
    # The marker must reconcile with its own breakdown, or it is not an
    # honest account of the medium.
    total = p.payload["n_kept"] + p.payload["n_review_only"] + sum(p.payload["n_rejected"].values())
    assert total == p.payload["n_detected"]


def test_rollup_records_review_only_counts(store):
    s, _ = store
    p = s.video_rollup_point(
        "vid",
        "cfg",
        n_detected=2,
        n_kept=1,
        rejected={},
        n_frames=1,
        n_review_only=1,
        review_only_reasons={"size": 1},
        n_dropped_noncanonical=0,
    )
    assert p.payload["n_review_only"] == 1
    assert p.payload["review_only_reasons"] == {"size": 1}
