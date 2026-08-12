import io
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from scalar_forensic.faces.chips import chip_paths, review_chip_paths
from scalar_forensic.faces.indexing import FacePipeline
from scalar_forensic.faces.store import FACE_VECTOR_NAME
from scalar_forensic.faces.types import FaceDetection

FRONTAL = np.array([[130, 130], [170, 130], [150, 155], [135, 175], [165, 175]], np.float32)


def _img_bytes() -> bytes:
    rng = np.random.default_rng(5)
    img = rng.integers(30, 220, (300, 300, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _det(x=100.0, y=100.0, w=100.0, h=None, conf=0.95, scale=1.0, lm=FRONTAL):
    return FaceDetection(
        bbox=(x, y, w, h if h is not None else w),
        landmarks=lm,
        confidence=conf,
        detect_scale=scale,
    )


def _pipeline(detections):
    detector = MagicMock()
    detector.detect.return_value = detections
    detector.detector_id = "yunet"
    detector.model_hash = "d" * 64
    embedder = MagicMock()
    embedder.embed.side_effect = lambda crops: np.eye(512, dtype=np.float32)[: len(crops)]
    embedder.embedding_norms = np.full(len(detections), 21.7, np.float32)
    store = MagicMock()
    # bbox enters the id: distinct faces in one image must not collide, or the
    # review-only id list would silently deduplicate.
    store.face_point_id.side_effect = lambda h, t, b: f"id-{h}-{t}-{int(b[0])}x{int(b[2])}"
    store.observation_key.side_effect = lambda h, t, b: f"{h}:{t or ''}:obs"
    return FacePipeline(
        detector=detector,
        embedder=embedder,
        store=store,
        audit=MagicMock(),
        cfg=MagicMock(config_hash="c" * 64, to_payload=lambda: {"pipeline_config_hash": "c" * 64}),
        min_conf=0.8,
        min_size=64,
        min_sharpness=25.0,
        max_clipped=0.6,
        max_pose=0.35,
        review_min_conf=0.6,
        review_min_size=24,
        crop_dilation=0.15,
        store_dir=None,
    )


def _embedded_crops(pipeline):
    """The aligned crops actually handed to the embedder."""
    if not pipeline.embedder.embed.call_args_list:
        return []
    return pipeline.embedder.embed.call_args_list[0].args[0]


def test_process_image_keeps_good_face_and_builds_payload():
    det = FaceDetection(
        bbox=(100, 100, 100, 100), landmarks=FRONTAL, confidence=0.95, detect_scale=1.0
    )
    result = _pipeline([det]).process_image(_img_bytes(), image_hash="h1", image_path="/x.png")
    assert result.n_detected == 1 and result.n_kept == 1
    payload = result.points[0].payload
    assert payload["is_face"] is True and payload["image_hash"] == "h1"
    assert payload["observation_key"] == "h1::obs"
    assert "quality_sharpness" in payload and payload["embedding_norm"] > 0
    assert 0.0 <= payload["quality"] <= 1.0  # composite the payload index serves


def test_process_image_counts_rejections_by_reason():
    weak = FaceDetection(
        bbox=(100, 100, 100, 100), landmarks=FRONTAL, confidence=0.3, detect_scale=1.0
    )
    result = _pipeline([weak]).process_image(_img_bytes(), image_hash="h1", image_path="/x.png")
    assert result.n_kept == 0 and result.rejected == {"confidence": 1}
    assert result.points == []  # rejected faces are never persisted


def test_zero_faces_is_a_valid_result():
    result = _pipeline([]).process_image(_img_bytes(), image_hash="h1", image_path="/x.png")
    assert result.n_detected == 0 and result.n_kept == 0


def test_video_frame_payload_carries_video_provenance():
    det = FaceDetection(
        bbox=(100, 100, 100, 100), landmarks=FRONTAL, confidence=0.95, detect_scale=1.0
    )
    result = _pipeline([det]).process_image(
        _img_bytes(),
        image_hash="frame1",
        image_path="/frames/f1.jpg",
        video_hash="v1",
        video_path="/v.mp4",
        frame_timecode_ms=4000,
    )
    payload = result.points[0].payload
    assert payload["video_hash"] == "v1" and payload["video_path"] == "/v.mp4"
    assert payload["frame_timecode_ms"] == 4000


def test_embedding_is_one_batch_per_image():
    dets = [
        FaceDetection(
            bbox=(10 + 120 * i, 100, 100, 100), landmarks=FRONTAL, confidence=0.95, detect_scale=1.0
        )
        for i in range(2)
    ]
    p = _pipeline(dets)
    result = p.process_image(_img_bytes(), image_hash="h1", image_path="/x.png")
    assert result.n_kept == 2
    assert p.embedder.embed.call_count == 1  # one batch, not one call per face


def test_interleaved_outcomes_pair_each_embedding_with_its_own_face():
    # SENTINEL: the embeddings array has one row per EMBEDDABLE face only.
    # Alternating outcomes is what breaks a naive enumerate() over a combined
    # list -- it would attach one person's vector to another's observation.
    dets = [
        _det(x=10.0, w=200.0),  # embeddable
        _det(x=220.0, w=50.0),  # review-only (below the 64px embedding floor)
        _det(x=10.0, w=200.0, y=10.0),  # embeddable
        _det(x=220.0, w=50.0, y=200.0),  # review-only
    ]
    p = _pipeline(dets)
    result = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")

    assert result.n_kept == 2
    assert result.n_review_only == 2
    embedded = [pt for pt in result.points if pt.payload["embedding_status"] == "embedded"]
    assert len(embedded) == 2
    # eye(512) row i is handed back for the i-th crop in the batch, so a
    # misaligned pairing shows up as the wrong one-hot index.
    assert embedded[0].vector[FACE_VECTOR_NAME][0] == 1.0
    assert embedded[1].vector[FACE_VECTOR_NAME][1] == 1.0
    assert len(_embedded_crops(p)) == 2
    # And each embedded point must describe the face whose vector it holds.
    assert [pt.payload["bbox"][2] for pt in embedded] == [200, 200]


def test_all_review_only_image_still_yields_observations():
    # Guards the early-return: this is exactly what danny1.jpeg produces.
    p = _pipeline([_det(x=10.0, w=50.0), _det(x=100.0, w=50.0), _det(x=200.0, w=50.0)])
    result = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")
    assert result.n_review_only == 3
    assert len(result.points) == 3
    assert p.embedder.embed.call_count == 0  # nothing to embed, no batch at all


def test_review_only_points_carry_no_vector():
    p = _pipeline([_det(w=50.0)])
    point = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg").points[0]
    assert point.vector == {}
    assert point.payload["embedding_status"] == "review_only"
    assert point.payload["embedding_exclusion_reason"] == "size"
    assert point.payload["is_face"] is True
    assert point.payload["aligned_chip_hash"] is None
    # Embedding-only quality subscores must be absent, not zero: a 0.0 pose
    # would read as "perfectly frontal" rather than "never measured".
    assert point.payload["quality_pose"] is None
    assert point.payload["quality_sharpness"] is None
    assert point.payload["quality"] is None
    assert point.payload["embedding_norm"] is None


def test_review_only_point_ids_are_collected_for_demotion():
    p = _pipeline([_det(x=10.0, w=50.0), _det(x=100.0, w=200.0)])
    r = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")
    review_only = [pt for pt in r.points if pt.payload["embedding_status"] == "review_only"]
    assert r.review_only_point_ids == [pt.id for pt in review_only]
    # Never an embedded point's id: clearing its vector would destroy data.
    embedded_ids = {pt.id for pt in r.points if pt.payload["embedding_status"] == "embedded"}
    assert not embedded_ids & set(r.review_only_point_ids)


def test_review_only_failures_are_not_counted_as_rejections():
    p = _pipeline([_det(x=10.0, w=50.0), _det(x=100.0, w=10.0)])
    r = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")
    assert r.n_review_only == 1
    assert r.review_only_reasons == {"size": 1}
    assert r.rejected == {"size": 1}
    # The partition is exhaustive: every detection lands in exactly one bucket.
    assert r.n_detected == r.n_kept + r.n_review_only + sum(r.rejected.values())


def test_degenerate_crop_is_rejected_not_retained():
    # A review-only observation whose crop does not exist is useless.
    p = _pipeline([_det(x=5000.0, y=5000.0, w=50.0)])
    r = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")
    assert r.n_review_only == 0
    assert r.rejected == {"size": 1}
    assert r.points == []


def test_embedded_points_record_both_chip_hashes(tmp_path):
    p = _pipeline([_det(w=200.0)])
    p.store_dir = tmp_path
    r = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")
    payload = r.points[0].payload
    assert payload["embedding_status"] == "embedded"
    assert payload["embedding_exclusion_reason"] is None
    # Distinct domains, and both files exist where the payload says they are.
    assert payload["aligned_chip_hash"] != payload["review_chip_hash"]
    png, _, _ = chip_paths(tmp_path, payload["aligned_chip_hash"])
    review, thumb = review_chip_paths(tmp_path, payload["review_chip_hash"])
    assert png.exists() and review.exists() and thumb.exists()


def test_review_only_chips_are_written_under_the_review_hash(tmp_path):
    p = _pipeline([_det(w=50.0)])
    p.store_dir = tmp_path
    r = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")
    payload = r.points[0].payload
    review, thumb = review_chip_paths(tmp_path, payload["review_chip_hash"])
    assert review.exists() and thumb.exists()
    # No aligned PNG is written for a face that was never aligned.
    assert not any(tmp_path.rglob("*.png"))


def test_noncanonical_drops_are_carried_onto_the_result():
    # The detector's counter is cumulative across images; the result must
    # record this image's delta, not the running total.
    p = _pipeline([_det(w=200.0)])
    p.detector.n_dropped_noncanonical = 7  # a previous image already dropped 7

    def _detect_and_drop(_img):
        p.detector.n_dropped_noncanonical += 2
        return [_det(w=200.0)]

    p.detector.detect.side_effect = _detect_and_drop
    r = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")
    assert r.n_dropped_noncanonical == 2


def test_detector_without_the_counter_is_tolerated():
    p = _pipeline([_det(w=200.0)])
    del p.detector.n_dropped_noncanonical
    p.detector.mock_add_spec(["detect", "detector_id", "model_hash"])
    p.detector.detect.return_value = [_det(w=200.0)]
    r = p.process_image(_img_bytes(), image_hash="hash", image_path="path.jpg")
    assert r.n_dropped_noncanonical == 0
