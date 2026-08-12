import io
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from scalar_forensic.faces.indexing import FacePipeline
from scalar_forensic.faces.types import FaceDetection

FRONTAL = np.array([[130, 130], [170, 130], [150, 155], [135, 175], [165, 175]], np.float32)


def _img_bytes() -> bytes:
    rng = np.random.default_rng(5)
    img = rng.integers(30, 220, (300, 300, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _pipeline(detections):
    detector = MagicMock()
    detector.detect.return_value = detections
    detector.detector_id = "yunet"
    detector.model_hash = "d" * 64
    embedder = MagicMock()
    embedder.embed.side_effect = lambda crops: np.eye(512, dtype=np.float32)[: len(crops)]
    embedder.embedding_norms = np.full(len(detections), 21.7, np.float32)
    store = MagicMock()
    store.face_point_id.side_effect = lambda h, t, b: f"id-{h}-{t}"
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
        crop_dilation=0.15,
        store_dir=None,
    )


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
