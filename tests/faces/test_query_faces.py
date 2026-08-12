"""Stage 1 (Phase 1b): session-scoped face detection on the uploaded query image.

The load-bearing assertion in this file is the first test: a query face that
fails the embedding gate carries ``vector=None``.  That is the same structural
exclusion guarantee the index path has (vectorless points), carried into the
query path — not a payload filter, not a flag consulted at search time.
"""

from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from scalar_forensic.faces.types import FaceDetection
from scalar_forensic.web.app import app
from scalar_forensic.web.pipeline.faces_query import detect_query_faces

client = TestClient(app)


def _det(size: float, conf: float) -> FaceDetection:
    return FaceDetection(
        bbox=(10.0, 10.0, size, size),
        landmarks=np.array(
            [[20.0, 25.0], [40.0, 25.0], [30.0, 35.0], [22.0, 45.0], [38.0, 45.0]],
            dtype=np.float32,
        ),
        confidence=conf,
        detect_scale=1.0,
    )


def _settings(tmp_path, **over):
    s = MagicMock()
    s.faces_enabled = True
    s.face_startup_error.return_value = None
    s.face_collection = "case1_faces"
    s.collection = "case1"
    s.face_store_dir = tmp_path
    s.face_detector_model = tmp_path / "yunet.onnx"
    s.face_embedder_model = tmp_path / "sface.onnx"
    s.face_detect_max_size = 1600
    s.face_min_conf = 0.8
    s.face_min_size = 64
    s.face_max_pose = 0.35
    s.face_min_sharpness = 25.0
    s.face_max_clipped = 0.6
    s.face_review_min_conf = 0.6
    s.face_review_min_size = 36
    s.face_crop_dilation = 0.25
    s.face_query_max_faces = 25
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _mock_models(n_vectors: int = 1):
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[1.0] + [0.0] * 127] * n_vectors, dtype=np.float32)
    embedder.model_hash = "e" * 64
    embedder.manifest_hash = "m" * 64
    embedder.manifest.embedding_dim = 128
    embedder.normalization_id = "affine-0.0-1.0"
    detector = MagicMock()
    detector.detector_id = "yunet"
    detector.model_hash = "d" * 64
    detector._score_threshold = 0.5
    return detector, embedder


def test_review_only_query_face_gets_no_vector(tmp_path):
    """The Phase 1 exclusion guarantee, carried into the query path."""
    # Textured noise so the sharpness gate passes on the embeddable face.
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, size=(400, 400, 3), dtype=np.uint8)
    detector, embedder = _mock_models()
    detector.detect.return_value = [_det(120.0, 0.95), _det(40.0, 0.9)]

    with (
        patch("scalar_forensic.web.pipeline.faces_query.load_for_detection", return_value=img),
        patch("scalar_forensic.web.pipeline.faces_query._detector", return_value=detector),
        patch("scalar_forensic.web.pipeline.faces_query._embedder", return_value=embedder),
    ):
        result = detect_query_faces(b"jpegbytes", _settings(tmp_path), max_faces=25)

    assert result.n_detected == 2
    embedded = [f for f in result.faces if f.embedding_status == "embedded"]
    review = [f for f in result.faces if f.embedding_status == "review_only"]
    assert len(embedded) == 1 and len(review) == 1
    assert embedded[0].vector is not None
    assert review[0].vector is None
    assert review[0].embedding_exclusion_reason == "size"
    # Unmeasured subscores are null, never 0.0 (spec §6.2)
    assert review[0].quality["sharpness"] is None
    assert review[0].quality["exposure"] is None
    assert result.n_searchable == 1
    assert result.n_review_only == 1


def test_query_faces_are_never_persisted(tmp_path):
    """No Qdrant client, no file under the chip store."""
    rng = np.random.default_rng(1)
    img = rng.integers(60, 200, size=(400, 400, 3), dtype=np.uint8)
    detector, embedder = _mock_models()
    detector.detect.return_value = [_det(120.0, 0.95)]

    with (
        patch("scalar_forensic.web.pipeline.faces_query.load_for_detection", return_value=img),
        patch("scalar_forensic.web.pipeline.faces_query._detector", return_value=detector),
        patch("scalar_forensic.web.pipeline.faces_query._embedder", return_value=embedder),
    ):
        result = detect_query_faces(b"jpegbytes", _settings(tmp_path), max_faces=25)

    assert result.faces[0].review_jpeg is not None  # held in memory
    assert list(tmp_path.rglob("*")) == []  # nothing written
    source = open("src/scalar_forensic/web/pipeline/faces_query.py", encoding="utf-8").read()
    assert "QdrantClient" not in source
    assert "write_review_chips" not in source
    assert "write_aligned_chips" not in source


def test_detection_below_the_review_gate_is_rejected_not_retained(tmp_path):
    rng = np.random.default_rng(2)
    img = rng.integers(60, 200, size=(400, 400, 3), dtype=np.uint8)
    detector, embedder = _mock_models()
    detector.detect.return_value = [_det(120.0, 0.4)]  # below review_min_conf 0.6

    with (
        patch("scalar_forensic.web.pipeline.faces_query.load_for_detection", return_value=img),
        patch("scalar_forensic.web.pipeline.faces_query._detector", return_value=detector),
        patch("scalar_forensic.web.pipeline.faces_query._embedder", return_value=embedder),
    ):
        result = detect_query_faces(b"j", _settings(tmp_path), max_faces=25)

    assert result.faces == []
    assert result.rejected == {"confidence": 1}
    assert result.n_detected == 1


def test_detection_is_truncated_at_the_configured_cap(tmp_path):
    rng = np.random.default_rng(3)
    img = rng.integers(60, 200, size=(400, 400, 3), dtype=np.uint8)
    detector, embedder = _mock_models(n_vectors=2)
    detector.detect.return_value = [_det(120.0, 0.95)] * 5

    with (
        patch("scalar_forensic.web.pipeline.faces_query.load_for_detection", return_value=img),
        patch("scalar_forensic.web.pipeline.faces_query._detector", return_value=detector),
        patch("scalar_forensic.web.pipeline.faces_query._embedder", return_value=embedder),
    ):
        result = detect_query_faces(b"j", _settings(tmp_path), max_faces=2)

    assert result.truncated is True
    assert result.n_detected == 5
    assert len(result.faces) == 2


def test_query_faces_endpoint_404s_on_unknown_session():
    with patch("scalar_forensic.web.routes.faces.get_session", return_value=None):
        resp = client.post("/api/faces/query-faces", data={"session_id": "nope", "file_id": "x"})
    assert resp.status_code == 404


def _query_face(**over):
    from scalar_forensic.web.session import QueryFace

    base = dict(
        index=0,
        bbox=(1.0, 2.0, 3.0, 4.0),
        landmarks=[[1.0, 2.0]] * 5,
        det_conf=0.9,
        detect_scale=1.0,
        quality={
            "confidence": 0.9,
            "size": 120.0,
            "pose": 0.1,
            "sharpness": 40.0,
            "exposure": 0.0,
        },
        embedding_status="embedded",
        embedding_exclusion_reason=None,
        vector=[1.0] + [0.0] * 127,
        review_jpeg=b"\xff\xd8",
    )
    base.update(over)
    return QueryFace(**base)


def test_query_faces_endpoint_never_returns_a_vector(tmp_path):
    face = _query_face()
    entry = MagicMock(is_video=False, temp_path=tmp_path / "q.jpg", query_faces=[face])
    (tmp_path / "q.jpg").write_bytes(b"jpegbytes")
    session = MagicMock(files=[entry])

    result = MagicMock(
        faces=[face],
        n_detected=1,
        n_searchable=1,
        n_review_only=0,
        rejected={},
        truncated=False,
        cfg=MagicMock(config_hash="9f2c"),
    )
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=session),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.detect_query_faces", return_value=result),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    ):
        body = client.post(
            "/api/faces/query-faces", data={"session_id": "s", "file_id": "f"}
        ).json()

    assert body["faces"][0]["searchable"] is True
    assert "vector" not in body["faces"][0]
    assert "review_jpeg" not in body["faces"][0]
    assert body["faces"][0]["chip_url"].endswith("/query-chip/s/f/0")
    assert body["n_detected"] == 1
    assert body["truncated"] is False


def test_query_faces_endpoint_marks_review_only_as_not_searchable(tmp_path):
    face = _query_face(
        embedding_status="review_only",
        embedding_exclusion_reason="size",
        vector=None,
        quality={
            "confidence": 0.7,
            "size": 40.0,
            "pose": 0.1,
            "sharpness": None,
            "exposure": None,
        },
    )
    entry = MagicMock(is_video=False, temp_path=tmp_path / "q.jpg", query_faces=[face])
    (tmp_path / "q.jpg").write_bytes(b"jpegbytes")
    result = MagicMock(
        faces=[face],
        n_detected=1,
        n_searchable=0,
        n_review_only=1,
        rejected={},
        truncated=False,
        cfg=MagicMock(config_hash="9f2c"),
    )
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.detect_query_faces", return_value=result),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    ):
        body = client.post(
            "/api/faces/query-faces", data={"session_id": "s", "file_id": "f"}
        ).json()

    assert body["faces"][0]["searchable"] is False
    assert body["faces"][0]["embedding_status"] == "review_only"
    assert body["faces"][0]["quality"]["sharpness"] is None
    assert body["n_searchable"] == 0


def test_query_chip_serves_the_in_memory_crop_with_no_store(tmp_path):
    face = _query_face(review_jpeg=b"\xff\xd8jpeg")
    entry = MagicMock(query_faces=[face])
    with (
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
    ):
        resp = client.get("/api/faces/query-chip/s/f/0")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.headers["cache-control"] == "no-store"
    assert resp.content == b"\xff\xd8jpeg"


def test_query_chip_404s_on_unknown_index(tmp_path):
    entry = MagicMock(query_faces=[_query_face()])
    with (
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
    ):
        resp = client.get("/api/faces/query-chip/s/f/7")
    assert resp.status_code == 404
