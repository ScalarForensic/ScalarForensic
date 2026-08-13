import io
import os
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from scalar_forensic.faces.chips import chip_paths, review_chip_paths
from scalar_forensic.faces.indexing import (
    DEFAULT_FACE_WORKERS,
    FacePipeline,
    decode_shared,
    process_media_threaded,
)
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


# ---------------------------------------------------------------------------
# Decode reuse (in-loop face detection shares the batch loop's decode)
# ---------------------------------------------------------------------------


def test_process_image_accepts_a_predecoded_image_without_touching_data():
    # The in-loop path hands over the batch loop's native-res decode; data=None
    # proves no second decode can happen (there are no bytes to decode).
    from scalar_forensic.faces.decode import load_for_detection

    data = _img_bytes()
    arr = load_for_detection(data)
    det = _det(w=200.0)

    from_bytes = _pipeline([det]).process_image(data, image_hash="h1", image_path="/x.png")
    p = _pipeline([det])
    from_array = p.process_image(None, image_hash="h1", image_path="/x.png", img=arr)

    np.testing.assert_array_equal(p.detector.detect.call_args.args[0], arr)
    assert from_array.n_detected == from_bytes.n_detected == 1
    assert from_array.n_kept == from_bytes.n_kept == 1
    assert [pt.id for pt in from_array.points] == [pt.id for pt in from_bytes.points]
    vec_a = from_array.points[0].vector[FACE_VECTOR_NAME]
    vec_b = from_bytes.points[0].vector[FACE_VECTOR_NAME]
    assert vec_a == vec_b


def test_process_image_without_img_still_decodes_data():
    p = _pipeline([_det(w=200.0)])
    with patch("scalar_forensic.faces.indexing.load_for_detection") as load:
        load.return_value = np.zeros((300, 300, 3), np.uint8)
        p.process_image(_img_bytes(), image_hash="h1", image_path="/x.png")
    load.assert_called_once()


def _png_bytes(size=(700, 500)) -> bytes:
    rng = np.random.default_rng(7)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (*size, 3), np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_shared_derives_the_exact_embed_input_for_png():
    # The derived image must be byte-identical to what preprocess_batch would
    # produce, or a faces-on run would embed different pixels than a faces-off
    # run of the same media — a comparability break, not just a quality nit.
    from scalar_forensic.embedder import _cap_short_side, _open_rgb
    from scalar_forensic.faces.decode import load_for_detection

    data = _png_bytes()
    arr, derived = decode_shared(data, cap=331)

    np.testing.assert_array_equal(arr, load_for_detection(data))
    assert derived is not None
    reference = _cap_short_side(_open_rgb(data, 331), 331)
    assert derived.size == reference.size
    np.testing.assert_array_equal(np.asarray(derived), np.asarray(reference))


def test_decode_shared_declines_to_derive_for_jpeg():
    # JPEG's draft() decode path produces slightly different pixels than a
    # full decode + resize; the embed input must come from the normal path.
    from scalar_forensic.faces.decode import load_for_detection

    rng = np.random.default_rng(9)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (500, 700, 3), np.uint8)).save(
        buf, format="JPEG", quality=90
    )
    data = buf.getvalue()

    arr, derived = decode_shared(data, cap=331)
    assert derived is None
    np.testing.assert_array_equal(arr, load_for_detection(data))


def test_decode_shared_orientation_edge_case_falls_back_but_detects_oriented():
    # A PNG carrying an eXIf Orientation tag: load_for_detection orients it,
    # embedder._open_rgb (PNG not in _EXIF_ORIENTATION_FORMATS) does not.
    # decode_shared must serve detection the oriented array and refuse to
    # derive the embed input rather than silently change embed pixels.
    from scalar_forensic.faces.decode import load_for_detection

    rng = np.random.default_rng(11)
    img = Image.fromarray(rng.integers(0, 255, (400, 600, 3), np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 270° CW on transpose
    buf = io.BytesIO()
    img.save(buf, format="PNG", exif=exif)
    data = buf.getvalue()

    oriented = load_for_detection(data)
    if oriented.shape[:2] == (400, 600):  # Pillow build without eXIf read support
        pytest.skip("Pillow did not round-trip the PNG eXIf orientation tag")

    arr, derived = decode_shared(data, cap=331)
    assert derived is None
    np.testing.assert_array_equal(arr, oriented)


# ---------------------------------------------------------------------------
# CLI wiring: detection folded into the batch loop, residual pass for the rest
# ---------------------------------------------------------------------------


@pytest.fixture
def inloop_env(tmp_path, monkeypatch):
    """A real index() invocation (fake Qdrant/embedder, mocked FacePipeline)."""
    from tests.test_ingest_characterization import (
        FakeEmbedder,
        FakeQdrantClient,
        FakeQdrantStore,
    )

    monkeypatch.chdir(tmp_path)
    dummy_model = tmp_path / "dummy-model"
    dummy_model.touch()
    det = tmp_path / "yunet.onnx"
    det.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"
    emb.write_bytes(b"x")
    (tmp_path / "emb.onnx.manifest.json").write_text("{}")
    monkeypatch.setenv("SFN_MODEL_DINO", str(dummy_model))
    monkeypatch.setenv("SFN_BATCH_SIZE", "4")
    monkeypatch.setenv("SFN_NORMALIZE_SIZE", "336")  # must match FakeEmbedder
    monkeypatch.setenv("SFN_THUMBNAIL_DIR", "")
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_FACE_DETECTOR_MODEL", str(det))
    monkeypatch.setenv("SFN_FACE_EMBEDDER_MODEL", str(emb))
    monkeypatch.setenv("SFN_EXAMINER_ID", "ex1")
    monkeypatch.setenv("SFN_FACE_STORE_DIR", str(tmp_path / "faces"))

    images = tmp_path / "evidence"
    images.mkdir()
    rng = np.random.default_rng(3)
    for name in ("red.png", "green.png", "blue.png"):
        Image.fromarray(rng.integers(0, 255, (16, 16, 3), np.uint8)).save(images / name)
    # exact duplicate → run-duplicate: skipped by the batch loop, so its face
    # processing must come from the residual pass.
    (images / "red_copy.png").write_bytes((images / "red.png").read_bytes())

    store = FakeQdrantStore()

    def make_pipeline():
        from scalar_forensic.faces.indexing import FaceIndexResult

        pipeline = MagicMock()
        pipeline.cfg.config_hash = "cfg1"
        pipeline.store.collection_is_new.return_value = False
        pipeline.store.check_compat.return_value = []
        pipeline.store.processed_hashes.return_value = set()
        pipeline.store.stale_face_points.return_value = []
        pipeline.process_image.return_value = FaceIndexResult()
        return pipeline

    def run(pipeline):
        from scalar_forensic.cli import index

        with (
            patch("scalar_forensic.indexer.QdrantClient", lambda **kw: FakeQdrantClient(store)),
            patch(
                "scalar_forensic.cli.load_embedder",
                lambda model, use_sscd, **kw: FakeEmbedder(dim=4, name="fake-dino"),
            ),
            patch(
                "scalar_forensic.faces.indexing.FacePipeline.from_settings",
                return_value=pipeline,
            ),
        ):
            index(
                input_dir=images,
                dino=True,
                sscd=False,
                faces=True,
                report=tmp_path / "report.csv",
                allow_online=False,
                reference=False,
                ignore_config_mismatch=False,
            )

    from types import SimpleNamespace

    return SimpleNamespace(run=run, make_pipeline=make_pipeline, store=store, images=images)


def test_batch_loop_media_get_faces_from_the_shared_decode(inloop_env):
    pipeline = inloop_env.make_pipeline()
    inloop_env.run(pipeline)

    calls = pipeline.process_image.call_args_list
    assert len(calls) == 4  # 3 unique + 1 run-duplicate
    inloop = [c for c in calls if c.kwargs.get("img") is not None]
    residual = [c for c in calls if c.kwargs.get("img") is None]
    # The three dedup winners ride the batch loop's decode: a pre-decoded
    # native-res array and NO bytes — a second decode is structurally impossible.
    assert len(inloop) == 3
    for c in inloop:
        assert c.args[0] is None
        assert isinstance(c.kwargs["img"], np.ndarray)
        assert c.kwargs["img"].shape == (16, 16, 3)
    # The run-duplicate never enters the batch loop; the residual pass reads it.
    assert len(residual) == 1
    assert isinstance(residual[0].args[0], bytes)


def test_inloop_faces_keep_points_clear_marker_ordering(inloop_env):
    # The marker-last / clear-between guarantee must survive the move into the
    # batch loop: per medium, points → vector clear → marker, on every path.
    pipeline = inloop_env.make_pipeline()
    inloop_env.run(pipeline)
    order = [
        c[0] for c in pipeline.store.method_calls if c[0] in ("upsert_faces", "clear_face_vector")
    ]
    assert order == ["upsert_faces", "clear_face_vector", "upsert_faces"] * 4


def test_already_indexed_media_still_get_faces_via_the_residual_pass(inloop_env):
    # An already-embedded case is exactly what the old standalone pass existed
    # for; the batch loop skips everything, so faces must come from residual.
    first = inloop_env.make_pipeline()
    inloop_env.run(first)

    second = inloop_env.make_pipeline()
    inloop_env.run(second)
    calls = second.process_image.call_args_list
    assert len(calls) == 4
    assert all(c.kwargs.get("img") is None for c in calls)
    assert all(isinstance(c.args[0], bytes) for c in calls)


def test_face_marker_skip_set_is_honoured_on_both_paths(inloop_env):
    # processed_hashes is the idempotency source; a marked medium must be
    # processed by neither the in-loop nor the residual path.
    from scalar_forensic.embedder import hash_file

    pipeline = inloop_env.make_pipeline()
    green_sha = hash_file(inloop_env.images / "green.png")
    red_sha = hash_file(inloop_env.images / "red.png")  # covers red_copy.png too
    pipeline.store.processed_hashes.return_value = {green_sha, red_sha}
    inloop_env.run(pipeline)

    calls = pipeline.process_image.call_args_list
    assert len(calls) == 1  # only blue.png is unmarked
    assert calls[0].kwargs["image_hash"] not in {green_sha, red_sha}


# ── Residual-pass threading (efficiency audit 2026-08-13 §4, fix 1) ──────────
# The residual pass fans decode+detect+embed out over a thread pool.  YuNet is
# not thread-safe, so each worker thread must build its own detector; the
# embedder session is shared and its embed()+embedding_norms read must be
# serialised; every yielded result reaches the caller on the calling thread.


class _ThreadDetector:
    """Fake detector recording which threads call it; optional rendezvous."""

    def __init__(self, detections, barrier=None):
        self._detections = detections
        self._barrier = barrier
        self.threads: set[int] = set()
        self.n_dropped_noncanonical = 0

    def detect(self, img):
        self.threads.add(threading.get_ident())
        if self._barrier is not None:
            try:
                self._barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass
        return list(self._detections)


class _SerialCheckingEmbedder:
    """Flags overlapping embed() calls; norms are per-call instance state."""

    def __init__(self):
        self._guard = threading.Lock()
        self._active = 0
        self.overlap = False
        self.embedding_norms = np.empty(0, np.float32)

    def embed(self, crops):
        with self._guard:
            self._active += 1
            if self._active > 1:
                self.overlap = True
        time.sleep(0.005)
        self.embedding_norms = np.full(len(crops), 21.7, np.float32)
        out = np.eye(512, dtype=np.float32)[: len(crops)]
        with self._guard:
            self._active -= 1
        return out


def _media_jobs(tmp_path, n):
    jobs = []
    for i in range(n):
        p = tmp_path / f"m{i}.png"
        p.write_bytes(_img_bytes())
        jobs.append((p, f"sha-{i}", None))
    return jobs


def test_threaded_results_arrive_in_job_order_with_correct_attribution(tmp_path):
    pipeline = _pipeline([])
    pipeline.detector_factory = lambda: _ThreadDetector([_det()])
    jobs = _media_jobs(tmp_path, 6)
    jobs[0] = (jobs[0][0], jobs[0][1], {"video_hash": "vh0", "frame_timecode_ms": 42})

    out = list(process_media_threaded(pipeline, jobs, max_workers=3))

    assert [(p, sha, vmeta) for p, sha, vmeta, _ in out] == jobs
    for _p, sha, _vmeta, res in out:
        assert not isinstance(res, Exception)
        assert res.n_kept == 1
        assert res.points[0].payload["image_hash"] == sha
    assert out[0][3].points[0].payload["video_hash"] == "vh0"
    assert out[0][3].points[0].payload["frame_timecode_ms"] == 42


def test_each_detector_instance_is_used_by_exactly_one_thread(tmp_path):
    barrier = threading.Barrier(2)
    created: list[_ThreadDetector] = []

    def factory():
        d = _ThreadDetector([_det()], barrier=barrier)
        created.append(d)
        return d

    pipeline = _pipeline([])
    pipeline.detector_factory = factory
    jobs = _media_jobs(tmp_path, 8)

    list(process_media_threaded(pipeline, jobs, max_workers=4))

    # The pipeline's own detector must never run in a worker thread.
    assert pipeline.detector.detect.call_count == 0
    # The barrier forces ≥2 threads into detect() simultaneously, so a shared
    # single instance would show a thread-count > 1 here.
    assert 2 <= len(created) <= 4
    assert all(len(d.threads) == 1 for d in created)


def test_shared_embedder_calls_never_overlap(tmp_path):
    pipeline = _pipeline([])
    pipeline.embedder = _SerialCheckingEmbedder()
    pipeline.detector_factory = lambda: _ThreadDetector([_det()], barrier=threading.Barrier(2))
    jobs = _media_jobs(tmp_path, 8)

    out = list(process_media_threaded(pipeline, jobs, max_workers=4))

    assert not pipeline.embedder.overlap
    assert all(res.n_kept == 1 for *_head, res in out)
    for *_head, res in out:
        assert res.points[0].payload["embedding_norm"] == pytest.approx(21.7)


def test_unreadable_file_yields_its_exception_and_spares_the_rest(tmp_path):
    pipeline = _pipeline([])
    pipeline.detector_factory = lambda: _ThreadDetector([_det()])
    jobs = _media_jobs(tmp_path, 4)
    jobs[2] = (tmp_path / "does-not-exist.png", "sha-missing", None)

    out = list(process_media_threaded(pipeline, jobs, max_workers=2))

    assert isinstance(out[2][3], Exception)
    for i, (_p, _sha, _vmeta, res) in enumerate(out):
        if i != 2:
            assert not isinstance(res, Exception)
            assert res.n_kept == 1


def test_without_a_factory_the_pass_runs_sequentially_on_the_calling_thread(tmp_path):
    shared = _ThreadDetector([_det()])
    pipeline = _pipeline([])
    pipeline.detector = shared
    pipeline.detector_factory = None
    jobs = _media_jobs(tmp_path, 3)

    out = list(process_media_threaded(pipeline, jobs))

    assert shared.threads == {threading.get_ident()}
    assert all(res.n_kept == 1 for *_head, res in out)


def test_default_worker_count_is_min_8_cpu():
    assert DEFAULT_FACE_WORKERS == min(8, os.cpu_count() or 1)
