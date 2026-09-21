"""Shared inference caches must load once and refuse configuration drift."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from scalar_forensic.config import Settings
from scalar_forensic.web.pipeline import _embedders, faces_query


@pytest.fixture(autouse=True)
def clear_caches():
    _embedders._embedder_cache.clear()
    faces_query._detector_cache.clear()
    faces_query._embedder_cache.clear()
    yield
    _embedders._embedder_cache.clear()
    faces_query._detector_cache.clear()
    faces_query._embedder_cache.clear()


def test_concurrent_model_requests_load_once(monkeypatch):
    calls = []
    model = SimpleNamespace(embed_images=lambda images: images)

    def load(**kwargs):
        calls.append(kwargs)
        time.sleep(0.05)
        return model

    monkeypatch.setattr(_embedders, "load_embedder", load)
    with ThreadPoolExecutor(4) as pool:
        models = list(pool.map(lambda _: _embedders._get_embedder("dino", Settings()), range(4)))
    assert len(calls) == 1
    assert all(item is model for item in models)


def test_cached_model_refuses_preprocessing_drift(monkeypatch):
    monkeypatch.setattr(
        _embedders,
        "load_embedder",
        lambda **kw: SimpleNamespace(embed_images=lambda images: images),
    )
    monkeypatch.setenv("SFN_NORMALIZE_SIZE", "512")
    _embedders._get_embedder("dino", Settings())
    monkeypatch.setenv("SFN_NORMALIZE_SIZE", "256")
    with pytest.raises(ValueError, match="cached.*configuration"):
        _embedders._get_embedder("dino", Settings())


def test_cached_inference_calls_are_serialized(monkeypatch):
    active = 0
    peak = 0
    guard = threading.Lock()

    def embed(images):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(active, peak)
        time.sleep(0.03)
        with guard:
            active -= 1
        return images

    monkeypatch.setattr(
        _embedders, "load_embedder", lambda **kw: SimpleNamespace(embed_images=embed)
    )
    model = _embedders._get_embedder("dino", Settings())
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(model.embed_images, [[1], [2], [3], [4]])) == [[1], [2], [3], [4]]
    assert peak == 1


def test_face_detector_concurrent_cache_load_once(monkeypatch):
    calls = []

    def load(*args):
        calls.append(args)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(faces_query, "YuNetDetector", load)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: faces_query._detector(Settings()), range(4)))
    assert len(calls) == 1


def test_face_detector_resize_and_detect_are_atomic(monkeypatch):
    import numpy as np

    class MutableDetector:
        shape = None

        def detect(self, image):
            self.shape = image.shape
            time.sleep(0.02)
            assert self.shape == image.shape
            return []

    detector = MutableDetector()
    monkeypatch.setattr(faces_query, "_detector", lambda _: detector)
    monkeypatch.setattr(faces_query, "_embedder", lambda _: object())
    monkeypatch.setattr(faces_query, "pipeline_config", lambda *args: object())
    monkeypatch.setattr(faces_query, "load_for_detection", lambda data: np.zeros((int(data), 8, 3)))
    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(
                lambda size: faces_query.detect_query_faces(
                    str(size).encode(), Settings(), max_faces=10
                ),
                [10, 20, 30, 40],
            )
        )
    assert all(result.n_detected == 0 for result in results)


def test_query_validation_refuses_drift_without_loading(monkeypatch):
    monkeypatch.setattr(
        _embedders,
        "load_embedder",
        lambda **kw: SimpleNamespace(embed_images=lambda images: images),
    )
    _embedders._get_embedder("dino", Settings())
    monkeypatch.setenv("SFN_NORMALIZE_SIZE", "789")
    with pytest.raises(ValueError, match="configuration"):
        _embedders.validate_cached_embedder_settings(Settings(), ["dino"])
