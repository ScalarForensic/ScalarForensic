"""Characterization tests for the ingest write path (audit T3).

These tests lock the *current* behavior of the code that writes evidence into
Qdrant — dedup decisions, point-ID derivation, payload construction, the
update-vs-insert split that protects the other model's named vector, and the
CSV report — before the planned decomposition of ``cli.index()`` (audit T7).
They are characterization tests: if one fails after a refactor, the refactor
changed observable ingest behavior.

All Qdrant traffic goes to an in-memory fake client; embedding is faked.
File scanning, hashing, preprocessing, and CSV writing are real.
"""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image
from qdrant_client.models import (
    Distance,
    FieldCondition,
    HasVectorCondition,
    VectorParams,
)

from scalar_forensic.cli import (
    _S_FAIL_EMB,
    _S_INDEXED,
    _S_SKIP_DUP,
    _S_SKIP_FRAME_DUP,
    _S_SKIP_IDX,
    _dedup_by_hash,
    _FileRecord,
    _video_status,
    _write_csv,
)
from scalar_forensic.embedder import hash_file_both
from scalar_forensic.indexer import Indexer, qdrant_scroll_all

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StoredPoint:
    def __init__(self, vectors: dict, payload: dict) -> None:
        self.vectors = dict(vectors)
        self.payload = dict(payload)


class _FakeCollection:
    def __init__(self, vectors_config) -> None:
        self.vectors_config = vectors_config  # dict[str, VectorParams] | VectorParams
        self.points: dict[str, _StoredPoint] = {}
        self.payload_indexes: dict[str, object] = {}


class FakeQdrantStore:
    """Shared in-memory backend — every FakeQdrantClient sees the same state,
    matching how two Indexer instances talk to one real server."""

    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}
        self.calls: list[tuple] = []  # (method, collection) in invocation order


def _matches(point: _StoredPoint, flt) -> bool:
    if flt is None:
        return True
    for cond in flt.must or []:
        if isinstance(cond, HasVectorCondition):
            if cond.has_vector not in point.vectors:
                return False
        elif isinstance(cond, FieldCondition):
            if point.payload.get(cond.key) != cond.match.value:
                return False
        else:  # pragma: no cover - guard against untested filter kinds
            raise NotImplementedError(f"filter condition not faked: {cond!r}")
    return True


class FakeQdrantClient:
    def __init__(self, store: FakeQdrantStore) -> None:
        self._store = store

    # -- collection management ------------------------------------------------
    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in self._store.collections]
        )

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self._store.calls.append(("create_collection", collection_name))
        self._store.collections[collection_name] = _FakeCollection(vectors_config)

    def get_collection(self, collection_name: str):
        coll = self._store.collections[collection_name]
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=coll.vectors_config)),
            payload_schema=dict(coll.payload_indexes),
        )

    def create_payload_index(self, collection_name: str, field_name: str, field_schema) -> None:
        self._store.calls.append(("create_payload_index", field_name))
        self._store.collections[collection_name].payload_indexes[field_name] = field_schema

    # -- point I/O -------------------------------------------------------------
    def scroll(
        self,
        collection_name: str,
        scroll_filter=None,
        limit: int = 10,
        with_payload=None,
        with_vectors: bool = False,
        offset=None,
    ):
        coll = self._store.collections[collection_name]
        hits = [
            SimpleNamespace(id=pid, payload=dict(p.payload))
            for pid, p in coll.points.items()
            if _matches(p, scroll_filter)
        ]
        return hits, None  # single page

    def retrieve(self, collection_name: str, ids, with_vectors=False, with_payload=False):
        coll = self._store.collections[collection_name]
        return [SimpleNamespace(id=pid) for pid in ids if pid in coll.points]

    def upsert(self, collection_name: str, points) -> None:
        self._store.calls.append(("upsert", collection_name, [p.id for p in points]))
        coll = self._store.collections[collection_name]
        for p in points:
            coll.points[p.id] = _StoredPoint(vectors=p.vector, payload=p.payload or {})

    def update_vectors(self, collection_name: str, points) -> None:
        self._store.calls.append(("update_vectors", collection_name, [p.id for p in points]))
        coll = self._store.collections[collection_name]
        for pv in points:
            coll.points[pv.id].vectors.update(pv.vector)

    def set_payload(self, collection_name: str, payload: dict, points) -> None:
        self._store.calls.append(("set_payload", collection_name))
        coll = self._store.collections[collection_name]
        if isinstance(points, list):
            targets = [coll.points[pid] for pid in points]
        else:  # Filter
            targets = [p for p in coll.points.values() if _matches(p, points)]
        for t in targets:
            t.payload.update(payload)


class FakeEmbedder:
    """Duck-types the embedder surface cli.index() touches."""

    def __init__(self, dim: int, name: str) -> None:
        self.embedding_dim = dim
        self.model_name = name
        self.model_hash = f"hash-{name}"
        self.normalize_size = 336
        self.inference_dtype = "float32"
        self.device = "cpu"
        self.compiled = False
        self._counter = 0

    def normalize_batch_bytes(self, pil_images):
        return list(pil_images)

    def embed_images(self, images):
        out = []
        for _ in images:
            self._counter += 1
            out.append([float(self._counter)] + [0.0] * (self.embedding_dim - 1))
        return out


@pytest.fixture
def store() -> FakeQdrantStore:
    return FakeQdrantStore()


@pytest.fixture
def fake_client_cls(store):
    """A QdrantClient substitute whose instances all share *store*."""

    def _factory(url=None, api_key=None, **kwargs):
        return FakeQdrantClient(store)

    return _factory


def make_indexer(fake_client_cls, store, vector_name="dino", dim=4, **kwargs) -> Indexer:
    with patch("scalar_forensic.indexer.QdrantClient", fake_client_cls):
        return Indexer(
            url="http://fake:6333",
            collection="case",
            vector_name=vector_name,
            embedding_dim=dim,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# qdrant_scroll_all
# ---------------------------------------------------------------------------


class _PagingClient:
    """Stub returning a fixed page sequence to lock pagination termination."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = 0

    def scroll(self, **kwargs):
        self.calls += 1
        return self._pages.pop(0)


def test_scroll_all_paginates_until_no_offset():
    a, b, c = (SimpleNamespace(id=i) for i in "abc")
    client = _PagingClient([([a, b], "next"), ([c], None)])
    got = list(qdrant_scroll_all(client, "case", scroll_filter=None, limit=2, with_payload=True))
    assert got == [a, b, c]
    assert client.calls == 2


def test_scroll_all_stops_on_empty_page():
    client = _PagingClient([([], "dangling-offset")])
    assert (
        list(qdrant_scroll_all(client, "case", scroll_filter=None, limit=2, with_payload=True))
        == []
    )
    assert client.calls == 1


# ---------------------------------------------------------------------------
# Indexer._ensure_collection
# ---------------------------------------------------------------------------

_ALL_PAYLOAD_INDEXES = {
    "image_hash",
    "image_path",
    "image_hash_md5",
    "video_hash",
    "video_path",
    "frame_timecode_ms",
    "is_video_frame",
    "is_video",
    "is_reference",
}


def test_new_collection_created_with_full_config_and_indexes(fake_client_cls, store):
    cfg = {
        "dino": VectorParams(size=4, distance=Distance.COSINE),
        "sscd": VectorParams(size=2, distance=Distance.COSINE),
    }
    make_indexer(fake_client_cls, store, initial_vectors_config=cfg)
    coll = store.collections["case"]
    assert coll.vectors_config is cfg  # both modalities registered in one shot
    assert set(coll.payload_indexes) == _ALL_PAYLOAD_INDEXES


def test_existing_collection_indexes_only_created_when_missing(fake_client_cls, store):
    make_indexer(fake_client_cls, store)
    n_creates = sum(1 for c in store.calls if c[0] == "create_payload_index")
    assert n_creates == len(_ALL_PAYLOAD_INDEXES)
    # Second Indexer against the same collection: schema is complete, no new calls.
    make_indexer(fake_client_cls, store)
    assert sum(1 for c in store.calls if c[0] == "create_payload_index") == n_creates


def test_legacy_single_vector_collection_is_rejected(fake_client_cls, store):
    store.collections["case"] = _FakeCollection(VectorParams(size=4, distance=Distance.COSINE))
    with pytest.raises(ValueError, match="legacy single-vector"):
        make_indexer(fake_client_cls, store)


def test_missing_named_vector_demands_drop_and_reindex(fake_client_cls, store):
    store.collections["case"] = _FakeCollection(
        {"dino": VectorParams(size=4, distance=Distance.COSINE)}
    )
    with pytest.raises(ValueError, match="Drop the collection"):
        make_indexer(fake_client_cls, store, vector_name="sscd", dim=2)


def test_dimension_mismatch_is_rejected(fake_client_cls, store):
    store.collections["case"] = _FakeCollection(
        {"dino": VectorParams(size=4, distance=Distance.COSINE)}
    )
    with pytest.raises(ValueError, match="dim=4"):
        make_indexer(fake_client_cls, store, dim=8)


# ---------------------------------------------------------------------------
# Indexer.upsert_batch — payloads, IDs, and the update-vs-insert split
# ---------------------------------------------------------------------------

_META = {
    "model_name": "fake-dino",
    "model_hash": "hash-fake-dino",
    "embedding_dim": 4,
    "normalize_size": 336,
    "inference_dtype": "float32",
    "library_versions": {"torch": "0"},
}


def test_upsert_batch_new_points_payload_and_deterministic_ids(fake_client_cls, store, tmp_path):
    idx = make_indexer(fake_client_cls, store)
    img = tmp_path / "a.png"
    img.touch()
    idx.upsert_batch(
        [img],
        ["sha-a"],
        [[1.0, 0.0, 0.0, 0.0]],
        _META,
        exif_payloads={img: {"exif_gps": True}},
        image_hashes_md5=["md5-a"],
    )
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, "sha-a"))
    point = store.collections["case"].points[pid]
    assert point.vectors == {"dino": [1.0, 0.0, 0.0, 0.0]}
    assert point.payload["image_hash"] == "sha-a"
    assert point.payload["image_hash_md5"] == "md5-a"
    assert point.payload["image_path"] == str(img.resolve())
    assert point.payload["exif_gps"] is True
    # provenance is prefixed with the vector name
    assert point.payload["dino_model_name"] == "fake-dino"
    assert point.payload["dino_model_hash"] == "hash-fake-dino"
    assert point.payload["dino_embedding_dim"] == 4
    assert "dino_indexed_at" in point.payload
    assert point.payload["library_versions"] == {"torch": "0"}
    # not a reference run, so no tag
    assert "is_reference" not in point.payload


def test_upsert_batch_reference_run_tags_points(fake_client_cls, store, tmp_path):
    idx = make_indexer(fake_client_cls, store, is_reference=True)
    img = tmp_path / "a.png"
    img.touch()
    idx.upsert_batch([img], ["sha-a"], [[1.0, 0.0, 0.0, 0.0]], _META)
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, "sha-a"))
    assert store.collections["case"].points[pid].payload["is_reference"] is True


def test_upsert_batch_sscd_n_crops_recorded_only_when_present(fake_client_cls, store, tmp_path):
    idx = make_indexer(fake_client_cls, store, vector_name="sscd", dim=2)
    img = tmp_path / "a.png"
    img.touch()
    idx.upsert_batch([img], ["sha-a"], [[1.0, 0.0]], {**_META, "sscd_n_crops": 3})
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, "sha-a"))
    assert store.collections["case"].points[pid].payload["sscd_n_crops"] == 3


def test_upsert_batch_existing_point_gets_vector_update_not_replacement(
    fake_client_cls, store, tmp_path
):
    """The core guarantee: indexing the second modality must not wipe the
    first modality's vector or the core payload."""
    store.collections["case"] = _FakeCollection(
        {
            "dino": VectorParams(size=4, distance=Distance.COSINE),
            "sscd": VectorParams(size=2, distance=Distance.COSINE),
        }
    )
    img = tmp_path / "a.png"
    img.touch()

    dino = make_indexer(fake_client_cls, store)
    dino.upsert_batch([img], ["sha-a"], [[1.0, 0.0, 0.0, 0.0]], _META)

    sscd = make_indexer(fake_client_cls, store, vector_name="sscd", dim=2)
    sscd_meta = {**_META, "model_name": "fake-sscd", "model_hash": "hash-fake-sscd"}
    sscd.upsert_batch([img], ["sha-a"], [[9.0, 9.0]], sscd_meta)

    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, "sha-a"))
    point = store.collections["case"].points[pid]
    # both vectors present — dino survived the sscd pass
    assert point.vectors == {"dino": [1.0, 0.0, 0.0, 0.0], "sscd": [9.0, 9.0]}
    # provenance for both models side by side
    assert point.payload["dino_model_name"] == "fake-dino"
    assert point.payload["sscd_model_name"] == "fake-sscd"
    # the second pass went through update_vectors + set_payload, not upsert
    methods = [c[0] for c in store.calls if c[0] in ("upsert", "update_vectors")]
    assert methods == ["upsert", "update_vectors"]


def test_upsert_batch_mixed_batch_splits_new_and_existing(fake_client_cls, store, tmp_path):
    idx = make_indexer(fake_client_cls, store)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.touch()
    b.touch()
    idx.upsert_batch([a], ["sha-a"], [[1.0, 0.0, 0.0, 0.0]], _META)
    idx.upsert_batch(
        [a, b], ["sha-a", "sha-b"], [[2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]], _META
    )
    pid_a = str(uuid.uuid5(uuid.NAMESPACE_URL, "sha-a"))
    pid_b = str(uuid.uuid5(uuid.NAMESPACE_URL, "sha-b"))
    upserts = [c for c in store.calls if c[0] == "upsert"]
    updates = [c for c in store.calls if c[0] == "update_vectors"]
    assert upserts[-1][2] == [pid_b]  # only the new point is inserted
    assert updates[-1][2] == [pid_a]  # the existing one is updated in place
    assert store.collections["case"].points[pid_a].vectors["dino"] == [2.0, 0.0, 0.0, 0.0]


def test_upsert_batch_video_frame_ids_derive_from_video_hash_and_timecode(
    fake_client_cls, store, tmp_path
):
    """Two videos with an identical frame must produce two distinct points."""
    idx = make_indexer(fake_client_cls, store)
    f1, f2 = tmp_path / "f1.jpg", tmp_path / "f2.jpg"
    f1.touch()
    f2.touch()
    vmeta = {
        "video_path": "/case/v1.mp4",
        "frame_timecode_ms": 40,
        "frame_index": 1,
        "extraction_fps": 1.0,
        "max_frames_cap": 100,
        "pyav_version": "12.0",
    }
    idx.upsert_batch(
        [f1, f2],
        ["same-sha", "same-sha"],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        _META,
        video_metadata=[
            {**vmeta, "video_hash": "vh-1"},
            {**vmeta, "video_hash": "vh-2", "video_path": "/case/v2.mp4"},
        ],
    )
    pid_1 = str(uuid.uuid5(uuid.NAMESPACE_URL, "vh-1:40"))
    pid_2 = str(uuid.uuid5(uuid.NAMESPACE_URL, "vh-2:40"))
    points = store.collections["case"].points
    assert set(points) == {pid_1, pid_2}
    p = points[pid_1]
    assert p.payload["is_video_frame"] is True
    assert p.payload["video_hash"] == "vh-1"
    assert p.payload["frame_timecode_ms"] == 40
    assert p.payload["frame_index"] == 1
    assert p.payload["pyav_version"] == "12.0"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"image_hashes": ["h"]}, "Batch length mismatch"),
        ({"image_hashes_md5": ["m1"]}, "MD5 hash list length mismatch"),
        ({"video_metadata": [None]}, "video_metadata length mismatch"),
    ],
)
def test_upsert_batch_length_mismatches_raise(fake_client_cls, store, tmp_path, kwargs, match):
    idx = make_indexer(fake_client_cls, store)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    base = dict(
        image_paths=[a, b],
        image_hashes=["ha", "hb"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        shared_metadata=_META,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        idx.upsert_batch(**base)


# ---------------------------------------------------------------------------
# Indexer — dedup lookups and video bookkeeping
# ---------------------------------------------------------------------------


def _put(store, pid, vectors, payload):
    store.collections["case"].points[pid] = _StoredPoint(vectors, payload)


def test_get_all_indexed_hashes_and_paths_only_count_own_vector(fake_client_cls, store):
    idx = make_indexer(fake_client_cls, store)
    _put(store, "p1", {"dino": [1.0]}, {"image_hash": "h1", "image_path": "/a"})
    _put(store, "p2", {"sscd": [1.0]}, {"image_hash": "h2", "image_path": "/b"})
    _put(store, "p3", {}, {"image_hash": "h3", "image_path": "/c"})  # vectorless
    assert idx.get_all_indexed_hashes() == {"h1"}
    assert idx.get_all_indexed_paths() == {"/a"}


def test_get_all_video_info_reports_completeness(fake_client_cls, store):
    idx = make_indexer(fake_client_cls, store)
    frame = {"is_video_frame": True, "extraction_fps": 1.0, "max_frames_cap": 100}
    _put(store, "f1", {"dino": [1.0]}, {**frame, "video_hash": "vh-done", "video_frames_total": 2})
    _put(store, "f2", {"dino": [1.0]}, {**frame, "video_hash": "vh-partial"})
    info = idx.get_all_video_info()
    assert info["vh-done"]["complete"] is True
    assert info["vh-partial"]["complete"] is False
    assert info["vh-done"]["extraction_fps"] == 1.0


def test_mark_video_complete_stamps_only_matching_frames(fake_client_cls, store):
    idx = make_indexer(fake_client_cls, store)
    _put(store, "f1", {"dino": [1.0]}, {"is_video_frame": True, "video_hash": "vh-1"})
    _put(store, "f2", {"dino": [1.0]}, {"is_video_frame": True, "video_hash": "vh-2"})
    _put(store, "i1", {"dino": [1.0]}, {"image_hash": "h1"})
    idx.mark_video_complete("vh-1", 7)
    pts = store.collections["case"].points
    assert pts["f1"].payload["video_frames_total"] == 7
    assert "video_frames_total" not in pts["f2"].payload
    assert "video_frames_total" not in pts["i1"].payload


def test_upsert_video_records_are_vectorless_anchors_with_stable_ids(fake_client_cls, store):
    idx = make_indexer(fake_client_cls, store)
    idx.upsert_video_records(
        [
            {
                "video_hash": "vh-1",
                "video_path": "/case/v1.mp4",
                "total_frames": 42,
                "extraction_fps": 1.0,
                "max_frames_cap": 100,
                "pyav_version": "12.0",
            }
        ]
    )
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, "video:vh-1"))
    point = store.collections["case"].points[pid]
    assert point.vectors == {}  # payload-only anchor — never reachable by search
    assert point.payload["is_video"] is True
    assert point.payload["total_frames"] == 42
    assert "indexed_at" in point.payload


def test_upsert_video_records_empty_list_is_a_noop(fake_client_cls, store):
    idx = make_indexer(fake_client_cls, store)
    before = len(store.calls)
    idx.upsert_video_records([])
    assert len(store.calls) == before


# ---------------------------------------------------------------------------
# cli._dedup_by_hash — winner election and skip classification
# ---------------------------------------------------------------------------


def _dedup_fixture(tmp_path, n_specs=1):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    paths = [a, b, c]
    hashes = {a: "h1", b: "h1", c: "h2"}  # b duplicates a within the run
    records = {p: _FileRecord(path=p) for p in paths}
    skipped = [0] * n_specs
    return paths, hashes, records, skipped


def test_dedup_elects_first_path_per_hash_and_marks_run_duplicates(tmp_path):
    paths, hashes, records, skipped = _dedup_fixture(tmp_path)
    needs, any_needs, n_run_dups, n_all_indexed = _dedup_by_hash(
        paths, hashes, records, [set()], [set()], 1, skipped
    )
    a, b, c = paths
    assert needs[0] == {a, c}
    assert any_needs == {a, c}
    assert (n_run_dups, n_all_indexed) == (1, 0)
    assert records[b].status == _S_SKIP_DUP
    assert records[b].reason == "duplicate in run (same SHA-256)"
    assert records[a].status == "pending"  # winners stay pending until embedded
    assert skipped == [1]


def test_dedup_skips_already_indexed_by_hash(tmp_path):
    paths, hashes, records, skipped = _dedup_fixture(tmp_path)
    needs, any_needs, n_run_dups, n_all_indexed = _dedup_by_hash(
        paths, hashes, records, [{"h2"}], [set()], 1, skipped
    )
    a, b, c = paths
    assert needs[0] == {a}
    assert (n_run_dups, n_all_indexed) == (1, 1)
    assert records[c].status == _S_SKIP_IDX
    assert records[c].reason == "already indexed in Qdrant"
    assert skipped == [2]  # run-dup + already-indexed both count as skips


def test_dedup_skips_already_indexed_by_resolved_path(tmp_path):
    paths, hashes, records, skipped = _dedup_fixture(tmp_path)
    c = paths[2]
    _dedup_by_hash(paths, hashes, records, [set()], [{str(c.resolve())}], 1, skipped)
    assert records[c].status == _S_SKIP_IDX


def test_dedup_per_spec_needs_differ_but_skip_counts_are_shared(tmp_path):
    """Characterizes current behavior: a file needed by only one spec is NOT
    counted as skipped for the other — n_all_indexed uses the union."""
    paths, hashes, records, skipped = _dedup_fixture(tmp_path, n_specs=2)
    needs, any_needs, n_run_dups, n_all_indexed = _dedup_by_hash(
        paths, hashes, records, [{"h2"}, set()], [set(), set()], 2, skipped
    )
    a, b, c = paths
    assert needs[0] == {a}
    assert needs[1] == {a, c}
    assert any_needs == {a, c}
    assert n_all_indexed == 0  # c is still needed by spec 1 → not "all indexed"
    assert records[c].status == "pending"
    assert skipped == [1, 1]


# ---------------------------------------------------------------------------
# cli._video_status — three-way split of a video's frame outcomes
# ---------------------------------------------------------------------------


def test_video_status_indexed_when_any_frame_produced_a_vector():
    assert _video_status(10, 3, 2, 5) == (_S_INDEXED, "10 frames extracted")


def test_video_status_already_indexed_when_every_frame_was_in_qdrant():
    status, reason = _video_status(10, 0, 10, 0)
    assert status == _S_SKIP_IDX
    assert reason == "all 10 extracted frames already indexed"


def test_video_status_in_run_duplicate_is_not_a_failure():
    """The campaign case: every frame was byte-identical to a frame of another
    video in the same run, so nothing was embedded — that is a duplicate, not a
    failed embedding."""
    status, reason = _video_status(10, 0, 0, 10)
    assert status == _S_SKIP_FRAME_DUP
    assert status != _S_FAIL_EMB
    assert reason == (
        "all 10 extracted frames were duplicates of frames from other media "
        "in this run (in-run duplicates)"
    )


def test_video_status_mixed_dup_and_already_indexed_names_both_counts():
    status, reason = _video_status(10, 0, 4, 6)
    assert status == _S_SKIP_FRAME_DUP
    assert "6 in-run duplicate(s), 4 already indexed" in reason


def test_video_status_still_fails_when_frames_are_unaccounted_for():
    status, reason = _video_status(10, 0, 2, 3)
    assert status == _S_FAIL_EMB
    assert reason == "10 frames extracted but no new vectors were indexed"


# ---------------------------------------------------------------------------
# cli._write_csv
# ---------------------------------------------------------------------------


def test_write_csv_rows_sorted_with_processed_flag(tmp_path):
    r1 = _FileRecord(path=tmp_path / "b.png", status=_S_INDEXED, md5="m1", sha256="s1")
    r2 = _FileRecord(
        path=tmp_path / "a.png",
        status=_S_SKIP_DUP,
        reason="duplicate in run (same SHA-256)",
        md5="m2",
        sha256="s2",
        is_frame=True,
    )
    out = tmp_path / "reports" / "report.csv"  # parent dir is created on demand
    _write_csv({r1.path: r1, r2.path: r2}, out)
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["path", "processed", "reason", "md5", "sha256", "is_video_frame"]
    assert rows[1] == [
        str(tmp_path / "a.png"),
        "no",
        "duplicate in run (same SHA-256)",
        "m2",
        "s2",
        "True",
    ]
    assert rows[2] == [str(tmp_path / "b.png"), "yes", "", "m1", "s1", "False"]


# ---------------------------------------------------------------------------
# End-to-end: cli.index() with fake embedders against the fake store
# ---------------------------------------------------------------------------


@pytest.fixture
def ingest_env(tmp_path, monkeypatch, fake_client_cls):
    """Isolated cwd + env for a real index() invocation with fakes."""
    for key in list(__import__("os").environ):
        if key.startswith("SFN_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    dummy_model = tmp_path / "dummy-model"
    dummy_model.touch()
    monkeypatch.setenv("SFN_MODEL_DINO", str(dummy_model))  # satisfies the offline check
    monkeypatch.setenv("SFN_BATCH_SIZE", "4")  # skip auto-calibration
    # Must match FakeEmbedder.normalize_size or the comparability safeguard
    # (correctly) aborts the second run against the same collection.
    monkeypatch.setenv("SFN_NORMALIZE_SIZE", "336")
    monkeypatch.setenv("SFN_THUMBNAIL_DIR", "")  # disable thumbnails

    images = tmp_path / "evidence"
    images.mkdir()
    colors = {"red.png": (255, 0, 0), "green.png": (0, 255, 0), "blue.png": (0, 0, 255)}
    for name, color in colors.items():
        Image.new("RGB", (16, 16), color).save(images / name)
    # exact duplicate of red.png under another name → run-duplicate
    (images / "red_copy.png").write_bytes((images / "red.png").read_bytes())

    embedders = {}

    def fake_load_embedder(model, use_sscd, **kwargs):
        name = "fake-sscd" if use_sscd else "fake-dino"
        embedders[name] = FakeEmbedder(dim=2 if use_sscd else 4, name=name)
        return embedders[name]

    with (
        patch("scalar_forensic.indexer.QdrantClient", fake_client_cls),
        patch("scalar_forensic.cli.load_embedder", fake_load_embedder),
    ):
        yield SimpleNamespace(images=images, tmp=tmp_path, embedders=embedders)


def _run_index(images: Path, csv_path: Path, *, dino=True, sscd=False):
    from scalar_forensic.cli import index

    index(
        input_dir=images,
        dino=dino,
        sscd=sscd,
        faces=False,
        report=csv_path,
        allow_online=False,
        reference=False,
        ignore_config_mismatch=False,
    )


def test_index_end_to_end_points_and_report(ingest_env, store):
    csv_path = ingest_env.tmp / "report.csv"
    _run_index(ingest_env.images, csv_path)

    points = store.collections["sfn"].points
    assert len(points) == 3  # red, green, blue — red_copy deduped

    by_hash = {p.payload["image_hash"]: p for p in points.values()}
    red_sha, red_md5 = hash_file_both(ingest_env.images / "red.png")
    assert red_sha in by_hash
    red_point = by_hash[red_sha]
    # point ID is uuid5 of the sha256
    assert str(uuid.uuid5(uuid.NAMESPACE_URL, red_sha)) in points
    assert red_point.payload["image_hash_md5"] == red_md5
    # the dedup winner is the first path encountered for the hash — either
    # red.png or red_copy.png depending on scan order, both resolve here
    assert red_point.payload["image_path"] in {
        str((ingest_env.images / "red.png").resolve()),
        str((ingest_env.images / "red_copy.png").resolve()),
    }
    assert red_point.payload["dino_model_name"] == "fake-dino"
    assert red_point.payload["dino_model_hash"] == "hash-fake-dino"
    assert len(red_point.vectors["dino"]) == 4

    # CSV: 4 files, one skipped as run-duplicate, three indexed
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 4
    by_name = {Path(r["path"]).name: r for r in rows}
    indexed = [n for n, r in by_name.items() if r["processed"] == "yes"]
    skipped = [n for n, r in by_name.items() if r["processed"] == "no"]
    assert len(indexed) == 3 and len(skipped) == 1
    assert skipped[0] in {"red.png", "red_copy.png"}
    assert by_name[skipped[0]]["reason"] == "duplicate in run (same SHA-256)"
    assert by_name[indexed[0]]["sha256"] and by_name[indexed[0]]["md5"]


def test_index_second_run_skips_everything(ingest_env, store):
    _run_index(ingest_env.images, ingest_env.tmp / "r1.csv")
    calls_after_first = len([c for c in store.calls if c[0] in ("upsert", "update_vectors")])

    _run_index(ingest_env.images, ingest_env.tmp / "r2.csv")
    calls_after_second = len([c for c in store.calls if c[0] in ("upsert", "update_vectors")])
    assert calls_after_second == calls_after_first  # nothing written on re-run

    rows = list(csv.DictReader((ingest_env.tmp / "r2.csv").open()))
    assert all(r["processed"] == "no" for r in rows)
    reasons = {r["reason"] for r in rows}
    assert "already indexed in Qdrant" in reasons


def test_index_dual_modality_run_shares_points_across_models(ingest_env, store):
    """--dino --sscd in one run: the second model's upsert must find the first
    model's points and update them rather than replace them."""
    _run_index(ingest_env.images, ingest_env.tmp / "report.csv", dino=True, sscd=True)

    coll = store.collections["sfn"]
    # collection was created once, with both named vectors registered together
    assert set(coll.vectors_config) == {"dino", "sscd"}
    assert len(coll.points) == 3
    for point in coll.points.values():
        assert set(point.vectors) == {"dino", "sscd"}
        assert len(point.vectors["dino"]) == 4
        assert len(point.vectors["sscd"]) == 2
        assert point.payload["dino_model_name"] == "fake-dino"
        assert point.payload["sscd_model_name"] == "fake-sscd"

    # SSCD runs first (models_to_run order), inserting; DINO then updates.
    methods = [c[0] for c in store.calls if c[0] in ("upsert", "update_vectors")]
    assert "update_vectors" in methods
    assert methods.index("upsert") < methods.index("update_vectors")
