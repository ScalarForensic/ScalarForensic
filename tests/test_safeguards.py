"""Tests for the embedding-configuration safeguards.

Covers :mod:`scalar_forensic.safeguards` directly without spinning up Qdrant —
the QdrantClient interface used by the safeguard is small enough (3 methods)
to mock cleanly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import VectorParams

from scalar_forensic.safeguards import (
    QdrantUnavailable,
    check_collection_compat,
    compute_remote_model_hash,
    compute_sscd_model_hash,
    expected_model_hashes_from_settings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DINO_HASH = "a" * 64
SSCD_HASH = "b" * 64


def _settings(
    *,
    normalize_size: int = 224,
    sscd_n_crops: int = 1,
    embedding_endpoint: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int = 0,
    model_dino: str = "/no/such/path",
    model_sscd: str = "/no/such/path",
) -> SimpleNamespace:
    """A duck-typed Settings stand-in.

    The real Settings class only does file I/O at import time, so a SimpleNamespace
    with the same attribute names is sufficient for unit-testing the safeguards.
    """
    return SimpleNamespace(
        normalize_size=normalize_size,
        sscd_n_crops=sscd_n_crops,
        embedding_endpoint=embedding_endpoint,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        model_dino=model_dino,
        model_sscd=model_sscd,
    )


def _make_client(
    *,
    collections: list[str],
    vectors_cfg: dict[str, VectorParams] | None,
    payload_by_vector: dict[str, dict | list[dict] | None],
    fail_on: str | None = None,
) -> MagicMock:
    """Build a MagicMock QdrantClient covering the surface used by the safeguard.

    *payload_by_vector* may map a vector name to a single payload dict (returned
    as one point), a list of payload dicts (returned as that many points across
    one scroll page), or ``None`` (no points).

    *fail_on* makes one of (``get_collections``, ``get_collection``, ``scroll``)
    raise ``ResponseHandlingException`` so we can exercise the error path.
    """
    client = MagicMock()

    def _get_collections():
        if fail_on == "get_collections":
            raise ResponseHandlingException("connection refused")
        return SimpleNamespace(collections=[SimpleNamespace(name=c) for c in collections])

    def _get_collection(name: str):
        if fail_on == "get_collection":
            raise ResponseHandlingException("connection refused")
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors_cfg)))

    def _scroll(
        *,
        collection_name,  # noqa: ARG001
        scroll_filter,
        with_payload,  # noqa: ARG001
        with_vectors,  # noqa: ARG001
        limit,  # noqa: ARG001
        offset=None,
    ):
        if fail_on == "scroll":
            raise ResponseHandlingException("connection refused")
        # Once we've returned the configured points, signal end-of-scroll so
        # the safeguard's pagination loop terminates.
        if offset == "DONE":
            return ([], None)
        vn = scroll_filter.must[0].has_vector
        entry = payload_by_vector.get(vn)
        if entry is None:
            return ([], None)
        payloads = entry if isinstance(entry, list) else [entry]
        points = [SimpleNamespace(payload=p) for p in payloads]
        return (points, "DONE")

    client.get_collections.side_effect = _get_collections
    client.get_collection.side_effect = _get_collection
    client.scroll.side_effect = _scroll
    return client


def _vp() -> VectorParams:
    """A VectorParams instance — exact fields don't matter for the safeguard."""
    from qdrant_client.models import Distance

    return VectorParams(size=4, distance=Distance.COSINE)


# ---------------------------------------------------------------------------
# Pure hash helpers
# ---------------------------------------------------------------------------


def test_compute_sscd_model_hash_matches_sha256(tmp_path: Path):
    f = tmp_path / "sscd.pt"
    payload = b"some torchscript bytes"
    f.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert compute_sscd_model_hash(f) == expected


def test_compute_remote_model_hash_is_deterministic_and_strips_trailing_slash():
    h1 = compute_remote_model_hash("https://api.example.com/", "model-x", 1024)
    h2 = compute_remote_model_hash("https://api.example.com", "model-x", 1024)
    assert h1 == h2

    h3 = compute_remote_model_hash("https://api.example.com", "model-y", 1024)
    assert h1 != h3


def test_compute_remote_model_hash_changes_with_dim():
    h1 = compute_remote_model_hash("https://x", "m", 512)
    h2 = compute_remote_model_hash("https://x", "m", 1024)
    assert h1 != h2


def test_compute_remote_model_hash_has_config_prefix():
    h = compute_remote_model_hash("https://api.example.com", "model-x", 1024)
    assert h.startswith("config:"), f"expected 'config:' prefix, got: {h!r}"
    # Remainder must be a valid SHA-256 hex digest (64 hex chars).
    assert len(h) == len("config:") + 64
    assert all(c in "0123456789abcdef" for c in h[len("config:"):])


# ---------------------------------------------------------------------------
# expected_model_hashes_from_settings
# ---------------------------------------------------------------------------


def test_expected_hashes_for_remote_returns_same_hash_per_vector():
    s = _settings(
        embedding_endpoint="https://embed.example.com",
        embedding_model="some-model",
        embedding_dim=768,
    )
    out = expected_model_hashes_from_settings(s, needed_vectors={"dino", "sscd"})
    assert set(out) == {"dino", "sscd"}
    assert out["dino"] == out["sscd"]


def test_expected_hashes_for_remote_missing_config_returns_empty():
    s = _settings(embedding_endpoint="https://embed.example.com", embedding_dim=0)
    assert expected_model_hashes_from_settings(s, needed_vectors={"dino"}) == {}


def test_expected_hashes_skips_missing_local_models(tmp_path: Path):
    s = _settings(model_dino=str(tmp_path / "missing"), model_sscd=str(tmp_path / "missing"))
    out = expected_model_hashes_from_settings(s, needed_vectors={"dino", "sscd"})
    # Both files missing — nothing to hash, but no exception either.
    assert out == {}


def test_expected_hashes_only_computes_requested_vectors(tmp_path: Path):
    sscd = tmp_path / "sscd.pt"
    sscd.write_bytes(b"x")
    s = _settings(model_sscd=str(sscd))
    out = expected_model_hashes_from_settings(s, needed_vectors={"sscd"})
    assert set(out) == {"sscd"}


# ---------------------------------------------------------------------------
# check_collection_compat — happy paths
# ---------------------------------------------------------------------------


def test_check_returns_empty_when_collection_missing():
    client = _make_client(
        collections=[],
        vectors_cfg=None,
        payload_by_vector={},
    )
    assert check_collection_compat(client, "sfn", _settings()) == []


def test_check_returns_empty_when_no_points():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp(), "sscd": _vp()},
        payload_by_vector={"dino": None, "sscd": None},
    )
    assert check_collection_compat(client, "sfn", _settings()) == []


def test_check_returns_empty_when_payload_lacks_provenance_fields():
    """Old indexes pre-date the provenance fields — must not flag a mismatch."""
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp(), "sscd": _vp()},
        payload_by_vector={"dino": {}, "sscd": {}},
    )
    s = _settings(normalize_size=512, sscd_n_crops=5)
    assert (
        check_collection_compat(
            client, "sfn", s, expected_dino_hash=DINO_HASH, expected_sscd_hash=SSCD_HASH
        )
        == []
    )


def test_check_returns_empty_when_everything_matches():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp(), "sscd": _vp()},
        payload_by_vector={
            "dino": {
                "dino_model_hash": DINO_HASH,
                "dino_normalize_size": 224,
            },
            "sscd": {
                "sscd_model_hash": SSCD_HASH,
                "sscd_n_crops": 1,
            },
        },
    )
    assert (
        check_collection_compat(
            client,
            "sfn",
            _settings(normalize_size=224, sscd_n_crops=1),
            expected_dino_hash=DINO_HASH,
            expected_sscd_hash=SSCD_HASH,
        )
        == []
    )


def test_check_handles_legacy_single_vector_collection():
    """Pre-named-vector collections expose vectors as VectorParams, not dict."""
    client = _make_client(
        collections=["sfn"],
        vectors_cfg=_vp(),  # not a dict — single-vector legacy schema
        payload_by_vector={},
    )
    assert check_collection_compat(client, "sfn", _settings()) == []


# ---------------------------------------------------------------------------
# check_collection_compat — mismatch detection
# ---------------------------------------------------------------------------


def test_check_detects_dino_normalize_size_mismatch():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp()},
        payload_by_vector={"dino": {"dino_normalize_size": 224}},
    )
    errors = check_collection_compat(client, "sfn", _settings(normalize_size=512))
    assert len(errors) == 1
    assert "normalize_size" in errors[0]
    assert "stored=224" in errors[0]
    assert "current=512" in errors[0]


def test_check_detects_sscd_n_crops_mismatch():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"sscd": _vp()},
        payload_by_vector={"sscd": {"sscd_n_crops": 5}},
    )
    errors = check_collection_compat(client, "sfn", _settings(sscd_n_crops=1))
    assert len(errors) == 1
    assert "sscd_n_crops" in errors[0]
    assert "stored=5" in errors[0]
    assert "current=1" in errors[0]


def test_check_detects_dino_model_hash_mismatch():
    stored = "c" * 64
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp()},
        payload_by_vector={"dino": {"dino_model_hash": stored}},
    )
    errors = check_collection_compat(client, "sfn", _settings(), expected_dino_hash=DINO_HASH)
    assert len(errors) == 1
    assert "model_hash" in errors[0]
    # Hashes are NOT truncated — the full 64-character digest must appear verbatim.
    assert stored in errors[0]
    assert DINO_HASH in errors[0]


def test_check_detects_sscd_model_hash_mismatch():
    stored = "d" * 64
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"sscd": _vp()},
        payload_by_vector={"sscd": {"sscd_model_hash": stored}},
    )
    errors = check_collection_compat(client, "sfn", _settings(), expected_sscd_hash=SSCD_HASH)
    assert len(errors) == 1
    assert stored in errors[0]
    assert SSCD_HASH in errors[0]


def test_check_skips_model_hash_when_no_expected_value():
    """Web startup does not compute the hash if no model is reachable —
    in that case the safeguard must not flag a stored hash as a mismatch."""
    stored = "e" * 64
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp()},
        payload_by_vector={"dino": {"dino_model_hash": stored}},
    )
    # No expected_dino_hash → no mismatch possible.
    assert check_collection_compat(client, "sfn", _settings()) == []


def test_check_aggregates_multiple_mismatches():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp(), "sscd": _vp()},
        payload_by_vector={
            "dino": {"dino_normalize_size": 224, "dino_model_hash": "x" * 64},
            "sscd": {"sscd_n_crops": 5, "sscd_model_hash": "y" * 64},
        },
    )
    errors = check_collection_compat(
        client,
        "sfn",
        _settings(normalize_size=512, sscd_n_crops=1),
        expected_dino_hash=DINO_HASH,
        expected_sscd_hash=SSCD_HASH,
    )
    # Two hashes + normalize_size + sscd_n_crops = 4 lines.
    assert len(errors) == 4


# ---------------------------------------------------------------------------
# check_collection_compat — collection-internal inconsistency
# ---------------------------------------------------------------------------


def test_check_detects_two_distinct_dino_hashes_in_collection():
    """A collection that mixes two model_hashes (e.g. from a previous
    sfn index --ignore-config-mismatch run) must be flagged regardless of
    what the current settings happen to match."""
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp()},
        payload_by_vector={
            "dino": [
                {"dino_model_hash": "a" * 64, "dino_normalize_size": 224},
                {"dino_model_hash": "b" * 64, "dino_normalize_size": 224},
            ],
        },
    )
    # Current settings happen to match the FIRST stored hash exactly — under
    # the old single-point check this would have returned no errors, masking
    # the mixed corpus.  Now it must flag the inconsistency.
    errors = check_collection_compat(
        client, "sfn", _settings(normalize_size=224), expected_dino_hash="a" * 64
    )
    assert len(errors) == 1
    assert "multiple embedding configurations" in errors[0]
    assert "a" * 64 in errors[0]
    assert "b" * 64 in errors[0]


def test_check_detects_mixed_normalize_size_in_collection():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp()},
        payload_by_vector={
            "dino": [
                {"dino_normalize_size": 224},
                {"dino_normalize_size": 512},
            ],
        },
    )
    errors = check_collection_compat(client, "sfn", _settings(normalize_size=224))
    assert len(errors) == 1
    assert "multiple embedding configurations" in errors[0]
    assert "normalize_size=224" in errors[0]
    assert "normalize_size=512" in errors[0]


def test_check_detects_mixed_sscd_n_crops_in_collection():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"sscd": _vp()},
        payload_by_vector={
            "sscd": [
                {"sscd_n_crops": 1},
                {"sscd_n_crops": 5},
            ],
        },
    )
    errors = check_collection_compat(client, "sfn", _settings(sscd_n_crops=1))
    assert len(errors) == 1
    assert "multiple embedding configurations" in errors[0]
    assert "sscd_n_crops=1" in errors[0]
    assert "sscd_n_crops=5" in errors[0]


def test_check_internal_inconsistency_skips_current_vs_stored_comparison():
    """When the collection itself is inconsistent there is no single 'stored'
    tuple to compare against; the safeguard must NOT also emit a
    current-vs-stored line that would confuse the user about which value
    is the real reference."""
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp()},
        payload_by_vector={
            "dino": [
                {"dino_model_hash": "a" * 64, "dino_normalize_size": 224},
                {"dino_model_hash": "b" * 64, "dino_normalize_size": 512},
            ],
        },
    )
    errors = check_collection_compat(
        client, "sfn", _settings(normalize_size=999), expected_dino_hash="c" * 64
    )
    assert len(errors) == 1
    assert "multiple embedding configurations" in errors[0]


def test_check_ignores_pre_provenance_points_in_inconsistency_scan():
    """An older index whose points lack the provenance fields entirely must
    not register as a phantom (None, None, None) tuple alongside a newer
    point with real provenance — that would falsely flag inconsistency."""
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp()},
        payload_by_vector={
            "dino": [
                {},  # pre-provenance point
                {"dino_model_hash": "a" * 64, "dino_normalize_size": 224},
            ],
        },
    )
    # Current settings match the one real tuple — should be clean.
    errors = check_collection_compat(
        client, "sfn", _settings(normalize_size=224), expected_dino_hash="a" * 64
    )
    assert errors == []


def test_check_normalize_size_flag_suppresses_dino_resolution_check_only():
    """The check_normalize_size knob exists for unit-testing isolation, not as
    a phase-discrimination flag — n_crops and hash checks must still run when
    it is False."""
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp(), "sscd": _vp()},
        payload_by_vector={
            "dino": {"dino_normalize_size": 224},
            "sscd": {"sscd_n_crops": 5},
        },
    )
    errors = check_collection_compat(
        client,
        "sfn",
        _settings(normalize_size=512, sscd_n_crops=1),
        check_normalize_size=False,
    )
    # normalize_size mismatch suppressed; n_crops mismatch still flagged.
    assert len(errors) == 1
    assert "sscd_n_crops" in errors[0]


def test_check_ignores_stored_normalize_size_for_sscd():
    """SSCD's normalize_size is fixed at 288 inside the embedder — only DINO's
    user-controlled normalize_size should ever be compared."""
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"sscd": _vp()},
        payload_by_vector={"sscd": {"sscd_normalize_size": 288}},
    )
    # SFN_NORMALIZE_SIZE=512 must not flag a mismatch on the sscd vector.
    assert check_collection_compat(client, "sfn", _settings(normalize_size=512)) == []


# ---------------------------------------------------------------------------
# check_collection_compat — Qdrant connectivity errors surface, not swallow
# ---------------------------------------------------------------------------


def test_check_raises_qdrant_unavailable_on_get_collections_failure():
    client = _make_client(
        collections=[],
        vectors_cfg=None,
        payload_by_vector={},
        fail_on="get_collections",
    )
    with pytest.raises(QdrantUnavailable):
        check_collection_compat(client, "sfn", _settings())


def test_check_raises_qdrant_unavailable_on_get_collection_failure():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg=None,
        payload_by_vector={},
        fail_on="get_collection",
    )
    with pytest.raises(QdrantUnavailable):
        check_collection_compat(client, "sfn", _settings())


def test_check_raises_qdrant_unavailable_on_scroll_failure():
    client = _make_client(
        collections=["sfn"],
        vectors_cfg={"dino": _vp()},
        payload_by_vector={"dino": {"dino_normalize_size": 224}},
        fail_on="scroll",
    )
    with pytest.raises(QdrantUnavailable):
        check_collection_compat(client, "sfn", _settings())


def test_check_raises_on_connection_error():
    client = MagicMock()
    client.get_collections.side_effect = ConnectionError("nope")
    with pytest.raises(QdrantUnavailable):
        check_collection_compat(client, "sfn", _settings())
