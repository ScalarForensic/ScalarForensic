"""Collection validation is repeated at use, including database recovery and references."""

from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from scalar_forensic.config import Settings
from scalar_forensic.safeguards import (
    CompatibilityError,
    QdrantUnavailable,
    require_collection_compatible,
)
from scalar_forensic.web.pipeline.query import query_session
from scalar_forensic.web.session import FileEntry, Session


def test_compatible_collection_then_changed_reference_is_blocked(tmp_path):
    client = QdrantClient(":memory:")
    settings = Settings()
    settings.normalize_size = 224
    for name in ("case", "reference"):
        client.create_collection(
            name, vectors_config={"dino": VectorParams(size=2, distance=Distance.COSINE)}
        )
        client.upsert(
            name,
            [
                PointStruct(
                    id=1,
                    vector={"dino": [1.0, 0.0]},
                    payload={"dino_model_hash": "good", "dino_normalize_size": 224},
                )
            ],
        )
    with patch(
        "scalar_forensic.safeguards.expected_model_hashes_from_settings",
        return_value={"dino": "good"},
    ):
        require_collection_compatible(client, "reference", settings, needed_vectors={"dino"})
        client.set_payload("reference", {"dino_model_hash": "bad"}, points=[1])
        with pytest.raises(CompatibilityError, match="model_hash"):
            require_collection_compatible(client, "reference", settings, needed_vectors={"dino"})
        require_collection_compatible(client, "case", settings, needed_vectors={"dino"})
    client.close()


def test_unknown_provenance_cannot_authorize_similarity():
    client = QdrantClient(":memory:")
    client.create_collection(
        "case", vectors_config={"dino": VectorParams(size=2, distance=Distance.COSINE)}
    )
    client.upsert("case", [PointStruct(id=1, vector={"dino": [1.0, 0.0]}, payload={})])
    with patch(
        "scalar_forensic.safeguards.expected_model_hashes_from_settings",
        return_value={"dino": "good"},
    ):
        with pytest.raises(CompatibilityError, match="provenance"):
            require_collection_compatible(client, "case", Settings(), needed_vectors={"dino"})
    client.close()


def test_recovered_database_is_checked_before_search(tmp_path):
    settings = Settings()
    settings.collection = "case"
    settings.reference_collection = "reference"
    session = Session("s", [FileEntry("f", "x.jpg", tmp_path / "x", dino_embedding=[1.0, 0.0])])
    with (
        patch("scalar_forensic.web.pipeline.query.QdrantClient") as factory,
        patch("scalar_forensic.web.pipeline.query.require_collection_compatible") as gate,
    ):
        gate.side_effect = ConnectionError("offline")
        with pytest.raises(ConnectionError):
            query_session(session, ["semantic"], 0.7, 0.5, 10, settings)
        gate.side_effect = CompatibilityError("incompatible after recovery")
        with pytest.raises(CompatibilityError):
            query_session(session, ["semantic"], 0.7, 0.5, 10, settings)
        factory.return_value.query_points.assert_not_called()


def test_reference_gate_runs_before_any_similarity(tmp_path):
    settings = Settings()
    settings.collection = "case"
    settings.reference_collection = "reference"
    session = Session("s", [FileEntry("f", "x.jpg", tmp_path / "x", dino_embedding=[1.0, 0.0])])
    calls = []

    def validate(client, collection, settings, **kwargs):
        calls.append(collection)
        if collection == "reference":
            raise CompatibilityError("reference drift")

    with (
        patch("scalar_forensic.web.pipeline.query.QdrantClient") as factory,
        patch(
            "scalar_forensic.web.pipeline.query.require_collection_compatible", side_effect=validate
        ),
    ):
        with pytest.raises(CompatibilityError):
            query_session(session, ["semantic"], 0.7, 0.5, 10, settings, include_reference=True)
        assert calls == ["case", "reference"]
        factory.return_value.query_points.assert_not_called()


def test_exact_hash_lookup_does_not_require_embedding_compatibility(tmp_path):
    session = Session("s", [FileEntry("f", "x.jpg", tmp_path / "x", file_hash="a" * 64)])
    with (
        patch("scalar_forensic.web.pipeline.query.QdrantClient"),
        patch("scalar_forensic.web.pipeline.query.require_collection_compatible") as gate,
        patch("scalar_forensic.web.pipeline.query._query_exact", return_value=([], [])),
    ):
        query_session(session, ["exact"], 0.7, 0.5, 10, Settings())
        gate.assert_not_called()


def test_asgi_lifespan_performs_compatibility_check(monkeypatch, tmp_path):
    import asyncio

    from scalar_forensic.web.app import app, lifespan

    monkeypatch.setenv("SFN_SESSION_TEMP_ROOT", str(tmp_path / "sessions"))
    with patch("scalar_forensic.web.app._check_collection_compat") as check:

        async def start_stop():
            async with lifespan(app):
                assert check.call_count >= 1

        asyncio.run(start_stop())


def test_stats_rejects_settings_drift_from_session_model(tmp_path):
    from scalar_forensic.web.pipeline import _embedders
    from scalar_forensic.web.pipeline.stats import query_semantic_stats

    settings = Settings()
    identity = _embedders._configuration("dino", settings)
    session = Session("s", [FileEntry("f", "x.jpg", tmp_path / "x", dino_embedding=[1.0, 0.0])])
    settings.normalize_size += 112
    with (
        patch.dict(_embedders._embedder_cache, {"dino": object()}, clear=True),
        patch.dict(_embedders._cache_configs, {"dino": identity}, clear=True),
        patch("scalar_forensic.web.pipeline.stats.QdrantClient") as factory,
    ):
        with pytest.raises(CompatibilityError, match="cached embedder"):
            query_semantic_stats(session, "f", settings)
        factory.assert_not_called()


@pytest.mark.parametrize(
    ("vector_name", "field", "replacement"),
    [
        ("dino", "dino_model_hash", "bad"),
        ("dino", "dino_normalize_size", 336),
        ("dino", "dino_model_hash", None),
        ("sscd", "sscd_model_hash", "bad"),
        ("sscd", "sscd_n_crops", None),
    ],
)
def test_strict_gate_rejects_incompatible_provenance_beyond_sample(vector_name, field, replacement):
    client = QdrantClient(":memory:")
    settings = Settings()
    settings.normalize_size = 224
    settings.sscd_n_crops = 5
    client.create_collection(
        "tail", vectors_config={vector_name: VectorParams(size=2, distance=Distance.COSINE)}
    )
    payload = {
        f"{vector_name}_model_hash": "good",
        f"{vector_name}_normalize_size": 224 if vector_name == "dino" else 288,
        "sscd_n_crops": 5,
    }
    client.upsert(
        "tail",
        [PointStruct(id=i, vector={vector_name: [1.0, 0.0]}, payload=payload) for i in range(1025)]
        + [
            PointStruct(id=9999, vector={}, payload={"meta": "vectorless provenance is irrelevant"})
        ],
    )
    try:
        require_collection_compatible(
            client,
            "tail",
            settings,
            needed_vectors={vector_name},
            expected_hashes={vector_name: "good"},
        )
        with (
            patch.object(client, "count", side_effect=ConnectionError("count unavailable")),
            pytest.raises(QdrantUnavailable, match="count unavailable"),
        ):
            require_collection_compatible(
                client,
                "tail",
                settings,
                needed_vectors={vector_name},
                expected_hashes={vector_name: "good"},
            )
        if replacement is None:
            client.delete_payload("tail", keys=[field], points=[1024])
        else:
            client.set_payload("tail", {field: replacement}, points=[1024])
        with (
            patch.object(client, "scroll", wraps=client.scroll) as scroll,
            pytest.raises(CompatibilityError, match="provenance"),
        ):
            require_collection_compatible(
                client,
                "tail",
                settings,
                needed_vectors={vector_name},
                expected_hashes={vector_name: "good"},
            )
        assert scroll.call_count == 4
        assert all(call.kwargs["limit"] == 256 for call in scroll.call_args_list)
    finally:
        client.close()
