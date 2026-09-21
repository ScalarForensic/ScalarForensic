"""Release checks against an explicitly selected disposable Qdrant server."""

import os
import uuid
from importlib.metadata import version

import httpx
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.common.version_check import is_compatible

_URL = os.environ.get("SFN_TEST_QDRANT_URL")
pytestmark = pytest.mark.skipif(not _URL, reason="SFN_TEST_QDRANT_URL not set")


def test_server_supports_locked_client():
    client = QdrantClient(url=_URL)
    try:
        server_version = client.info().version
        client_version = version("qdrant-client")
        assert is_compatible(client_version, server_version), (
            f"Unsupported Qdrant pair: client {client_version}, server {server_version}"
        )
    finally:
        client.close()


def test_snapshot_restore_preserves_vectors_payloads_and_search():
    client = QdrantClient(url=_URL, timeout=30)
    source = f"sfn_test_snapshot_{uuid.uuid4().hex}"
    restored = f"{source}_restored"
    created = []
    try:
        for name in (source, restored):
            client.create_collection(
                name,
                vectors_config={
                    "face": models.VectorParams(size=4, distance=models.Distance.COSINE)
                },
            )
            created.append(name)
        client.upsert(
            source,
            points=[
                models.PointStruct(
                    id=1,
                    vector={"face": [1.0, 0.0, 0.0, 0.0]},
                    payload={
                        "embedding_status": "embedded",
                        "source_video_hash": "synthetic",
                        "timecode_ms": 42,
                    },
                ),
                models.PointStruct(
                    id=2,
                    vector={},
                    payload={
                        "embedding_status": "review_only",
                        "embedding_exclusion_reason": "size",
                    },
                ),
            ],
            wait=True,
        )
        snapshot = client.create_snapshot(source, wait=True)
        assert snapshot is not None
        with httpx.Client(base_url=_URL, timeout=30) as http:
            download = http.get(f"/collections/{source}/snapshots/{snapshot.name}")
            download.raise_for_status()
            upload = http.post(
                f"/collections/{restored}/snapshots/upload",
                params={"wait": "true", "priority": "snapshot"},
                files={"snapshot": (snapshot.name, download.content, "application/octet-stream")},
            )
            upload.raise_for_status()
            assert upload.json()["result"] is True
        assert client.count(restored, exact=True).count == 2
        before = client.retrieve(source, ids=[1, 2], with_payload=True, with_vectors=True)
        after = client.retrieve(restored, ids=[1, 2], with_payload=True, with_vectors=True)
        assert sorted(before, key=lambda p: p.id) == sorted(after, key=lambda p: p.id)
        hits = client.query_points(restored, query=[1.0, 0.0, 0.0, 0.0], using="face").points
        assert [hit.id for hit in hits] == [1]
        assert hits[0].score == pytest.approx(1.0)
    finally:
        try:
            for name in reversed(created):
                client.delete_collection(name)
        finally:
            client.close()
