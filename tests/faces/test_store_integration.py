"""Live-Qdrant checks for the review-only exclusion guarantee.

The guarantee — a review-only observation is structurally unreachable by
similarity search because it carries no vector — cannot be observed by a
hermetic test: asserting ``PointStruct.vector == {}`` inspects a constructor,
not Qdrant's storage or its search behaviour.  These are the only tests that
can catch a demotion regression.

Skipped unless SFN_TEST_QDRANT_URL is set.  Qdrant is deliberately not
published to the host (docker-compose.yml); add a local override with
  services: {qdrant: {ports: ["127.0.0.1:6333:6333"]}}
then run:
  SFN_TEST_QDRANT_URL=http://localhost:6333 uv run pytest tests/faces/test_store_integration.py -v
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from scalar_forensic.faces.store import FACE_VECTOR_NAME, FaceStore

_URL = os.environ.get("SFN_TEST_QDRANT_URL")
pytestmark = pytest.mark.skipif(not _URL, reason="SFN_TEST_QDRANT_URL not set")


@pytest.fixture
def live_collection():
    """A throwaway face collection on a real server, dropped afterwards."""
    client = QdrantClient(url=_URL)
    name = f"sfn_test_faces_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        collection_name=name,
        vectors_config={FACE_VECTOR_NAME: VectorParams(size=4, distance=Distance.COSINE)},
    )
    try:
        yield client, name
    finally:
        client.delete_collection(collection_name=name)


def _search(client, name):
    return client.query_points(
        collection_name=name, query=[1.0, 0.0, 0.0, 0.0], using=FACE_VECTOR_NAME, limit=10
    ).points


def test_demoted_point_is_not_returned_by_vector_search(live_collection):
    client, name = live_collection
    pid = str(uuid.uuid4())
    client.upsert(
        collection_name=name,
        points=[
            PointStruct(
                id=pid,
                vector={FACE_VECTOR_NAME: [1.0, 0.0, 0.0, 0.0]},
                payload={"is_face": True, "embedding_status": "embedded"},
            )
        ],
        wait=True,
    )
    assert [h.id for h in _search(client, name)] == [pid], (
        "precondition: the embedded point is findable"
    )

    # Demote exactly as the pipeline does: rewrite the point vectorless, then
    # clear through the store.  An upsert with vector={} alone must never be
    # trusted to drop a vector the previous run stored at the same id.
    client.upsert(
        collection_name=name,
        points=[
            PointStruct(
                id=pid,
                vector={},
                payload={
                    "is_face": True,
                    "embedding_status": "review_only",
                    "embedding_exclusion_reason": "size",
                },
            )
        ],
        wait=True,
    )
    FaceStore(client, name, "case1", 4).clear_face_vector([pid])

    assert _search(client, name) == [], "a review-only point must be unreachable by vector search"

    got = client.retrieve(collection_name=name, ids=[pid], with_payload=True)
    assert got[0].payload["embedding_status"] == "review_only", "payload must survive demotion"
    assert got[0].payload["embedding_exclusion_reason"] == "size"


def test_vectorless_upsert_also_drops_the_stored_vector(live_collection):
    """Measured server behaviour, recorded rather than relied upon.

    On qdrant 1.17 a re-upsert with ``vector={}`` already removes the named
    vector, so the explicit clear is belt-and-braces here.  It stays anyway:
    this is a storage-engine detail, not a documented API guarantee, and the
    exclusion guarantee must not rest on it.  If a later version keeps the
    vector this test flips and the clear becomes the only thing standing
    between a review-only observation and the search space — which is exactly
    why the call site does not skip it.
    """
    client, name = live_collection
    pid = str(uuid.uuid4())
    client.upsert(
        collection_name=name,
        points=[PointStruct(id=pid, vector={FACE_VECTOR_NAME: [1.0, 0.0, 0.0, 0.0]}, payload={})],
        wait=True,
    )
    client.upsert(
        collection_name=name,
        points=[PointStruct(id=pid, vector={}, payload={"embedding_status": "review_only"})],
        wait=True,
    )
    assert _search(client, name) == []


def test_clear_face_vector_tolerates_points_that_never_held_a_vector(live_collection):
    """The call site passes every review-only id, not only demoted ones.

    Most of those points have never held a vector — a first-time review-only
    face is upserted vectorless and then cleared.  That must not error.
    """
    client, name = live_collection
    vectorless = str(uuid.uuid4())
    client.upsert(
        collection_name=name,
        points=[PointStruct(id=vectorless, vector={}, payload={"is_face": True})],
        wait=True,
    )
    FaceStore(client, name, "case1", 4).clear_face_vector([vectorless])

    got = client.retrieve(collection_name=name, ids=[vectorless], with_payload=True)
    assert got[0].payload["is_face"] is True, "payload must survive a no-op clear"


def test_clear_face_vector_raises_on_ids_that_do_not_exist(live_collection):
    """Contradicts the plan's assumption — the server does NOT no-op here.

    ``delete_vectors`` on an unknown id returns 404, so clear_face_vector's
    precondition is that every id was upserted first.  The CLI satisfies it:
    ``upsert_faces`` runs immediately before the clear and qdrant-client's
    upsert defaults to wait=True, so the points are committed.  Loud failure
    is the right behaviour to keep — a silently swallowed 404 could mean a
    review-only point kept its vector and stayed searchable, which is the one
    outcome this design exists to prevent.
    """
    client, name = live_collection
    store = FaceStore(client, name, "case1", 4)
    with pytest.raises(Exception, match="(?i)not found"):
        store.clear_face_vector([str(uuid.uuid4())])


def test_unreferenced_chip_hashes_reads_the_real_payload_projection(live_collection):
    """The scroll projection must actually name both hash fields.

    A projection that omitted aligned_chip_hash would make every aligned PNG
    look unreferenced and purge would unlink chips that authenticate surviving
    observations.  Against a mock this passes either way; against a real
    server it does not.
    """
    client, name = live_collection
    store = FaceStore(client, name, "case1", 4)
    client.upsert(
        collection_name=name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector={FACE_VECTOR_NAME: [1.0, 0.0, 0.0, 0.0]},
                payload={
                    "is_face": True,
                    "aligned_chip_hash": "a" * 64,
                    "review_chip_hash": "r" * 64,
                },
            )
        ],
        wait=True,
    )
    unreferenced = store.unreferenced_chip_hashes(["a" * 64, "r" * 64, "z" * 64])
    assert unreferenced == ["z" * 64]
