from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from scalar_forensic.faces.store import FaceStore
from scalar_forensic.web.pipeline.faces_stats import face_score_stats


def test_negative_scores_are_present_in_top_neighbor_statistics():
    client = QdrantClient(":memory:")
    client.create_collection(
        "faces", vectors_config={"face": VectorParams(size=2, distance=Distance.COSINE)}
    )
    client.upsert(
        "faces",
        [
            PointStruct(id=i, vector={"face": v}, payload={"is_face": True})
            for i, v in enumerate(([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]))
        ],
    )
    stats = face_score_stats(FaceStore(client, "faces", "case", 2), [1.0, 0.0], sample_size=10)
    assert stats.count == 3
    assert stats.min_score == -1.0
    assert sum(stats.histogram) == 3
    client.close()
