"""Model provenance extraction from Qdrant payloads."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from scalar_forensic.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full indexing provenance for a single image (on-demand, for Audit modal)
# ---------------------------------------------------------------------------

_PROVENANCE_FIELD_NAMES = [
    "model_name",
    "model_hash",
    "indexed_at",
    "inference_dtype",
    "normalize_size",
    "embedding_dim",
]

# Per-model provenance is stored with a vector-name prefix in the unified collection.
# Map: (vector_name, mode_label) pairs to resolve at query time.
_VECTOR_MODE_MAP = [
    ("dino", "semantic"),
    ("sscd", "altered"),
]


def _payload_model_provenance(payload: dict) -> dict[str, dict]:
    """Extract model provenance from a Qdrant payload using _VECTOR_MODE_MAP.

    Returns a dict keyed by mode label (e.g. "semantic", "altered") for every
    model whose name or hash is present in *payload*.
    """
    mp: dict[str, dict] = {}
    for vn, mode in _VECTOR_MODE_MAP:
        name = payload.get(f"{vn}_model_name", "")
        h = payload.get(f"{vn}_model_hash", "")
        if name or h:
            mp[mode] = {"name": name, "hash": h}
    return mp


def get_hit_qdrant_provenance(image_hash: str, settings: Settings) -> dict[str, dict]:
    """Fetch full indexing-time provenance for one image hash from Qdrant.

    Returns a dict keyed by mode ("altered", "semantic") containing all
    provenance fields stored in the point payload when the image was indexed.
    Fields are retrieved from the prefixed payload keys (e.g. ``dino_model_name``).
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    result: dict[str, dict] = {}
    try:
        # Collect all prefixed field names we want from the payload.
        payload_fields = [
            f"{vn}_{field}" for vn, _ in _VECTOR_MODE_MAP for field in _PROVENANCE_FIELD_NAMES
        ] + ["sscd_n_crops", "library_versions"]
        records, _ = client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="image_hash", match=MatchValue(value=image_hash))]
            ),
            limit=1,
            with_payload=payload_fields,
            with_vectors=False,
        )
        if records:
            p = records[0].payload
            for vn, mode in _VECTOR_MODE_MAP:
                mode_data = {field: p.get(f"{vn}_{field}") for field in _PROVENANCE_FIELD_NAMES}
                # Only include the mode entry if at least one provenance field is populated.
                if any(v is not None for v in mode_data.values()):
                    mode_data["library_versions"] = p.get("library_versions")
                    if vn == "sscd":
                        mode_data["sscd_n_crops"] = p.get("sscd_n_crops")
                    result[mode] = mode_data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Provenance query failed on %s: %s", settings.collection, exc)
    return result
