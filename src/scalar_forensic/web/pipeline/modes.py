"""Collection availability probing for the web frontend."""

from __future__ import annotations

import asyncio
import logging

from qdrant_client import QdrantClient

from scalar_forensic.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection availability
# ---------------------------------------------------------------------------


async def get_available_modes(settings: Settings) -> tuple[list[str], bool, str | None]:
    """Return which query modes are usable based on existing Qdrant collections.

    Retries up to 4 times (initial + 3 retries) with exponential backoff (1s/2s/4s)
    so transient startup delays don't immediately surface as errors.
    Returns a tuple of (modes, has_reference, error_message); error_message is None on success.
    has_reference is True only when SFN_REFERENCE_COLLECTION is set AND that collection exists.
    """
    _delays = [1, 2, 4]
    last_exc: Exception | None = None

    for attempt in range(4):
        if attempt > 0:
            await asyncio.sleep(_delays[attempt - 1])
        try:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            existing = {c.name for c in client.get_collections().collections}
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Qdrant connection attempt %d/4 failed at %s: %s",
                attempt + 1,
                settings.qdrant_url,
                exc,
            )
    else:
        return [], False, str(last_exc)

    if settings.collection not in existing:
        return [], False, None

    has_reference = (
        bool(settings.reference_collection) and settings.reference_collection in existing
    )

    modes: list[str] = ["exact"]  # exact works as long as the collection exists
    try:
        info = client.get_collection(settings.collection)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            if "sscd" in vectors_config:
                modes.append("altered")
            if "dino" in vectors_config:
                modes.append("semantic")
    except Exception as exc:  # noqa: BLE001
        err = f"Could not inspect vector config for {settings.collection}: {exc}"
        logger.warning(err)
        return modes, has_reference, err
    return modes, has_reference, None
