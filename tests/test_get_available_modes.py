"""Tests for get_available_modes() in scalar_forensic.web.pipeline (issue #8)."""

import asyncio
from unittest.mock import MagicMock, patch

from scalar_forensic.config import Settings
from scalar_forensic.web.pipeline import get_available_modes


def _settings(**kwargs) -> Settings:
    """Build a minimal Settings instance with sensible defaults."""
    defaults = dict(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection_sscd="sfn-sscd",
        collection_dino="sfn-dinov2",
    )
    defaults.update(kwargs)
    s = MagicMock(spec=Settings)
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _make_collection(name: str) -> MagicMock:
    c = MagicMock()
    c.name = name
    return c


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


def test_get_available_modes_both_collections():
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = [
        _make_collection("sfn-sscd"),
        _make_collection("sfn-dinov2"),
    ]

    with patch("scalar_forensic.web.pipeline.QdrantClient") as MockClient:
        MockClient.return_value.get_collections.return_value = collections_response
        modes, error = asyncio.run(get_available_modes(settings))

    assert error is None
    assert set(modes) == {"exact", "altered", "semantic"}


def test_get_available_modes_only_sscd():
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = [_make_collection("sfn-sscd")]

    with patch("scalar_forensic.web.pipeline.QdrantClient") as MockClient:
        MockClient.return_value.get_collections.return_value = collections_response
        modes, error = asyncio.run(get_available_modes(settings))

    assert error is None
    assert "exact" in modes
    assert "altered" in modes
    assert "semantic" not in modes


def test_get_available_modes_empty_collections():
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = []

    with patch("scalar_forensic.web.pipeline.QdrantClient") as MockClient:
        MockClient.return_value.get_collections.return_value = collections_response
        modes, error = asyncio.run(get_available_modes(settings))

    assert modes == []
    assert error is None


# ---------------------------------------------------------------------------
# Failure paths (issue #8: retry with exponential backoff)
# ---------------------------------------------------------------------------


def test_get_available_modes_all_retries_exhausted():
    """When every attempt fails, returns ([], error_message) after 4 attempts."""
    settings = _settings()
    boom = ConnectionError("timed out")

    call_count = 0

    def _raising_client(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise boom

    async def _noop_sleep(_):
        pass  # skip real delays

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient", side_effect=_raising_client),
        patch("scalar_forensic.web.pipeline.asyncio.sleep", side_effect=_noop_sleep),
    ):
        modes, error = asyncio.run(get_available_modes(settings))

    assert modes == []
    assert error is not None
    assert "timed out" in error
    assert call_count == 4  # initial + 3 retries


def test_get_available_modes_succeeds_on_second_attempt():
    """When first attempt fails but second succeeds, returns correct modes."""
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = [
        _make_collection("sfn-sscd"),
        _make_collection("sfn-dinov2"),
    ]

    attempt = 0

    def _flaky_client(*args, **kwargs):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise ConnectionError("first attempt failed")
        client = MagicMock()
        client.get_collections.return_value = collections_response
        return client

    async def _noop_sleep(_):
        pass

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient", side_effect=_flaky_client),
        patch("scalar_forensic.web.pipeline.asyncio.sleep", side_effect=_noop_sleep),
    ):
        modes, error = asyncio.run(get_available_modes(settings))

    assert error is None
    assert set(modes) == {"exact", "altered", "semantic"}
    assert attempt == 2


def test_get_available_modes_retry_delays_are_exponential():
    """Verify that backoff delays follow the 1s/2s/4s pattern."""
    settings = _settings()
    boom = ConnectionError("down")
    sleep_args: list[float] = []

    async def _record_sleep(delay):
        sleep_args.append(delay)

    with (
        patch(
            "scalar_forensic.web.pipeline.QdrantClient",
            side_effect=lambda *a, **kw: (_ for _ in ()).throw(boom),
        ),
        patch("scalar_forensic.web.pipeline.asyncio.sleep", side_effect=_record_sleep),
    ):
        asyncio.run(get_available_modes(settings))

    assert sleep_args == [1, 2, 4]
