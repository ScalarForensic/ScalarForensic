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
        collection="sfn",
        reference_collection=None,
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


def _make_collection_info(vector_names: list[str]) -> MagicMock:
    """Return a mock get_collection() response with the given named vectors."""
    info = MagicMock()
    info.config.params.vectors = {name: MagicMock() for name in vector_names}
    return info


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


def test_get_available_modes_both_vectors():
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = [_make_collection("sfn")]

    with patch("scalar_forensic.web.pipeline.QdrantClient") as MockClient:
        MockClient.return_value.get_collections.return_value = collections_response
        MockClient.return_value.get_collection.return_value = _make_collection_info(
            ["dino", "sscd"]
        )
        modes, has_ref, error = asyncio.run(get_available_modes(settings))

    assert error is None
    assert has_ref is False
    assert set(modes) == {"exact", "altered", "semantic"}


def test_get_available_modes_with_reference_collection():
    settings = _settings(reference_collection="sfn_ref")
    collections_response = MagicMock()
    collections_response.collections = [
        _make_collection("sfn"),
        _make_collection("sfn_ref"),
    ]

    with patch("scalar_forensic.web.pipeline.QdrantClient") as MockClient:
        MockClient.return_value.get_collections.return_value = collections_response
        MockClient.return_value.get_collection.return_value = _make_collection_info(
            ["dino", "sscd"]
        )
        modes, has_ref, error = asyncio.run(get_available_modes(settings))

    assert error is None
    assert has_ref is True
    assert set(modes) == {"exact", "altered", "semantic"}


def test_get_available_modes_only_sscd():
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = [_make_collection("sfn")]

    with patch("scalar_forensic.web.pipeline.QdrantClient") as MockClient:
        MockClient.return_value.get_collections.return_value = collections_response
        MockClient.return_value.get_collection.return_value = _make_collection_info(["sscd"])
        modes, _has_ref, error = asyncio.run(get_available_modes(settings))

    assert error is None
    assert "exact" in modes
    assert "altered" in modes
    assert "semantic" not in modes


def test_get_available_modes_only_dino():
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = [_make_collection("sfn")]

    with patch("scalar_forensic.web.pipeline.QdrantClient") as MockClient:
        MockClient.return_value.get_collections.return_value = collections_response
        MockClient.return_value.get_collection.return_value = _make_collection_info(["dino"])
        modes, _has_ref, error = asyncio.run(get_available_modes(settings))

    assert error is None
    assert "exact" in modes
    assert "semantic" in modes
    assert "altered" not in modes


def test_get_available_modes_collection_absent():
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = []

    with patch("scalar_forensic.web.pipeline.QdrantClient") as MockClient:
        MockClient.return_value.get_collections.return_value = collections_response
        modes, _has_ref, error = asyncio.run(get_available_modes(settings))

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
        modes, _has_ref, error = asyncio.run(get_available_modes(settings))

    assert modes == []
    assert error is not None
    assert "timed out" in error
    assert call_count == 4  # initial + 3 retries


def test_get_available_modes_succeeds_on_second_attempt():
    """When first attempt fails but second succeeds, returns correct modes."""
    settings = _settings()
    collections_response = MagicMock()
    collections_response.collections = [_make_collection("sfn")]

    attempt = 0

    def _flaky_client(*args, **kwargs):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise ConnectionError("first attempt failed")
        client = MagicMock()
        client.get_collections.return_value = collections_response
        client.get_collection.return_value = _make_collection_info(["dino", "sscd"])
        return client

    async def _noop_sleep(_):
        pass

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient", side_effect=_flaky_client),
        patch("scalar_forensic.web.pipeline.asyncio.sleep", side_effect=_noop_sleep),
    ):
        modes, _has_ref, error = asyncio.run(get_available_modes(settings))

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
