"""Tests for the sfn-web startup compatibility pre-flight.

Covers ``scalar_forensic.web.app._check_collection_compat`` with focus on the
``--ignore-config-mismatch`` escape hatch: a mismatch with ``ignore_mismatch=False``
must call ``sys.exit(1)``; the same mismatch with ``ignore_mismatch=True`` must
log a warning and return without exiting.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scalar_forensic.web.app import _check_collection_compat


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key": None,
        "collection": "sfn",
        "normalize_size": 224,
        "sscd_n_crops": 1,
        "embedding_endpoint": None,
        "embedding_model": None,
        "embedding_dim": 0,
        "model_dino": "/no/such/path",
        "model_sscd": "/no/such/path",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_no_mismatch_returns_silently(capsys):
    """Empty errors list → no exit, no error output."""
    fake_client = MagicMock()
    with (
        patch("scalar_forensic.web.app.QdrantClient", return_value=fake_client),
        patch(
            "scalar_forensic.safeguards.check_collection_compat",
            return_value=[],
        ),
        patch(
            "scalar_forensic.safeguards.expected_model_hashes_from_settings",
            return_value={},
        ),
    ):
        # Make the pre-hash collection introspection succeed and report
        # the collection exists with both vectors.
        fake_client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="sfn")]
        )
        fake_client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors={"dino": object()}))
        )

        _check_collection_compat(_settings(), ignore_mismatch=False)

    err = capsys.readouterr().err
    assert "[ERROR]" not in err
    assert "[WARN]" not in err


def test_mismatch_without_override_exits_with_code_1(capsys):
    fake_client = MagicMock()
    fake_client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="sfn")]
    )
    fake_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors={"dino": object()}))
    )

    with (
        patch("scalar_forensic.web.app.QdrantClient", return_value=fake_client),
        patch(
            "scalar_forensic.safeguards.check_collection_compat",
            return_value=["[dino] normalize_size: stored=224  current=512"],
        ),
        patch(
            "scalar_forensic.safeguards.expected_model_hashes_from_settings",
            return_value={},
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        _check_collection_compat(_settings(normalize_size=512), ignore_mismatch=False)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "[ERROR]" in err
    assert "Embedding configuration mismatch" in err
    assert "normalize_size: stored=224  current=512" in err
    assert "--ignore-config-mismatch" in err


def test_mismatch_with_override_warns_and_returns(capsys):
    """--ignore-config-mismatch must downgrade the failure to a warning and
    let the server start, so investigators can read a known-incompatible
    collection without re-indexing."""
    fake_client = MagicMock()
    fake_client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="sfn")]
    )
    fake_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors={"dino": object()}))
    )

    with (
        patch("scalar_forensic.web.app.QdrantClient", return_value=fake_client),
        patch(
            "scalar_forensic.safeguards.check_collection_compat",
            return_value=["[dino] normalize_size: stored=224  current=512"],
        ),
        patch(
            "scalar_forensic.safeguards.expected_model_hashes_from_settings",
            return_value={},
        ),
    ):
        # Must NOT raise SystemExit.
        _check_collection_compat(_settings(normalize_size=512), ignore_mismatch=True)

    err = capsys.readouterr().err
    assert "[WARN]" in err
    assert "--ignore-config-mismatch" in err
    assert "normalize_size: stored=224  current=512" in err
    # Still warns about the consequences — silently wrong scores.
    assert "silently meaningless" in err


def test_fresh_install_does_not_block(capsys):
    """Collection absent from Qdrant → return without checking anything."""
    fake_client = MagicMock()
    fake_client.get_collections.return_value = SimpleNamespace(collections=[])

    with patch("scalar_forensic.web.app.QdrantClient", return_value=fake_client):
        _check_collection_compat(_settings(), ignore_mismatch=False)

    err = capsys.readouterr().err
    assert "[ERROR]" not in err


def test_qdrant_unreachable_warns_and_returns(capsys):
    """A connectivity error must log a warning and return — never silently
    treated as 'fresh install', and never as a hard failure."""
    fake_client = MagicMock()
    fake_client.get_collections.side_effect = ConnectionError("nope")

    with patch("scalar_forensic.web.app.QdrantClient", return_value=fake_client):
        _check_collection_compat(_settings(), ignore_mismatch=False)

    err = capsys.readouterr().err
    assert "[WARN]" in err
    assert "unreachable" in err.lower()
