"""Tests for the indexer guard that refuses --reference when SFN_REFERENCE_COLLECTION
matches SFN_COLLECTION.

Indexing into the case collection with --reference would silently stamp
``is_reference=true`` onto already-existing case points (the indexer's trailing
``set_payload`` writes the ref tag to all batch points).  The CLI must refuse
the run before any model is loaded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from scalar_forensic.cli import index


def _settings(case: str, reference: str | None) -> MagicMock:
    s = MagicMock()
    s.collection = case
    s.reference_collection = reference
    s._env_file = None
    return s


def test_reference_equals_case_collection_aborts(tmp_path: Path):
    """--reference with SFN_REFERENCE_COLLECTION == SFN_COLLECTION must exit
    with code 1 before any indexing or model load happens."""
    with patch(
        "scalar_forensic.cli.Settings",
        return_value=_settings(case="sfn", reference="sfn"),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            index(
                input_dir=tmp_path,
                dino=True,
                sscd=False,
                report=None,
                allow_online=False,
                reference=True,
            )
    assert exc_info.value.exit_code == 1


def test_reference_unset_aborts(tmp_path: Path):
    """--reference without SFN_REFERENCE_COLLECTION configured must also fail."""
    with patch(
        "scalar_forensic.cli.Settings",
        return_value=_settings(case="sfn", reference=None),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            index(
                input_dir=tmp_path,
                dino=True,
                sscd=False,
                report=None,
                allow_online=False,
                reference=True,
            )
    assert exc_info.value.exit_code == 1
