"""Per-run ingestion manifest (runbook work item, 2026-08-13).

Written at run start — after discovery and the upfront hash pass, before any
embedding or upsert — next to the CSV report as ``<report_stem>.manifest.json``.
The location is therefore fully derived from configuration (``SFN_REPORT_DIR``
or ``--report``), never hardcoded.

The manifest records what the run was *configured* to do: a full Settings
snapshot, the identity and hash of every model involved, and the discovered
input file list — so a finished collection can be traced back to the exact
configuration and inputs that produced it.  Stdlib-only by design.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scalar_forensic.config import Settings

MANIFEST_VERSION = 1

# Credential values must never land in an evidence-directory artefact.  The
# *presence* of a credential is still recorded (redaction marker vs None).
_SECRET_SETTINGS = frozenset({"qdrant_api_key", "embedding_api_key"})
_REDACTED = "***redacted***"


def manifest_path_for(csv_path: Path) -> Path:
    """The manifest path paired with a CSV report path (same dir, same stem)."""
    return csv_path.with_name(csv_path.stem + ".manifest.json")


def settings_snapshot(settings: Settings) -> dict[str, Any]:
    """JSON-safe snapshot of every public Settings field, secrets redacted."""
    snap: dict[str, Any] = {}
    for key, value in vars(settings).items():
        if key.startswith("_"):
            continue
        if key in _SECRET_SETTINGS:
            snap[key] = _REDACTED if value else None
        elif isinstance(value, Path):
            snap[key] = str(value)
        else:
            snap[key] = value
    return snap


def input_file_entry(path: Path, sha256: str | None = None) -> dict[str, Any]:
    """One discovered-input entry: path + size, plus sha256 when already computed.

    Size is best-effort: a file that vanished or cannot be stat'ed since
    discovery is still listed (that fact is itself worth recording), with
    ``size: null``.
    """
    entry: dict[str, Any] = {"path": str(path)}
    try:
        entry["size"] = path.stat().st_size
    except OSError:
        entry["size"] = None
    if sha256:
        entry["sha256"] = sha256
    return entry


def write_run_manifest(
    csv_path: Path,
    *,
    settings: Settings,
    target_collection: str,
    input_root: Path,
    files: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
) -> Path:
    """Write the per-run manifest next to *csv_path* and return its path.

    *models* maps modality → identity dict: for image embedders
    ``{model_name, model_hash, embedding_dim}`` keyed by vector name; for the
    face modality the full PipelineConfig payload (detector/embedder/manifest
    hashes plus ``pipeline_config_hash``) under the key ``"faces"``.
    """
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "collection": target_collection,
        "report_csv": str(csv_path),
        "settings": settings_snapshot(settings),
        "models": models,
        "input": {
            "root": str(input_root),
            "file_count": len(files),
            "files": files,
        },
    }
    out = manifest_path_for(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out
