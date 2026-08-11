"""Helpers shared by more than one web route module."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from scalar_forensic.config import Settings


def _check_allowed_path(p: Path) -> None:
    """Raise 403 unless *p* is under an allowed root (input_dir or frame_store_dir).

    File-serving endpoints require at least one configured root.  When neither
    is set we fail closed rather than serving arbitrary host paths.
    """
    settings = Settings()
    allowed_roots: list[Path] = []
    if settings.input_dir is not None:
        allowed_roots.append(settings.input_dir.resolve())
    if settings.frame_store_dir is not None:
        allowed_roots.append(settings.frame_store_dir.resolve())
    if not allowed_roots:
        raise HTTPException(
            status_code=403,
            detail=(
                "File serving is disabled: neither SFN_INPUT_DIR nor "
                "SFN_FRAME_STORE_DIR is configured"
            ),
        )
    for root in allowed_roots:
        try:
            p.relative_to(root)
            return
        except ValueError:
            continue
    raise HTTPException(status_code=403, detail="Path is outside the allowed directories")
