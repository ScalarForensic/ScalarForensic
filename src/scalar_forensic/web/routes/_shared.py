"""Helpers shared by more than one web route module."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from scalar_forensic.config import Settings
from scalar_forensic.video import VIDEO_EXTENSIONS


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


def _resolve_video_path(path: str) -> Path:
    """Validate *path* as a servable video file and return it resolved.

    Shared by ``/api/video-frame`` (the indexing side) and the three playback
    routes, so it lives here beside :func:`_check_allowed_path`: by the same
    one-implementation rule, a security control gets exactly one definition.

    Same allowed-root check as /api/hit-image (``_check_allowed_path``); the set
    of acceptable extensions is the scanner's own ``VIDEO_EXTENSIONS``, never a
    hand-copied list, so what can be played can never drift from what can be
    indexed.
    """
    raw = Path(path)
    if not raw.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path")
    p = raw.resolve()
    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a video file")
    _check_allowed_path(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return p
