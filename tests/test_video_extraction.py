"""Tests for video frame extraction validation (scalar_forensic.video)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scalar_forensic.video import extract_frames

_NONEXISTENT = Path("/nonexistent/video.mp4")


# ---------------------------------------------------------------------------
# fps validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_fps", [0, -1, -0.001, -100.0])
def test_extract_frames_invalid_fps_raises_value_error(bad_fps):
    """fps <= 0 must raise ValueError before any file I/O."""
    with pytest.raises(ValueError, match="fps"):
        list(extract_frames(_NONEXISTENT, fps=bad_fps))


# ---------------------------------------------------------------------------
# max_frames validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_max", [-1, -100])
def test_extract_frames_negative_max_frames_raises_value_error(bad_max):
    """max_frames < 0 must raise ValueError before any file I/O."""
    with pytest.raises(ValueError, match="max_frames"):
        list(extract_frames(_NONEXISTENT, fps=1.0, max_frames=bad_max))


def test_extract_frames_zero_max_frames_does_not_raise():
    """max_frames=0 means 'no cap' — the guard must not reject it.

    A RuntimeError from file-open is expected, not a ValueError.
    """
    with pytest.raises(Exception) as exc_info:
        list(extract_frames(_NONEXISTENT, fps=1.0, max_frames=0))

    assert not isinstance(exc_info.value, ValueError), (
        "max_frames=0 is the 'no cap' sentinel and must not be rejected"
    )


def test_extract_frames_positive_fps_reaches_file_open():
    """With valid fps > 0 the guard passes and execution reaches file-open
    (which raises RuntimeError for a nonexistent path, not ValueError)."""
    with pytest.raises(Exception) as exc_info:
        list(extract_frames(_NONEXISTENT, fps=1.0))

    assert not isinstance(exc_info.value, ValueError)
