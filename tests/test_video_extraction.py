"""Tests for video frame extraction validation (scalar_forensic.video)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scalar_forensic.video import extract_frame_at, extract_frames

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


# ---------------------------------------------------------------------------
# extract_frame_at: overshoot-seeking behaviour
# ---------------------------------------------------------------------------


def _make_av_frame(pts: int) -> MagicMock:
    frame = MagicMock()
    frame.pts = pts
    frame.to_image.return_value = MagicMock(name=f"image_pts{pts}")
    return frame


def _make_av_container(frames: list, time_base: float = 0.001, avg_rate: float = 30.0) -> MagicMock:
    stream = MagicMock()
    stream.time_base = time_base
    stream.average_rate = avg_rate

    container = MagicMock()
    container.streams.video = [stream]
    container.decode.return_value = iter(frames)
    return container


def test_extract_frame_at_returns_previous_frame_on_overshoot():
    """If decoding overshoots the target by more than one frame interval,
    extract_frame_at must return the last frame decoded before the overshoot."""
    # target = 1000 ms (1.0 s), time_base = 1 ms/unit, fps = 30
    # frame_dur ≈ 0.0333 s  →  threshold = 1.0 + 0.0333 ≈ 1.0333 s
    # prev_frame at 966 ms (0.966 s) is just before the detection window
    # overshot_frame at 1100 ms (1.1 s) exceeds the threshold → return prev
    prev_frame = _make_av_frame(pts=966)
    overshot_frame = _make_av_frame(pts=1100)
    container = _make_av_container([prev_frame, overshot_frame])

    with patch("av.open", return_value=container):
        result = extract_frame_at(Path("/fake/video.mp4"), timecode_ms=1000)

    assert result is prev_frame.to_image.return_value


def test_extract_frame_at_returns_exact_frame_when_within_range():
    """A frame within ±1 frame interval of the target must be returned directly."""
    # prev_frame at 966 ms is before the window; target_frame at 1000 ms is within it
    prev_frame = _make_av_frame(pts=966)
    target_frame = _make_av_frame(pts=1000)
    container = _make_av_container([prev_frame, target_frame])

    with patch("av.open", return_value=container):
        result = extract_frame_at(Path("/fake/video.mp4"), timecode_ms=1000)

    assert result is target_frame.to_image.return_value


def test_extract_frame_at_returns_none_for_unopenable_video():
    with patch("av.open", side_effect=RuntimeError("cannot open")):
        result = extract_frame_at(Path("/nonexistent/video.mp4"), timecode_ms=500)
    assert result is None
