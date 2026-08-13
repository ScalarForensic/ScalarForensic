"""Tests for the CLI ingestion run-progress display (bar + rate + ETA).

The display is an operator-facing progress readout for campaign runs, kept by
operator ruling (2026-08-13). These tests pin two things:

1. the mechanics — bar rendering, EWMA rate, ETA extrapolation;
2. the *honesty* of what is shown: the estimator publishes no uncertainty band
   and claims no calibration, per the project's no-uncalibrated-numbers rule.
"""

from __future__ import annotations

import inspect

from scalar_forensic.cli import _fmt_duration, _progress_bar, _RateTracker

# ---------------------------------------------------------------------------
# _progress_bar
# ---------------------------------------------------------------------------


def test_bar_is_empty_at_zero_and_full_at_hundred():
    assert _progress_bar(0.0, width=10) == "░" * 10
    assert _progress_bar(100.0, width=10) == "█" * 10


def test_bar_fills_proportionally():
    assert _progress_bar(50.0, width=10) == "█" * 5 + "░" * 5
    assert _progress_bar(30.0, width=10) == "█" * 3 + "░" * 7


def test_bar_clamps_out_of_range_percentages():
    # A percentage can exceed 100 when the frame total is only an estimate.
    assert _progress_bar(140.0, width=8) == "█" * 8
    assert _progress_bar(-5.0, width=8) == "░" * 8


def test_bar_keeps_its_width_at_every_percentage():
    for pct in range(0, 101):
        assert len(_progress_bar(float(pct), width=28)) == 28


# ---------------------------------------------------------------------------
# _RateTracker
# ---------------------------------------------------------------------------


def test_rate_is_unknown_before_any_observation():
    t = _RateTracker()
    assert t.rate is None
    assert t.eta(1000) is None


def test_first_observation_seeds_the_rate_but_not_the_eta():
    t = _RateTracker()
    t.update(100, 2.0)  # 50 items/s
    assert t.rate == 50.0
    # One batch is not a rate worth extrapolating from.
    assert t.eta(1000) is None


def test_second_observation_enables_the_eta():
    t = _RateTracker()
    t.update(100, 2.0)  # 50 items/s
    t.update(100, 2.0)  # 50 items/s — EWMA stays at 50
    assert t.rate == 50.0
    assert t.eta(500) == 10.0


def test_ewma_blends_old_estimate_and_new_observation_in_half():
    t = _RateTracker()
    t.update(100, 1.0)  # 100 items/s
    t.update(200, 1.0)  # 200 items/s -> 100 + 0.5 * (200 - 100)
    assert t.rate == 150.0
    t.update(50, 1.0)  # 50 items/s  -> 150 + 0.5 * (50 - 150)
    assert t.rate == 100.0


def test_degenerate_observations_are_ignored():
    t = _RateTracker()
    t.update(0, 1.0)  # no items
    t.update(10, 0.0)  # no elapsed time
    t.update(10, -1.0)  # negative elapsed time
    assert t.rate is None
    t.update(10, 1.0)
    t.update(0, 5.0)  # must not disturb a good estimate, nor count as an update
    assert t.rate == 10.0
    assert t.eta(100) is None  # still only one real observation


def test_eta_is_remaining_over_the_current_rate():
    t = _RateTracker()
    t.update(10, 1.0)
    t.update(10, 1.0)  # 10 items/s
    assert t.eta(0) == 0.0
    assert t.eta(45) == 4.5


# ---------------------------------------------------------------------------
# Honesty of the displayed numbers
# ---------------------------------------------------------------------------


def test_tracker_publishes_no_uncertainty_band():
    """The removed Kalman version exposed √P/K and a ±1σ ETA band.

    Q and R were hand-picked constants, so that band described the constants,
    not the run. Nothing may re-expose it (face-cosine precedent).
    """
    t = _RateTracker()
    t.update(10, 1.0)
    t.update(10, 1.0)
    for banned in ("rate_std", "kalman_gain", "sigma", "variance", "stddev"):
        assert not hasattr(t, banned), f"{banned} is an uncalibrated number"
    # eta() returns a scalar, never a (value, spread) pair.
    assert isinstance(t.eta(100), float)


def test_tracker_documents_why_the_band_was_dropped():
    """The docstring may discuss calibration only to disclaim it.

    A reviewer reading this class must not find a calibration claim; they
    should find the reason the previous one was withdrawn.
    """
    doc = (inspect.getdoc(_RateTracker) or "").lower()
    assert "not a calibrated" in doc
    assert "hand-picked constants" in doc
    # No sentence asserting the estimate *is* calibrated or a confidence bound.
    assert "yields a calibrated" not in doc
    assert "confidence band" not in doc
    assert "confidence interval collides" in doc  # only as the disclaimer


def test_eta_is_labelled_as_a_current_rate_extrapolation():
    """The operator must read the ETA as an extrapolation, not a prediction."""
    src = inspect.getsource(__import__("scalar_forensic.cli", fromlist=["index"]).index)
    assert src.count("remaining at current rate") == 2  # image path + frame path
    assert "σ_η" not in src
    assert "√P" not in src


def test_progress_box_renders_with_the_pieces_the_operator_reads():
    t = _RateTracker()
    t.update(100, 1.0)
    t.update(100, 1.0)
    pct = 25.0
    line = (
        f"  [{_progress_bar(pct)}]  1,000 / 4,000  ({pct:.1f}%)\n"
        f"  {t.rate:.1f} img/s  ·  ~{_fmt_duration(t.eta(3000))} remaining at current rate"
    )
    assert "█" in line and "░" in line
    assert "1,000 / 4,000" in line
    assert "100.0 img/s" in line
    assert "~30s remaining at current rate" in line
