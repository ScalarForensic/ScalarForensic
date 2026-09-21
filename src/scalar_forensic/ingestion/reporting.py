"""Stable operator CSV and terminal reports."""

import csv
from collections import Counter
from pathlib import Path

import typer

from .models import (
    _S_FAIL_EMB,
    _S_FAIL_PRE,
    _S_FAIL_READ,
    _S_INDEXED,
    _S_SKIP_DUP,
    _S_SKIP_FRAME_DUP,
    _S_SKIP_IDX,
    _S_UNSUPPORTED,
    _FileRecord,
)


def _write_csv(records: dict[Path, "_FileRecord"], csv_path: Path) -> None:
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "processed", "reason", "md5", "sha256", "is_video_frame"])
            for rec in sorted(records.values(), key=lambda r: str(r.path)):
                processed = "yes" if rec.status == _S_INDEXED else "no"
                writer.writerow(
                    [str(rec.path), processed, rec.reason, rec.md5, rec.sha256, rec.is_frame]
                )
    except OSError as exc:
        typer.echo(f"[ERROR] Could not write CSV report to {csv_path}: {exc}", err=True)


def _print_summary(
    records: dict[Path, "_FileRecord"],
    resolved_input: Path,
    csv_path: Path,
    specs: list,
    indexed_counts: list[int],
    skipped_counts: list[int],
    failed_counts: list[int],
    n_images: int = 0,
    n_videos: int = 0,
    n_video_frames_indexed: int = 0,
    n_video_frames_skipped: int = 0,
) -> None:
    counts = Counter(r.status for r in records.values())
    total = len(records)
    n_image = n_images

    # Per-model breakdown
    typer.echo("\nDone.")
    for spec_idx, (embedder, _, _) in enumerate(specs):
        typer.echo(
            f"  [{type(embedder).__name__}]  indexed={indexed_counts[spec_idx]}"
            f"  skipped={skipped_counts[spec_idx]}  embed_failed={failed_counts[spec_idx]}"
        )

    # Overall file summary table
    sep = "─" * 52
    typer.echo(f"\nIngestion summary  ({resolved_input})")
    typer.echo(sep)
    rows: list[tuple[str, int | None]] = [
        ("Total files found", total),
        ("  image files", n_image),
        ("  video files", n_videos),
        ("  non-image files (unsupported)", counts[_S_UNSUPPORTED]),
        ("", None),
        ("Indexed (new)", counts[_S_INDEXED]),
        ("Skipped — already indexed", counts[_S_SKIP_IDX]),
        ("Skipped — duplicate in batch", counts[_S_SKIP_DUP]),
        ("Skipped — all frames dup in run", counts[_S_SKIP_FRAME_DUP]),
        ("Failed  — read error", counts[_S_FAIL_READ]),
        ("Failed  — preprocessing", counts[_S_FAIL_PRE]),
        ("Failed  — embedding", counts[_S_FAIL_EMB]),
    ]
    if n_videos > 0:
        rows += [
            ("", None),
            ("Video frames indexed (new)", n_video_frames_indexed),
            ("Video frames skipped (dup)", n_video_frames_skipped),
        ]
    for label, value in rows:
        if value is None:
            typer.echo("")
        else:
            typer.echo(f"  {label:<36} {value:>6,}")
    typer.echo(sep)
    typer.echo(f"  CSV report → {csv_path}")


def _fmt_rate(count: int, seconds: float, unit: str) -> str:
    if seconds <= 0:
        return "—"
    return f"{count / seconds:.1f} {unit}/s"


def _fmt_mbps(bytes_total: int, seconds: float) -> str:
    if seconds <= 0:
        return "—"
    return f"{bytes_total / 1e6 / seconds:.1f} MB/s"


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _progress_bar(pct: float, width: int = 28) -> str:
    """Unicode block-element progress bar."""
    filled = round(width * min(max(pct, 0.0), 100.0) / 100)
    return "█" * filled + "░" * (width - filled)


class _RateTracker:
    """Smoothed throughput estimate driving the run-progress line. Θ(1) per update.

        x̂ₜ = x̂ₜ₋₁ + α(zₜ − x̂ₜ₋₁),  α = ½,  x̂₁ = z₁

    where zₜ = items / elapsed_s for one batch (or one block of frames).

    The previous version of this class was a scalar Kalman filter that also
    published a ±1σ band around the ETA. That band was not a calibrated
    uncertainty: Q and R were hand-picked constants, so at steady state the
    filter is exactly this α = ½ EWMA and √P is a function of those constants,
    not of the run. Reporting it as a confidence interval collides with the
    project rule against displaying uncalibrated numbers (precedent: the raw
    face cosine, `docs/specs/face-pipeline.md` §10), so the band is gone and
    the ETA is labelled for what it is — an extrapolation at the current rate.
    """

    _ALPHA: float = 0.5  # smoothing factor: half old estimate, half new observation

    def __init__(self) -> None:
        self._x: float | None = None  # x̂: current rate estimate (items/s)
        self._n: int = 0  # number of updates applied

    def update(self, n_items: int, elapsed_s: float) -> None:
        """Incorporate one batch/block observation."""
        if elapsed_s <= 0 or n_items <= 0:
            return
        z = n_items / elapsed_s
        self._n += 1
        self._x = z if self._x is None else self._x + self._ALPHA * (z - self._x)

    @property
    def rate(self) -> float | None:
        """x̂ₜ — current smoothed rate (items/s), or None before the first update."""
        return self._x

    def eta(self, remaining: int) -> float | None:
        """Seconds to finish `remaining` items at the current rate.

        None until two observations exist — a single batch is not a rate the
        display should extrapolate from. No error bound is returned: this is a
        straight-line extrapolation, not a prediction with a known spread.
        """
        if self._x is None or self._x <= 0 or self._n < 2:
            return None
        return remaining / self._x


def _stale_path_listing(stale_points: "list[dict]", limit: int = 20) -> "list[str]":
    """Indented lines naming the files a stale-observation set belongs to.

    One line per distinct source file, most-affected first, so the operator
    deciding whether to delete sees *which* evidence is involved rather than a
    bare total.  Long listings are capped at *limit* files and the remainder is
    reported as a count.
    """
    counts: dict[str, int] = {}
    for sp in stale_points:
        src = sp.get("source_path") or "(source file unknown)"
        counts[src] = counts.get(src, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [f"    {n:>4} × {src}" for src, n in ordered[:limit]]
    if len(ordered) > limit:
        lines.append(f"    … and {len(ordered) - limit:,} more file(s)")
    return lines
