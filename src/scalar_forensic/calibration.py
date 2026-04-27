"""Batch-size auto-calibration via Michaelis-Menten throughput model.

Given an embedder and a directory of representative sample images, this module
probes batch sizes exponentially and fits the hyperbolic saturation model::

    T(b) = T_max · b / (b + K)

to find the 95%-efficiency operating point b* = K · η / (1 − η).

Fitting is done via the Lineweaver-Burk (double-reciprocal) linearisation::

    b / T(b)  =  (1 / T_max) · b  +  K / T_max

which reduces to a two-parameter OLS problem solvable in pure Python.

The result is persisted to ``data/sfn_batch_cache.json`` so subsequent runs
(CLI and web) avoid re-calibrating on every startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from scalar_forensic.embedder import _SHARED_CAP

if TYPE_CHECKING:
    from scalar_forensic.embedder import AnyEmbedder

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_CACHE_FILE = Path("data/sfn_batch_cache.json")
_EFFICIENCY_TARGET = 0.95  # fraction of T_max to accept as "optimal"
_MAX_BATCH = 256  # hard ceiling — 512 triggers GPU hangs on some ROCm cards
_MIN_WARMUP_IMAGES = 16  # warm-up images — fixed, never scales with probe_b
_MAX_MEASURE_IMAGES = 32  # measurement window cap — keeps each probe fast;
#   the shuffle ensures these 32 are representative
_PROBE_REPEATS = 3  # timed runs per batch; median is reported — suppresses noise
_MIN_GAIN = 0.03  # converged: positive gain < 3 % → saturation reached
_PEAK_FRAC = 0.70  # post-peak guard: stop after this many consecutive
_POST_PEAK_DROPS = 3  #   measurements below _PEAK_FRAC × best seen so far
_PLATEAU_WINDOW = 4  # band-of-oscillation check: last N measurements…
_PLATEAU_CV = 0.10  #   …with coefficient of variation below this → plateau
_BAR_WIDTH = 30  # characters for the throughput bar


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def load_cached_batch_size(cache_file: Path = _CACHE_FILE) -> int | None:
    """Return the cached batch size, or *None* if no valid cache exists."""
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        v = int(data["batch_size"])
        return v if v > 0 else None
    except (KeyError, ValueError, json.JSONDecodeError, OSError):
        return None


def save_cached_batch_size(
    batch_size: int,
    *,
    device: str = "",
    throughput_img_per_s: float = 0.0,
    t_max: float = 0.0,
    k_half: float = 0.0,
    cache_file: Path = _CACHE_FILE,
) -> None:
    """Persist the calibrated batch size and model parameters to *cache_file*."""
    import os
    import tempfile
    from datetime import datetime

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_size": batch_size,
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "throughput_img_per_s": round(throughput_img_per_s, 2),
        "model_t_max": round(t_max, 2),
        "model_k_half": round(k_half, 2),
    }
    fd, tmp_path = tempfile.mkstemp(dir=cache_file.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2))
        Path(tmp_path).replace(cache_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Internal probe helper
# ---------------------------------------------------------------------------


def _run_batches(
    embedders: list[AnyEmbedder],
    images: list[bytes],
    batch_size: int,
    *,
    cap: int = _SHARED_CAP,
) -> None:
    """Pass *images* through the full combined pipeline in batches of *batch_size*.

    Mirrors the real ingestion pipeline: hash → preprocess → (normalize →
    embed) × len(embedders).  All embedders share a single preprocess_batch
    call per chunk, matching what the indexer does.  Running them together
    also captures the combined VRAM pressure that a single-embedder probe
    would underestimate.

    *cap* should match the effective cap used by the real indexing pipeline
    (``max(_SSCD_SCALE, settings.normalize_size)``).  The default is
    ``_SHARED_CAP`` (331 px) for backward compatibility, but callers should
    pass the real cap so that calibration accurately represents production
    throughput when ``normalize_size > 331``.
    """
    from scalar_forensic.embedder import hash_bytes, hash_bytes_md5, preprocess_batch

    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        for data in chunk:
            hash_bytes(data)
            hash_bytes_md5(data)
        pils = [r for r in preprocess_batch(chunk, cap=cap) if not isinstance(r, Exception)]
        if pils:
            for embedder in embedders:
                norm = embedder.normalize_batch_bytes(pils)
                embedder.embed_images(norm)


def _probe(
    embedders: list[AnyEmbedder],
    warmup_bytes: list[bytes],
    measure_bytes: list[bytes],
    batch_size: int,
    *,
    repeats: int = 1,
    cap: int = _SHARED_CAP,
) -> float:
    """Warm up once then time *measure_bytes* for *repeats* runs; return median img/s.

    Taking the median of multiple runs suppresses timing noise from GPU
    scheduler jitter and variable kernel compile times — especially important
    at larger batch sizes where a single lucky/unlucky run can skew the probe
    by 20%+.

    Keeping warmup and measure windows separate (and always the same
    *measure_bytes* across all probes) ensures every batch size is measured
    on an identical image distribution — critical when the sample set contains
    images of varying sizes.
    """
    import statistics

    _run_batches(embedders, warmup_bytes, batch_size, cap=cap)  # warm-up (discarded)
    tps: list[float] = []
    for _ in range(repeats):
        t0 = perf_counter()
        _run_batches(embedders, measure_bytes, batch_size, cap=cap)
        elapsed = perf_counter() - t0
        tps.append(len(measure_bytes) / elapsed if elapsed > 0 else 0.0)
    return statistics.median(tps)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def calibrate(
    embedders: AnyEmbedder | list[AnyEmbedder],
    sample_dir: Path,
    *,
    cache_file: Path | None = _CACHE_FILE,
    cap: int = _SHARED_CAP,
) -> int:
    """Probe batch sizes, fit the saturation model, display results, and return
    the optimal batch size.

    The returned value is also persisted to *cache_file* so future runs skip
    calibration and load the cached value directly.

    Parameters
    ----------
    embedders:
        One or more loaded embedders.  When multiple are given every probe
        runs all of them on the same preprocessed batch — matching the real
        indexing pipeline where preprocessing is shared and each model runs
        sequentially on the same images.  This captures the true combined VRAM
        pressure instead of calibrating each model in isolation.
    sample_dir:
        Directory containing representative JPEG/PNG images for probing.
        Images are tiled in memory — no repeated disk I/O per probe.
    cache_file:
        Destination for the JSON result cache.  Pass ``None`` to skip
        writing the cache.
    cap:
        Short-side pixel cap passed to ``preprocess_batch`` for each probe.
        Should match the effective cap used by the real indexing pipeline:
        ``max(_SSCD_SCALE, settings.normalize_size)``.  Defaults to
        ``_SHARED_CAP`` (331 px).  When ``normalize_size > 331``, passing the
        correct cap ensures calibration measures throughput at production image
        sizes rather than underestimating it on smaller inputs.
    """
    import typer

    emb_list: list[AnyEmbedder] = (
        [embedders] if not isinstance(embedders, list) else embedders  # type: ignore[list-item]
    )
    primary = emb_list[0]

    # ── Load sample images into memory (once) ─────────────────────────────
    image_paths: list[Path] = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(sorted(sample_dir.glob(pat)))

    raw_bytes: list[bytes] = []
    for p in image_paths:
        try:
            raw_bytes.append(p.read_bytes())
        except OSError as exc:
            typer.echo(f"[WARN] Skipping unreadable sample image {p.name}: {exc}", err=True)

    if not raw_bytes:
        typer.echo(
            f"[WARN] No readable images in {sample_dir} — using default batch_size=32",
            err=True,
        )
        return 32

    n_samples = len(raw_bytes)

    # Shuffle once with a fixed seed so every probe sees the same image
    # distribution.  Without this, sorted filenames give small images to small
    # batch sizes and large images to large ones, making throughput numbers
    # incomparable across probes.
    import random as _random

    _random.Random(0).shuffle(raw_bytes)

    # ── Header ────────────────────────────────────────────────────────────
    sep = "─" * 62
    model_names = "  +  ".join(type(e).__name__ for e in emb_list)
    typer.echo("")
    typer.echo("SFN_BATCH_SIZE unset — calibrating optimal batch size")
    typer.echo("  T(b) = T_max·b / (b+K)  [Michaelis-Menten saturation]")
    typer.echo(f"  sample: {sample_dir}  ({n_samples} images)  ·  device: {primary.device}")
    typer.echo(f"  pipeline: {model_names}")
    typer.echo(sep)

    # ── Exponential probe ─────────────────────────────────────────────────
    measurements: list[tuple[int, float]] = []
    prev_tp: float | None = None

    probe_b = 1
    while probe_b <= _MAX_BATCH:
        # Measurement window: capped at _MAX_MEASURE_IMAGES for speed, but
        # always at least probe_b so large batch sizes get one full batch.
        # The fixed shuffle above ensures any subset is representative.
        measure_count = max(min(n_samples, _MAX_MEASURE_IMAGES), probe_b)
        reps_m = (measure_count // n_samples) + 1
        measure_bytes = (raw_bytes * reps_m)[:measure_count]

        # Warm-up: fixed size — does NOT scale with probe_b.
        warmup_bytes = (raw_bytes * 2)[:_MIN_WARMUP_IMAGES]

        try:
            tp = _probe(
                emb_list, warmup_bytes, measure_bytes, probe_b, repeats=_PROBE_REPEATS, cap=cap
            )
        except Exception:  # noqa: BLE001 — catches CUDA/ROCm OOM and JIT errors
            typer.echo(f"  batch={probe_b:>4}  {'':>{_BAR_WIDTH + 2}}  out of memory — stopped")
            break

        if tp <= 0:
            typer.echo(
                f"  batch={probe_b:>4}  {'':>{_BAR_WIDTH + 2}}  non-positive throughput — stopped"
            )
            break

        measurements.append((probe_b, tp))

        max_tp = max(t for _, t in measurements)
        bar_fill = round(_BAR_WIDTH * tp / max_tp)
        bar = "█" * bar_fill + "░" * (_BAR_WIDTH - bar_fill)
        if prev_tp is not None:
            gain_pct = (tp / prev_tp - 1) * 100
            gain_str = f"  Δ{gain_pct:+6.1f}%"
        else:
            gain_str = ""

        # ── Stopping conditions ───────────────────────────────────────────
        # 1. Saturation: small *positive* gain — throughput has plateaued.
        if prev_tp is not None and 0.0 <= (tp - prev_tp) / prev_tp < _MIN_GAIN:
            typer.echo(
                f"  batch={probe_b:>4}  {bar}  {tp:7.1f} img/s{gain_str}"
                f"  ← Δ < {_MIN_GAIN * 100:.0f}%, converged"
            )
            break

        # 2. Post-peak decay: several consecutive measurements well below the
        #    running best (covers noise spikes and genuine GPU memory pressure).
        if len(measurements) >= _POST_PEAK_DROPS:
            recent_below = sum(
                1 for _, t in measurements[-_POST_PEAK_DROPS:] if t < max_tp * _PEAK_FRAC
            )
            if recent_below == _POST_PEAK_DROPS:
                typer.echo(
                    f"  batch={probe_b:>4}  {bar}  {tp:7.1f} img/s{gain_str}"
                    f"  ← {_POST_PEAK_DROPS}× below peak, stopped"
                )
                break

        # 3. Stable plateau: last _PLATEAU_WINDOW measurements oscillate within
        #    a tight band — throughput has settled regardless of trend direction.
        if len(measurements) >= _PLATEAU_WINDOW + 1:
            recent = [t for _, t in measurements[-_PLATEAU_WINDOW:]]
            mean_tp = sum(recent) / len(recent)
            if mean_tp > 0:
                variance = sum((t - mean_tp) ** 2 for t in recent) / len(recent)
                cv = variance**0.5 / mean_tp
                if cv < _PLATEAU_CV:
                    typer.echo(
                        f"  batch={probe_b:>4}  {bar}  {tp:7.1f} img/s{gain_str}"
                        f"  ← plateau (CV={cv:.2f})"
                    )
                    break

        typer.echo(f"  batch={probe_b:>4}  {bar}  {tp:7.1f} img/s{gain_str}")
        prev_tp = tp
        probe_b *= 2

    # ── Fit Michaelis-Menten via Lineweaver-Burk OLS ──────────────────────
    t_max_fit = 0.0
    k_half_fit = 0.0
    b_star = 0.0
    optimal = 32

    if len(measurements) >= 2:
        # Lineweaver-Burk: y_i = b_i / T_i,  x_i = b_i
        # y = a·x + c   →   a = 1/T_max,  c = K/T_max
        xs = [float(b) for b, _ in measurements]
        ys = [b / tp for b, tp in measurements]
        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sxx = sum(x * x for x in xs)
        denom = n * sxx - sx * sx
        if denom != 0:
            a = (n * sxy - sx * sy) / denom  # 1 / T_max
            c = (sy - a * sx) / n  # K / T_max
            if a > 0:
                t_max_fit = 1.0 / a
                k_half_fit = max(c / a, 0.0)
                eta = _EFFICIENCY_TARGET
                b_star = k_half_fit * eta / (1.0 - eta)

    if b_star >= 1.0:
        # Largest power-of-2 ≤ b*, capped to the largest successfully probed
        # batch size so we never return an untested value that could OOM.
        max_probed = max(b for b, _ in measurements)
        p2 = 1
        while p2 * 2 <= b_star and p2 * 2 <= max_probed:
            p2 *= 2
        optimal = p2
    elif measurements:
        optimal = max(measurements, key=lambda m: m[1])[0]

    best_tp = max(tp for _, tp in measurements) if measurements else 0.0

    # ── Summary ────────────────────────────────────────────────────────────
    typer.echo(sep)
    if t_max_fit > 0 and b_star >= 1.0:
        typer.echo(f"  Fitted:   T_max = {t_max_fit:.1f} img/s  ·  K_½ = {k_half_fit:.1f}")
        typer.echo(
            f"  Target:   {int(_EFFICIENCY_TARGET * 100)}% of T_max"
            f"  →  b* = {b_star:.1f}"
            f"  →  largest 2ⁿ ≤ b*: {optimal}"
        )
    else:
        typer.echo(f"  Best measured: {best_tp:.1f} img/s at batch={optimal}")
    typer.echo(f"  ✓ Optimal batch size: {optimal}")

    if cache_file is not None:
        save_cached_batch_size(
            optimal,
            device=primary.device,
            throughput_img_per_s=best_tp,
            t_max=t_max_fit,
            k_half=k_half_fit,
            cache_file=cache_file,
        )
        typer.echo(f"  Cached → {cache_file}")
    typer.echo(sep)
    typer.echo("")

    return optimal
