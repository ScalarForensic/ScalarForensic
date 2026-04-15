"""CLI entry point for ScalarForensic."""

import csv
import os
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import batched
from pathlib import Path
from time import perf_counter

import typer
from PIL import Image

from scalar_forensic.config import Settings
from scalar_forensic.embedder import (
    AnyEmbedder,
    ExifInfo,
    extract_exif,
    get_library_versions,
    hash_bytes,
    hash_bytes_md5,
    hash_file,
    hash_file_md5,
    load_embedder,
    preprocess_batch,
    preprocess_pil_batch,
)
from scalar_forensic.indexer import Indexer
from scalar_forensic.scanner import _HEIF_AVAILABLE, scan_all_files
from scalar_forensic.video import (
    extract_frames,
    get_pyav_version,
    make_virtual_path,
)

# ── file-level status codes ──────────────────────────────────────────────────
_S_INDEXED = "indexed"
_S_SKIP_DUP = "skipped_dup_batch"
_S_SKIP_IDX = "skipped_indexed"
_S_FAIL_READ = "failed_read"
_S_FAIL_PRE = "failed_preprocessing"
_S_FAIL_EMB = "failed_embedding"
_S_UNSUPPORTED = "unsupported"


@dataclass
class _FileRecord:
    path: Path
    status: str = "pending"
    reason: str = ""
    md5: str = ""
    sha256: str = ""


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


class _ETATracker:
    """Kalman-filtered throughput estimator — Θ(1) time and space per update.

    State space: x ∈ ℝ₊ (throughput, img/s), A = H = 1 (scalar random walk):

        Predict:  x̂ₜ⁻  = x̂ₜ₋₁                         Φ := 1
                  Pₜ⁻  = Pₜ₋₁ + Q,    Q ∈ ℝ₊

        Update:   Kₜ   = Pₜ⁻ (Pₜ⁻ + R)⁻¹               Kₜ ∈ (0, 1)
                  x̂ₜ   = x̂ₜ⁻ + Kₜ(zₜ − x̂ₜ⁻)
                  Pₜ   = (1 − Kₜ)Pₜ⁻                   (Joseph form, H = 1)

        DARE (t → ∞, unique ℝ₊ root of P∞² + QP∞ − QR = 0):
                  P∞   = ½(√(Q² + 4QR) − Q)
                  K∞   = Q / (Q + √(Q² + 4QR))
          ∀ Q = R/2 :  K∞ = ½                           equal-weight equilibrium ✓

        δ-method (first-order error propagation, η̂ := N_rem / x̂):
                  Var[η̂] ≈ (∂η/∂x)²|_{x=x̂} · Pₜ
                           = (N_rem · x̂⁻²)² · Pₜ
                  σ_η    = N_rem · √Pₜ / x̂²            ±1σ confidence band

        Contrast with a rolling-window mean (O(w) space, no variance) or a
        simple EWMA (O(1) space, but no principled variance propagation):
        the Kalman formulation is optimal under Gaussian assumptions and
        yields a calibrated uncertainty estimate at no extra cost.
    """

    _Q: float = 50.0  # process-noise variance  (img/s)²
    _R: float = 100.0  # measurement-noise variance (img/s)²

    def __init__(self) -> None:
        self._x: float | None = None  # x̂: current rate estimate (img/s)
        self._P: float = 1e8  # P: estimate error variance (diffuse prior)
        self._k: float = 1.0  # Kₜ: Kalman gain at last update (1 = full trust)
        self._n: int = 0  # number of updates applied

    def update(self, n_imgs: int, elapsed_s: float) -> None:
        """Incorporate a new batch observation.  Θ(1) — scalar predict-update cycle."""
        if elapsed_s <= 0 or n_imgs <= 0:
            return
        z = n_imgs / elapsed_s  # zₜ: observed throughput
        self._n += 1
        if self._x is None:
            self._x = z
            self._P = self._R  # P₁ = R: certainty = measurement quality
            return
        p_pred = self._P + self._Q  # Pₜ⁻ = Pₜ₋₁ + Q
        k = p_pred / (p_pred + self._R)  # Kₜ = Pₜ⁻(Pₜ⁻ + R)⁻¹
        self._x = self._x + k * (z - self._x)  # x̂ₜ = x̂ₜ⁻ + Kₜ(zₜ − x̂ₜ⁻)
        self._P = (1.0 - k) * p_pred  # Pₜ = (1 − Kₜ)Pₜ⁻
        self._k = k

    @property
    def rate(self) -> float | None:
        """x̂ₜ — current optimal rate estimate (img/s)."""
        return self._x

    @property
    def rate_std(self) -> float:
        """√Pₜ — 1σ uncertainty on the rate estimate (img/s)."""
        return self._P**0.5

    @property
    def kalman_gain(self) -> float:
        """Kₜ — Kalman gain at the most recent update.
        Converges toward K∞ = ½ at steady state (Q = R/2).
        """
        return self._k

    def eta(self, remaining: int) -> tuple[float, float] | None:
        """Return (η̂, σ_η) in seconds, or None if not enough data.

        Θ(1) — closed-form δ-method propagation:
            η̂   = N_rem / x̂
            σ_η = N_rem · √Pₜ / x̂²
        """
        if self._x is None or self._x <= 0 or self._n < 2:
            return None
        eta_s = remaining / self._x  # η̂
        sigma_s = remaining * self.rate_std / self._x**2  # σ_η
        return eta_s, sigma_s


def _apply_dedup(
    pairs: list[tuple[Path, str]],
    indexer: Indexer,
    settings: Settings,
) -> list[tuple[Path, str]]:
    """Filter already-indexed images according to the configured dedup mode."""
    mode = settings.duplicate_check_mode
    hashes = [h for _, h in pairs]
    str_paths = [str(p.resolve()) for p, _ in pairs]

    indexed_hashes: set[str] = set()
    indexed_paths: set[str] = set()

    if mode in ("hash", "both"):
        indexed_hashes = indexer.get_indexed_hashes(hashes)
    if mode in ("filepath", "both"):
        indexed_paths = indexer.get_indexed_paths(str_paths)

    return [
        (p, h)
        for p, h in pairs
        if h not in indexed_hashes and str(p.resolve()) not in indexed_paths
    ]


def _write_thumbnail(img: "Image.Image", dest: Path, size: int) -> None:
    """Save a thumbnail JPEG of *img* at *dest* (size×size, aspect-ratio preserved)."""
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    if thumb.mode not in ("RGB", "L"):
        thumb = thumb.convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(dest, format="JPEG", quality=85, optimize=True)


def _write_csv(records: dict[Path, "_FileRecord"], csv_path: Path) -> None:
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "processed", "reason", "md5", "sha256"])
            for rec in sorted(records.values(), key=lambda r: str(r.path)):
                processed = "yes" if rec.status == _S_INDEXED else "no"
                writer.writerow([str(rec.path), processed, rec.reason, rec.md5, rec.sha256])
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


def _frame_stream(
    video_paths: list[Path],
    records: dict[Path, "_FileRecord"],
    settings: "Settings",
    pyav_version: str,
    pre_hashes: dict[Path, str] | None = None,
) -> Iterator[tuple[Path, str, "Image.Image", dict, Path]]:
    """Yield ``(virtual_path, frame_hash, pil, vmeta, source_video_path)`` for every
    frame across all videos, streaming across video boundaries so the caller can
    batch frames from multiple videos into a single GPU batch with ``batched()``.

    When *pre_hashes* is provided (populated by the video pre-check pass) the
    per-video SHA-256 is taken from there instead of being recomputed here.
    """
    for video_path in video_paths:
        rec = records[video_path]
        if pre_hashes is not None and video_path in pre_hashes:
            video_hash = pre_hashes[video_path]
        else:
            try:
                video_hash = hash_file(video_path)
                video_hash_md5 = hash_file_md5(video_path)
            except OSError as exc:
                typer.echo(f"[WARN] Cannot read video {video_path.name}: {exc}", err=True)
                rec.status = _S_FAIL_READ
                rec.reason = f"read error: {exc}"
                continue
            rec.sha256 = video_hash
            rec.md5 = video_hash_md5
        video_abs_path = str(video_path.resolve())
        try:
            for frame in extract_frames(
                video_path, fps=settings.video_fps, max_frames=settings.video_max_frames
            ):
                virtual_path = Path(make_virtual_path(video_path, frame))
                vmeta = {
                    "video_hash": video_hash,
                    "video_path": video_abs_path,
                    "frame_timecode_ms": frame.timecode_ms,
                    "frame_index": frame.frame_index,
                    "extraction_fps": settings.video_fps,
                    "max_frames_cap": settings.video_max_frames,
                    "pyav_version": pyav_version,
                }
                yield virtual_path, frame.frame_hash, frame.image, vmeta, video_path
        except RuntimeError as exc:
            typer.echo(f"[WARN] Frame extraction failed for {video_path.name}: {exc}", err=True)
            if rec.status not in (_S_FAIL_READ, _S_INDEXED, _S_SKIP_IDX):
                rec.status = _S_FAIL_PRE
                rec.reason = f"frame extraction error: {exc}"


def index(
    input_dir: Path | None = typer.Argument(
        default=None, help="Root directory of images (overrides SFN_INPUT_DIR)"
    ),
    dino: bool = typer.Option(False, "--dino", help="Use DINOv2 backend (1024-dim semantic)"),
    sscd: bool = typer.Option(False, "--sscd", help="Use SSCD backend (512-dim copy-detection)"),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="CSV report output path (default: sfn_ingestion_<timestamp>.csv in cwd)",
    ),
    allow_online: bool = typer.Option(
        False,
        "--allow-online",
        help=(
            "Allow outward internet connections (e.g. to HuggingFace Hub for first-time "
            "model downloads). Offline by default — see SFN_ALLOW_ONLINE in .env."
        ),
    ),
) -> None:
    """Embed all images under INPUT_DIR and store vectors in Qdrant."""
    # Write back to os.environ so Settings() reads the correct value,
    # and any subprocess inherits the flag without an explicit argument.
    if allow_online:
        os.environ["SFN_ALLOW_ONLINE"] = "true"

    settings = Settings()

    # Apply HuggingFace offline guard before any model loading occurs.
    settings.apply_network_policy()

    env_source = str(settings._env_file) if settings._env_file else "(no .env, using defaults)"
    heif_status = "enabled (pillow-heif)" if _HEIF_AVAILABLE else "disabled (install pillow-heif)"
    typer.echo(f"Config: {env_source}")
    typer.echo(f"HEIF/HEIC support: {heif_status}")
    typer.echo(
        f"Dedup mode: {settings.duplicate_check_mode}  |  EXIF extraction: {settings.extract_exif}"
    )

    resolved_input = input_dir or settings.input_dir
    if resolved_input is None:
        typer.echo(
            "[ERROR] No input directory given. "
            "Pass one as an argument or set SFN_INPUT_DIR in .env.",
            err=True,
        )
        raise typer.Exit(1)
    if not resolved_input.is_dir():
        typer.echo(f"[ERROR] Not a directory: {resolved_input}", err=True)
        raise typer.Exit(1)

    if not dino and not sscd:
        typer.echo("[ERROR] Specify at least one of --dino or --sscd.", err=True)
        raise typer.Exit(1)

    # Pre-flight: fail fast if a HuggingFace Hub model ID is configured while offline.
    err = settings.offline_model_error(need_dino=dino)
    if err:
        typer.echo(f"[ERROR] {err}", err=True)
        raise typer.Exit(1)

    # Resolve the CSV report path early so the user knows where it will land.
    csv_path = report or Path(f"sfn_ingestion_{datetime.now():%Y%m%d_%H%M%S}.csv")

    # Build list of (use_sscd, model_name, collection_name) for each requested backend.
    models_to_run: list[tuple[bool, str, str]] = []
    if sscd:
        models_to_run.append((True, settings.model_sscd, settings.collection_sscd))
    if dino:
        models_to_run.append((False, settings.model_dino, settings.collection_dino))

    # Load all models upfront — fail fast before scanning.
    specs: list[tuple[AnyEmbedder, Indexer, str]] = []
    for use_sscd, model_name, collection in models_to_run:
        backend_name = "SSCD" if use_sscd else "DINOv2"
        if settings.embedding_endpoint:
            if not settings.embedding_model:
                typer.echo(
                    "[ERROR] SFN_EMBEDDING_MODEL must be set when"
                    " SFN_EMBEDDING_ENDPOINT is configured.",
                    err=True,
                )
                raise typer.Exit(1)
            effective_model = settings.embedding_model
            typer.echo(
                f"Using remote {backend_name} embedder at {settings.embedding_endpoint!r}"
                f" (model={effective_model!r}) ..."
            )
        else:
            effective_model = model_name
            typer.echo(
                f"Loading {backend_name} model {model_name!r} on device={settings.device!r} ..."
            )
        try:
            embedder = load_embedder(
                model=effective_model,
                use_sscd=use_sscd,
                device=settings.device,
                normalize_size=settings.normalize_size,
                remote_endpoint=settings.embedding_endpoint,
                remote_api_key=settings.embedding_api_key,
                embedding_dim=settings.embedding_dim,
                local_files_only=not settings.allow_online,
            )
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"[ERROR] {exc}", err=True)
            raise typer.Exit(1)
        fp16 = embedder.device == "cuda"
        compiled = getattr(embedder, "compiled", False)
        typer.echo(
            f"  backend={type(embedder).__name__}  dim={embedder.embedding_dim}"
            f"  device={embedder.device}  fp16={fp16}  compiled={compiled}"
        )
        if compiled:
            typer.echo("  (first batch will be slow — torch.compile warm-up)")

        typer.echo(f"Connecting to Qdrant  collection={collection!r} ...")
        try:
            indexer = Indexer(
                url=settings.qdrant_url,
                collection=collection,
                embedding_dim=embedder.embedding_dim,
                api_key=settings.qdrant_api_key,
            )
        except ValueError as exc:
            typer.echo(f"[ERROR] {exc}", err=True)
            raise typer.Exit(1)

        typer.echo("Computing model hash (may take a moment) ...")
        model_hash = embedder.model_hash
        typer.echo(f"  model_hash={model_hash[:16]}...")

        specs.append((embedder, indexer, model_hash))

    # ── Batch size: explicit config > calibration cache > auto-calibrate ─────
    if settings.batch_size is None:
        from scalar_forensic.calibration import (
            calibrate,
            load_cached_batch_size,
            save_cached_batch_size,
        )

        _sample_dir = Path("data/sample_images")
        cached = load_cached_batch_size()
        if cached is not None:
            settings.batch_size = cached
            typer.echo(f"Batch size: {cached}  (calibration cache)")
        elif _sample_dir.is_dir():
            from scalar_forensic.embedder import RemoteEmbedder

            # RemoteEmbedder calibration would send every sample image to the
            # remote endpoint on each probe iteration — expensive and surprising.
            # Skip it and require an explicit SFN_BATCH_SIZE in remote mode.
            local_specs = [
                (emb, idx, mh) for emb, idx, mh in specs if not isinstance(emb, RemoteEmbedder)
            ]
            if not local_specs:
                settings.batch_size = 32
                typer.echo(
                    "[WARN] Auto-calibration skipped for remote embedder — "
                    "set SFN_BATCH_SIZE explicitly.  Defaulting to batch_size=32.",
                    err=True,
                )
            else:
                # Calibrate each local embedder; use the minimum so the chosen
                # batch size is safe for every model that will run in production.
                # When multiple embedders are active, suppress per-embedder cache
                # writes (cache_file=None) and write a single entry for the safe
                # minimum after all probes complete — prevents the last embedder's
                # (potentially larger) value from overwriting the minimum.
                if len(local_specs) == 1:
                    sizes = [calibrate(local_specs[0][0], _sample_dir)]
                else:
                    sizes = [
                        calibrate(emb, _sample_dir, cache_file=None) for emb, _, _ in local_specs
                    ]
                    save_cached_batch_size(min(sizes))
                settings.batch_size = min(sizes)
                if len(local_specs) > 1:
                    per = "  ".join(
                        f"{type(emb).__name__}={sz}" for (emb, _, _), sz in zip(local_specs, sizes)
                    )
                    typer.echo(f"  ({per}  →  using minimum)")
            if len(sizes) > 1:
                per = "  ".join(
                    f"{type(emb).__name__}={sz}" for (emb, _, _), sz in zip(specs, sizes)
                )
                typer.echo(f"  ({per}  →  using minimum)")
        else:
            settings.batch_size = 32
            typer.echo(
                f"[WARN] {_sample_dir} not found — using batch_size=32. "
                "Add sample images there to enable auto-calibration.",
                err=True,
            )
    else:
        typer.echo(f"Batch size: {settings.batch_size}  (SFN_BATCH_SIZE)")

    # ── Pre-scan: collect all files and classify image / video / unsupported ──
    typer.echo(f"Scanning {resolved_input} ...")
    records: dict[Path, _FileRecord] = {}
    image_paths: list[Path] = []
    video_paths: list[Path] = []
    for path, file_type in scan_all_files(resolved_input):
        rec = _FileRecord(path=path)
        records[path] = rec
        if file_type == "image":
            image_paths.append(path)
        elif file_type == "video":
            video_paths.append(path)
            rec.status = _S_UNSUPPORTED  # placeholder — overwritten after video processing
            rec.reason = "video (pending)"
        else:
            ext = path.suffix.lower() or "(no extension)"
            rec.status = _S_UNSUPPORTED
            rec.reason = f"unsupported extension: {ext}"

    n_unsupported = len(records) - len(image_paths) - len(video_paths)
    typer.echo(
        f"  {len(records):,} files found  "
        f"({len(image_paths):,} image, {len(video_paths):,} video, {n_unsupported:,} other)"
    )

    # ── Per-model counters ────────────────────────────────────────────────────
    indexed_counts = [0] * len(specs)
    skipped_counts = [0] * len(specs)
    failed_counts = [0] * len(specs)

    total_read_s = total_hash_s = 0.0
    total_bytes = 0
    batch_num = 0
    imgs_processed_so_far = 0
    total_image_count = len(image_paths)
    tracker = _ETATracker()

    if total_image_count > 0:
        typer.echo(
            "  ETA  x̂ₜ = x̂ₜ⁻ + Kₜ(zₜ − x̂ₜ⁻)"
            "  ·  Kₜ = Pₜ⁻(Pₜ⁻ + R)⁻¹"
            "  ·  σ_η = N_rem · √Pₜ / x̂²"
            "  [Θ(1) Kalman]"
        )

    for batch_paths in batched(iter(image_paths), settings.batch_size):
        batch_num += 1
        batch_wall_t0 = perf_counter()

        # ── Read (shared) ────────────────────────────────────────────────────
        t0 = perf_counter()
        raw: list[tuple[Path, bytes]] = []
        batch_bytes = 0
        for p in batch_paths:
            try:
                data = p.read_bytes()
                raw.append((p, data))
                batch_bytes += len(data)
            except OSError as exc:
                typer.echo(f"[WARN] Cannot read {p}: {exc}", err=True)
                records[p].status = _S_FAIL_READ
                records[p].reason = f"read error: {exc}"
        read_s = perf_counter() - t0
        total_read_s += read_s
        total_bytes += batch_bytes
        # Count after reading so the denominator reflects files we actually
        # attempted to open — not files we merely iterated over in the plan.
        imgs_processed_so_far += len(batch_paths)

        if not raw:
            continue

        # ── Hash (shared) ────────────────────────────────────────────────────
        t0 = perf_counter()
        path_hash_pairs_full = [(p, hash_bytes(data), hash_bytes_md5(data)) for p, data in raw]
        path_hash_pairs = [(p, sha) for p, sha, _ in path_hash_pairs_full]
        md5_by_sha256 = {sha: md5 for _, sha, md5 in path_hash_pairs_full}
        hash_s = perf_counter() - t0
        total_hash_s += hash_s

        # Populate hash fields for every successfully-read file.
        for p, sha, md5 in path_hash_pairs_full:
            records[p].sha256 = sha
            records[p].md5 = md5

        # ── EXIF (shared, once per batch if enabled) ─────────────────────────
        exif_data: dict[Path, ExifInfo] | None = None
        if settings.extract_exif:
            data_by_path_for_exif = {p: data for p, data in raw}
            exif_data = {p: extract_exif(data_by_path_for_exif[p]) for p, _ in path_hash_pairs}

        # ── Within-batch hash dedup ───────────────────────────────────────────
        unique_path_by_hash: dict[str, Path] = {}
        for p, h in path_hash_pairs:
            unique_path_by_hash.setdefault(h, p)
        unique_pairs = [(p, h) for h, p in unique_path_by_hash.items()]

        winner_paths = {p for p, _ in unique_pairs}
        for p, _ in path_hash_pairs:
            if p not in winner_paths and records[p].status == "pending":
                records[p].status = _S_SKIP_DUP
                records[p].reason = "duplicate in batch (same SHA-256)"

        data_by_path = {p: data for p, data in raw}

        # ── Pre-dedup: determine which images actually need work ─────────────────
        # Run dedup for every spec before preprocessing so we only decode/resize
        # images that will be embedded or need a thumbnail generated.
        to_embed_per_spec: list[list[tuple[Path, str]]] = []
        needs_embed: set[Path] = set()
        for _, indexer, _ in specs:
            te = _apply_dedup(unique_pairs, indexer, settings)
            to_embed_per_spec.append(te)
            needs_embed.update(p for p, _ in te)

        needs_thumbnail: set[Path] = set()
        if settings.thumbnail_dir is not None:
            for p, h in unique_pairs:
                if not (settings.thumbnail_dir / f"{h}.jpg").exists():
                    needs_thumbnail.add(p)

        needs_pre = needs_embed | needs_thumbnail

        # ── Shared pre-processing (only images that need it) ──────────────────
        t0 = perf_counter()
        paths_to_pre = [p for p, _ in unique_pairs if p in needs_pre]
        pre_results = (
            preprocess_batch([data_by_path[p] for p in paths_to_pre]) if paths_to_pre else []
        )

        pre_by_path: dict[Path, Image.Image] = {}
        for p, result in zip(paths_to_pre, pre_results, strict=True):
            if isinstance(result, Exception):
                typer.echo(f"[WARN] Preprocessing failed for {p.name}: {result}", err=True)
                if records[p].status == "pending":
                    records[p].status = _S_FAIL_PRE
                    records[p].reason = f"preprocessing error: {result}"
            else:
                pre_by_path[p] = result

        # ── Thumbnail generation (shared, after preprocessing) ────────────────
        if settings.thumbnail_dir is not None:
            hash_by_path = {p: h for p, h in unique_pairs}
            for p, img in pre_by_path.items():
                sha = hash_by_path.get(p)
                if sha:
                    thumb_path = settings.thumbnail_dir / f"{sha}.jpg"
                    if not thumb_path.exists():
                        try:
                            _write_thumbnail(img, thumb_path, settings.thumbnail_size)
                        except Exception as exc:  # noqa: BLE001
                            typer.echo(f"[WARN] Thumbnail failed for {p.name}: {exc}", err=True)

        # Propagate preprocessing failure to batch-duplicates of a failed winner.
        # Those duplicates were marked _S_SKIP_DUP earlier; update them so the
        # final report accurately reflects that their hash-winner couldn't be processed.
        pre_failures: set[Path] = {p for p in paths_to_pre if p not in pre_by_path}
        failed_pre_hashes = {h for p, h in unique_pairs if p in pre_failures}
        if failed_pre_hashes:
            for p, h in path_hash_pairs:
                if h in failed_pre_hashes and records[p].status in ("pending", _S_SKIP_DUP):
                    records[p].status = _S_FAIL_PRE
                    if not records[p].reason:
                        records[p].reason = "duplicate of image that failed preprocessing"

        pre_s = perf_counter() - t0

        # Exclude preprocessing failures from the per-model loop; images that were
        # skipped pre-dedup (already indexed + thumbnail present) stay in unique_pairs
        # so the per-model loop can mark them _S_SKIP_IDX and log them correctly.
        unique_pairs = [(p, h) for p, h in unique_pairs if p not in pre_failures]

        if not unique_pairs:
            continue

        # Recompute the in-batch duplicate skip count after preprocessing-failure
        # propagation: some records originally marked _S_SKIP_DUP may have been
        # reclassified to _S_FAIL_PRE above, so using the old duplicate_hashes_in_batch
        # count would inflate the per-model skipped_counts.
        duplicate_skips_in_batch = sum(
            1 for p, _ in path_hash_pairs if records[p].status == _S_SKIP_DUP
        )

        # ── Per-model loop: normalize + embed, collect upsert jobs ────────────
        model_segments: list[str] = []
        upsert_jobs: list = []

        for spec_idx, (embedder, indexer, model_hash) in enumerate(specs):
            # Use pre-computed dedup result, filtered to exclude any preprocessing failures.
            to_embed = [(p, h) for p, h in to_embed_per_spec[spec_idx] if p not in pre_failures]
            n_skipped = duplicate_skips_in_batch + (len(unique_pairs) - len(to_embed))
            skipped_counts[spec_idx] += n_skipped

            # Mark already-indexed files for this model (don't overwrite "indexed").
            to_embed_set = {p for p, _ in to_embed}
            for p, _ in unique_pairs:
                if p not in to_embed_set and records[p].status != _S_INDEXED:
                    records[p].status = _S_SKIP_IDX
                    records[p].reason = "already indexed in Qdrant"

            backend = type(embedder).__name__

            if not to_embed:
                model_segments.append(f"{backend} all skipped ({n_skipped})")
                continue

            paths, hashes = zip(*to_embed)
            n = len(paths)
            model_pil = [pre_by_path[p] for p in paths]

            # ── Normalize ────────────────────────────────────────────────────
            t0 = perf_counter()
            try:
                norm_images = embedder.normalize_batch_bytes(model_pil)
            except Exception as exc:  # noqa: BLE001
                typer.echo(
                    f"[ERROR] Normalization failed for batch of {n} [{backend}]: {exc}", err=True
                )
                failed_counts[spec_idx] += n
                for p in paths:
                    if records[p].status != _S_INDEXED:
                        records[p].status = _S_FAIL_EMB
                        records[p].reason = f"normalization error: {exc}"
                continue
            norm_s = perf_counter() - t0

            # ── Embed ─────────────────────────────────────────────────────────
            t0 = perf_counter()
            try:
                embeddings = embedder.embed_images(norm_images)
            except Exception as exc:  # noqa: BLE001
                typer.echo(
                    f"[ERROR] Embedding failed for batch of {n} [{backend}]: {exc}", err=True
                )
                failed_counts[spec_idx] += n
                for p in paths:
                    if records[p].status != _S_INDEXED:
                        records[p].status = _S_FAIL_EMB
                        records[p].reason = f"embedding error: {exc}"
                continue
            embed_s = perf_counter() - t0

            indexed_counts[spec_idx] += n
            for p in paths:
                records[p].status = _S_INDEXED
                records[p].reason = ""

            model_segments.append(
                f"{backend} norm {norm_s:.2f}s"
                f"  embed {embed_s:.2f}s ({_fmt_rate(n, embed_s, 'img')})"
                f"  +{n}"
            )

            # Collect upsert work — executed in parallel after all models embed.
            shared_metadata = {
                "model_name": embedder.model_name,
                "model_hash": model_hash,
                "embedding_dim": embedder.embedding_dim,
                "normalize_size": embedder.normalize_size,
                "inference_dtype": embedder.inference_dtype,
                "library_versions": get_library_versions(),
            }
            exif_for_batch: dict[Path, dict] | None = None
            if exif_data is not None:
                exif_for_batch = {p: dict(exif_data[p]) for p in paths}

            hashes_md5 = [md5_by_sha256[h] for h in hashes]

            def _make_upsert(idx, ps, hs, hs_md5, embs, meta, exif):
                def _job():
                    idx.upsert_batch(ps, hs, embs, meta, exif, hs_md5)

                return _job

            upsert_jobs.append(
                _make_upsert(
                    indexer,
                    list(paths),
                    list(hashes),
                    hashes_md5,
                    embeddings,
                    shared_metadata,
                    exif_for_batch,
                )
            )

        # ── Parallel upserts ──────────────────────────────────────────────────
        upsert_wall_s = 0.0
        if upsert_jobs:
            t0 = perf_counter()
            with ThreadPoolExecutor(max_workers=len(upsert_jobs)) as pool:
                list(pool.map(lambda j: j(), upsert_jobs))
            upsert_wall_s = perf_counter() - t0

        if model_segments:
            wall_s = perf_counter() - batch_wall_t0
            n_imgs = len(path_hash_pairs)
            tracker.update(n_imgs, wall_s)
            upsert_str = f"  │  upsert {upsert_wall_s:.2f}s" if upsert_jobs else ""
            shared = (
                f"  ▸ {batch_num:04d}  {n_imgs} imgs  {batch_bytes / 1e6:.1f} MB"
                f"  {_fmt_rate(n_imgs, wall_s, 'img')}"
                f"  │  read {read_s:.2f}s ({_fmt_mbps(batch_bytes, read_s)})"
                f"  hash {hash_s:.2f}s  pre {pre_s:.2f}s"
            )
            typer.echo(shared + "  │  " + "  │  ".join(model_segments) + upsert_str)

            if batch_num % 10 == 0 and total_image_count > 0:
                result = tracker.eta(total_image_count - imgs_processed_so_far)
                if result is not None:
                    eta_s, sigma_s = result
                    pct = imgs_processed_so_far / total_image_count * 100
                    bar = _progress_bar(pct)
                    sep = "─" * 68
                    typer.echo(
                        f"  {sep}\n"
                        f"  [{bar}]  {imgs_processed_so_far:,} / {total_image_count:,}"
                        f"  ({pct:.1f}%)\n"
                        f"  x̂ = {tracker.rate:.1f} img/s"
                        f"  √P = {tracker.rate_std:.1f}"
                        f"  K = {tracker.kalman_gain:.3f}"
                        f"  ·  η̂ ~ {_fmt_duration(eta_s)}"
                        f"  σ_η ± {_fmt_duration(sigma_s)}\n"
                        f"  {sep}"
                    )

    # ── Video processing pass ─────────────────────────────────────────────────
    n_video_frames_indexed = 0
    n_video_frames_skipped = 0

    if video_paths:
        pyav_version = get_pyav_version()
        typer.echo(f"\nProcessing {len(video_paths):,} video file(s) (PyAV {pyav_version}) ...")

        # ── Pre-check: skip videos already fully indexed ───────────────────────
        # Hash each video once and query each model's collection at the video level.
        # If all models already have the video, skip frame extraction entirely.
        # If only some models have it, only those models skip during the batch loop.
        pre_hashes: dict[Path, str] = {}  # vpath → sha256
        skip_by_spec: dict[Path, set[int]] = {}  # vpath → spec indices that already have it

        for vp in video_paths:
            rec = records[vp]
            try:
                vh = hash_file(vp)
                vm = hash_file_md5(vp)
            except OSError as exc:
                typer.echo(f"[WARN] Cannot read video {vp.name}: {exc}", err=True)
                rec.status = _S_FAIL_READ
                rec.reason = f"read error: {exc}"
                continue
            rec.sha256 = vh
            rec.md5 = vm
            pre_hashes[vp] = vh

            already_in = {
                spec_idx
                for spec_idx, (_, indexer, _) in enumerate(specs)
                if indexer.is_video_complete(vh, settings.video_fps, settings.video_max_frames)
            }
            if already_in:
                skip_by_spec[vp] = already_in
            if len(already_in) == len(specs):
                rec.status = _S_SKIP_IDX
                rec.reason = "video already indexed"

        videos_to_process = [
            vp
            for vp in video_paths
            if vp in pre_hashes and records[vp].status not in (_S_FAIL_READ, _S_SKIP_IDX)
        ]

        # Per-video frame counters — only for videos being processed this run.
        vf_total: dict[Path, int] = {vp: 0 for vp in videos_to_process}
        vf_indexed: dict[Path, int] = {vp: 0 for vp in videos_to_process}
        vf_skipped: dict[Path, int] = {vp: 0 for vp in videos_to_process}
        # Per-spec, per-video indexed counts — used after the batch loop to
        # call mark_video_complete() only for specs that fully indexed a video.
        vf_indexed_by_spec: dict[int, dict[Path, int]] = {
            si: {vp: 0 for vp in videos_to_process} for si in range(len(specs))
        }

        for raw_frame_batch in batched(
            _frame_stream(videos_to_process, records, settings, pyav_version, pre_hashes),
            settings.batch_size,
        ):
            batch_num += 1
            batch_wall_t0 = perf_counter()

            frames_batch = list(raw_frame_batch)
            n_items = len(frames_batch)

            virtual_paths = [t[0] for t in frames_batch]
            frame_hashes = [t[1] for t in frames_batch]
            frame_images_raw = [t[2] for t in frames_batch]
            frame_vmetas = [t[3] for t in frames_batch]
            source_vpaths = [t[4] for t in frames_batch]

            for sv in source_vpaths:
                vf_total[sv] += 1

            # ── Preprocess: cap short side (same transform as preprocess_batch) ──
            t0 = perf_counter()
            pre_images = preprocess_pil_batch(frame_images_raw)
            pre_s = perf_counter() - t0

            # ── Thumbnails ────────────────────────────────────────────────────
            if settings.thumbnail_dir is not None:
                for fhash, img in zip(frame_hashes, pre_images):
                    thumb_path = settings.thumbnail_dir / f"{fhash}.jpg"
                    if not thumb_path.exists():
                        try:
                            _write_thumbnail(img, thumb_path, settings.thumbnail_size)
                        except Exception as exc:  # noqa: BLE001
                            typer.echo(f"[WARN] Thumbnail failed for frame: {exc}", err=True)

            # ── Frame store (size-capped cache for cross-host thumbnail regen) ──
            if settings.frame_store_dir is not None:
                for fhash, raw_img in zip(frame_hashes, frame_images_raw):
                    frame_path = settings.frame_store_dir / f"{fhash}.jpg"
                    if not frame_path.exists():
                        try:
                            _write_thumbnail(raw_img, frame_path, settings.frame_store_size)
                        except Exception as exc:  # noqa: BLE001
                            typer.echo(f"[WARN] Frame store failed: {exc}", err=True)

            # ── Per-model loop: dedup → normalize → embed → collect upsert ───
            model_segments: list[str] = []
            upsert_jobs: list = []

            for spec_idx, (embedder, indexer, model_hash) in enumerate(specs):
                # Skip frames from videos already indexed for this model.
                # The pre-check pass determined which spec indices have each video;
                # no per-frame Qdrant query needed.
                to_embed_idx = [
                    i
                    for i in range(n_items)
                    if spec_idx not in skip_by_spec.get(source_vpaths[i], set())
                ]
                n_skipped = n_items - len(to_embed_idx)
                skipped_counts[spec_idx] += n_skipped

                if spec_idx == 0:
                    n_video_frames_skipped += n_skipped
                # Update per-video skip flag for all specs so that a video only
                # indexed by spec 1 (not spec 0) is still reflected in vf_skipped.
                for i in set(range(n_items)) - set(to_embed_idx):
                    vf_skipped[source_vpaths[i]] += 1

                backend = type(embedder).__name__

                if not to_embed_idx:
                    model_segments.append(f"{backend} all skipped ({n_skipped})")
                    continue

                embed_vpaths = [virtual_paths[i] for i in to_embed_idx]
                embed_hashes = [frame_hashes[i] for i in to_embed_idx]
                embed_pil = [pre_images[i] for i in to_embed_idx]
                embed_vmetas = [frame_vmetas[i] for i in to_embed_idx]
                embed_src = [source_vpaths[i] for i in to_embed_idx]
                n = len(to_embed_idx)

                # ── Normalize ────────────────────────────────────────────────
                t0 = perf_counter()
                try:
                    norm_images = embedder.normalize_batch_bytes(embed_pil)
                except Exception as exc:  # noqa: BLE001
                    typer.echo(
                        f"[ERROR] Normalization failed for frame batch [{backend}]: {exc}",
                        err=True,
                    )
                    failed_counts[spec_idx] += n
                    continue
                norm_s = perf_counter() - t0

                # ── Embed ─────────────────────────────────────────────────────
                t0 = perf_counter()
                try:
                    embeddings = embedder.embed_images(norm_images)
                except Exception as exc:  # noqa: BLE001
                    typer.echo(
                        f"[WARN] Embedding failed for {n} frames [{backend}]: {exc}", err=True
                    )
                    failed_counts[spec_idx] += n
                    continue
                embed_s = perf_counter() - t0

                indexed_counts[spec_idx] += n
                if spec_idx == 0:
                    n_video_frames_indexed += n
                # Update per-video indexed flag for all specs so that a video only
                # indexed by spec 1 (not spec 0) is still marked as _S_INDEXED.
                for sv in embed_src:
                    vf_indexed[sv] += 1
                    vf_indexed_by_spec[spec_idx][sv] += 1

                model_segments.append(
                    f"{backend} norm {norm_s:.2f}s"
                    f"  embed {embed_s:.2f}s ({_fmt_rate(n, embed_s, 'img')})"
                    f"  +{n}"
                )

                shared_metadata = {
                    "model_name": embedder.model_name,
                    "model_hash": model_hash,
                    "embedding_dim": embedder.embedding_dim,
                    "normalize_size": embedder.normalize_size,
                    "inference_dtype": embedder.inference_dtype,
                    "library_versions": get_library_versions(),
                }

                def _make_frame_upsert(idx, eps, ehs, embs, meta, vmetas):
                    def _job():
                        idx.upsert_batch(eps, ehs, embs, meta, video_metadata=vmetas)

                    return _job

                upsert_jobs.append(
                    _make_frame_upsert(
                        indexer,
                        embed_vpaths,
                        embed_hashes,
                        embeddings,
                        shared_metadata,
                        embed_vmetas,
                    )
                )

            # ── Parallel upserts ──────────────────────────────────────────────
            upsert_wall_s = 0.0
            if upsert_jobs:
                t0 = perf_counter()
                with ThreadPoolExecutor(max_workers=len(upsert_jobs)) as pool:
                    list(pool.map(lambda j: j(), upsert_jobs))
                upsert_wall_s = perf_counter() - t0

            if model_segments:
                wall_s = perf_counter() - batch_wall_t0
                upsert_str = f"  │  upsert {upsert_wall_s:.2f}s" if upsert_jobs else ""
                typer.echo(
                    f"  ▸ {batch_num:04d}  {n_items} frames"
                    f"  {_fmt_rate(n_items, wall_s, 'frame')}"
                    f"  │  pre {pre_s:.2f}s"
                    f"  │  " + "  │  ".join(model_segments) + upsert_str
                )

        # ── Mark videos as fully indexed (per spec) ──────────────────────────
        # Called only when all frames of a video were successfully embedded and
        # upserted by a given spec.  Writes video_frames_total onto every frame
        # payload so is_video_complete() can distinguish a finished index from
        # an interrupted partial one.
        for vp in videos_to_process:
            total = vf_total.get(vp, 0)
            if total == 0:
                continue
            vh = pre_hashes[vp]
            already_done = skip_by_spec.get(vp, set())
            for spec_idx, (_, indexer, _) in enumerate(specs):
                if spec_idx in already_done:
                    continue  # was complete before this run
                if vf_indexed_by_spec[spec_idx].get(vp, 0) >= total:
                    indexer.mark_video_complete(vh, total)

        # ── Finalise per-video file records from accumulated frame counts ─────
        for vp in video_paths:
            rec = records[vp]
            if rec.status in (_S_FAIL_READ, _S_FAIL_PRE, _S_SKIP_IDX):
                continue  # already set by pre-check or _frame_stream
            total = vf_total[vp]
            if total == 0:
                typer.echo(f"  [WARN] No frames extracted from {vp.name}", err=True)
                rec.status = _S_UNSUPPORTED
                rec.reason = "no frames extracted"
            elif vf_indexed[vp] > 0:
                rec.status = _S_INDEXED
                rec.reason = f"{total} frames extracted"
            elif vf_skipped[vp] >= total:
                rec.status = _S_SKIP_IDX
                rec.reason = f"all {total} extracted frames already indexed"
            else:
                rec.status = _S_FAIL_EMB
                rec.reason = f"{total} frames extracted but no new vectors were indexed"

    # ── Write CSV report ──────────────────────────────────────────────────────
    _write_csv(records, csv_path)

    # ── Print summary table ───────────────────────────────────────────────────
    _print_summary(
        records,
        resolved_input,
        csv_path,
        specs,
        indexed_counts,
        skipped_counts,
        failed_counts,
        n_images=len(image_paths),
        n_videos=len(video_paths),
        n_video_frames_indexed=n_video_frames_indexed,
        n_video_frames_skipped=n_video_frames_skipped,
    )


def main() -> None:
    typer.run(index)
