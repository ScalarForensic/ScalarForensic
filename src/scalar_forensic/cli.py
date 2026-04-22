"""CLI entry point for ScalarForensic."""

import csv
import hashlib
import io
import os
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import batched
from pathlib import Path
from time import perf_counter

import typer
from PIL import Image
from qdrant_client.models import Distance, VectorParams

from scalar_forensic.config import ENV_ALLOW_ONLINE, Settings
from scalar_forensic.embedder import (
    AnyEmbedder,
    ExifInfo,
    HashCache,
    SSCDEmbedder,
    effective_preprocessing_cap,
    extract_exif,
    get_library_versions,
    hash_bytes_md5,
    hash_file,
    hash_file_both,
    load_embedder,
    preprocess_batch,
    write_thumbnail,
)
from scalar_forensic.indexer import Indexer
from scalar_forensic.scanner import _HEIF_AVAILABLE, scan_all_files
from scalar_forensic.video import (
    extract_frames,
    frame_disk_path,
    get_pyav_version,
    get_video_info,
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
    is_frame: bool = False  # True for frame JPEGs written during slicing


@dataclass
class _BatchCtx:
    """Carries Phase-A results (read/hash/dedup) across one loop iteration.

    The preprocessing Future is submitted at the end of Phase A and resolved
    at the start of the next iteration's finish step — so CPU preprocessing
    of batch N+1 overlaps with GPU embedding of batch N.
    """

    batch_num: int
    batch_bytes: int
    read_s: float
    hash_s: float
    imgs_at_batch: int  # items_processed_so_far snapshot — used for ETA display
    path_hash_pairs: list[tuple[Path, str]]
    md5_by_sha256: dict[str, str]
    unique_pairs: list[tuple[Path, str]]
    to_embed_per_spec: list[list[tuple[Path, str]]]
    exif_data: "dict[Path, ExifInfo] | None"
    paths_to_pre: list[Path]
    pre_future: "Future[tuple[list, float]] | None"  # tuple[list[Image.Image | Exception], float]


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


def _dedup_by_hash(
    paths: "list[Path]",
    hash_lookup: "dict[Path, str]",
    records: "dict[Path, _FileRecord]",
    dedup_hashes: "list[set[str]]",
    dedup_paths: "list[set[str]]",
    n_specs: int,
    skipped_counts: "list[int]",
) -> "tuple[list[set[Path]], set[Path], int, int]":
    """Elect one winner per unique SHA-256, mark duplicates and already-indexed files.

    Returns ``(needs_per_spec, any_needs, n_run_dups, n_all_indexed)``.
    Mutates *records* (sets status/reason) and *skipped_counts* in place.
    """
    unique_by_hash: dict[str, Path] = {}
    for p in paths:
        unique_by_hash.setdefault(hash_lookup[p], p)
    winners: set[Path] = set(unique_by_hash.values())

    for p in paths:
        if p not in winners:
            records[p].status = _S_SKIP_DUP
            records[p].reason = "duplicate in run (same SHA-256)"

    needs_per_spec: list[set[Path]] = [set() for _ in range(n_specs)]
    for si in range(n_specs):
        needs_per_spec[si] = {
            p
            for p in winners
            if hash_lookup[p] not in dedup_hashes[si] and str(p.resolve()) not in dedup_paths[si]
        }
    any_needs = set().union(*needs_per_spec) if n_specs > 0 else set()

    n_run_dups = len(paths) - len(winners)
    n_all_indexed = len(winners) - len(any_needs)
    for si in range(n_specs):
        skipped_counts[si] += n_run_dups + n_all_indexed

    for p in winners:
        if p not in any_needs:
            records[p].status = _S_SKIP_IDX
            records[p].reason = "already indexed in Qdrant"

    return needs_per_spec, any_needs, n_run_dups, n_all_indexed


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
        ("  image files", n_images),
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
        os.environ[ENV_ALLOW_ONLINE] = "true"

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

    # Build list of (use_sscd, model_name, vector_name) for each requested backend.
    models_to_run: list[tuple[bool, str, str]] = []
    if sscd:
        models_to_run.append((True, settings.model_sscd, "sscd"))
    if dino:
        models_to_run.append((False, settings.model_dino, "dino"))

    # ── Pass 1: load all embedders upfront so we know every vector's dimension ──
    # When the collection does not yet exist, passing all selected vector types in
    # a single create_collection call is more efficient than adding them one by one.
    _loaded: list[tuple[AnyEmbedder, str]] = []  # (embedder, vector_name)
    for use_sscd, model_name, vector_name in models_to_run:
        backend_name = "SSCD" if use_sscd else "DINOv2"
        try:
            effective_model = settings.resolve_embedding_model(model_name)
        except ValueError as exc:
            typer.echo(f"[ERROR] {exc}", err=True)
            raise typer.Exit(1)
        if settings.embedding_endpoint:
            typer.echo(
                f"Using remote {backend_name} embedder at {settings.embedding_endpoint!r}"
                f" (model={effective_model!r}) ..."
            )
        else:
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
                n_crops=settings.sscd_n_crops,
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
        _loaded.append((embedder, vector_name))

    # Build the full vectors config for this run so the collection is created
    # with all named vector types in a single call.
    _initial_vectors_config: dict[str, VectorParams] = {
        vn: VectorParams(size=emb.embedding_dim, distance=Distance.COSINE) for emb, vn in _loaded
    }

    # ── Pass 2: create Indexer instances with the full vectors config ──────────
    specs: list[tuple[AnyEmbedder, Indexer, str]] = []
    for embedder, vector_name in _loaded:
        typer.echo(
            f"Connecting to Qdrant  collection={settings.collection!r}  vector={vector_name!r} ..."
        )
        try:
            indexer = Indexer(
                url=settings.qdrant_url,
                collection=settings.collection,
                vector_name=vector_name,
                embedding_dim=embedder.embedding_dim,
                api_key=settings.qdrant_api_key,
                initial_vectors_config=_initial_vectors_config,
            )
        except ValueError as exc:
            typer.echo(f"[ERROR] {exc}", err=True)
            raise typer.Exit(1)

        typer.echo("Computing model hash (may take a moment) ...")
        model_hash = embedder.model_hash
        typer.echo(f"  model_hash={model_hash[:16]}...")

        specs.append((embedder, indexer, model_hash))

    # Effective short-side cap for preprocessing: must satisfy both SSCD (≥331 px)
    # and DINOv2 (≥normalize_size px).  Computed here so it is available to
    # calibrate() before the batch loops.
    _effective_cap = effective_preprocessing_cap(settings.normalize_size)

    # ── Batch size: explicit config > calibration cache > auto-calibrate ─────
    if settings.batch_size is None:
        from scalar_forensic.calibration import (
            calibrate,
            load_cached_batch_size,
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
            local_embedders = [emb for emb, _, _ in specs if not isinstance(emb, RemoteEmbedder)]
            if not local_embedders:
                settings.batch_size = 32
                typer.echo(
                    "[WARN] Auto-calibration skipped for remote embedder — "
                    "set SFN_BATCH_SIZE explicitly.  Defaulting to batch_size=32.",
                    err=True,
                )
            else:
                # Calibrate the full combined pipeline in one pass: all local
                # embedders share a single preprocess_batch call per probe batch,
                # then each model runs normalize + embed in sequence — exactly
                # mirroring what the indexer does.  This captures true combined
                # VRAM pressure rather than underestimating it by probing models
                # in isolation.
                settings.batch_size = calibrate(local_embedders, _sample_dir, cap=_effective_cap)
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

    # Library versions don't change during a run — compute once and reuse.
    _library_versions = get_library_versions()

    # ── Per-model counters ────────────────────────────────────────────────────
    indexed_counts = [0] * len(specs)
    skipped_counts = [0] * len(specs)
    failed_counts = [0] * len(specs)

    total_read_s = total_hash_s = 0.0
    total_bytes = 0
    batch_num = 0
    imgs_processed_so_far = 0
    tracker = _ETATracker()

    # ── Mutable containers shared between slicing pass and _finish_batch ─────
    # Pre-declared so the closure captures them by reference; filled later.
    vmeta_by_path: dict[Path, dict] = {}  # frame disk path → video metadata
    _frame_source: dict[Path, Path] = {}  # frame disk path → source video path
    vf_total: dict[Path, int] = {}  # source video path → total frames extracted
    vf_indexed_by_spec: dict[int, dict[Path, int]] = {}  # spec_idx → {source vp → n_indexed}

    # Per-spec needs sets — images populated during image dedup, frames during slicing.
    _needs_per_spec: list[set[Path]] = [set() for _ in specs]
    _frame_needs_per_spec: list[set[Path]] = [set() for _ in specs]

    # ── Pipelined batch loop ──────────────────────────────────────────────────
    # Pipeline depth = 2: Phase A (read / hash / dedup) runs up to two batches
    # ahead of the GPU.  Each Phase A immediately submits a preprocessing Future
    # so CPU pre-processing starts as early as possible.  _finish_batch() (the
    # GPU embed step) is called only when the pipeline is full, meaning Phase A
    # for batches N+1 and N+2 overlaps with embed(N) regardless of whether CPU
    # or GPU is the bottleneck.
    _PIPELINE_DEPTH = 2
    _pipeline: deque[_BatchCtx] = deque()

    def _finish_batch(ctx: _BatchCtx) -> None:
        """Resolve the preprocessing Future and run embed + upsert for *ctx*."""
        t_finish = perf_counter()

        # ── Wait for preprocessing (usually already done while GPU ran) ───────
        # pre_s is the actual CPU preprocessing time recorded inside the worker;
        # the future itself resolves instantly when GPU is the bottleneck.
        pre_results: list[Image.Image | Exception]
        pre_results, pre_s = ctx.pre_future.result() if ctx.pre_future is not None else ([], 0.0)

        pre_by_path: dict[Path, Image.Image] = {}
        for p, result in zip(ctx.paths_to_pre, pre_results, strict=True):
            if isinstance(result, Exception):
                typer.echo(f"[WARN] Preprocessing failed for {p.name}: {result}", err=True)
                if records[p].status == "pending":
                    records[p].status = _S_FAIL_PRE
                    records[p].reason = f"preprocessing error: {result}"
            else:
                pre_by_path[p] = result

        # ── Thumbnail generation ───────────────────────────────────────────────
        if settings.thumbnail_dir is not None:
            hash_by_path = {p: h for p, h in ctx.unique_pairs}
            for p, img in pre_by_path.items():
                sha = hash_by_path.get(p)
                if sha:
                    thumb_path = settings.thumbnail_dir / f"{sha}.jpg"
                    if not thumb_path.exists():
                        try:
                            write_thumbnail(img, thumb_path, settings.thumbnail_size)
                        except Exception as exc:  # noqa: BLE001
                            typer.echo(f"[WARN] Thumbnail failed for {p.name}: {exc}", err=True)

        # ── Propagate preprocessing failures ──────────────────────────────────
        pre_failures: set[Path] = {p for p in ctx.paths_to_pre if p not in pre_by_path}
        failed_pre_hashes = {h for p, h in ctx.unique_pairs if p in pre_failures}
        if failed_pre_hashes:
            for p, h in ctx.path_hash_pairs:
                if h in failed_pre_hashes and records[p].status in ("pending", _S_SKIP_DUP):
                    records[p].status = _S_FAIL_PRE
                    if not records[p].reason:
                        records[p].reason = "duplicate of image that failed preprocessing"

        unique_pairs = [(p, h) for p, h in ctx.unique_pairs if p not in pre_failures]
        if not unique_pairs:
            return

        # ── Per-model loop: normalize + embed, collect upsert jobs ────────────
        n_frames_in_batch = sum(1 for p, _ in ctx.path_hash_pairs if p in vmeta_by_path)
        n_plain_in_batch = len(ctx.path_hash_pairs) - n_frames_in_batch
        model_segments: list[str] = []
        upsert_jobs: list = []

        for spec_idx, (embedder, indexer, model_hash) in enumerate(specs):
            to_embed = [(p, h) for p, h in ctx.to_embed_per_spec[spec_idx] if p not in pre_failures]
            n_skipped = len(unique_pairs) - len(to_embed)
            skipped_counts[spec_idx] += n_skipped

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
                if p in vmeta_by_path:
                    sv = _frame_source[p]
                    vf_indexed_by_spec.setdefault(spec_idx, {})
                    vf_indexed_by_spec[spec_idx][sv] = vf_indexed_by_spec[spec_idx].get(sv, 0) + 1

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
                "library_versions": _library_versions,
                **(
                    {"sscd_n_crops": embedder.n_crops} if isinstance(embedder, SSCDEmbedder) else {}
                ),
            }
            exif_for_batch: dict[Path, dict] | None = None
            if ctx.exif_data is not None:
                exif_for_batch = {p: dict(ctx.exif_data[p]) for p in paths if p in ctx.exif_data}

            hashes_md5 = [ctx.md5_by_sha256[h] for h in hashes]
            video_metadata_for_batch = [vmeta_by_path.get(p) for p in list(paths)]
            has_vmeta = any(v is not None for v in video_metadata_for_batch)

            def _make_upsert(idx, ps, hs, hs_md5, embs, meta, exif, vmetas):
                def _job():
                    idx.upsert_batch(ps, hs, embs, meta, exif, hs_md5, vmetas)

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
                    video_metadata_for_batch if has_vmeta else None,
                )
            )

        # ── Serialized upserts ────────────────────────────────────────────────
        # Must be sequential: both models target the same unified collection and
        # share point IDs.  Concurrent upserts race on the retrieve→upsert TOCTOU
        # window — the second thread may see a point as "new" and overwrite the
        # first thread's named vector.  Sequential execution guarantees the second
        # call sees the first's point as "existing" and uses update_vectors.
        upsert_wall_s = 0.0
        if upsert_jobs:
            t0 = perf_counter()
            for j in upsert_jobs:
                j()
            upsert_wall_s = perf_counter() - t0

        if model_segments:
            # wall_s = read + hash + everything in _finish_batch (pre-wait, normalize,
            # embed, upsert).  Using perf_counter() - t_finish for the finish phase
            # avoids double-counting pre_s, which is already included in that interval.
            # Dedup time (~ms) is intentionally excluded.
            wall_s = ctx.read_s + ctx.hash_s + (perf_counter() - t_finish)
            n_items = len(ctx.path_hash_pairs)
            tracker.update(n_items, wall_s)
            items_str = (
                f"{n_plain_in_batch} imgs + {n_frames_in_batch} frames"
                if n_frames_in_batch > 0
                else f"{n_items} imgs"
            )
            upsert_str = f"  │  upsert {upsert_wall_s:.2f}s" if upsert_jobs else ""
            shared = (
                f"  ▸ {ctx.batch_num:04d}  {items_str}  {ctx.batch_bytes / 1e6:.1f} MB"
                f"  {_fmt_rate(n_items, wall_s, 'img')}"
                f"  │  read {ctx.read_s:.2f}s ({_fmt_mbps(ctx.batch_bytes, ctx.read_s)})"
                f"  hash {ctx.hash_s:.2f}s  pre {pre_s:.2f}s"
            )
            typer.echo(shared + "  │  " + "  │  ".join(model_segments) + upsert_str)

            if ctx.batch_num % 10 == 0 and total_image_count > 0:
                result = tracker.eta(total_image_count - ctx.imgs_at_batch)
                if result is not None:
                    eta_s, sigma_s = result
                    pct = ctx.imgs_at_batch / total_image_count * 100
                    bar = _progress_bar(pct)
                    sep = "─" * 68
                    typer.echo(
                        f"  {sep}\n"
                        f"  [{bar}]  {ctx.imgs_at_batch:,} / {total_image_count:,}"
                        f"  ({pct:.1f}%)\n"
                        f"  x̂ = {tracker.rate:.1f} img/s"
                        f"  √P = {tracker.rate_std:.1f}"
                        f"  K = {tracker.kalman_gain:.3f}"
                        f"  ·  η̂ ~ {_fmt_duration(eta_s)}"
                        f"  σ_η ± {_fmt_duration(sigma_s)}\n"
                        f"  {sep}"
                    )

    def _timed_preprocess(data: list[bytes], cap: int) -> tuple[list, float]:
        """Run preprocess_batch and return (results, elapsed_s) for display."""
        t0 = perf_counter()
        result = preprocess_batch(data, cap=cap)
        return result, perf_counter() - t0

    # ── Persistent hash cache (shared across image and video passes) ─────────
    _hash_cache: HashCache | None = (
        HashCache(settings.hash_cache_path) if settings.hash_cache_path is not None else None
    )

    # ── Pre-load Qdrant dedup indices ─────────────────────────────────────────
    _dedup_hashes: list[set[str]] = []
    _dedup_paths: list[set[str]] = []
    _video_info: list[dict[str, dict]] = []

    if image_paths or video_paths:
        _mode = settings.duplicate_check_mode
        typer.echo("Pre-loading dedup index from Qdrant ...")
        for _, _idx, _ in specs:
            _dedup_hashes.append(
                _idx.get_all_indexed_hashes() if _mode in ("hash", "both") else set()
            )
            _dedup_paths.append(
                _idx.get_all_indexed_paths() if _mode in ("filepath", "both") else set()
            )
        typer.echo(
            "  "
            + " | ".join(
                f"{type(emb).__name__}: {len(h):,} hashes"
                for (emb, _, _), h in zip(specs, _dedup_hashes)
            )
        )

    if video_paths:
        typer.echo("Pre-loading video index from Qdrant ...")
        _video_info = [indexer.get_all_video_info() for _, indexer, _ in specs]
        typer.echo(
            "  "
            + " | ".join(
                f"{type(emb).__name__}: {len(vi):,} videos"
                for (emb, _, _), vi in zip(specs, _video_info)
            )
        )

    # ── Upfront hash pass (images + videos) ──────────────────────────────────
    # Hash every file before any embedding so the full dedup picture is known
    # upfront.  The HashCache skips the disk read for unchanged files on
    # subsequent runs.  Images are hashed in parallel; videos sequentially
    # (few files, often large — parallel read would thrash the drive).
    _file_hashes: dict[Path, str] = {}  # image / frame path → sha256
    _pre_hashes: dict[Path, str] = {}  # video path → sha256
    _n_cache_hits = 0

    _hash_label_parts = []
    if image_paths:
        _hash_label_parts.append(f"{len(image_paths):,} image files")
    if video_paths:
        _hash_label_parts.append(f"{len(video_paths):,} video files")
    if _hash_label_parts:
        typer.echo(f"Hashing {' and '.join(_hash_label_parts)} ...")
    _t_hash0 = perf_counter()

    if image_paths:

        def _hash_one(p: Path) -> tuple[Path, str | None, str | None, bool]:
            try:
                if _hash_cache is not None:
                    sha, cached = _hash_cache.get_or_hash(p)
                    return p, sha, None, cached
                return p, hash_file(p), None, False
            except OSError as exc:
                return p, None, str(exc), False

        _n_hash_workers = min(32, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=_n_hash_workers) as _hpool:
            for _p, _sha, _err, _cached in _hpool.map(_hash_one, image_paths):
                if _err:
                    records[_p].status = _S_FAIL_READ
                    records[_p].reason = f"read error: {_err}"
                else:
                    _file_hashes[_p] = _sha
                    records[_p].sha256 = _sha
                    if _cached:
                        _n_cache_hits += 1

    for _vp in video_paths:
        _vrec = records[_vp]
        try:
            if _hash_cache is not None:
                _vh, _vm, _vcached = _hash_cache.get_or_hash_both(_vp)
            else:
                _vh, _vm = hash_file_both(_vp)
                _vcached = False
            if _vcached:
                _n_cache_hits += 1
        except OSError as _exc:
            typer.echo(f"[WARN] Cannot read video {_vp.name}: {_exc}", err=True)
            _vrec.status = _S_FAIL_READ
            _vrec.reason = f"read error: {_exc}"
            continue
        _vrec.sha256 = _vh
        _vrec.md5 = _vm
        _pre_hashes[_vp] = _vh

    _hash_elapsed = perf_counter() - _t_hash0

    # ── Image dedup ───────────────────────────────────────────────────────────
    _paths_to_batch: list[Path] = []
    _any_needs: set[Path] = set()
    _n_run_dups = 0
    _n_all_indexed = 0

    if image_paths:
        _needs_per_spec, _any_needs, _n_run_dups, _n_all_indexed = _dedup_by_hash(
            paths=list(_file_hashes.keys()),
            hash_lookup=_file_hashes,
            records=records,
            dedup_hashes=_dedup_hashes,
            dedup_paths=_dedup_paths,
            n_specs=len(specs),
            skipped_counts=skipped_counts,
        )
        # Build the ordered list of image paths that actually need processing.
        _paths_to_batch = [p for p in image_paths if p in _any_needs]

    # ── Video dedup ───────────────────────────────────────────────────────────
    _skip_by_spec: dict[Path, set[int]] = {}
    _videos_to_process: list[Path] = []

    for _vp, _vh in _pre_hashes.items():
        _already_in = {
            _si
            for _si, _vi in enumerate(_video_info)
            if (_vinfo := _vi.get(_vh)) is not None
            and _vinfo["extraction_fps"] == settings.video_fps
            and _vinfo["max_frames_cap"] == settings.video_max_frames
            and _vinfo["complete"]
        }
        if _already_in:
            _skip_by_spec[_vp] = _already_in
        if len(_already_in) == len(specs):
            records[_vp].status = _S_SKIP_IDX
            records[_vp].reason = "video already indexed"

    _videos_to_process = [
        vp
        for vp in video_paths
        if vp in _pre_hashes and records[vp].status not in (_S_FAIL_READ, _S_SKIP_IDX)
    ]

    # ── Hash pass summary ─────────────────────────────────────────────────────
    if _hash_label_parts:
        _cache_hits_str = (
            f"  │  {_n_cache_hits:,} cache hits"
            f"  {(len(_file_hashes) + len(_pre_hashes)) - _n_cache_hits:,} hashed"
            if _hash_cache is not None
            else ""
        )
        if image_paths:
            typer.echo(
                f"  {len(_file_hashes):,} images in {_hash_elapsed:.1f}s"
                f"{_cache_hits_str}"
                f"  │  {len(_any_needs):,} to embed"
                f"  │  {_n_run_dups:,} run-dups"
                f"  │  {_n_all_indexed:,} already indexed"
                f"  │  {len(image_paths) - len(_file_hashes):,} read errors"
            )
        if video_paths:
            typer.echo(
                f"  {len(_pre_hashes):,} videos"
                + ("" if image_paths else f" in {_hash_elapsed:.1f}s{_cache_hits_str}")
                + f"  │  {len(_videos_to_process):,} to process"
                f"  │  {len(video_paths) - len(_pre_hashes):,} read errors"
            )

    # ── Video slicing pass ────────────────────────────────────────────────────
    # Extract frames from each video that needs processing, save each as a JPEG
    # on disk at the canonical path, and register them in the unified ingestion
    # queue so they go through exactly the same pipeline as regular images.
    _frame_paths: list[Path] = []

    if _videos_to_process:
        if settings.frame_store_dir is None:
            typer.echo(
                "[ERROR] SFN_FRAME_STORE_DIR is required when video files need indexing. "
                "Set it in .env or configure frame_store_dir.",
                err=True,
            )
            raise typer.Exit(1)

        _frame_store = settings.frame_store_dir
        _pyav_version = get_pyav_version()
        _n_vids = len(_videos_to_process)
        typer.echo(
            f"\nSlicing {_n_vids:,} video(s) into frames  (PyAV {_pyav_version})  →  {_frame_store}"
        )

        # Pre-probe durations for ETA estimation (container open only — no decoding).
        typer.echo("  Probing video durations ...")
        _expected_per_video: dict[Path, int] = {}
        for _vp_probe in _videos_to_process:
            _probe = get_video_info(_vp_probe)
            _dur = _probe.get("duration_s")
            if _dur and _dur > 0:
                _exp = int(_dur * settings.video_fps)
                if settings.video_max_frames > 0:
                    _exp = min(_exp, settings.video_max_frames)
                _expected_per_video[_vp_probe] = max(_exp, 1)
        _total_expected_frames = sum(_expected_per_video.values())
        if _total_expected_frames > 0:
            typer.echo(f"  ~{_total_expected_frames:,} frames estimated across {_n_vids} video(s)")

        _slice_tracker = _ETATracker()
        _slice_total_frames = 0  # running total across all videos
        _SLICE_BLOCK = 50  # frames per Kalman update + progress line

        _video_records_to_upsert: list[dict] = []

        for _vi, _vp in enumerate(_videos_to_process):
            _vh = _pre_hashes[_vp]
            _vp_abs = str(_vp.resolve())
            _n_frames_this_video = 0
            _exp_this = _expected_per_video.get(_vp, 0)
            _exp_suffix = f"  (est. {_exp_this:,} frames)" if _exp_this > 0 else ""

            typer.echo(f"\n▶ [{_vi + 1}/{_n_vids}]  {_vp.name}{_exp_suffix}")

            _t_video_start = perf_counter()
            _t_block_start = perf_counter()
            _block_frames = 0
            _block_count = 0  # how many complete blocks emitted — for ETA box cadence

            try:
                for _frame in extract_frames(
                    _vp, fps=settings.video_fps, max_frames=settings.video_max_frames
                ):
                    _fp = frame_disk_path(_frame_store, _vh, _frame.timecode_ms)

                    # Encode frame to JPEG bytes; reuse existing file if present
                    # (keeps hashes consistent with previous runs).
                    if _fp.exists():
                        try:
                            _jpeg_bytes = _fp.read_bytes()
                        except OSError as _exc:
                            typer.echo(
                                f"[WARN] Cannot read existing frame {_fp.name}: {_exc}", err=True
                            )
                            continue
                    else:
                        _buf = io.BytesIO()
                        _img_rgb = (
                            _frame.image.convert("RGB")
                            if _frame.image.mode != "RGB"
                            else _frame.image
                        )
                        _img_rgb.save(_buf, format="JPEG", quality=85, optimize=True)
                        _jpeg_bytes = _buf.getvalue()
                        try:
                            _fp.parent.mkdir(parents=True, exist_ok=True)
                            _fp.write_bytes(_jpeg_bytes)
                        except OSError as _exc:
                            typer.echo(f"[WARN] Cannot write frame {_fp}: {_exc}", err=True)
                            continue

                    _frame_sha256 = hashlib.sha256(_jpeg_bytes).hexdigest()
                    _frame_md5 = hash_bytes_md5(_jpeg_bytes)

                    _file_hashes[_fp] = _frame_sha256
                    _frame_rec = _FileRecord(
                        path=_fp,
                        sha256=_frame_sha256,
                        md5=_frame_md5,
                        is_frame=True,
                    )
                    records[_fp] = _frame_rec

                    vmeta_by_path[_fp] = {
                        "video_hash": _vh,
                        "video_path": _vp_abs,
                        "frame_timecode_ms": _frame.timecode_ms,
                        "frame_index": _frame.frame_index,
                        "extraction_fps": settings.video_fps,
                        "max_frames_cap": settings.video_max_frames,
                        "pyav_version": _pyav_version,
                    }
                    _frame_source[_fp] = _vp
                    _frame_paths.append(_fp)
                    _n_frames_this_video += 1
                    _slice_total_frames += 1
                    _block_frames += 1

                    if _block_frames >= _SLICE_BLOCK:
                        _block_s = perf_counter() - _t_block_start
                        _slice_tracker.update(_block_frames, _block_s)
                        _block_frames = 0
                        _block_count += 1
                        _t_block_start = perf_counter()

                        # Per-block progress line
                        _tc_s = _frame.timecode_ms / 1000
                        _tc_str = (
                            f"{int(_tc_s // 3600)}:"
                            f"{int((_tc_s % 3600) // 60):02d}:"
                            f"{int(_tc_s % 60):02d}"
                        )
                        typer.echo(
                            f"  ▸ frame {_n_frames_this_video:,}"
                            f"  timecode {_tc_str}"
                            f"  │  {_fmt_rate(_SLICE_BLOCK, _block_s, 'fps')}"
                        )

                        # Kalman ETA box every 5 blocks (= 250 frames)
                        if _block_count % 5 == 0 and _total_expected_frames > 0:
                            _remaining = max(_total_expected_frames - _slice_total_frames, 0)
                            _eta_result = _slice_tracker.eta(_remaining)
                            if _eta_result is not None:
                                _pct = _slice_total_frames / _total_expected_frames * 100
                                _bar = _progress_bar(_pct)
                                _eta_s, _sigma_s = _eta_result
                                _sep = "─" * 68
                                typer.echo(
                                    f"  {_sep}\n"
                                    f"  [{_bar}]  {_slice_total_frames:,}"
                                    f" / ~{_total_expected_frames:,}"
                                    f"  ({_pct:.1f}%)\n"
                                    f"  x̂ = {_slice_tracker.rate:.1f} fps"
                                    f"  √P = {_slice_tracker.rate_std:.1f}"
                                    f"  K = {_slice_tracker.kalman_gain:.3f}"
                                    f"  ·  η̂ ~ {_fmt_duration(_eta_s)}"
                                    f"  σ_η ± {_fmt_duration(_sigma_s)}\n"
                                    f"  {_sep}"
                                )

            except RuntimeError as _exc:
                typer.echo(f"[WARN] Frame extraction failed for {_vp.name}: {_exc}", err=True)
                records[_vp].status = _S_FAIL_PRE
                records[_vp].reason = f"frame extraction error: {_exc}"
                continue

            # Flush any remaining sub-block frames into the tracker.
            if _block_frames > 0:
                _block_s = perf_counter() - _t_block_start
                _slice_tracker.update(_block_frames, _block_s)

            _video_s = perf_counter() - _t_video_start
            typer.echo(
                f"  ✓ {_n_frames_this_video:,} frames"
                f"  {_fmt_duration(_video_s)}"
                f"  {_fmt_rate(_n_frames_this_video, _video_s, 'fps')}"
            )

            vf_total[_vp] = _n_frames_this_video

            if _n_frames_this_video > 0:
                _video_records_to_upsert.append(
                    {
                        "video_hash": _vh,
                        "video_path": _vp_abs,
                        "total_frames": _n_frames_this_video,
                        "extraction_fps": settings.video_fps,
                        "max_frames_cap": settings.video_max_frames,
                        "pyav_version": _pyav_version,
                    }
                )

        _n_total_frames = len(_frame_paths)
        typer.echo(f"\n  {_n_total_frames:,} frames from {_n_vids:,} video(s)")

        # Upsert one payload-only Qdrant record per video (no vectors).
        if _video_records_to_upsert and specs:
            specs[0][1].upsert_video_records(_video_records_to_upsert)

    # ── Frame dedup (hash-based, same pipeline as images) ────────────────────
    _frame_any_needs: set[Path] = set()
    if _frame_paths:
        _frame_needs_per_spec, _frame_any_needs, _n_frame_run_dups, _n_frame_all_indexed = (
            _dedup_by_hash(
                paths=_frame_paths,
                hash_lookup=_file_hashes,
                records=records,
                dedup_hashes=_dedup_hashes,
                dedup_paths=_dedup_paths,
                n_specs=len(specs),
                skipped_counts=skipped_counts,
            )
        )
        _paths_to_batch.extend(fp for fp in _frame_paths if fp in _frame_any_needs)

    # ── Combined per-spec needs ───────────────────────────────────────────────
    _combined_needs_per_spec: list[set[Path]] = [
        _needs_per_spec[si] | _frame_needs_per_spec[si] for si in range(len(specs))
    ]

    total_image_count = len(_paths_to_batch)
    if total_image_count > 0:
        n_img_items = len([p for p in _paths_to_batch if p not in vmeta_by_path])
        n_frame_items = total_image_count - n_img_items
        item_desc = (
            f"{n_img_items:,} images + {n_frame_items:,} frames"
            if n_frame_items > 0
            else f"{total_image_count:,} images"
        )
        typer.echo(
            f"\nEmbedding {item_desc}"
            "  ·  ETA  x̂ₜ = x̂ₜ⁻ + Kₜ(zₜ − x̂ₜ⁻)"
            "  ·  Kₜ = Pₜ⁻(Pₜ⁻ + R)⁻¹"
            "  ·  σ_η = N_rem · √Pₜ / x̂²"
            "  [Θ(1) Kalman]"
        )

    with ThreadPoolExecutor(max_workers=1) as _pre_pool:
        for batch_paths in batched(iter(_paths_to_batch), settings.batch_size):
            batch_num += 1

            # ── Read (shared) ─────────────────────────────────────────────────
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
            imgs_processed_so_far += len(batch_paths)

            if not raw:
                continue

            # ── Hash: SHA-256 from pre-computed lookup, MD5 from in-memory bytes ─
            t0 = perf_counter()
            path_hash_pairs_full = [(p, _file_hashes[p], hash_bytes_md5(data)) for p, data in raw]
            path_hash_pairs = [(p, sha) for p, sha, _ in path_hash_pairs_full]
            md5_by_sha256 = {sha: md5 for _, sha, md5 in path_hash_pairs_full}
            hash_s = perf_counter() - t0
            total_hash_s += hash_s

            for p, _, md5 in path_hash_pairs_full:
                records[p].md5 = md5

            # ── EXIF (shared, once per batch if enabled, images only) ─────────
            exif_data: dict[Path, ExifInfo] | None = None
            if settings.extract_exif:
                data_by_path_for_exif = {p: data for p, data in raw if p not in vmeta_by_path}
                exif_pairs = [(p, h) for p, h in path_hash_pairs if p not in vmeta_by_path]
                if exif_pairs:
                    exif_data = {
                        p: extract_exif(data_by_path_for_exif[p])
                        for p, _ in exif_pairs
                        if p in data_by_path_for_exif
                    }

            # All paths in _paths_to_batch are unique — no within-batch dedup needed.
            unique_pairs = path_hash_pairs

            data_by_path = {p: data for p, data in raw}

            # ── Per-spec: use pre-computed combined needs sets ─────────────────
            to_embed_per_spec: list[list[tuple[Path, str]]] = []
            needs_embed: set[Path] = set()
            for spec_i, _ in enumerate(specs):
                te = [(p, h) for p, h in unique_pairs if p in _combined_needs_per_spec[spec_i]]
                to_embed_per_spec.append(te)
                needs_embed.update(p for p, _ in te)

            needs_thumbnail: set[Path] = set()
            if settings.thumbnail_dir is not None:
                for p, h in unique_pairs:
                    if not (settings.thumbnail_dir / f"{h}.jpg").exists():
                        needs_thumbnail.add(p)

            needs_pre = needs_embed | needs_thumbnail

            # ── Submit preprocessing in background ────────────────────────────
            # Cap so DINOv2 gets its configured resolution and SSCD ≥ 331 px.
            paths_to_pre = [p for p, _ in unique_pairs if p in needs_pre]
            bytes_to_pre = [data_by_path[p] for p in paths_to_pre]
            pre_future: Future[tuple[list, float]] | None = (
                _pre_pool.submit(_timed_preprocess, bytes_to_pre, _effective_cap)
                if paths_to_pre
                else None
            )

            # ── Drain oldest batch when pipeline is full ─────────────────────
            # Phase A for this batch (and the one before it) has already run,
            # so embed(oldest) overlaps with both of those Phase A passes.
            if len(_pipeline) >= _PIPELINE_DEPTH:
                _finish_batch(_pipeline.popleft())
            _pipeline.append(
                _BatchCtx(
                    batch_num=batch_num,
                    batch_bytes=batch_bytes,
                    read_s=read_s,
                    hash_s=hash_s,
                    imgs_at_batch=imgs_processed_so_far,
                    path_hash_pairs=path_hash_pairs,
                    md5_by_sha256=md5_by_sha256,
                    unique_pairs=unique_pairs,
                    to_embed_per_spec=to_embed_per_spec,
                    exif_data=exif_data,
                    paths_to_pre=paths_to_pre,
                    pre_future=pre_future,
                )
            )

        # ── Drain remaining batches ───────────────────────────────────────────
        while _pipeline:
            _finish_batch(_pipeline.popleft())

    # ── Mark videos as fully indexed (per spec) ──────────────────────────────
    # Called only when all frames of a video were successfully embedded for a
    # given spec.  Writes the video_frames_total marker onto the stored frame
    # payloads so future runs can distinguish a finished index from an
    # interrupted partial one via get_all_video_info().
    if _videos_to_process:
        # Count dup-skipped frames per source video; these are already present
        # in Qdrant (via a hash-identical frame that was embedded) so they count
        # toward the "fully indexed" threshold even though they weren't embedded
        # in this run.  Only count within-video dups: if the dedup winner came
        # from a different video its Qdrant point carries that video's metadata,
        # not this one's, so this video's timecode slot was never written.
        _winner_by_hash: dict[str, Path] = {}
        for _fp in _frame_source:
            if records[_fp].status != _S_SKIP_DUP:
                _winner_by_hash.setdefault(_file_hashes[_fp], _fp)

        _vf_dup_count: dict[Path, int] = {}
        for _fp, _sv in _frame_source.items():
            if records[_fp].status == _S_SKIP_DUP:
                _winner = _winner_by_hash.get(_file_hashes[_fp])
                if _winner is not None and _frame_source.get(_winner) == _sv:
                    _vf_dup_count[_sv] = _vf_dup_count.get(_sv, 0) + 1

        for _vp in _videos_to_process:
            _total = vf_total.get(_vp, 0)
            if _total == 0:
                continue
            _vh = _pre_hashes[_vp]
            _already_done = _skip_by_spec.get(_vp, set())
            for _spec_idx, (_, _indexer, _) in enumerate(specs):
                if _spec_idx in _already_done:
                    continue  # was complete before this run
                _indexed = vf_indexed_by_spec.get(_spec_idx, {}).get(_vp, 0)
                _dups = _vf_dup_count.get(_vp, 0)
                if _indexed + _dups >= _total:
                    _indexer.mark_video_complete(_vh, _total)

    # ── Finalise per-video source file records ────────────────────────────────
    if video_paths:
        # Build per-video aggregate counts from frame record statuses.
        _vf_indexed_total: dict[Path, int] = {}
        _vf_skipped_total: dict[Path, int] = {}
        for _fp, _sv in _frame_source.items():
            if records[_fp].status == _S_INDEXED:
                _vf_indexed_total[_sv] = _vf_indexed_total.get(_sv, 0) + 1
            elif records[_fp].status == _S_SKIP_IDX:
                _vf_skipped_total[_sv] = _vf_skipped_total.get(_sv, 0) + 1

        for _vp in video_paths:
            _vrec = records[_vp]
            if _vrec.status in (_S_FAIL_READ, _S_FAIL_PRE, _S_SKIP_IDX):
                continue  # already set by hash pass, slicing, or pre-check
            _total = vf_total.get(_vp, 0)
            if _total == 0:
                if _vp in _videos_to_process:
                    typer.echo(f"  [WARN] No frames extracted from {_vp.name}", err=True)
                    _vrec.status = _S_UNSUPPORTED
                    _vrec.reason = "no frames extracted"
                continue
            _n_idx = _vf_indexed_total.get(_vp, 0)
            _n_skp = _vf_skipped_total.get(_vp, 0)
            if _n_idx > 0:
                _vrec.status = _S_INDEXED
                _vrec.reason = f"{_total} frames extracted"
            elif _n_skp >= _total:
                _vrec.status = _S_SKIP_IDX
                _vrec.reason = f"all {_total} extracted frames already indexed"
            else:
                _vrec.status = _S_FAIL_EMB
                _vrec.reason = f"{_total} frames extracted but no new vectors were indexed"

    # ── Reclassify run-duplicates whose winner failed preprocessing ───────────
    # Non-winners are marked _S_SKIP_DUP upfront and never enter any batch, so
    # _finish_batch cannot reclassify them.  Do a single post-batch pass here.
    if image_paths and _file_hashes:
        _fail_pre_hashes = {
            _file_hashes[p]
            for p in _file_hashes
            if not records[p].is_frame and records[p].status == _S_FAIL_PRE
        }
        if _fail_pre_hashes:
            for _p, _sha in _file_hashes.items():
                if (
                    not records[_p].is_frame
                    and _sha in _fail_pre_hashes
                    and records[_p].status == _S_SKIP_DUP
                ):
                    records[_p].status = _S_FAIL_PRE
                    records[_p].reason = "duplicate of image that failed preprocessing"

    # ── Close hash cache (close() performs the final flush) ──────────────────
    if _hash_cache is not None:
        _hash_cache.close()

    # ── Video frames summary counters ─────────────────────────────────────────
    n_video_frames_indexed = sum(_vf_indexed_total.values()) if video_paths else 0
    n_video_frames_skipped = sum(_vf_skipped_total.values()) if video_paths else 0

    # ── Write CSV report ──────────────────────────────────────────────────────
    _write_csv(records, csv_path)

    # ── Print summary table (user-supplied files only, no frame records) ──────
    _user_records = {p: r for p, r in records.items() if not r.is_frame}
    _print_summary(
        _user_records,
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
