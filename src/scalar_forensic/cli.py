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
from scalar_forensic.manifest import input_file_entry, write_run_manifest
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
# Video source file whose every extracted frame was a byte-identical duplicate
# of a frame from other media in the same run — nothing new to index, but not a
# failure either.  Distinct from _S_SKIP_DUP: the video file itself is not a
# duplicate, its frames are.
_S_SKIP_FRAME_DUP = "skipped_frames_dup_run"
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
    imgs_at_batch: int  # items_processed_so_far snapshot — used for the progress line
    path_hash_pairs: list[tuple[Path, str]]
    md5_by_sha256: dict[str, str]
    unique_pairs: list[tuple[Path, str]]
    to_embed_per_spec: list[list[tuple[Path, str]]]
    exif_data: "dict[Path, ExifInfo] | None"
    paths_to_pre: list[Path]
    # tuple[list[Image.Image | Exception], list[face results], float] — see
    # _timed_preprocess; face results are (path, sha256, vmeta,
    # FaceIndexResult | Exception) for this batch's in-loop face detection.
    pre_future: "Future[tuple[list, list, float]] | None"


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


def _video_status(
    n_total: int, n_indexed: int, n_already: int, n_run_dup: int
) -> "tuple[str, str]":
    """Classify a video source file from the outcomes of its extracted frames.

    ``n_total`` frames were extracted; of those ``n_indexed`` produced new
    vectors, ``n_already`` were already in Qdrant, and ``n_run_dup`` were
    byte-identical to a frame of other media processed in the same run.
    Returns ``(status, reason)`` — both are court-facing, so each branch states
    the fact that holds, never a failure standing in for a duplicate.
    """
    if n_indexed > 0:
        return _S_INDEXED, f"{n_total} frames extracted"
    if n_already >= n_total:
        return _S_SKIP_IDX, f"all {n_total} extracted frames already indexed"
    if n_already + n_run_dup >= n_total:
        detail = (
            f"{n_run_dup} in-run duplicate(s), {n_already} already indexed"
            if n_already
            else "in-run duplicates"
        )
        return (
            _S_SKIP_FRAME_DUP,
            f"all {n_total} extracted frames were duplicates of frames from "
            f"other media in this run ({detail})",
        )
    return _S_FAIL_EMB, f"{n_total} frames extracted but no new vectors were indexed"


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


def _check_collection_compat_cli(
    specs: "list[tuple[AnyEmbedder, Indexer, str]]",
    settings: "Settings",
) -> list[str]:
    """Return mismatch descriptions between current config and existing indexed points.

    Thin adapter over :func:`scalar_forensic.safeguards.check_collection_compat`
    that pulls expected hashes from the loaded embedders (already computed for
    the upsert payload) and uses each spec's own Indexer client.
    """
    from scalar_forensic.safeguards import QdrantUnavailable, check_collection_compat

    if not specs:
        return []

    expected_dino_hash: str | None = None
    expected_sscd_hash: str | None = None
    for _embedder, indexer, model_hash in specs:
        if indexer.vector_name == "dino":
            expected_dino_hash = model_hash
        elif indexer.vector_name == "sscd":
            expected_sscd_hash = model_hash

    client = specs[0][1].client
    collection = specs[0][1].collection
    try:
        return check_collection_compat(
            client,
            collection,
            settings,
            expected_dino_hash=expected_dino_hash,
            expected_sscd_hash=expected_sscd_hash,
        )
    except QdrantUnavailable as exc:
        # During `sfn index` Qdrant is *required* — surface the failure rather
        # than swallow it.  The Indexer constructor would have failed earlier
        # if Qdrant were truly down, so reaching this branch implies a transient
        # error worth reporting.
        return [f"Qdrant unreachable while validating compatibility: {exc}"]


def index(
    input_dir: Path | None = typer.Argument(
        default=None, help="Root directory of images (overrides SFN_INPUT_DIR)"
    ),
    dino: bool = typer.Option(False, "--dino", help="Use DINOv2 backend (1024-dim semantic)"),
    sscd: bool = typer.Option(False, "--sscd", help="Use SSCD backend (512-dim copy-detection)"),
    faces: bool = typer.Option(
        False,
        "--faces",
        help="Also detect, embed and store faces (requires SFN_FACES_ENABLED=true)",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help=(
            "CSV report output path "
            "(default: <SFN_REPORT_DIR>/sfn_ingestion_<timestamp>.csv, "
            "SFN_REPORT_DIR defaults to data/reports)"
        ),
    ),
    allow_online: bool = typer.Option(
        False,
        "--allow-online",
        help=(
            "Allow outward internet connections (e.g. to HuggingFace Hub for first-time "
            "model downloads). Offline by default — see SFN_ALLOW_ONLINE in .env."
        ),
    ),
    reference: bool = typer.Option(
        False,
        "--reference",
        help=(
            "Index into the reference collection instead of the case collection. "
            "Requires SFN_REFERENCE_COLLECTION to be set in .env. "
            "Points are tagged with is_reference=true in their payload."
        ),
    ),
    ignore_config_mismatch: bool = typer.Option(
        False,
        "--ignore-config-mismatch",
        help=(
            "Skip embedding-configuration compatibility checks against existing"
            " collection points. Mixing incompatible embeddings in one collection"
            " will corrupt similarity results."
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

    if not dino and not sscd and not faces:
        typer.echo("[ERROR] Specify at least one of --dino, --sscd or --faces.", err=True)
        raise typer.Exit(1)

    # ── Face modality startup check (spec §13: fail at startup, not mid-run) ──
    face_pipeline = None
    _faces_done: set[str] = set()
    if faces:
        _face_err = (
            settings.face_startup_error()
            if settings.faces_enabled
            else "SFN_FACES_ENABLED must be 'true' to use --faces "
            "(see docs/specs/face-pipeline.md)."
        )
        if _face_err:
            typer.echo(f"[ERROR] {_face_err}", err=True)
            raise typer.Exit(1)
        for _note in settings.face_threshold_notes():
            typer.secho(f"note: {_note}", fg=typer.colors.YELLOW)
        from scalar_forensic.faces.indexing import FacePipeline  # deferred: optional deps

        face_pipeline = FacePipeline.from_settings(settings)
        if face_pipeline.store.collection_is_new():
            _auth_ref = typer.prompt(
                "First face-collection activation. Authorization reference (free text,"
                " recorded in the enablement record; empty allowed)",
                default="",
            )
            if not _auth_ref:
                typer.echo(
                    "[WARN] No authorization reference recorded for this activation.", err=True
                )
            face_pipeline.store.ensure_collection(
                face_pipeline.cfg, settings.examiner_id, _auth_ref or None
            )
            face_pipeline.audit.append(
                "enablement",
                examiner_id=settings.examiner_id,
                authorization_ref=_auth_ref or None,
                face_collection=settings.face_collection,
                case_collection=settings.collection,
            )
        for _warn in face_pipeline.store.check_compat(face_pipeline.cfg):
            typer.echo(f"[WARN] face collection config differs — {_warn}", err=True)
        _faces_done = face_pipeline.store.processed_hashes(face_pipeline.cfg.config_hash)

    if reference:
        if not settings.reference_collection:
            typer.echo(
                "[ERROR] --reference requires SFN_REFERENCE_COLLECTION to be set in .env",
                err=True,
            )
            raise typer.Exit(1)
        if settings.reference_collection == settings.collection:
            typer.echo(
                "[ERROR] SFN_REFERENCE_COLLECTION must differ from SFN_COLLECTION",
                err=True,
            )
            raise typer.Exit(1)
        target_collection = settings.reference_collection
    else:
        target_collection = settings.collection

    # Pre-flight: fail fast if a HuggingFace Hub model ID is configured while offline.
    err = settings.offline_model_error(need_dino=dino)
    if err:
        typer.echo(f"[ERROR] {err}", err=True)
        raise typer.Exit(1)

    # Resolve the CSV report path early so the user knows where it will land.
    # The per-run manifest is written next to it (see scalar_forensic.manifest).
    csv_path = report or settings.report_dir / f"sfn_ingestion_{datetime.now():%Y%m%d_%H%M%S}.csv"

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
            f"Connecting to Qdrant  collection={target_collection!r}  vector={vector_name!r} ..."
        )
        try:
            indexer = Indexer(
                url=settings.qdrant_url,
                collection=target_collection,
                vector_name=vector_name,
                embedding_dim=embedder.embedding_dim,
                api_key=settings.qdrant_api_key,
                initial_vectors_config=_initial_vectors_config,
                is_reference=reference,
            )
        except ValueError as exc:
            typer.echo(f"[ERROR] {exc}", err=True)
            raise typer.Exit(1)

        typer.echo("Computing model hash (may take a moment) ...")
        model_hash = embedder.model_hash
        typer.echo(f"  model_hash={model_hash}")

        specs.append((embedder, indexer, model_hash))

    mismatches = _check_collection_compat_cli(specs, settings)
    if mismatches:
        detail = "\n  ".join(mismatches)
        if ignore_config_mismatch:
            typer.echo(
                "[WARN] Embedding configuration mismatch"
                f" (--ignore-config-mismatch set):\n  {detail}",
                err=True,
            )
        else:
            typer.echo("[ERROR] Embedding configuration mismatch — indexing aborted.", err=True)
            typer.echo(f"  {detail}", err=True)
            typer.echo("", err=True)
            typer.echo(
                "Mixing incompatible embeddings in the same collection"
                " corrupts similarity results.",
                err=True,
            )
            typer.echo("Options:", err=True)
            typer.echo("  • Restore the original settings in .env, OR", err=True)
            typer.echo("  • Re-create the collection from scratch and re-index, OR", err=True)
            typer.echo(
                "  • Download the exact model version recorded in this collection"
                " (local DINOv2/SSCD only):",
                err=True,
            )
            typer.echo(
                "      uv run python scripts/download_models.py --dino --hash <stored_hash>",
                err=True,
            )
            typer.echo(
                "      uv run python scripts/download_models.py --sscd --hash <stored_hash>",
                err=True,
            )
            typer.echo("    (use the stored= value from the mismatch detail above), OR", err=True)
            typer.echo(
                "  • Pass --ignore-config-mismatch to proceed anyway (not recommended)", err=True
            )
            raise typer.Exit(1)

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
    tracker = _RateTracker()

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
        pre_results, face_results, pre_s = (
            ctx.pre_future.result() if ctx.pre_future is not None else ([], [], 0.0)
        )

        pre_by_path: dict[Path, Image.Image] = {}
        for p, result in zip(ctx.paths_to_pre, pre_results, strict=True):
            if isinstance(result, Exception):
                typer.echo(f"[WARN] Preprocessing failed for {p.name}: {result}", err=True)
                if records[p].status == "pending":
                    records[p].status = _S_FAIL_PRE
                    records[p].reason = f"preprocessing error: {result}"
            else:
                pre_by_path[p] = result

        # ── Commit this batch's in-loop face results (main thread only) ───────
        # Before the unique_pairs early-return below: face results are
        # independent of whether any image survived embed preprocessing.
        for _fp, _fsha, _fvmeta, _fres in face_results:
            if isinstance(_fres, Exception):
                _face_stats["failed"] += 1
                typer.echo(f"[WARN] Face processing failed for {_fp.name}: {_fres}", err=True)
            else:
                _commit_face_result(_fp, _fsha, _fvmeta, _fres)

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

        # Recompute duplicate-skip count after failure reclassification.
        duplicate_skips_in_batch = sum(
            1 for p, _ in ctx.path_hash_pairs if records[p].status == _S_SKIP_DUP
        )

        # ── Per-model loop: normalize + embed, collect upsert jobs ────────────
        n_frames_in_batch = sum(1 for p, _ in ctx.path_hash_pairs if p in vmeta_by_path)
        n_plain_in_batch = len(ctx.path_hash_pairs) - n_frames_in_batch
        model_segments: list[str] = []
        upsert_jobs: list = []

        for spec_idx, (embedder, indexer, model_hash) in enumerate(specs):
            to_embed = [(p, h) for p, h in ctx.to_embed_per_spec[spec_idx] if p not in pre_failures]
            n_skipped = duplicate_skips_in_batch + (len(unique_pairs) - len(to_embed))
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
                pct = ctx.imgs_at_batch / total_image_count * 100
                bar = _progress_bar(pct)
                sep = "─" * 68
                eta_s = tracker.eta(total_image_count - ctx.imgs_at_batch)
                rate_str = f"{tracker.rate:.1f} img/s" if tracker.rate is not None else "— img/s"
                eta_str = (
                    f"  ·  ~{_fmt_duration(eta_s)} remaining at current rate"
                    if eta_s is not None
                    else ""
                )
                typer.echo(
                    f"  {sep}\n"
                    f"  [{bar}]  {ctx.imgs_at_batch:,} / {total_image_count:,}"
                    f"  ({pct:.1f}%)\n"
                    f"  {rate_str}{eta_str}\n"
                    f"  {sep}"
                )

    def _timed_preprocess(
        paths: "list[Path]",
        data_by: "dict[Path, bytes]",
        cap: int,
        face_jobs: "list[tuple[Path, str, dict | None, bool]]",
    ) -> tuple[list, list, float]:
        """Preprocess for embedding + in-loop face detection in one worker pass.

        For each face job ``(path, sha256, vmeta, want_pre)`` the image is
        decoded ONCE at native resolution (decode_shared): the array feeds
        YuNet detection + alignment + SFace embedding, and — where that is
        byte-identical to the normal preprocess path — the downscaled embed
        image is derived from it instead of decoding the bytes a second time.
        Detection runs sequentially here: this pool has one worker, the
        residual pass only starts after the pool is drained, and OpenCV's
        YuNet instance is not thread-safe.  Face-store I/O stays with the
        caller (main thread).

        Returns ``(pre_results_aligned_with_paths, face_results, elapsed_s)``
        where face results are ``(path, sha256, vmeta, FaceIndexResult |
        Exception)``.
        """
        t0 = perf_counter()
        face_set = {j[0] for j in face_jobs}
        plain = [p for p in paths if p not in face_set]
        by_path: dict[Path, object] = dict(
            zip(plain, preprocess_batch([data_by[p] for p in plain], cap=cap), strict=True)
        )
        face_results: list[tuple[Path, str, dict | None, object]] = []
        for p, sha, vmeta, want_pre in face_jobs:
            img = None
            derived: object = None
            try:
                img, derived = _face_decode_shared(data_by[p], cap)
            except Exception as exc:  # noqa: BLE001 — one bad file must not end the run
                derived = exc
            if want_pre:
                if isinstance(derived, Exception):
                    by_path[p] = derived
                elif derived is None:
                    # JPEG (draft() shortcut) or an orientation edge case:
                    # decode the embed input separately so it stays
                    # byte-identical to a faces-off run of the same media.
                    by_path[p] = preprocess_batch([data_by[p]], cap=cap)[0]
                else:
                    by_path[p] = derived
            if img is None:
                face_results.append((p, sha, vmeta, derived))  # the decode Exception
                continue
            try:
                fres: object = face_pipeline.process_image(
                    None,
                    image_hash=sha,
                    image_path=str(p.resolve()),
                    video_hash=(vmeta or {}).get("video_hash"),
                    video_path=(vmeta or {}).get("video_path"),
                    frame_timecode_ms=(vmeta or {}).get("frame_timecode_ms"),
                    img=img,
                )
            except Exception as exc:  # noqa: BLE001
                fres = exc
            face_results.append((p, sha, vmeta, fres))
        return [by_path[p] for p in paths], face_results, perf_counter() - t0

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

    # ── Per-run manifest ──────────────────────────────────────────────────────
    # Config snapshot + model hashes + discovered input list, written before any
    # embedding or upsert.  Placed after the hash pass so every readable input
    # already carries its sha256; unreadable/unsupported files appear as
    # path+size only.  Records at this point hold only *input* files — video
    # frames are derived later, during slicing, and belong in Qdrant payloads,
    # not here.
    _manifest_models: dict[str, dict] = {
        idx.vector_name: {
            "model_name": emb.model_name,
            "model_hash": mh,
            "embedding_dim": emb.embedding_dim,
        }
        for emb, idx, mh in specs
    }
    if face_pipeline is not None:
        _manifest_models["faces"] = face_pipeline.cfg.to_payload()
    _manifest_path = write_run_manifest(
        csv_path,
        settings=settings,
        target_collection=target_collection,
        input_root=resolved_input,
        files=[input_file_entry(p, sha256=rec.sha256 or None) for p, rec in records.items()],
        models=_manifest_models,
    )
    typer.echo(f"Run manifest → {_manifest_path}")

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

        _slice_tracker = _RateTracker()
        _slice_total_frames = 0  # running total across all videos
        _SLICE_BLOCK = 50  # frames per rate update + progress line

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
            _block_count = 0  # how many complete blocks emitted — for progress-box cadence

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

                        # Run-wide progress box every 5 blocks (= 250 frames);
                        # the total is an estimate from the probed durations.
                        if _block_count % 5 == 0 and _total_expected_frames > 0:
                            _remaining = max(_total_expected_frames - _slice_total_frames, 0)
                            _pct = _slice_total_frames / _total_expected_frames * 100
                            _bar = _progress_bar(_pct)
                            _sep = "─" * 68
                            _eta_s = _slice_tracker.eta(_remaining)
                            _rate = _slice_tracker.rate
                            _rate_str = f"{_rate:.1f} fps" if _rate is not None else "— fps"
                            _eta_str = (
                                f"  ·  ~{_fmt_duration(_eta_s)} remaining at current rate"
                                if _eta_s is not None
                                else ""
                            )
                            typer.echo(
                                f"  {_sep}\n"
                                f"  [{_bar}]  {_slice_total_frames:,}"
                                f" / ~{_total_expected_frames:,}"
                                f"  ({_pct:.1f}%)\n"
                                f"  {_rate_str}{_eta_str}\n"
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

    # ── Face work: in-loop detection + residual pass ──────────────────────────
    # Media the batch loop visits get their face detection folded into its
    # background preprocess worker, reusing one native-resolution decode for
    # detection and (where byte-identical) the embed input.  Media the loop
    # does not visit — already embedded, run-duplicates, --faces-only runs —
    # keep the original decode-per-item pass after the loop.  The face markers
    # remain the sole idempotency mechanism on both paths, and every store
    # write (points → vector clear → marker) happens on the main thread with
    # the ordering guarantees unchanged.
    _face_pending: dict[Path, tuple[str, dict | None]] = {}
    _face_stats = {"detected": 0, "kept": 0, "review_only": 0, "dropped_noncanon": 0, "failed": 0}
    _face_rejected: dict[str, int] = {}
    _face_review_reasons: dict[str, int] = {}
    # Collected across the run, acted on once at the end: prompting per
    # medium would ask the same question hundreds of times, and deleting
    # biometric observations without showing what they are first is not a
    # decision the tool gets to make on the operator's behalf.
    _stale_points: list[dict] = []
    _video_rollup: dict[str, dict] = {}
    _n_face_media = 0
    if face_pipeline is not None:
        from scalar_forensic.faces.indexing import decode_shared as _face_decode_shared

        for _p, _sha in _file_hashes.items():
            if _sha not in _faces_done:
                _face_pending[_p] = (_sha, vmeta_by_path.get(_p))
        _n_face_media = len(_face_pending)
        typer.echo(
            f"\nFaces: processing {_n_face_media:,} media item(s)  →  {settings.face_collection}"
        )

    def _commit_face_result(_p: Path, _sha: str, _vmeta: "dict | None", _fres) -> None:
        """Store one medium's face results and its marker; update run counters."""
        _marker = face_pipeline.store.marker_point(
            _sha,
            (_vmeta or {}).get("video_hash"),
            face_pipeline.cfg.config_hash,
            _fres.n_detected,
            _fres.n_kept,
            _fres.rejected,
            n_review_only=_fres.n_review_only,
            review_only_reasons=_fres.review_only_reasons,
            n_dropped_noncanonical=_fres.n_dropped_noncanonical,
        )
        # Points first, marker last, with the vector clear between them.
        # The marker is this medium's idempotency record: once it is
        # committed for this config hash, the medium is never reprocessed.
        # Committing it in the same call as the points would make a failed
        # clear permanent and invisible — a point whose payload says
        # review-only while its vector is still live in the index, which is
        # the one state this design exists to prevent.  clear_face_vector
        # can raise (delete_vectors 404s on an unknown id, see
        # tests/faces/test_store_integration.py), so the ordering is
        # load-bearing, not stylistic.
        #
        # Checked before the upsert, while the collection still holds only
        # what previous runs wrote: point ids come from the bbox, not from
        # any threshold, so a threshold change rewrites a point in place
        # and leaves nothing stale.  What does survive is a face that has
        # dropped below the review gate (no point produced at all) or an
        # observation whose bbox moved because the detector changed — in
        # both cases the old point is still there, still carrying its old
        # provenance, and if it was embedded it is still searchable.
        # Tag each stale point with the file it came from while that is still
        # known here — the stored payload fields fetched for a stale point do
        # not include a path, and the end-of-run prompt has to name the files
        # whose observations would be deleted, not just count them.  For a
        # video frame the operator's file is the video, not the frame JPEG.
        _stale_here = face_pipeline.store.stale_face_points(
            _sha, {str(pt.id) for pt in _fres.points}
        )
        _stale_src = (_vmeta or {}).get("video_path") or str(_p)
        for _sp in _stale_here:
            _sp["source_path"] = _stale_src
        _stale_points.extend(_stale_here)
        face_pipeline.store.upsert_faces(_fres.points)
        # Every review-only point, not only genuinely demoted ones:
        # delete_vectors is idempotent and ignores absent vectors, so
        # first-time review-only observations cost nothing.  An upsert with
        # vector={} must not be trusted to clear a vector a previous run
        # stored at the same point id -- a review-only point that kept its
        # vector would still be returned by similarity search.  Only ever
        # pass review-only ids: clearing an embedded point's vector
        # destroys data recoverable only by a full re-index.
        face_pipeline.store.clear_face_vector(_fres.review_only_point_ids)
        face_pipeline.store.upsert_faces([_marker])
        _face_stats["detected"] += _fres.n_detected
        _face_stats["kept"] += _fres.n_kept
        _face_stats["review_only"] += _fres.n_review_only
        _face_stats["dropped_noncanon"] += _fres.n_dropped_noncanonical
        for _reason, _n in _fres.rejected.items():
            _face_rejected[_reason] = _face_rejected.get(_reason, 0) + _n
        for _reason, _n in _fres.review_only_reasons.items():
            _face_review_reasons[_reason] = _face_review_reasons.get(_reason, 0) + _n
        _vh_roll = (_vmeta or {}).get("video_hash")
        if _vh_roll:
            _agg = _video_rollup.setdefault(
                _vh_roll,
                {
                    "n_frames": 0,
                    "n_detected": 0,
                    "n_kept": 0,
                    "rejected": {},
                    "n_review_only": 0,
                    "review_only_reasons": {},
                    "n_dropped_noncanonical": 0,
                },
            )
            _agg["n_frames"] += 1
            _agg["n_detected"] += _fres.n_detected
            _agg["n_kept"] += _fres.n_kept
            _agg["n_review_only"] += _fres.n_review_only
            _agg["n_dropped_noncanonical"] += _fres.n_dropped_noncanonical
            for _reason, _n in _fres.rejected.items():
                _agg["rejected"][_reason] = _agg["rejected"].get(_reason, 0) + _n
            for _reason, _n in _fres.review_only_reasons.items():
                _agg["review_only_reasons"][_reason] = (
                    _agg["review_only_reasons"].get(_reason, 0) + _n
                )

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
            "  ·  progress every 10 batches; ETA extrapolates the current smoothed rate"
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

            # ── Submit preprocessing (+ in-loop face detection) in background ─
            # Cap so DINOv2 gets its configured resolution and SSCD ≥ 331 px.
            paths_to_pre = [p for p, _ in unique_pairs if p in needs_pre]
            face_jobs: list[tuple[Path, str, dict | None, bool]] = []
            if face_pipeline is not None:
                for p, _h in unique_pairs:
                    _face_info = _face_pending.pop(p, None)
                    if _face_info is not None:
                        face_jobs.append((p, _face_info[0], _face_info[1], p in needs_pre))
            pre_future: Future[tuple[list, list, float]] | None = (
                _pre_pool.submit(
                    _timed_preprocess, paths_to_pre, data_by_path, _effective_cap, face_jobs
                )
                if paths_to_pre or face_jobs
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
        _vf_rundup_total: dict[Path, int] = {}
        for _fp, _sv in _frame_source.items():
            if records[_fp].status == _S_INDEXED:
                _vf_indexed_total[_sv] = _vf_indexed_total.get(_sv, 0) + 1
            elif records[_fp].status == _S_SKIP_IDX:
                _vf_skipped_total[_sv] = _vf_skipped_total.get(_sv, 0) + 1
            elif records[_fp].status == _S_SKIP_DUP:
                _vf_rundup_total[_sv] = _vf_rundup_total.get(_sv, 0) + 1

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
            _vrec.status, _vrec.reason = _video_status(
                _total,
                _vf_indexed_total.get(_vp, 0),
                _vf_skipped_total.get(_vp, 0),
                _vf_rundup_total.get(_vp, 0),
            )

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

    # ── Face residual pass ───────────────────────────────────────────────────
    # The batch loop only iterates *not-yet-embedded* media, so it cannot be
    # the only place faces happen — that would silently yield zero faces on an
    # already-indexed case.  Whatever the loop did not consume from
    # _face_pending (already-embedded media, run-duplicates, read failures,
    # --faces-only runs) is processed here: read + decode + detect + embed fan
    # out over a thread pool with per-thread YuNet detectors (efficiency audit
    # 2026-08-13 §4 fix 1), while every store write still happens below on the
    # main thread, per medium, in the points → vector clear → marker order.
    if face_pipeline is not None:
        from scalar_forensic.faces.indexing import process_media_threaded

        _residual_jobs = [(_p, _fi[0], _fi[1]) for _p, _fi in _face_pending.items()]
        for _p, _sha, _vmeta, _fres in process_media_threaded(face_pipeline, _residual_jobs):
            if isinstance(_fres, Exception):  # one bad file must not end the run
                _face_stats["failed"] += 1
                typer.echo(f"[WARN] Face processing failed for {_p.name}: {_fres}", err=True)
                continue
            _commit_face_result(_p, _sha, _vmeta, _fres)

        # Per-video rollup markers, written once each after their frames.
        for _vh_roll, _agg in _video_rollup.items():
            face_pipeline.store.upsert_faces(
                [
                    face_pipeline.store.video_rollup_point(
                        _vh_roll,
                        face_pipeline.cfg.config_hash,
                        _agg["n_detected"],
                        _agg["n_kept"],
                        _agg["rejected"],
                        _agg["n_frames"],
                        n_review_only=_agg["n_review_only"],
                        review_only_reasons=_agg["review_only_reasons"],
                        n_dropped_noncanonical=_agg["n_dropped_noncanonical"],
                    )
                ]
            )
        # ── Stale observations: show, ask, then delete ───────────────────────
        _n_stale_removed = 0
        _n_stale_chip_files = 0
        if _stale_points:
            _by_status: dict[str, int] = {}
            _by_cfg: dict[str, int] = {}
            for _sp in _stale_points:
                _st = _sp.get("embedding_status") or "embedded"
                _by_status[_st] = _by_status.get(_st, 0) + 1
                _cfg_h = _sp.get("pipeline_config_hash") or "unknown"
                _by_cfg[_cfg_h] = _by_cfg.get(_cfg_h, 0) + 1
            typer.echo("")
            typer.secho(
                f"{len(_stale_points):,} stale face observation(s) found: stored by an earlier "
                "run and not produced again by this one.",
                fg=typer.colors.YELLOW,
            )
            typer.echo(
                "  by kind:   "
                + ", ".join(
                    f"{_n} {'review-only' if _s == 'review_only' else _s}"
                    for _s, _n in sorted(_by_status.items())
                )
            )
            typer.echo(
                "  by config: "
                + ", ".join(f"{_n} under {_c[:12]}" for _c, _n in sorted(_by_cfg.items()))
            )
            _stale_lines = _stale_path_listing(_stale_points)
            _n_stale_files = len({_sp.get("source_path") for _sp in _stale_points})
            typer.echo(f"  by file ({_n_stale_files:,}):")
            for _line in _stale_lines:
                typer.echo(_line)
            if _by_status.get("embedded"):
                typer.echo(
                    "  Embedded ones are still returned by similarity search, under thresholds "
                    "this run no longer applies."
                )
            typer.echo(
                "  Adjudications reference observation_key, not point ids, so they are not "
                "deleted — but a deleted observation's key stops resolving."
            )
            # An unattended run has no answer to give: click aborts on EOF,
            # which here would kill the run at the very end, after every marker
            # was already written.  Catch it and leave the observations alone —
            # deleting biometric data is not something to infer from a closed
            # stdin, and "not deleted, and here is why" is the safe outcome.
            try:
                _do_delete = typer.confirm("Delete these stale observations?", default=False)
            except (typer.Abort, EOFError):
                _do_delete = False
                typer.echo("")
                typer.secho(
                    "  Non-interactive run: not deleting. Re-run from a terminal to confirm.",
                    fg=typer.colors.YELLOW,
                )
            if _do_delete:
                _ids = [str(_sp["id"]) for _sp in _stale_points]
                _stale_result = face_pipeline.store.delete_face_points(_ids)
                _n_stale_removed = _stale_result.n_points
                if settings.face_store_dir is not None:
                    from scalar_forensic.faces.chips import chip_paths as _chip_paths
                    from scalar_forensic.faces.chips import (
                        review_chip_paths as _review_chip_paths,
                    )

                    for _chash in face_pipeline.store.unreferenced_chip_hashes(
                        _stale_result.chip_hashes
                    ):
                        for _path in dict.fromkeys(
                            (
                                *_chip_paths(Path(settings.face_store_dir), _chash),
                                *_review_chip_paths(Path(settings.face_store_dir), _chash),
                            )
                        ):
                            if _path.exists():
                                _path.unlink()
                                _n_stale_chip_files += 1
            else:
                typer.secho(
                    f"  Left in place: {len(_stale_points):,} stale observation(s) remain in "
                    f"{settings.face_collection}. Re-run and confirm, or purge the media.",
                    fg=typer.colors.YELLOW,
                )

        # "kept" no longer names one population: a review-only observation is
        # kept on disk and in the collection but is not comparable.  The summary
        # must therefore reconcile — detected = comparable + retained + rejected
        # — or it understates how many biometric crops the run wrote.
        _rej_str = ", ".join(f"{_n} {_r}" for _r, _n in sorted(_face_rejected.items()))
        _rev_str = ", ".join(f"{_n} {_r}" for _r, _n in sorted(_face_review_reasons.items()))
        typer.echo(
            f"faces: {_face_stats['detected']:,} detected"
            f"  │  {_face_stats['kept']:,} comparable"
            + (
                f"  │  {_face_stats['review_only']:,} retained for review"
                + (f" ({_rev_str})" if _rev_str else "")
                if _face_stats["review_only"]
                else ""
            )
            + (f"  │  {sum(_face_rejected.values()):,} rejected: {_rej_str}" if _rej_str else "")
            + (f"  │  {_face_stats['failed']:,} failed" if _face_stats["failed"] else "")
            + (f"  │  {_n_stale_removed:,} stale removed" if _n_stale_removed else "")
        )
        face_pipeline.audit.append(
            "index_run",
            examiner_id=settings.examiner_id,
            input_dir=str(resolved_input),
            n_media=_n_face_media,
            n_detected=_face_stats["detected"],
            n_kept=_face_stats["kept"],
            n_review_only=_face_stats["review_only"],
            review_only_reasons=_face_review_reasons,
            n_rejected=_face_rejected,
            n_dropped_noncanonical=_face_stats["dropped_noncanon"],
            n_failed=_face_stats["failed"],
            # Detected and removed are recorded separately on purpose: a run
            # where the operator declined must be distinguishable in the audit
            # trail from one where nothing was stale.
            n_stale_detected=len(_stale_points),
            n_stale_removed=_n_stale_removed,
            n_stale_chip_files=_n_stale_chip_files,
            stale_observation_keys=[
                _sp.get("observation_key") for _sp in _stale_points if _sp.get("observation_key")
            ],
            pipeline_config_hash=face_pipeline.cfg.config_hash,
        )

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


# ── Face-modality maintenance (separate console script) ──────────────────────
# `sfn` itself stays a single-command `typer.run(index)` app, so adding a named
# command here would break every documented `sfn <dir>` invocation.  Purge ships
# as its own entry point instead, mirroring the sfn / sfn-web split.
faces_app = typer.Typer(help="Face-modality maintenance commands.")


@faces_app.callback()
def _faces_callback() -> None:
    """Keep this a command group.

    Typer collapses a single-command app into the top level, which would
    make the invocation `sfn-faces --media ...` instead of the documented
    `sfn-faces purge --media ...`.
    """


@faces_app.command()
def purge(
    media: str = typer.Option(None, "--media", help="Purge faces for one media sha256"),
    all_: bool = typer.Option(False, "--all", help="Purge ALL face observations"),
) -> None:
    """Delete stored face observations and their chip files."""
    if bool(media) == bool(all_):
        typer.echo("[ERROR] Specify exactly one of --media <sha256> or --all.", err=True)
        raise typer.Exit(1)

    settings = Settings()
    err = (
        settings.face_startup_error()
        if settings.faces_enabled
        else "SFN_FACES_ENABLED must be 'true' to use face commands."
    )
    if err:
        typer.echo(f"[ERROR] {err}", err=True)
        raise typer.Exit(1)
    for _note in settings.face_threshold_notes():
        typer.secho(f"note: {_note}", fg=typer.colors.YELLOW)

    from qdrant_client import QdrantClient

    from scalar_forensic.faces.audit import AuditLog
    from scalar_forensic.faces.chips import chip_paths, review_chip_paths
    from scalar_forensic.faces.store import FaceStore

    if all_ and not typer.confirm(
        "Delete ALL face observations in "
        f"{settings.face_collection}? The enablement record is preserved."
    ):
        typer.echo("Aborted.")
        raise typer.Exit(1)

    store = FaceStore(
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        settings.face_collection,
        settings.collection,
        0,  # dim unused on the purge path — no collection is created here
    )
    result = store.purge_all() if all_ else store.purge_media(media)

    # Chips are content-addressed, so a hash freed by this purge may still be
    # referenced by a surviving observation (an exact-duplicate medium, or an
    # embedded face whose source crop matched a review-only one).  Filter through
    # the store before unlinking.  Three limits are deployment properties, not
    # code defects, and are documented rather than engineered away:
    #   1. The check is collection-scoped but the chip store is not — purging
    #      case A can unlink a chip case B references unless SFN_FACE_STORE_DIR
    #      is set per case (INSTALL.md; the rule check_compat enforces for vectors).
    #   2. Check-then-unlink: a concurrent index run can write a referencing
    #      point between the two, leaving a dangling reference.  Single writer.
    #   3. Both are mitigated, not cured, by chips being re-derivable from media.
    n_chip_files = 0
    if settings.face_store_dir is not None:
        for chash in store.unreferenced_chip_hashes(result.chip_hashes):
            # Both builders, deliberately: a hash may be an aligned hash (PNG)
            # or a review hash (JPEG + thumb), and the caller cannot tell which.
            # The two overlap on the review pair; exists() keeps the count honest.
            for path in dict.fromkeys(
                (
                    *chip_paths(Path(settings.face_store_dir), chash),
                    *review_chip_paths(Path(settings.face_store_dir), chash),
                )
            ):
                if path.exists():
                    path.unlink()
                    n_chip_files += 1

    store_dir = Path(settings.face_store_dir) if settings.face_store_dir else Path("data")
    audit_dir = store_dir.parent if settings.face_store_dir else store_dir
    AuditLog(audit_dir / "face_audit.log").append(
        "purge",
        examiner_id=settings.examiner_id,
        scope="all" if all_ else "media",
        image_hash=None if all_ else media,
        n_points=result.n_points,
        n_chip_files=n_chip_files,
    )
    typer.echo(f"Purged {result.n_points:,} face point(s) and {n_chip_files:,} chip file(s).")


def faces_main() -> None:
    faces_app()


# ── Viewing-copy retention (separate console script) ─────────────────────────
# Same shape and same reason as sfn-faces above: `sfn` stays a single-command
# app, so retention gets its own entry point rather than a subcommand.
video_app = typer.Typer(help="Viewing-copy (video playback) maintenance commands.")


@video_app.callback()
def _video_callback() -> None:
    """Keep this a command group (see `_faces_callback`)."""


@video_app.command("purge")
def video_purge(
    media: str = typer.Option(None, "--media", help="Purge viewing copies for one media sha256"),
    all_: bool = typer.Option(False, "--all", help="Purge ALL viewing copies"),
) -> None:
    """Delete derived viewing copies from the video cache (spec §13).

    The LRU ceiling is the only automatic retention mechanism; this is the
    explicit one, and it is the answer to "what derived renderings existed and
    when were they destroyed".  A background TTL sweep was considered and dropped
    for exactly that reason (§6.4).

    Nothing here touches evidence.  The cache holds viewing copies only — the
    source files and the frame JPEGs written at index time are untouched, and a
    purged rendering can be regenerated by playing the video again.
    """
    if bool(media) == bool(all_):
        typer.echo("[ERROR] Specify exactly one of --media <sha256> or --all.", err=True)
        raise typer.Exit(1)

    from scalar_forensic.video_playback.cache import purge as purge_cache

    settings = Settings()
    if settings.video_cache_dir is None:
        typer.echo("[ERROR] SFN_VIDEO_CACHE_DIR is unset: there is no cache to purge.", err=True)
        raise typer.Exit(1)
    cache_dir = settings.video_cache_dir

    if all_ and not typer.confirm(f"Delete ALL viewing copies in {cache_dir}?"):
        typer.echo("Aborted.")
        raise typer.Exit(1)

    try:
        result = purge_cache(cache_dir, media=media, all_=all_)
    except ValueError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1) from exc

    # Filed, not only printed (§13, §7.3): "what derived renderings existed and
    # when were they destroyed" is a question asked after the terminal that
    # printed this has been closed.  The log lives beside the cache directory,
    # so a purge does not delete the record of itself.
    from scalar_forensic.video_playback.audit import record_purge

    record_purge(
        settings,
        examiner_id=settings.examiner_id,
        scope="all" if all_ else "media",
        video_sha256=None if all_ else media,
        videos=result.videos,
        files=result.files,
        bytes_freed=result.bytes_freed,
    )
    typer.echo(
        f"Purged {result.videos:,} video(s), {result.files:,} file(s), "
        f"{result.bytes_freed:,} bytes from {cache_dir} "
        f"(examiner: {settings.examiner_id or 'unset'})."
    )
    for digest in result.digests:
        typer.echo(f"  {digest}")


@video_app.command("render")
def video_render(
    path: str = typer.Option(..., "--path", help="The source video file"),
    at: float = typer.Option(
        0.0, "--at", help="Timecode in seconds; the chunk containing it is described"
    ),
    full: bool = typer.Option(
        False, "--full", help="Describe the full viewing copy instead of a chunk"
    ),
) -> None:
    """Print the exact invocation that produced a rendering (spec §7.2).

    A rendering an analyst watched is an interpretation of evidence — lossy,
    possibly tone-mapped, possibly rescaled — and §7.2's answer to that is that a
    reviewer must be able to reproduce it.  This prints the recorded argv, the
    pipeline it ran under, when it ran and on whose act.

    With no record for that window, it prints what *this host would run now*,
    labelled as a reproduction recipe rather than a record: the two are different
    claims and are never printed in the same shape.  A source whose bytes have
    changed since the recorded rendering is reported as stale (§7.1) rather than
    answered as though nothing had happened.

    No allowed-root containment check here, deliberately, and this is **not** a
    second implementation of ``_resolve_video_path``: that control exists to stop
    a *remote* caller naming a file outside ``SFN_INPUT_DIR``, and an operator at
    this shell already has the filesystem.
    """
    from scalar_forensic.video import VIDEO_EXTENSIONS
    from scalar_forensic.video_playback.audit import SCOPE_CHUNK, SCOPE_FULL, reproduction_report

    settings = Settings()
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        typer.echo(f"[ERROR] {p.name} is not a video file this tool indexes or plays.", err=True)
        raise typer.Exit(1)
    if not p.is_file():
        typer.echo(f"[ERROR] No such file: {p}", err=True)
        raise typer.Exit(1)
    if full and at:
        # A full copy has no window, so an --at beside it would be silently
        # ignored — and a reviewer would read a timecode the answer never used.
        typer.echo("[ERROR] --at describes a chunk; --full has no timecode.", err=True)
        raise typer.Exit(1)

    for line in reproduction_report(
        settings, p, scope=SCOPE_FULL if full else SCOPE_CHUNK, at=None if full else at
    ):
        typer.echo(line)


def video_main() -> None:
    video_app()
