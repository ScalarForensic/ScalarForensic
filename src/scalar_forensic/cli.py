"""CLI entry point for ScalarForensic."""

import csv
import os
from collections import Counter
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
    n_videos: int = 0,
    n_video_frames_indexed: int = 0,
    n_video_frames_skipped: int = 0,
) -> None:
    counts = Counter(r.status for r in records.values())
    total = len(records)
    n_image = total - counts[_S_UNSUPPORTED] - n_videos

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


def _index_video(
    video_path: Path,
    v_idx: int,
    v_total: int,
    records: dict[Path, "_FileRecord"],
    specs: list,
    settings: Settings,
    indexed_counts: list[int],
    skipped_counts: list[int],
    failed_counts: list[int],
    pyav_version: str,
) -> tuple[int, int]:
    """Extract frames from one video, embed them, and upsert into Qdrant.

    Returns (frames_indexed, frames_skipped) across all models.
    """
    rec = records[video_path]

    # Hash the video file in chunks — avoids loading multi-GB files into RAM.
    try:
        video_hash = hash_file(video_path)
        video_hash_md5 = hash_file_md5(video_path)
    except OSError as exc:
        typer.echo(f"[WARN] Cannot read video {video_path.name}: {exc}", err=True)
        rec.status = _S_FAIL_READ
        rec.reason = f"read error: {exc}"
        return 0, 0

    rec.sha256 = video_hash
    rec.md5 = video_hash_md5

    video_abs_path = str(video_path.resolve())

    def _vmeta(frame) -> dict:
        return {
            "video_hash": video_hash,
            "video_path": video_abs_path,
            "frame_timecode_ms": frame.timecode_ms,
            "frame_index": frame.frame_index,
            "extraction_fps": settings.video_fps,
            "pyav_version": pyav_version,
        }

    # Shared metadata is the same for all frames / all batches.
    shared_meta_per_spec = [
        {
            "model_name": embedder.model_name,
            "model_hash": model_hash,
            "embedding_dim": embedder.embedding_dim,
            "normalize_size": embedder.normalize_size,
            "inference_dtype": embedder.inference_dtype,
            "library_versions": get_library_versions(),
        }
        for embedder, _, model_hash in specs
    ]

    frames_indexed = 0
    frames_skipped = 0
    total_frames_seen = 0

    # Stream frames in batches of batch_size — keeps peak memory bounded to
    # O(batch_size * frame_pixels) regardless of max_frames.
    try:
        frame_gen = extract_frames(
            video_path, fps=settings.video_fps, max_frames=settings.video_max_frames
        )
        frame_batch_gen = batched(frame_gen, settings.batch_size)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[WARN] Frame extraction setup failed for {video_path.name}: {exc}", err=True)
        rec.status = _S_FAIL_PRE
        rec.reason = f"frame extraction error: {exc}"
        return 0, 0

    any_frames = False
    try:
        for raw_batch in frame_batch_gen:
            frames = list(raw_batch)
            if not frames:
                continue
            any_frames = True
            total_frames_seen += len(frames)

            virtual_paths = [Path(make_virtual_path(video_path, f)) for f in frames]
            frame_hashes = [f.frame_hash for f in frames]
            frame_images = [f.image for f in frames]

            # Thumbnails
            if settings.thumbnail_dir is not None:
                for frame, fhash in zip(frames, frame_hashes):
                    thumb_path = settings.thumbnail_dir / f"{fhash}.jpg"
                    if not thumb_path.exists():
                        try:
                            _write_thumbnail(frame.image, thumb_path, settings.thumbnail_size)
                        except Exception as exc:  # noqa: BLE001
                            typer.echo(
                                f"[WARN] Thumbnail failed for frame {frame.frame_index} of "
                                f"{video_path.name}: {exc}",
                                err=True,
                            )

            for spec_idx, (embedder, indexer, _) in enumerate(specs):
                # Dedup by virtual path (video identity + timecode)
                vpath_strings = [str(p) for p in virtual_paths]
                already = indexer.get_indexed_paths(vpath_strings)
                to_embed_idx = [i for i, p in enumerate(vpath_strings) if p not in already]
                if spec_idx == 0:
                    frames_skipped += len(frames) - len(to_embed_idx)

                if not to_embed_idx:
                    continue

                pre = preprocess_pil_batch([frame_images[i] for i in to_embed_idx])
                embed_paths = [virtual_paths[i] for i in to_embed_idx]
                embed_hashes = [frame_hashes[i] for i in to_embed_idx]
                embed_vmeta = [_vmeta(frames[i]) for i in to_embed_idx]

                try:
                    norm = embedder.normalize_batch_bytes(pre)
                    embeddings = embedder.embed_images(norm)
                except Exception as exc:  # noqa: BLE001
                    typer.echo(
                        f"[WARN] Embedding failed for {len(pre)} frames of "
                        f"{video_path.name} [{type(embedder).__name__}]: {exc}",
                        err=True,
                    )
                    failed_counts[spec_idx] += len(pre)
                    continue

                indexer.upsert_batch(
                    embed_paths,
                    embed_hashes,
                    embeddings,
                    shared_meta_per_spec[spec_idx],
                    video_metadata=embed_vmeta,
                )
                indexed_counts[spec_idx] += len(pre)
                if spec_idx == 0:
                    frames_indexed += len(pre)

    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[WARN] Frame extraction failed for {video_path.name}: {exc}", err=True)
        rec.status = _S_FAIL_PRE
        rec.reason = f"frame extraction error: {exc}"
        return frames_indexed, frames_skipped

    if not any_frames:
        typer.echo(f"  [WARN] No frames extracted from {video_path.name}", err=True)
        rec.status = _S_UNSUPPORTED
        rec.reason = "no frames extracted"
        return 0, 0

    rec.status = _S_INDEXED
    rec.reason = f"{total_frames_seen} frames extracted"

    # frames_skipped tracks only spec_idx==0 (symmetric with frames_indexed).
    n_skip = frames_skipped
    typer.echo(
        f"  video {v_idx}/{v_total} [{video_path.name}]: "
        f"{total_frames_seen} frames, {frames_indexed} embedded, {n_skip} skipped (dup)"
    )
    return frames_indexed, n_skip


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
        typer.echo(
            f"  backend={type(embedder).__name__}  dim={embedder.embedding_dim}"
            f"  device={embedder.device}  fp16={fp16}  compiled=True"
        )
        if fp16:
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
            upsert_str = f"  |  upsert {upsert_wall_s:.2f}s" if upsert_jobs else ""
            shared = (
                f"  batch {batch_num} [{n_imgs} imgs  {batch_bytes / 1e6:.1f} MB"
                f"  {_fmt_rate(n_imgs, wall_s, 'img')} total]"
                f"  read {read_s:.2f}s ({_fmt_mbps(batch_bytes, read_s)})"
                f"  hash {hash_s:.2f}s  pre {pre_s:.2f}s"
            )
            typer.echo(shared + "  |  " + "  |  ".join(model_segments) + upsert_str)

    # ── Video processing pass ─────────────────────────────────────────────────
    n_video_frames_indexed = 0
    n_video_frames_skipped = 0

    if video_paths:
        pyav_version = get_pyav_version()
        typer.echo(f"\nProcessing {len(video_paths):,} video file(s) (PyAV {pyav_version}) ...")
        for v_idx, video_path in enumerate(video_paths, 1):
            vi, vs = _index_video(
                video_path=video_path,
                v_idx=v_idx,
                v_total=len(video_paths),
                records=records,
                specs=specs,
                settings=settings,
                indexed_counts=indexed_counts,
                skipped_counts=skipped_counts,
                failed_counts=failed_counts,
                pyav_version=pyav_version,
            )
            n_video_frames_indexed += vi
            n_video_frames_skipped += vs

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
        n_videos=len(video_paths),
        n_video_frames_indexed=n_video_frames_indexed,
        n_video_frames_skipped=n_video_frames_skipped,
    )


def main() -> None:
    typer.run(index)
