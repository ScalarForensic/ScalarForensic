"""CLI entry point for ScalarForensic."""

from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path
from time import perf_counter

import typer

from scalar_forensic.config import Settings
from scalar_forensic.embedder import (
    AnyEmbedder,
    ExifInfo,
    extract_exif,
    get_library_versions,
    hash_bytes,
    load_embedder,
    preprocess_batch,
)
from scalar_forensic.indexer import Indexer
from scalar_forensic.scanner import _HEIF_AVAILABLE, scan_images


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


def index(
    input_dir: Path | None = typer.Argument(
        default=None, help="Root directory of images (overrides SFN_INPUT_DIR)"
    ),
    dino: bool = typer.Option(False, "--dino", help="Use DINOv2 backend (1024-dim semantic)"),
    sscd: bool = typer.Option(False, "--sscd", help="Use SSCD backend (512-dim copy-detection)"),
) -> None:
    """Embed all images under INPUT_DIR and store vectors in Qdrant."""
    settings = Settings()

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
        typer.echo(f"Loading {backend_name} model {model_name!r} on device={settings.device!r} ...")
        if settings.embedding_endpoint:
            if not settings.embedding_model:
                typer.echo(
                    "[ERROR] SFN_EMBEDDING_MODEL must be set when"
                    " SFN_EMBEDDING_ENDPOINT is configured.",
                    err=True,
                )
                raise typer.Exit(1)
            effective_model = settings.embedding_model
        else:
            effective_model = model_name
        try:
            embedder = load_embedder(
                model=effective_model,
                use_sscd=use_sscd,
                device=settings.device,
                normalize_size=settings.normalize_size,
                remote_endpoint=settings.embedding_endpoint,
                remote_api_key=settings.embedding_api_key,
                embedding_dim=settings.embedding_dim,
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

    typer.echo(f"Scanning images in {resolved_input} ...")

    # Per-model counters; index mirrors specs list.
    indexed_counts = [0] * len(specs)
    skipped_counts = [0] * len(specs)
    failed_count = 0

    total_read_s = total_hash_s = 0.0
    total_bytes = 0
    batch_num = 0

    for batch_paths in batched(scan_images(resolved_input), settings.batch_size):
        batch_num += 1
        batch_wall_t0 = perf_counter()

        # --- Read (shared) ---
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
                failed_count += 1
        read_s = perf_counter() - t0
        total_read_s += read_s
        total_bytes += batch_bytes

        if not raw:
            continue

        # --- Hash (shared) ---
        t0 = perf_counter()
        path_hash_pairs = [(p, hash_bytes(data)) for p, data in raw]
        hash_s = perf_counter() - t0
        total_hash_s += hash_s

        # --- EXIF (shared, once per batch if enabled) ---
        exif_data: dict[Path, ExifInfo] | None = None
        if settings.extract_exif:
            data_by_path_for_exif = {p: data for p, data in raw}
            exif_data = {p: extract_exif(data_by_path_for_exif[p]) for p, _ in path_hash_pairs}

        # --- Within-batch hash dedup (shared — avoids embedding identical content twice) ---
        unique_path_by_hash: dict[str, Path] = {}
        for p, h in path_hash_pairs:
            unique_path_by_hash.setdefault(h, p)
        unique_pairs = [(p, h) for h, p in unique_path_by_hash.items()]
        duplicate_hashes_in_batch = len(path_hash_pairs) - len(unique_pairs)

        data_by_path = {p: data for p, data in raw}

        # --- Shared pre-processing: open RGB + cap short side to 331 px (threaded) ---
        t0 = perf_counter()
        unique_paths_list = [p for p, _ in unique_pairs]
        pre_images = preprocess_batch([data_by_path[p] for p in unique_paths_list])
        pre_s = perf_counter() - t0
        pre_by_path = dict(zip(unique_paths_list, pre_images))

        # --- Per-model loop: normalize + embed, collect upsert jobs ---
        model_segments: list[str] = []
        upsert_jobs: list = []

        for spec_idx, (embedder, indexer, model_hash) in enumerate(specs):
            to_embed = _apply_dedup(unique_pairs, indexer, settings)
            n_skipped = duplicate_hashes_in_batch + (len(unique_pairs) - len(to_embed))
            skipped_counts[spec_idx] += n_skipped

            backend = type(embedder).__name__

            if not to_embed:
                model_segments.append(f"{backend} all skipped ({n_skipped})")
                continue

            paths, hashes = zip(*to_embed)
            n = len(paths)
            model_pil = [pre_by_path[p] for p in paths]

            # --- Normalize ---
            t0 = perf_counter()
            try:
                norm_images = embedder.normalize_batch_bytes(model_pil)
            except Exception as exc:  # noqa: BLE001
                typer.echo(
                    f"[ERROR] Normalization failed for batch of {n} [{backend}]: {exc}", err=True
                )
                failed_count += n
                continue
            norm_s = perf_counter() - t0

            # --- Embed ---
            t0 = perf_counter()
            try:
                embeddings = embedder.embed_images(norm_images)
            except Exception as exc:  # noqa: BLE001
                typer.echo(
                    f"[ERROR] Embedding failed for batch of {n} [{backend}]: {exc}", err=True
                )
                failed_count += n
                continue
            embed_s = perf_counter() - t0

            indexed_counts[spec_idx] += n
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

            def _make_upsert(idx, ps, hs, embs, meta, exif):
                def _job():
                    idx.upsert_batch(ps, hs, embs, meta, exif)

                return _job

            upsert_jobs.append(
                _make_upsert(
                    indexer, list(paths), list(hashes), embeddings, shared_metadata, exif_for_batch
                )
            )

        # --- Parallel upserts ---
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

    typer.echo("\nDone.")
    for spec_idx, (embedder, _, _) in enumerate(specs):
        typer.echo(
            f"  [{type(embedder).__name__}]  indexed={indexed_counts[spec_idx]}"
            f"  skipped={skipped_counts[spec_idx]}"
        )
    if failed_count:
        typer.echo(f"  failed={failed_count}")


def main() -> None:
    typer.run(index)
