"""CLI entry point for ScalarForensic."""

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
        f"Dedup mode: {settings.duplicate_check_mode}"
        f"  |  EXIF extraction: {settings.extract_exif}"
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
        try:
            embedder = load_embedder(
                model=model_name,
                use_sscd=use_sscd,
                device=settings.device,
                normalize_size=settings.normalize_size,
            )
        except FileNotFoundError as exc:
            typer.echo(f"[ERROR] {exc}", err=True)
            raise typer.Exit(1)
        typer.echo(
            f"  backend={type(embedder).__name__}  dim={embedder.embedding_dim}"
            f"  device={embedder.device}"
        )

        typer.echo(f"Connecting to Qdrant  collection={collection!r} ...")
        try:
            indexer = Indexer(
                url=settings.qdrant_url,
                collection=collection,
                embedding_dim=embedder.embedding_dim,
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

        # --- Per-model loop ---
        for spec_idx, (embedder, indexer, model_hash) in enumerate(specs):
            to_embed = _apply_dedup(unique_pairs, indexer, settings)
            n_skipped = duplicate_hashes_in_batch + (len(unique_pairs) - len(to_embed))
            skipped_counts[spec_idx] += n_skipped

            if not to_embed:
                typer.echo(
                    f"  batch {batch_num} [{type(embedder).__name__}]: "
                    f"{len(path_hash_pairs)} skipped ({n_skipped} already indexed / duplicate)"
                )
                continue

            paths, hashes = zip(*to_embed)
            n = len(paths)
            embed_data = [data_by_path[p] for p in paths]

            # --- Normalize ---
            t0 = perf_counter()
            try:
                images = embedder.normalize_batch_bytes(embed_data)
            except Exception as exc:  # noqa: BLE001
                typer.echo(
                    f"[ERROR] Normalization failed for batch of {n}"
                    f" [{type(embedder).__name__}]: {exc}",
                    err=True,
                )
                failed_count += n
                continue
            norm_s = perf_counter() - t0

            # --- Embed ---
            t0 = perf_counter()
            try:
                embeddings = embedder.embed_images(images)
            except Exception as exc:  # noqa: BLE001
                typer.echo(
                    f"[ERROR] Embedding failed for batch of {n} [{type(embedder).__name__}]: {exc}",
                    err=True,
                )
                failed_count += n
                continue
            embed_s = perf_counter() - t0

            # --- Upsert ---
            shared_metadata = {
                "model_name": embedder.model_name,
                "model_hash": model_hash,
                "embedding_dim": embedder.embedding_dim,
                "normalize_size": embedder.normalize_size,
                "library_versions": get_library_versions(),
            }
            exif_for_batch: dict[Path, dict] | None = None
            if exif_data is not None:
                exif_for_batch = {p: dict(exif_data[p]) for p in paths}

            t0 = perf_counter()
            indexer.upsert_batch(
                list(paths), list(hashes), embeddings, shared_metadata, exif_for_batch
            )
            upsert_s = perf_counter() - t0

            indexed_counts[spec_idx] += n
            backend = type(embedder).__name__
            typer.echo(
                f"  batch {batch_num} [{backend}]: {n} imgs  {batch_bytes / 1e6:.1f} MB"
                f"  |  read {read_s:.2f}s ({_fmt_mbps(batch_bytes, read_s)})"
                f"  hash {hash_s:.2f}s"
                f"  normalize {norm_s:.2f}s ({_fmt_rate(n, norm_s, 'img')})"
                f"  embed {embed_s:.2f}s ({_fmt_rate(n, embed_s, 'img')})"
                f"  upsert {upsert_s:.2f}s"
                f"  |  total indexed={indexed_counts[spec_idx]}"
            )

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
