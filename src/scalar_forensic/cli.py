"""CLI entry point for ScalarForensic."""
from itertools import batched
from pathlib import Path
from time import perf_counter

import typer

from scalar_forensic.embedder import (
    DEFAULT_MODEL_DINOV2,
    DEFAULT_MODEL_SSCD,
    DEFAULT_NORMALIZE_SIZE,
    get_library_versions,
    hash_bytes,
    load_embedder,
)
from scalar_forensic.indexer import Indexer
from scalar_forensic.scanner import scan_images


def _fmt_rate(count: int, seconds: float, unit: str) -> str:
    if seconds <= 0:
        return "—"
    return f"{count / seconds:.1f} {unit}/s"


def _fmt_mbps(bytes_total: int, seconds: float) -> str:
    if seconds <= 0:
        return "—"
    return f"{bytes_total / 1e6 / seconds:.1f} MB/s"


app = typer.Typer(add_completion=False)


@app.command(no_args_is_help=True)
def index(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Root directory of images"),
    sscd: bool = typer.Option(False, "--sscd", help=f"Use SSCD backend (default model: {DEFAULT_MODEL_SSCD})"),
    dino: bool = typer.Option(False, "--dino", help=f"Use DINOv2 backend (default model: {DEFAULT_MODEL_DINOV2})"),
    model: str | None = typer.Option(None, help="Override the model path / HuggingFace identifier"),
    normalize_size: int = typer.Option(DEFAULT_NORMALIZE_SIZE, help="DINOv2: resize images to N×N before embedding"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL"),
    collection: str | None = typer.Option(None, help="Qdrant collection name (default: sfn-sscd or sfn-dinov2)"),
    batch_size: int = typer.Option(32, min=1, help="Images per embedding batch"),
    device: str = typer.Option("auto", help="Compute device: auto | cuda | cpu | mps"),
    skip_existing: bool = typer.Option(True, help="Skip images already present in the collection"),
) -> None:
    """Embed all images under INPUT_DIR and store vectors in Qdrant."""
    if sscd and dino:
        typer.echo("[ERROR] --sscd and --dino are mutually exclusive.", err=True)
        raise typer.Exit(1)

    use_sscd = sscd or not dino  # SSCD is default when neither flag is given
    resolved_model = model or (DEFAULT_MODEL_SSCD if use_sscd else DEFAULT_MODEL_DINOV2)
    resolved_collection = collection or ("sfn-sscd" if use_sscd else "sfn-dinov2")

    typer.echo(f"Loading model {resolved_model!r} on device={device!r} ...")
    try:
        embedder = load_embedder(model=resolved_model, device=device, normalize_size=normalize_size)
    except FileNotFoundError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"  backend={type(embedder).__name__}  dim={embedder.embedding_dim}  device={embedder.device}")

    typer.echo(f"Connecting to Qdrant  collection={resolved_collection!r} ...")
    try:
        indexer = Indexer(url=qdrant_url, collection=resolved_collection, embedding_dim=embedder.embedding_dim)
    except ValueError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1)

    typer.echo("Computing model hash (may take a moment) ...")
    model_hash = embedder.model_hash
    typer.echo(f"  model_hash={model_hash[:16]}...")

    shared_metadata = {
        "model_name": embedder.model_name,
        "model_hash": model_hash,
        "embedding_dim": embedder.embedding_dim,
        "normalize_size": embedder.normalize_size,
        "library_versions": get_library_versions(),
    }

    typer.echo(f"Scanning images in {input_dir} ...")

    indexed = skipped = failed = batch_num = 0
    total_read_s = total_hash_s = total_norm_s = total_embed_s = total_upsert_s = 0.0
    total_bytes = 0

    for batch_paths in batched(scan_images(input_dir), batch_size):
        batch_num += 1

        # --- Read ---
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
                failed += 1
        read_s = perf_counter() - t0
        total_read_s += read_s
        total_bytes += batch_bytes

        if not raw:
            continue

        # --- Hash ---
        t0 = perf_counter()
        path_hash_pairs = [(p, hash_bytes(data)) for p, data in raw]
        hash_s = perf_counter() - t0
        total_hash_s += hash_s

        # --- Dedup via indexed Qdrant payload ---
        if skip_existing:
            already_indexed = indexer.get_indexed_hashes([h for _, h in path_hash_pairs])
        else:
            already_indexed = set()

        data_by_path = {p: data for p, data in raw}
        to_embed = [(p, h) for p, h in path_hash_pairs if h not in already_indexed]
        skipped += len(path_hash_pairs) - len(to_embed)

        if not to_embed:
            typer.echo(f"  batch {batch_num}: {len(path_hash_pairs)} skipped (already indexed)")
            continue

        paths, hashes = zip(*to_embed)
        n = len(paths)
        embed_data = [data_by_path[p] for p in paths]

        # --- Normalize ---
        t0 = perf_counter()
        try:
            images = embedder.normalize_batch_bytes(embed_data)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[ERROR] Normalization failed for batch of {n}: {exc}", err=True)
            failed += n
            continue
        norm_s = perf_counter() - t0
        total_norm_s += norm_s

        # --- Embed ---
        t0 = perf_counter()
        try:
            embeddings = embedder.embed_images(images)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[ERROR] Embedding failed for batch of {n}: {exc}", err=True)
            failed += n
            continue
        embed_s = perf_counter() - t0
        total_embed_s += embed_s

        # --- Upsert ---
        t0 = perf_counter()
        indexer.upsert_batch(list(paths), list(hashes), embeddings, shared_metadata)
        upsert_s = perf_counter() - t0
        total_upsert_s += upsert_s

        indexed += n
        batch_total_s = read_s + hash_s + norm_s + embed_s + upsert_s
        typer.echo(
            f"  batch {batch_num}: {n} imgs  {batch_bytes / 1e6:.1f} MB"
            f"  |  read {read_s:.2f}s ({_fmt_mbps(batch_bytes, read_s)})"
            f"  hash {hash_s:.2f}s ({_fmt_mbps(batch_bytes, hash_s)})"
            f"  normalize {norm_s:.2f}s ({_fmt_rate(n, norm_s, 'img')})"
            f"  embed {embed_s:.2f}s ({_fmt_rate(n, embed_s, 'img')})"
            f"  upsert {upsert_s:.2f}s"
            f"  |  {batch_total_s / n * 1000:.0f} ms/img"
            f"  total indexed={indexed}"
        )

    total_s = total_read_s + total_hash_s + total_norm_s + total_embed_s + total_upsert_s
    typer.echo(f"\nDone.  indexed={indexed}  skipped={skipped}  failed={failed}")
    if indexed > 0:
        typer.echo(
            f"Timing totals:"
            f"  read {total_read_s:.2f}s ({_fmt_mbps(total_bytes, total_read_s)})"
            f"  |  hash {total_hash_s:.2f}s ({_fmt_mbps(total_bytes, total_hash_s)})"
            f"  |  normalize {total_norm_s:.2f}s ({_fmt_rate(indexed, total_norm_s, 'img')})"
            f"  |  embed {total_embed_s:.2f}s ({_fmt_rate(indexed, total_embed_s, 'img')})"
            f"  |  upsert {total_upsert_s:.2f}s"
            f"  |  pipeline {total_s:.2f}s  {_fmt_rate(indexed, total_s, 'img')}  {total_s / indexed * 1000:.0f} ms/img"
        )


def main() -> None:
    app(prog_name="sfn")
