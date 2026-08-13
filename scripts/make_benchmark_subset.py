#!/usr/bin/env python3
"""Build a deterministic stratified benchmark subset of a media corpus.

Samples a fixed fraction (default 10%) of the files in a source directory,
stratified by media kind (heic / jpg / other_image / video) so the subset
preserves the corpus' format proportions, and *copies* the selected files
into a destination directory (copies, not symlinks — decode benchmarks need
real I/O). A JSON manifest (seed, per-stratum counts, chosen paths) is
written into the destination so any run is auditable and reproducible.

Determinism guarantee: the candidate list is sorted before sampling and each
stratum uses its own seeded RNG, so the same source tree + same seed always
selects the byte-identical set of paths — and adding files to one stratum
never changes the selection in another.

Usage:
    uv run python scripts/make_benchmark_subset.py SRC_DIR DEST_DIR \
        [--fraction 0.10] [--seed 20260813]

DEST_DIR must lie outside SRC_DIR (otherwise the campaign scanner would
pick the subset up as new corpus files). The source is only ever read.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: allow running without an editable install
# ---------------------------------------------------------------------------
_src = Path(__file__).parent.parent / "src"
if _src.exists():
    sys.path.insert(0, str(_src))

DEFAULT_SEED = 20260813
DEFAULT_FRACTION = 0.10
MANIFEST_NAME = "benchmark_subset_manifest.json"

# Extension → stratum. Anything not listed (sidecars like .aae, CSV reports,
# …) is counted as skipped and never copied.
_HEIC_EXTS = {".heic", ".heif"}
_JPG_EXTS = {".jpg", ".jpeg"}
_OTHER_IMAGE_EXTS = {".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

STRATA = ("heic", "jpg", "other_image", "video")


def classify(path: Path) -> str | None:
    """Return the stratum name for *path*, or None if it is not benchmarked."""
    ext = path.suffix.lower()
    if ext in _HEIC_EXTS:
        return "heic"
    if ext in _JPG_EXTS:
        return "jpg"
    if ext in _OTHER_IMAGE_EXTS:
        return "other_image"
    if ext in _VIDEO_EXTS:
        return "video"
    return None


def collect_strata(source: Path) -> tuple[dict[str, list[str]], int]:
    """Walk *source* and bucket every file into its stratum.

    Returns ({stratum: sorted relative POSIX paths}, skipped_count).
    Paths are sorted so sampling is independent of filesystem enumeration
    order.
    """
    strata: dict[str, list[str]] = {name: [] for name in STRATA}
    skipped = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        stratum = classify(path)
        if stratum is None:
            skipped += 1
            continue
        strata[stratum].append(path.relative_to(source).as_posix())
    for files in strata.values():
        files.sort()
    return strata, skipped


def sample_stratum(files: list[str], fraction: float, seed: int, stratum: str) -> list[str]:
    """Deterministically pick ~fraction of *files* (at least 1 if non-empty).

    *files* must already be sorted. Each stratum derives its own RNG from
    (seed, stratum name) so the selection in one stratum is unaffected by
    the contents of the others.
    """
    if not files:
        return []
    k = min(len(files), max(1, round(len(files) * fraction)))
    rng = random.Random(f"{seed}:{stratum}")
    return sorted(rng.sample(files, k))


def build_selection(
    source: Path, fraction: float, seed: int
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Return (manifest dict, {stratum: selected relative paths})."""
    strata, skipped = collect_strata(source)
    selected = {name: sample_stratum(files, fraction, seed, name) for name, files in strata.items()}
    manifest = {
        "source": str(source),
        "seed": seed,
        "fraction": fraction,
        "skipped_non_media": skipped,
        "strata": {
            name: {
                "total": len(strata[name]),
                "selected": len(selected[name]),
                "files": selected[name],
            }
            for name in STRATA
        },
    }
    return manifest, selected


def copy_selection(source: Path, dest: Path, selected: dict[str, list[str]]) -> int:
    """Copy every selected file from *source* into *dest*, preserving layout."""
    copied = 0
    for files in selected.values():
        for rel in files:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, target)
            copied += 1
    return copied


def make_subset(
    source: Path,
    dest: Path,
    fraction: float = DEFAULT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Build the subset under *dest* and return the manifest dict."""
    source = source.resolve()
    dest = dest.resolve()
    if not source.is_dir():
        raise SystemExit(f"ERROR: source is not a directory: {source}")
    if dest == source or source in dest.parents:
        raise SystemExit(
            f"ERROR: destination {dest} lies inside source {source} — "
            "the corpus scanner would pick the subset up; use a sibling directory."
        )
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"ERROR: destination {dest} exists and is not empty.")

    manifest, selected = build_selection(source, fraction, seed)
    dest.mkdir(parents=True, exist_ok=True)
    copied = copy_selection(source, dest, selected)
    manifest_path = dest / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Copied {copied} files to {dest}")
    for name in STRATA:
        s = manifest["strata"][name]
        print(f"  {name:12s} {s['selected']:>5} / {s['total']}")
    print(f"  skipped (non-media): {manifest['skipped_non_media']}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Corpus directory (read-only)")
    parser.add_argument(
        "dest",
        type=Path,
        help="Subset output directory — must be OUTSIDE the source directory",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help=f"Fraction to sample per stratum (default {DEFAULT_FRACTION})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed; same inputs + seed = identical selection (default {DEFAULT_SEED})",
    )
    args = parser.parse_args()
    make_subset(args.source, args.dest, fraction=args.fraction, seed=args.seed)


if __name__ == "__main__":
    main()
