#!/usr/bin/env python3
"""Pre-download both models for airgapped / offline deployment.

Run once on an internet-connected machine:

    uv run python scripts/download_models.py [--sscd] [--dino] [--all]

With no flags, both models are downloaded (equivalent to --all).

For vendoring Python dependencies, run scripts/download_deps.sh first.
Then copy the entire project (including models/ and vendor/) to the offline
system and install with:

    uv pip install --no-index --find-links vendor/ -r requirements.txt
    uv pip install --no-deps -e .
"""

import argparse
import sys
import urllib.request
from pathlib import Path

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_DEST = Path("models/sscd_disc_mixup.torchscript.pt")

DINO_MODEL_ID = "facebook/dinov2-large"
DINO_DEST = Path("models/dinov2-large")
# Pin to a specific commit hash for reproducible offline bundles, e.g.:
#   DINO_REVISION = "fc526a6"
# None uses the current default branch (non-reproducible across time).
DINO_REVISION: str | None = None


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = min(block_num * block_size, total_size)
    if total_size > 0:
        pct = downloaded * 100 // total_size
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_sscd(force: bool = False) -> None:
    SSCD_DEST.parent.mkdir(parents=True, exist_ok=True)
    if SSCD_DEST.exists() and not force:
        print(f"SSCD: already present at {SSCD_DEST} — skipping.")
        print("  (pass --force to re-download if the file is corrupted or incomplete)")
        return
    tmp = SSCD_DEST.with_suffix(".tmp")
    print(f"Downloading SSCD checkpoint → {SSCD_DEST} ...")
    try:
        urllib.request.urlretrieve(SSCD_URL, tmp, reporthook=_progress)
        tmp.rename(SSCD_DEST)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"\nSSCD: saved ({SSCD_DEST.stat().st_size / 1e6:.1f} MB)")


def _dino_is_complete() -> bool:
    """Return True only when config.json and at least one weight file are present."""
    if not DINO_DEST.exists():
        return False
    if not (DINO_DEST / "config.json").exists():
        return False
    return bool(list(DINO_DEST.glob("*.safetensors")) or list(DINO_DEST.glob("*.bin")))


def download_dino() -> None:
    if _dino_is_complete():
        print(f"DINOv2: already present at {DINO_DEST} — skipping.")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not found. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    DINO_DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DINOv2 snapshot ({DINO_MODEL_ID}) → {DINO_DEST} ...")
    print("  (this is ~1.2 GB; progress is shown per-file by huggingface_hub)")
    snapshot_download(
        DINO_MODEL_ID,
        revision=DINO_REVISION,
        local_dir=str(DINO_DEST),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"DINOv2: saved to {DINO_DEST}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sscd", action="store_true", help="Download SSCD checkpoint only")
    parser.add_argument("--dino", action="store_true", help="Download DINOv2 snapshot only")
    parser.add_argument(
        "--all", dest="all_models", action="store_true", help="Download both (default)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists (use when a prior run was interrupted or the file is corrupted)",
    )
    args = parser.parse_args()

    do_sscd = args.sscd or args.all_models or not (args.sscd or args.dino)
    do_dino = args.dino or args.all_models or not (args.sscd or args.dino)

    if do_sscd:
        download_sscd(force=args.force)
    if do_dino:
        download_dino()

    print("\nAll requested models are ready.")
    print("Make sure your .env contains:")
    if do_sscd:
        print(f"  SFN_MODEL_SSCD={SSCD_DEST}")
    if do_dino:
        print(f"  SFN_MODEL_DINO={DINO_DEST}")


if __name__ == "__main__":
    main()
