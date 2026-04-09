#!/usr/bin/env python3
"""Pre-download both models for airgapped / offline deployment.

Run once on an internet-connected machine, then copy the entire project
(including models/) to the offline system and run with uv sync --offline.

    uv run python scripts/download_models.py [--sscd] [--dino] [--all]

With no flags, both models are downloaded (equivalent to --all).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_DEST = Path("models/sscd_disc_mixup.torchscript.pt")

DINO_MODEL_ID = "facebook/dinov2-large"
DINO_DEST = Path("models/dinov2-large")


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = min(block_num * block_size, total_size)
    if total_size > 0:
        pct = downloaded * 100 // total_size
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_sscd() -> None:
    SSCD_DEST.parent.mkdir(parents=True, exist_ok=True)
    if SSCD_DEST.exists():
        print(f"SSCD: already present at {SSCD_DEST} — skipping.")
        return
    print(f"Downloading SSCD checkpoint → {SSCD_DEST} ...")
    urllib.request.urlretrieve(SSCD_URL, SSCD_DEST, reporthook=_progress)
    print(f"\nSSCD: saved ({SSCD_DEST.stat().st_size / 1e6:.1f} MB)")


def download_dino() -> None:
    if DINO_DEST.exists() and any(DINO_DEST.iterdir()):
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
    snapshot_download(DINO_MODEL_ID, local_dir=str(DINO_DEST))
    print(f"DINOv2: saved to {DINO_DEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sscd", action="store_true", help="Download SSCD checkpoint only")
    parser.add_argument("--dino", action="store_true", help="Download DINOv2 snapshot only")
    parser.add_argument("--all", dest="all_models", action="store_true", help="Download both (default)")
    args = parser.parse_args()

    do_sscd = args.sscd or args.all_models or not (args.sscd or args.dino)
    do_dino = args.dino or args.all_models or not (args.sscd or args.dino)

    if do_sscd:
        download_sscd()
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
