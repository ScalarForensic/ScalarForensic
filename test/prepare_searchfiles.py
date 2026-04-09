"""
Pick a random selection of images from the Unsplash dataset and copy them to
test/searchfiles/ so they can be used as query images in the web UI.

Run from project root:
    uv run python test/prepare_searchfiles.py
    uv run python test/prepare_searchfiles.py --count 20 --seed 7
"""

import argparse
import random
import shutil
from pathlib import Path

SRC = Path("data/images/unsplash-images-collection")
OUT = Path("test/searchfiles")

DEFAULT_COUNT = 10
DEFAULT_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare test search files")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of images to copy (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    if not SRC.exists():
        print(f"[ERROR] Source directory not found: {SRC}")
        print("        Run 'bash test/download_data.sh' first.")
        raise SystemExit(1)

    all_images = sorted(SRC.glob("*.jpg"))
    if not all_images:
        print(f"[ERROR] No .jpg files found in {SRC}")
        raise SystemExit(1)

    if args.count > len(all_images):
        print(
            f"[WARN] Requested {args.count} images but only {len(all_images)} available. "
            "Using all of them."
        )
        args.count = len(all_images)

    random.seed(args.seed)
    selected = random.sample(all_images, args.count)

    OUT.mkdir(parents=True, exist_ok=True)

    # Clear previous contents so re-runs are deterministic
    for old in OUT.glob("*.jpg"):
        old.unlink()

    for src in selected:
        shutil.copy2(src, OUT / src.name)

    searchfiles_abs = OUT.resolve()

    print(f"Copied {args.count} images to {OUT}/")
    print()
    print("=" * 60)
    print("  Next steps to run ScalarForensic")
    print("=" * 60)
    print()
    print("1. Install dependencies:")
    print("     uv sync --group web")
    print()
    print("2. Download the SSCD model (if not already present):")
    print("     mkdir -p models")
    print("     # Download sscd_disc_mixup.torchscript.pt from the SSCD release")
    print("     # and place it at:  models/sscd_disc_mixup.torchscript.pt")
    print()
    print("3. Start Qdrant (requires Docker):")
    print("     docker run -d --name qdrant -p 6333:6333 qdrant/qdrant")
    print()
    print("4. Index the dataset images:")
    print("     ./run.sh sfn data/images/unsplash-images-collection --dino --sscd")
    print()
    print("5. Start the web server:")
    print("     ./run.sh sfn-web")
    print()
    print("6. Open the UI:")
    print("     http://localhost:8080")
    print()
    print("7. Upload test images for analysis from:")
    print(f"     {searchfiles_abs}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
