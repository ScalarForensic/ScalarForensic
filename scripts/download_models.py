#!/usr/bin/env python3
"""Pre-download both models for airgapped / offline deployment.

Run once on an internet-connected machine:

    uv run python scripts/download_models.py [--sscd] [--dino] [--all]

With no flags, both models are downloaded (equivalent to --all).

Pass --hash to verify that the downloaded model produces the same content hash
shown in sfn / sfn-web startup mismatch warnings (the "stored=…" value):

    uv run python scripts/download_models.py --dino --hash <64-char-hex>
    uv run python scripts/download_models.py --sscd --hash <64-char-hex>

For DINOv2, if the requested hash is present in KNOWN_DINO_REVISIONS the
correct HuggingFace commit is pinned automatically; otherwise the script
downloads the current default branch and reports the actual hash so you can
verify manually or add an entry to the table for future use.

Remote endpoint hashes (prefixed "config:") cannot be used here — they
identify endpoint configuration, not model weights, and have no downloadable
artefact.

For vendoring Python dependencies, run scripts/download_deps.sh first.
Then copy the entire project (including models/ and vendor/) to the offline
system and install with:

    uv pip install --no-index --find-links vendor/ -r requirements.txt
    uv pip install --no-deps -e .
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_DEST = Path("models/sscd_disc_mixup.torchscript.pt")

DINO_MODEL_ID = "facebook/dinov2-large"
DINO_DEST = Path("models/dinov2-large")

# Maps ScalarForensic content hash → HuggingFace commit SHA for facebook/dinov2-large.
# When --hash resolves to an entry here, the matching revision is fetched automatically
# so the downloaded snapshot is guaranteed to produce the expected content hash.
#
# To populate: run the script after a fresh download and copy the printed
# "content hash: …" value; find the corresponding HuggingFace commit at
# https://huggingface.co/facebook/dinov2-large/commits/main and add it here.
KNOWN_DINO_REVISIONS: dict[str, str] = {
    # "<64-char SHA-256 content hash>": "<HuggingFace git commit SHA>",
}


# ---------------------------------------------------------------------------
# Hash helpers (mirror safeguards.compute_dino_model_hash /
#               safeguards.compute_sscd_model_hash without importing the package)
# ---------------------------------------------------------------------------


def _compute_dino_hash(local_dir: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(local_dir.rglob("*")):
        if not file.is_file():
            continue
        h.update(file.name.encode())
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def _compute_sscd_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = min(block_num * block_size, total_size)
    if total_size > 0:
        pct = downloaded * 100 // total_size
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_sscd(force: bool = False, expected_hash: str | None = None) -> None:
    SSCD_DEST.parent.mkdir(parents=True, exist_ok=True)
    if SSCD_DEST.exists() and not force:
        print(f"SSCD: already present at {SSCD_DEST}")
        if expected_hash:
            actual = _compute_sscd_hash(SSCD_DEST)
            print(f"  content hash: {actual}")
            if actual != expected_hash:
                print(f"  [ERROR] hash mismatch — expected: {expected_hash}")
                print("  Pass --force to re-download.")
                sys.exit(1)
            print("  hash verified ✓")
        else:
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
    actual = _compute_sscd_hash(SSCD_DEST)
    print(f"  content hash: {actual}")
    if expected_hash and actual != expected_hash:
        print(f"  [ERROR] hash mismatch — expected: {expected_hash}")
        print("  The file at the source URL may have changed.  Contact the maintainers.")
        sys.exit(1)
    elif expected_hash:
        print("  hash verified ✓")


def _dino_is_complete() -> bool:
    """Return True only when config.json and at least one weight file are present."""
    if not DINO_DEST.exists():
        return False
    if not (DINO_DEST / "config.json").exists():
        return False
    return bool(list(DINO_DEST.glob("*.safetensors")) or list(DINO_DEST.glob("*.bin")))


def download_dino(
    force: bool = False,
    revision: str | None = None,
    expected_hash: str | None = None,
) -> None:
    # Resolve revision from the lookup table when --hash is known and --dino-revision
    # was not explicitly provided.
    resolved_revision = revision
    if expected_hash and resolved_revision is None:
        resolved_revision = KNOWN_DINO_REVISIONS.get(expected_hash)
        if resolved_revision:
            print(f"DINOv2: resolved hash → HuggingFace revision {resolved_revision}")

    if _dino_is_complete() and not force:
        print(f"DINOv2: already present at {DINO_DEST}")
        if expected_hash:
            actual = _compute_dino_hash(DINO_DEST)
            print(f"  content hash: {actual}")
            if actual != expected_hash:
                print(f"  [ERROR] hash mismatch — expected: {expected_hash}")
                hint = f" --dino-revision {resolved_revision}" if resolved_revision else ""
                print(f"  Pass --force{hint} to re-download.")
                sys.exit(1)
            print("  hash verified ✓")
        else:
            print("  (pass --force to re-download if the file is corrupted or incomplete)")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not found. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    if force and DINO_DEST.exists():
        shutil.rmtree(DINO_DEST)
    DINO_DEST.mkdir(parents=True, exist_ok=True)
    rev_label = f" @ {resolved_revision}" if resolved_revision else ""
    print(f"Downloading DINOv2 snapshot ({DINO_MODEL_ID}{rev_label}) → {DINO_DEST} ...")
    print("  (this is ~1.2 GB; progress is shown per-file by huggingface_hub)")
    snapshot_download(
        DINO_MODEL_ID,
        revision=resolved_revision,
        local_dir=str(DINO_DEST),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"DINOv2: saved to {DINO_DEST}")
    actual = _compute_dino_hash(DINO_DEST)
    print(f"  content hash: {actual}")
    if expected_hash and actual != expected_hash:
        print(f"  [ERROR] hash mismatch — expected: {expected_hash}")
        if not resolved_revision:
            print(
                f"  The default branch of {DINO_MODEL_ID} does not match the requested hash.\n"
                "  Find the HuggingFace commit that produced this hash and pass it via\n"
                "  --dino-revision, or add it to KNOWN_DINO_REVISIONS in this script."
            )
        sys.exit(1)
    elif expected_hash:
        print("  hash verified ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        help=(
            "Re-download even if the file already exists "
            "(use when a prior run was interrupted or the file is corrupted)"
        ),
    )
    parser.add_argument(
        "--hash",
        metavar="SHA256",
        help=(
            "Expected content hash (the 'stored=…' value from a startup mismatch warning). "
            "For DINOv2, automatically selects the matching HuggingFace revision when "
            "listed in KNOWN_DINO_REVISIONS. "
            "Exits non-zero if the downloaded model does not match. "
            "Remote endpoint hashes (prefixed 'config:') are not accepted here."
        ),
    )
    parser.add_argument(
        "--dino-revision",
        metavar="REV",
        help=(
            "HuggingFace commit SHA, branch, or tag for the DINOv2 snapshot download. "
            "Overrides automatic revision lookup from --hash."
        ),
    )
    args = parser.parse_args()

    if args.hash and args.hash.startswith("config:"):
        print(
            "[ERROR] --hash value starts with 'config:' — this is a remote endpoint "
            "configuration hash, not a model weights hash.  Remote endpoint hashes are "
            "derived from endpoint URL + model name + embedding dimension and have no "
            "downloadable artefact.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.hash:
        args.hash = args.hash.lower()
        if len(args.hash) != 64 or not all(c in "0123456789abcdef" for c in args.hash):
            print(
                "[ERROR] --hash must be a 64-character hex string (SHA-256 digest).",
                file=sys.stderr,
            )
            sys.exit(1)

    do_sscd = args.sscd or args.all_models or not (args.sscd or args.dino)
    do_dino = args.dino or args.all_models or not (args.sscd or args.dino)

    if args.hash and do_sscd and do_dino:
        print(
            "[ERROR] --hash requires exactly one of --sscd or --dino "
            "(the two models have different hashes).",
            file=sys.stderr,
        )
        sys.exit(1)

    if do_sscd:
        download_sscd(force=args.force, expected_hash=args.hash)
    if do_dino:
        download_dino(force=args.force, revision=args.dino_revision, expected_hash=args.hash)

    print("\nAll requested models are ready.")
    print("Make sure your .env contains:")
    if do_sscd:
        print(f"  SFN_MODEL_SSCD={SSCD_DEST}")
    if do_dino:
        print(f"  SFN_MODEL_DINO={DINO_DEST}")


if __name__ == "__main__":
    main()
