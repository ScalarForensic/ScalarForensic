#!/usr/bin/env bash
# Download all Python dependency wheels for offline / airgapped deployment.
#
# Run on an internet-connected machine that already has uv available.
# Transfer the generated vendor/ directory and requirements.txt to the
# offline system alongside the rest of the project folder.
#
# IMPORTANT: wheels are platform- and Python-version-specific.
# Run this script on a machine with the same OS, CPU architecture, and
# Python version as the airgapped target (e.g. linux/x86_64, Python 3.12).
# Downloading on macOS or a different Python version and transferring to a
# Linux/x86_64 target will result in install failures on the offline machine.
#
# Usage:
#   bash scripts/download_deps.sh              # core deps → vendor/
#   bash scripts/download_deps.sh --web        # include web-UI group
#   bash scripts/download_deps.sh --heif       # include HEIF/HEIC group
#   bash scripts/download_deps.sh --web --heif # all optional groups
#   bash scripts/download_deps.sh --dest=pkg   # write wheels to pkg/ instead of vendor/
#
# The offline install commands are printed at the end of the script using the
# actual destination path chosen at runtime (vendor/ by default, or --dest value).

set -euo pipefail

DEST="vendor"
GROUPS=()

for arg in "$@"; do
    case "$arg" in
        --web)  GROUPS+=("--group" "web") ;;
        --heif) GROUPS+=("--group" "heif") ;;
        --dest=*) DEST="${arg#--dest=}" ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

echo "==> Exporting locked dependency list ..."
uv export --frozen --no-emit-project "${GROUPS[@]}" -o requirements.txt
echo "    Written: requirements.txt"

echo "==> Downloading wheels → ${DEST}/ ..."
echo "    (torch and torchvision are fetched from the configured PyTorch index — this may take a while)"
mkdir -p "${DEST}"

# uv pip download reads the index configuration from pyproject.toml,
# so it correctly resolves torch/torchvision from whichever PyTorch index is active.
uv pip download -r requirements.txt -d "${DEST}"

# Include the build backend so `uv pip install -e .` works fully offline.
uv pip download hatchling -d "${DEST}"

echo ""
echo "Done. Files to transfer to the offline machine:"
echo "  ${DEST}/          ($(ls "${DEST}" | wc -l) wheels)"
echo "  requirements.txt"
echo "  (and the rest of the project folder)"
echo ""
echo "On the offline machine:"
echo "  uv venv"
echo "  uv pip install --no-index --find-links \"${DEST}/\" -r requirements.txt"
echo "  uv pip install --no-index --find-links \"${DEST}/\" --no-deps -e ."
