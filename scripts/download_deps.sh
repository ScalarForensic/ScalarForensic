#!/usr/bin/env bash
# Download all Python dependency wheels for offline / airgapped deployment.
#
# Run on an internet-connected machine that already has uv available.
# Transfer the generated vendor/ directory and requirements.txt to the
# offline system alongside the rest of the project folder.
#
# Usage:
#   bash scripts/download_deps.sh              # core deps → vendor/
#   bash scripts/download_deps.sh --web        # include web-UI group
#   bash scripts/download_deps.sh --heif       # include HEIF/HEIC group
#   bash scripts/download_deps.sh --web --heif # all optional groups
#
# On the offline machine, install with:
#   uv venv
#   uv pip install --no-index --find-links vendor/ -r requirements.txt
#   uv pip install --no-deps -e .

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
echo "    (torch and torchvision are fetched from the PyTorch CUDA index — this may take a while)"
mkdir -p "${DEST}"

# uv pip download reads the index configuration from pyproject.toml,
# so it correctly resolves torch/torchvision from the pytorch-cu128 index.
uv pip download -r requirements.txt -d "${DEST}"

echo ""
echo "Done. Files to transfer to the offline machine:"
echo "  ${DEST}/          ($(ls "${DEST}" | wc -l) wheels)"
echo "  requirements.txt"
echo "  (and the rest of the project folder)"
echo ""
echo "On the offline machine:"
echo "  uv venv"
echo "  uv pip install --no-index --find-links ${DEST}/ -r requirements.txt"
echo "  uv pip install --no-deps -e ."
