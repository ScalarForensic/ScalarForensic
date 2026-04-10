#!/usr/bin/env bash
# Build a self-contained Docker image for offline / airgapped deployment.
#
# Run on an internet-connected machine that already has uv and Docker available.
# The script downloads models and Python wheels, builds the image, and saves it
# as a gzip'd tarball ready to transfer to the offline machine.
#
# Usage:
#   bash scripts/build_airgap_image.sh                   # tag: latest
#   bash scripts/build_airgap_image.sh --tag 1.0.0       # custom tag
#   bash scripts/build_airgap_image.sh --no-models        # skip model download
#   bash scripts/build_airgap_image.sh --no-deps          # skip wheel download
#   bash scripts/build_airgap_image.sh --no-models --no-deps  # build only
#
# On the offline machine:
#   docker load < scalarforensic-<tag>.tar.gz
#   cp .env.example .env  # edit SFN_IMAGES_DIR and any other settings
#   docker compose up -d
#   docker compose run --rm sfn-web sfn /images --dino --sscd

set -euo pipefail

TAG="latest"
SKIP_MODELS=false
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)       TAG="$2"; shift 2 ;;
        --tag=*)     TAG="${1#--tag=}"; shift ;;
        --no-models) SKIP_MODELS=true; shift ;;
        --no-deps)   SKIP_DEPS=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

IMAGE="scalarforensic:${TAG}"
OUTPUT="scalarforensic-${TAG}.tar.gz"

echo "==> ScalarForensic airgapped image builder"
echo "    image:  ${IMAGE}"
echo "    output: ${OUTPUT}"
echo ""

# ── Step 1: Models ────────────────────────────────────────────────────────────
if [ "$SKIP_MODELS" = false ]; then
    echo "==> [1/4] Downloading models ..."
    uv run python scripts/download_models.py
    echo ""
else
    echo "==> [1/4] Skipping model download (--no-models)"
fi

# ── Step 2: Python wheels ─────────────────────────────────────────────────────
if [ "$SKIP_DEPS" = false ]; then
    echo "==> [2/4] Downloading Python wheels (core + web group) ..."
    bash scripts/download_deps.sh --web
    echo ""
else
    echo "==> [2/4] Skipping wheel download (--no-deps)"
fi

# ── Step 3: Docker build ──────────────────────────────────────────────────────
echo "==> [3/4] Building Docker image (${IMAGE}) ..."
echo "    (the first build takes a while — PyTorch layers are large)"
docker build --tag "${IMAGE}" .
echo ""

# ── Step 4: Save ─────────────────────────────────────────────────────────────
echo "==> [4/4] Saving image → ${OUTPUT} ..."
echo "    (compression may take several minutes)"
docker save "${IMAGE}" | gzip > "${OUTPUT}"

FILESIZE=$(du -sh "${OUTPUT}" | cut -f1)
SHA256=$(sha256sum "${OUTPUT}" | awk '{print $1}')

echo ""
echo "Done."
printf "  %-10s %s\n" "Image:"  "${IMAGE}"
printf "  %-10s %s  (%s)\n" "Output:" "${OUTPUT}" "${FILESIZE}"
printf "  %-10s %s\n" "SHA256:" "${SHA256}"
echo ""
echo "Transfer ${OUTPUT} (and docker-compose.yml + .env.example) to the airgapped"
echo "machine, then follow the steps in INSTALL.md § 'Docker bundle'."
