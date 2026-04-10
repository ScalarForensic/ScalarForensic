#!/usr/bin/env bash
# Build a self-contained Docker bundle for offline / airgapped deployment.
#
# Run on an internet-connected machine that already has uv and Docker available.
# The script downloads models and Python wheels, builds the ScalarForensic image,
# pulls the Qdrant image, and saves both into a single gzip'd tarball.
#
# Qdrant is a separate application (https://qdrant.tech) that ScalarForensic
# uses as its vector database. It is included in the bundle purely for
# airgapped convenience; it is not part of this project.
#
# Usage:
#   bash scripts/build_airgap_image.sh                   # tag: latest
#   bash scripts/build_airgap_image.sh --tag 1.0.0       # custom tag
#   bash scripts/build_airgap_image.sh --no-models        # skip model download
#   bash scripts/build_airgap_image.sh --no-deps          # skip wheel download
#   bash scripts/build_airgap_image.sh --no-models --no-deps  # build only
#
# On the offline machine:
#   docker load < scalarforensic-<tag>.tar.gz   # loads both images
#   cp .env.example .env
#   SCALARFORENSIC_IMAGE=scalarforensic:<tag> docker compose up -d
#   docker compose run --rm sfn-web sfn /images --dino --sscd

set -euo pipefail

QDRANT_IMAGE="qdrant/qdrant:v1.17.1"
TAG="latest"
SKIP_MODELS=false
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "Error: --tag requires a non-empty value" >&2
                exit 1
            fi
            TAG="$2"; shift 2 ;;
        --tag=*)
            TAG="${1#--tag=}"
            if [[ -z "$TAG" ]]; then
                echo "Error: --tag= requires a non-empty value" >&2
                exit 1
            fi
            shift ;;
        --no-models) SKIP_MODELS=true; shift ;;
        --no-deps)   SKIP_DEPS=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

IMAGE="scalarforensic:${TAG}"
OUTPUT="scalarforensic-${TAG}.tar.gz"

echo "==> ScalarForensic airgapped image builder"
echo "    app image:    ${IMAGE}"
echo "    qdrant image: ${QDRANT_IMAGE}"
echo "    output:       ${OUTPUT}"
echo ""

# ── Step 1: Models ────────────────────────────────────────────────────────────
if [ "$SKIP_MODELS" = false ]; then
    echo "==> [1/5] Downloading models ..."
    uv run python scripts/download_models.py
    echo ""
else
    echo "==> [1/5] Skipping model download (--no-models)"
fi

# ── Step 2: Python wheels ─────────────────────────────────────────────────────
if [ "$SKIP_DEPS" = false ]; then
    echo "==> [2/5] Downloading Python wheels (core + web group) ..."
    bash scripts/download_deps.sh --web
    echo ""
else
    echo "==> [2/5] Skipping wheel download (--no-deps)"
fi

# ── Step 3: Build ScalarForensic image ───────────────────────────────────────
echo "==> [3/5] Building Docker image (${IMAGE}) ..."
echo "    (the first build takes a while — PyTorch layers are large)"
docker build --tag "${IMAGE}" .
echo ""

# ── Step 4: Pull Qdrant ───────────────────────────────────────────────────────
# Qdrant is a separate application used as the vector database backend.
# Pull it here so it can be bundled into the same archive for offline transfer.
echo "==> [4/5] Pulling Qdrant image (${QDRANT_IMAGE}) ..."
docker pull "${QDRANT_IMAGE}"
echo ""

# ── Step 5: Save both images into one archive ─────────────────────────────────
echo "==> [5/5] Saving both images → ${OUTPUT} ..."
echo "    (compression may take several minutes)"
docker save "${IMAGE}" "${QDRANT_IMAGE}" | gzip > "${OUTPUT}"

FILESIZE=$(du -sh "${OUTPUT}" | cut -f1)
SHA256=$(sha256sum "${OUTPUT}" | awk '{print $1}')

echo ""
echo "Done."
printf "  %-14s %s\n" "App image:"    "${IMAGE}"
printf "  %-14s %s\n" "Qdrant image:" "${QDRANT_IMAGE}"
printf "  %-14s %s  (%s)\n" "Output:" "${OUTPUT}" "${FILESIZE}"
printf "  %-14s %s\n" "SHA256:" "${SHA256}"
echo ""
echo "Transfer ${OUTPUT} (and docker-compose.yml + .env.example) to the airgapped machine."
echo ""
if [ "${TAG}" != "latest" ]; then
    echo "The app image was tagged '${TAG}' (not 'latest')."
    echo "Set SCALARFORENSIC_IMAGE=${IMAGE} when running Compose:"
    echo "  SCALARFORENSIC_IMAGE=${IMAGE} docker compose up -d"
    echo "Or export it for the session:  export SCALARFORENSIC_IMAGE=${IMAGE}"
    echo ""
fi
echo "Then follow the steps in INSTALL.md § 'Docker bundle'."
