#!/usr/bin/env bash
# Download the Unsplash random images collection from Kaggle and unzip it.
# Run from the project root:  bash test/download_data.sh
#
# Prerequisites:
#   - curl is installed
#   - Kaggle dataset is publicly accessible (no API key required for this dataset)

set -euo pipefail

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/lprdosmil/unsplash-random-images-collection"
ZIP_DEST="data/unsplash-random-images-collection.zip"
EXTRACT_DEST="data/images"

mkdir -p data "$EXTRACT_DEST"

if [ -d "data/images/unsplash-images-collection" ]; then
    echo "Dataset already extracted at data/images/unsplash-images-collection — skipping download."
    exit 0
fi

echo "Downloading dataset to $ZIP_DEST ..."
curl -L -o "$ZIP_DEST" "$DATASET_URL"

echo "Unzipping to $EXTRACT_DEST ..."
unzip -q "$ZIP_DEST" -d "$EXTRACT_DEST"

echo "Cleaning up zip ..."
rm "$ZIP_DEST"

echo ""
echo "Done. Images are in: $EXTRACT_DEST"
echo "Next step: run 'uv run python test/prepare_searchfiles.py' to create test search files."
