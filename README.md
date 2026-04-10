# ScalarForensic

Large-scale image vector indexer for forensic use. Embeds images into a Qdrant vector database for similarity search and deduplication — designed for evidence rooms where you need to find exact copies, edited versions, and visually related images across tens of thousands of files.

<img width="1895" height="1272" alt="image" src="https://github.com/user-attachments/assets/cd51803e-cce1-4501-a4f5-d70833ae1afd" />
<img width="1895" height="1272" alt="image" src="https://github.com/user-attachments/assets/4d592a75-50dd-4059-9559-2c4709c58397" />


**→ [Installation & setup guide](INSTALL.md)**

**→ [Performance assessment](docs/performance.md)**

---

## What it does

ScalarForensic indexes a directory of images into a Qdrant vector database using two complementary neural models, then lets you query that database through a web UI to find matches.

### Indexing (CLI)

```bash
uv run sfn <image-dir> --dino --sscd
```

Scans a directory recursively, hashes each image to skip duplicates, and upserts embeddings into Qdrant. Both models can be run in a single pass — images are read and decoded once.

### Querying (Web UI)

```bash
uv run sfn-web   # → http://localhost:8080
```

Upload one or more query images, select search modes, and explore results with threshold and result-count sliders. Hit images are streamed from the server filesystem on demand.

---

## Search modes

| Mode | What it finds | Model |
|------|--------------|-------|
| **Exact** | Byte-identical files (SHA-256 match) | — |
| **Altered** | Cropped, recolored, or lightly modified copies | SSCD ResNet-50 (Meta), 512-dim |
| **Semantic** | Same scene, subject, or visual style | DINOv2 ViT-L (Meta), 1024-dim |

Modes are automatically disabled in the UI if the corresponding collection has not been indexed yet. All three can be combined in a single query.

---

## Features

**Supported image formats:** `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp` `.gif` `.jp2` `.ico` `.psd` — plus `.heic`/`.heif` via an optional dependency group.

**Deduplication modes** (controlled by `SFN_DUPLICATE_CHECK_MODE`):

| Mode | Behaviour |
|------|-----------|
| `hash` | Skip images whose SHA-256 is already in the collection _(default)_ |
| `filepath` | Skip images whose absolute path is already stored — faster for large files |
| `both` | Skip if either hash or path is already indexed |

**EXIF metadata extraction** (`SFN_EXTRACT_EXIF=true`): stores two boolean fields on every indexed point — `exif` (any EXIF data present) and `exif_geo_data` (GPS coordinates present).

**Offline / airgapped deployment:** models and all Python wheels can be pre-downloaded and bundled so the entire stack runs without internet access. See [INSTALL.md § Offline deployment](INSTALL.md#offline--airgapped-deployment).

**GPU acceleration:** NVIDIA CUDA (default) and AMD ROCm are both supported, including RDNA 4 (RX 9070 / 9070 XT) natively on ROCm 6.4. See [INSTALL.md § GPU setup](INSTALL.md#gpu--hardware-acceleration).

---

## Quick start

```bash
git clone https://github.com/ScalarForensic/ScalarForensic
cd ScalarForensic
uv sync --group web
cp .env.example .env

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Download models (one-time)
uv run python scripts/download_models.py

# Index a folder
uv run sfn <image-dir> --dino --sscd

# Start the web UI
uv run sfn-web   # → http://localhost:8080
```

For full setup instructions, GPU configuration, offline deployment, and configuration
reference, see **[INSTALL.md](INSTALL.md)**.

---

## Third-party licenses

Bundled assets and their licenses are listed in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). All are compatible with GPL-3.
