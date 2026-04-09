# ScalarForensic

Large-scale image vector indexer for forensic use. Embeds images into a Qdrant vector database for similarity search and deduplication.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- Qdrant: `docker run -p 6333:6333 qdrant/qdrant`

## Setup

```bash
git clone https://github.com/ScalarForensic/ScalarForensic
cd ScalarForensic
uv sync
cp .env.example .env   # edit to match your environment
```

## Configuration

All settings live in `.env`. Copy `.env.example` to get started — every key is documented there. Environment variables already set in the shell take precedence over the file.

| Variable | Default | Description |
|----------|---------|-------------|
| `SFN_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `SFN_COLLECTION_DINO` | `sfn-dinov2` | Collection name for DINOv2 vectors |
| `SFN_COLLECTION_SSCD` | `sfn-sscd` | Collection name for SSCD vectors |
| `SFN_MODEL_DINO` | `facebook/dinov2-large` | DINOv2 model identifier |
| `SFN_MODEL_SSCD` | `models/sscd_disc_mixup.torchscript.pt` | Path to SSCD checkpoint |
| `SFN_NORMALIZE_SIZE` | `512` | DINOv2 resize dimension (N×N px) |
| `SFN_BATCH_SIZE` | `32` | Images per embedding batch |
| `SFN_DEVICE` | `auto` | Compute device: `auto` \| `cuda` \| `cpu` \| `mps` |
| `SFN_INPUT_DIR` | _(none)_ | Default input folder (can be passed as CLI argument instead) |
| `SFN_DUPLICATE_CHECK_MODE` | `hash` | Dedup strategy: `hash` \| `filepath` \| `both` |
| `SFN_EXTRACT_EXIF` | `false` | Store EXIF presence flags in the database |

## Usage

### Web UI (Phase 2)

```bash
uv sync --group web
uv run sfn-web          # → http://localhost:8080
```

Upload images or folders, select query modes (Exact / Altered / Semantic), and explore results with interactive threshold and result-count sliders. Hit images are loaded on demand from the server filesystem.

### Indexing CLI (Phase 1)

```bash
uv run sfn <image-dir> --sscd
uv run sfn <image-dir> --dino
uv run sfn <image-dir> --dino --sscd   # run both models in one pass
```

`<image-dir>` is optional if `SFN_INPUT_DIR` is set in `.env`.

All files in all subdirectories of the given folder are scanned recursively.

### Backends

| Flag | Model | Dim | Best for |
|------|-------|-----|----------|
| `--sscd` | SSCD ResNet-50 (Meta) | 512 | Copy detection, near-duplicates |
| `--dino` | DINOv2 ViT-L (Meta) | 1024 | Semantic similarity |

Both flags can be specified together. Images are read and hashed once; each model embeds and upserts to its own collection.

### SSCD setup (one-time)

SSCD is not on HuggingFace — download the checkpoint manually:

```bash
mkdir -p models
wget -P models https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
```

The default path is `models/sscd_disc_mixup.torchscript.pt` relative to where you run the command. Override with `SFN_MODEL_SSCD=/absolute/path` in `.env`.

DINOv2 downloads automatically on first run into the HuggingFace cache. For offline use, snapshot it first:

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-large', cache_dir='models/dinov2-large')"
```

Then set `SFN_MODEL_DINO=models/dinov2-large` in `.env`.

### Supported image formats

`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp` `.gif` `.jp2` `.ico` `.psd`

**HEIC/HEIF** (iPhone photos etc.) is supported as an optional extension:

```bash
uv sync --group heif   # installs pillow-heif
```

Once installed, `.heic` and `.heif` files are picked up automatically with no further configuration.

### Deduplication modes

Controlled by `SFN_DUPLICATE_CHECK_MODE` in `.env`:

| Mode | Behaviour |
|------|-----------|
| `hash` | Skip images whose SHA-256 hash is already in the collection _(default)_ |
| `filepath` | Skip images whose absolute path is already stored — useful for large files (e.g. video) where hashing is expensive |
| `both` | Skip if either the hash or the path is already indexed |

### EXIF metadata

Set `SFN_EXTRACT_EXIF=true` to store two boolean fields on every indexed point:

| Field | Meaning |
|-------|---------|
| `exif` | Whether the image contains any EXIF data |
| `exif_geo_data` | Whether the image contains GPS/geolocation data |

---

## Web UI query modes

| Mode | Logic | Requires |
|------|-------|---------|
| **Exact** | SHA-256 hash match — byte-identical file | Any indexed collection |
| **Altered** | SSCD vector search — cropped, recolored, or lightly modified copies | `sfn-sscd` collection |
| **Semantic** | DINOv2 vector search — same scene, subject, or style | `sfn-dinov2` collection |

Modes are automatically disabled in the UI if the corresponding collection has not been indexed yet.

## Third-party licenses

Bundled assets and their licenses are listed in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). All are compatible with GPL-3.
