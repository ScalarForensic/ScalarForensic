# ScalarForensic

Large-scale image vector indexer for forensic use. Embeds images into a Qdrant vector database for similarity search and deduplication.

## Requirements

- Python 3.12 (pinned — see [GPU setup](#gpu--hardware-acceleration) for why)
- [uv](https://github.com/astral-sh/uv)
- Qdrant: `docker run -p 6333:6333 qdrant/qdrant`

## Setup

```bash
git clone https://github.com/ScalarForensic/ScalarForensic
cd ScalarForensic
uv sync
cp .env.example .env   # edit to match your environment
```

## GPU / hardware acceleration

The repo ships pre-configured for **AMD ROCm 6.4**. The NVIDIA CUDA index is present but commented out — swap the active blocks in `pyproject.toml` to switch backends.

### AMD ROCm (current default)

**System requirements:** ROCm 6.4 installed system-wide (`rocm-smi --showdriverversion`).

**Supported GPU families (ROCm 6.4):**
- RDNA 2: gfx1030 / 1031 / 1032 (RX 6000 series)
- RDNA 3: gfx1100 / 1101 / 1102 (RX 7000 series)
- RDNA 4: gfx1201 (RX 9070 / 9070 XT) — supported natively, no `HSA_OVERRIDE_GFX_VERSION` needed

In `SFN_DEVICE`, both `auto` and `cuda` resolve to the ROCm/HIP backend — PyTorch uses the CUDA interface for ROCm.

**Why Python 3.12 is pinned:**
PyTorch ships `pytorch-triton-rocm` with a plain `linux_x86_64` wheel tag instead of the `manylinux` tag uv expects. uv's index resolver rejects it as platform-incompatible. The workaround in `pyproject.toml` pins it as a direct project dependency with a hard-coded wheel URL so uv skips tag validation — and direct URL sources in uv only work for packages listed in project dependencies, which requires a fixed Python version to select the right `cpXYZ` wheel.

**Upgrading PyTorch or switching ROCm versions:**

1. Pick the target index — available versions are listed at `https://download.pytorch.org/whl/torch/`.
2. In `pyproject.toml`, update `[[tool.uv.index]]` to point at the new ROCm index (e.g. `rocm7.2`) and update the `[tool.uv.sources]` `index` references to match.
3. Find the `pytorch-triton-rocm` version that the new torch requires (visible in the wheel's `.metadata` file at `https://download.pytorch.org/whl/<rocm-index>/torch-<version>+rocm<X.Y>-cpXYZ-cpXYZ-linux_x86_64.whl.metadata`).
4. Update the `pytorch-triton-rocm==X.Y.Z` pin in `[project.dependencies]` and the direct URL in `[tool.uv.sources]` to match. The URL pattern is:
   ```
   https://download-r2.pytorch.org/whl/pytorch_triton_rocm-X.Y.Z-cpABC-cpABC-linux_x86_64.whl
   ```
5. Run `uv sync --reinstall-package torch --reinstall-package torchvision --reinstall-package pytorch-triton-rocm`.

### NVIDIA CUDA

1. In `pyproject.toml`, comment out the `pytorch-rocm64` index block and uncomment the `pytorch-cu128` block.
2. In `[tool.uv.sources]`, point `torch` and `torchvision` at `pytorch-cu128` and remove the `pytorch-triton-rocm` entry entirely (not needed on CUDA).
3. Remove the `pytorch-triton-rocm` line from `[project.dependencies]`.
4. Run `uv sync --reinstall-package torch --reinstall-package torchvision`.

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

### Model setup (one-time)

Use the download script to fetch both models in one step:

```bash
uv run python scripts/download_models.py          # both models
uv run python scripts/download_models.py --sscd   # SSCD only
uv run python scripts/download_models.py --dino   # DINOv2 only
```

This places the files at the default paths (`models/sscd_disc_mixup.torchscript.pt` and `models/dinov2-large/`). For offline use, set `SFN_MODEL_DINO=models/dinov2-large` in `.env` so the app loads the local snapshot instead of fetching remotely (SSCD is always loaded from disk and needs no change).

**Manual alternative — SSCD:**

```bash
mkdir -p models
wget -P models https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
```

**Manual alternative — DINOv2:**

DINOv2 downloads automatically on first run into the HuggingFace cache. To snapshot it for offline use:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov2-large', local_dir='models/dinov2-large', local_dir_use_symlinks=False)
"
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

## Test data / quick-start sample

The `test/` folder contains two scripts to set up a local sample dataset. The downloaded images and generated search files are gitignored — only the scripts are committed.

```bash
# 1. Download and unzip the Unsplash sample dataset (~200 MB) into data/
bash test/download_data.sh

# 2. Copy 10 random images to test/searchfiles/ and print the full setup guide
uv run python test/prepare_searchfiles.py

# optional: choose a different count or seed
uv run python test/prepare_searchfiles.py --count 20 --seed 7
```

`prepare_searchfiles.py` prints step-by-step instructions at the end:
install deps → download SSCD model → start Qdrant → index → start web server → visit `localhost:8080` → upload from `test/searchfiles/`.

## Offline / airgapped deployment

Both models and all Python dependencies can be pre-downloaded and bundled with the project so the entire folder runs without internet access.

### On the internet-connected machine

**1. Download models:**

```bash
uv run python scripts/download_models.py
```

Models land at `models/sscd_disc_mixup.torchscript.pt` and `models/dinov2-large/`. Set `SFN_MODEL_DINO=models/dinov2-large` in `.env` so the app loads the local DINOv2 snapshot instead of fetching it remotely.

**2. Download Python wheels:**

```bash
bash scripts/download_deps.sh
```

This runs `uv export --frozen` to capture the locked dependency list, then downloads all wheels into `vendor/` using `uv pip download`. Because it goes through uv, the custom PyTorch CUDA index configured in `pyproject.toml` is respected automatically — torch and torchvision are fetched from the correct source.

> **Platform requirement:** wheels are specific to OS, CPU architecture, and Python version. Run this script on a machine that matches the airgapped target (e.g. `linux/x86_64`, Python 3.11). Downloading on macOS or a different Python version and transferring to a Linux target will cause install failures on the offline machine.

For optional groups add flags:

```bash
bash scripts/download_deps.sh --web        # include web-UI dependencies
bash scripts/download_deps.sh --heif       # include HEIF/HEIC support
bash scripts/download_deps.sh --web --heif # all optional groups
```

**3. Save the Qdrant Docker image:**

```bash
docker pull qdrant/qdrant:v1.17.1
docker save qdrant/qdrant:v1.17.1 | gzip > qdrant.tar.gz
```

**4. Transfer** the entire project folder to the offline machine, including:

| Path | Contents |
|------|----------|
| `models/` | SSCD checkpoint + DINOv2 snapshot |
| `vendor/` | Python wheels |
| `requirements.txt` | Locked dependency list (written by the script) |
| `qdrant.tar.gz` | Qdrant Docker image |

---

### On the airgapped machine

**5. Install Python dependencies from the local wheelhouse:**

```bash
uv venv
uv pip install --no-index --find-links vendor/ -r requirements.txt
uv pip install --no-deps -e .
```

**6. Load and start Qdrant:**

```bash
docker load < qdrant.tar.gz
docker run -p 6333:6333 qdrant/qdrant:v1.17.1
```

**7. Run:**

```bash
source .venv/bin/activate
sfn <image-dir> --dino --sscd
```

## Third-party licenses

Bundled assets and their licenses are listed in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). All are compatible with GPL-3.
