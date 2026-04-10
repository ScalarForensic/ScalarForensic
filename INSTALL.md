# Installation

## Requirements

- Python 3.12 (required — see [why](#why-python-312-is-required))
- [uv](https://github.com/astral-sh/uv)
- Qdrant: `docker run -p 6333:6333 qdrant/qdrant`

## Setup

```bash
git clone https://github.com/ScalarForensic/ScalarForensic
cd ScalarForensic
uv sync --group web   # add --group dev to include test/lint tools
cp .env.example .env  # edit to match your environment
# Note: a Qdrant service must be running at SFN_QDRANT_URL before starting the app.
```

`uv sync` alone only installs the base CLI dependencies. The `--group web` flag is
required to get FastAPI, Uvicorn, and python-multipart for `sfn-web`. Use `--group heif`
to add HEIC/HEIF support (see [below](#heicheif-support)).

## GPU / hardware acceleration

The repo ships pre-configured for **NVIDIA CUDA 12.8**. The AMD ROCm index is present
but commented out — swap the active blocks in `pyproject.toml` to switch backends.

### NVIDIA CUDA (default)

No extra steps. Run `uv sync` and everything installs from the CUDA index automatically.

To switch CUDA versions, change the `pytorch-cu128` index URL (e.g. `cu124`) and
update the matching `[tool.uv.sources]` references, then run:

```bash
uv sync --reinstall-package torch --reinstall-package torchvision
```

### AMD ROCm

**System requirements:** ROCm 6.4 installed system-wide (`rocm-smi --showdriverversion`).

**Supported GPU families (ROCm 6.4):**

| Family | GFX | Cards |
|--------|-----|-------|
| RDNA 2 | gfx1030 / 1031 / 1032 | RX 6000 series |
| RDNA 3 | gfx1100 / 1101 / 1102 | RX 7000 series |
| RDNA 4 | gfx1201 | RX 9070 / 9070 XT (natively supported, no `HSA_OVERRIDE_GFX_VERSION` needed) |

In `SFN_DEVICE`, both `auto` and `cuda` resolve to the ROCm/HIP backend — PyTorch uses
the CUDA interface for ROCm.

**Switching to ROCm:**

1. In `pyproject.toml`, comment out the `pytorch-cu128` index block and uncomment `pytorch-rocm64`.
2. In `[tool.uv.sources]`, comment out the CUDA `torch`/`torchvision` lines and uncomment the ROCm ones (including `pytorch-triton-rocm`).
3. In `[project.dependencies]`, uncomment the `pytorch-triton-rocm==...` line.
4. Run:
   ```bash
   uv sync --reinstall-package torch --reinstall-package torchvision \
            --reinstall-package pytorch-triton-rocm
   ```

**Why Python 3.12 is required:**

`pyproject.toml` enforces `requires-python = "==3.12.*"` for all users. The pin exists
because of the ROCm workaround: PyTorch ships `pytorch-triton-rocm` with a plain
`linux_x86_64` wheel tag instead of the `manylinux` tag uv expects. The workaround pins
it as a direct project dependency with a hard-coded wheel URL so uv skips tag validation
— but direct URL sources in uv only apply to packages explicitly listed in
`[project.dependencies]`, which requires a fixed Python version to resolve the correct
`cpXYZ` wheel filename. CUDA users are unaffected in practice; the constraint may be
relaxed once uv gains better support for non-standard wheel tags.

**Upgrading PyTorch or switching ROCm versions:**

1. Pick the target index — available versions are listed at `https://download.pytorch.org/whl/torch/`.
2. In `pyproject.toml`, update `[[tool.uv.index]]` to point at the new ROCm index (e.g. `rocm7.2`) and update the `[tool.uv.sources]` `index` references to match.
3. Find the `pytorch-triton-rocm` version that the new torch requires (visible in the wheel's `.metadata` file at `https://download.pytorch.org/whl/<rocm-index>/torch-<version>+rocm<X.Y>-cpXYZ-cpXYZ-linux_x86_64.whl.metadata`).
4. Update the `pytorch-triton-rocm==X.Y.Z` pin in `[project.dependencies]` and the direct URL in `[tool.uv.sources]`. The URL pattern is:
   ```
   https://download-r2.pytorch.org/whl/pytorch_triton_rocm-X.Y.Z-cpABC-cpABC-linux_x86_64.whl
   ```
5. Run `uv sync --reinstall-package torch --reinstall-package torchvision --reinstall-package pytorch-triton-rocm`.

## Configuration

All settings live in `.env`. Copy `.env.example` to get started — every key is documented
there. Environment variables already set in the shell take precedence over the file.

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
| `SFN_ALLOW_ONLINE` | `false` | Allow HuggingFace Hub connections (for first-time model downloads only) |

## Model setup (one-time)

ScalarForensic runs **offline by default** — the HuggingFace SDK is blocked from making
any network requests at runtime (see [Network policy](#network-policy) below).  This means
models must be downloaded explicitly before first use.

Use the download script to fetch both models in one step:

```bash
uv run python scripts/download_models.py          # both models
uv run python scripts/download_models.py --sscd   # SSCD only
uv run python scripts/download_models.py --dino   # DINOv2 only
```

This places the files at the default paths (`models/sscd_disc_mixup.torchscript.pt` and
`models/dinov2-large/`). After downloading DINOv2, set `SFN_MODEL_DINO=models/dinov2-large`
in `.env` so the app loads the local snapshot. SSCD is always loaded from a local file and
needs no change.

**Manual alternative — SSCD:**

```bash
mkdir -p models
wget -P models https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
```

**Manual alternative — DINOv2:**

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov2-large', local_dir='models/dinov2-large', local_dir_use_symlinks=False)
"
```

Then set `SFN_MODEL_DINO=models/dinov2-large` in `.env`.

## Network policy

ScalarForensic is designed for **airgapped / offline environments**.  By default:

- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set at process startup, preventing
  the HuggingFace SDK from making any network connections.
- Connections to `SFN_QDRANT_URL` and `SFN_EMBEDDING_ENDPOINT` are always allowed — these
  are explicit user-configured endpoints, not internet traffic.
- If `SFN_MODEL_DINO` is still set to a HuggingFace Hub ID (e.g. `facebook/dinov2-large`)
  and no local snapshot exists, the app will **refuse to start** with a clear error message
  pointing to the download script.

To allow internet connections for a first-time model download, use the `--allow-online`
flag or set `SFN_ALLOW_ONLINE=true`:

```bash
# CLI indexer
sfn --allow-online /path/to/images --dino --sscd

# Web UI
sfn-web --allow-online

# Via environment variable (persists for the session)
SFN_ALLOW_ONLINE=true sfn-web
```

Once models are cached locally and `SFN_MODEL_DINO` points to a local directory, remove
`--allow-online` and leave `SFN_ALLOW_ONLINE=false` (or unset) for all subsequent runs.

## HEIC/HEIF support

HEIC/HEIF (iPhone photos etc.) is an optional extension:

```bash
uv sync --group heif   # installs pillow-heif
```

Once installed, `.heic` and `.heif` files are picked up automatically with no further
configuration.

## Test data / quick-start sample

The `test/` folder contains two scripts to set up a local sample dataset. The downloaded
images and generated search files are gitignored — only the scripts are committed.

```bash
# 1. Download and unzip the Unsplash sample dataset (~200 MB) into data/
bash test/download_data.sh

# 2. Copy 10 random images to test/searchfiles/ and print the full setup guide
uv run python test/prepare_searchfiles.py

# optional: choose a different count or seed
uv run python test/prepare_searchfiles.py --count 20 --seed 7
```

`prepare_searchfiles.py` prints step-by-step instructions at the end:
install deps → download SSCD model → start Qdrant → index → start web server →
visit `localhost:8080` → upload from `test/searchfiles/`.

## Offline / airgapped deployment

Both models and all Python dependencies can be pre-downloaded and bundled with the project
so the entire folder runs without internet access.

### On the internet-connected machine

**1. Download models:**

```bash
uv run python scripts/download_models.py
```

Models land at `models/sscd_disc_mixup.torchscript.pt` and `models/dinov2-large/`. Set
`SFN_MODEL_DINO=models/dinov2-large` in `.env` so the app loads the local DINOv2 snapshot.
Leave `SFN_ALLOW_ONLINE=false` (or unset) — the app enforces offline mode by default.

**2. Download Python wheels:**

```bash
bash scripts/download_deps.sh
```

This runs `uv export --frozen` to capture the locked dependency list, then downloads all
wheels into `vendor/` using `uv pip download`. Because it goes through uv, the custom
PyTorch index configured in `pyproject.toml` is respected automatically.

> **Platform requirement:** wheels are specific to OS, CPU architecture, and Python
> version. Run this script on a machine that matches the airgapped target (e.g.
> `linux/x86_64`, Python 3.12). Downloading on macOS or a different Python version and
> transferring to a Linux target will cause install failures on the offline machine.

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

### On the airgapped machine

**5. Install Python dependencies from the local wheelhouse:**

```bash
uv venv
uv pip install --no-index --find-links vendor/ -r requirements.txt
uv pip install --no-index --find-links vendor/ --no-deps -e .
```

**6. Load and start Qdrant:**

```bash
docker load < qdrant.tar.gz
docker run -p 6333:6333 qdrant/qdrant:v1.17.1
```

**7. Run:**

```bash
source .venv/bin/activate
sfn <image-dir> --dino --sscd   # offline by default — no flag needed
sfn-web                          # same
```

### Docker bundle (single-file transfer)

If Docker is available on both machines, the entire environment — Python
interpreter, all wheels, and both models — can be baked into a single Docker
image and saved as one gzip'd tarball. This replaces the seven steps above with
two: `docker load` and `docker compose up`.

**When to prefer this over the wheel-based approach:**

- You want a single file to hand off, with no dependency on uv or Python being
  installed on the airgapped machine.
- You want Qdrant and the app to start together with one command and persist
  data automatically across restarts.
- You need to deploy to multiple airgapped machines from the same artifact.

#### On the internet-connected machine

Run the all-in-one build script:

```bash
bash scripts/build_airgap_image.sh            # tag: latest
bash scripts/build_airgap_image.sh --tag 1.0  # pin a version
```

This downloads models and wheels (if not already present), builds the Docker
image, and saves it as `scalarforensic-<tag>.tar.gz`. The build takes a while
the first time; subsequent code-only rebuilds reuse the cached model and
dependency layers.

**Transfer** to the offline machine:

| File | Purpose |
|------|---------|
| `scalarforensic-<tag>.tar.gz` | Docker image (everything bundled) |
| `docker-compose.yml` | Service definitions |
| `.env.example` | Config template |

#### On the airgapped machine

**1. Load the image:**

```bash
docker load < scalarforensic-<tag>.tar.gz
```

**2. Create your config file:**

```bash
cp .env.example .env
# Set SFN_IMAGES_DIR to the host path containing the images to index:
export SFN_IMAGES_DIR=/path/to/evidence/images
```

`SFN_IMAGES_DIR` controls which host directory is mounted into the container as
`/images`. Set it in your shell before running Compose, or add it to `.env`.

**3. Start Qdrant and the web UI:**

```bash
docker compose up -d
```

**4. Index images:**

```bash
docker compose run --rm sfn-web sfn --dino --sscd
# /images is the default input dir inside the container (set by docker-compose.yml).
# Pass a subdirectory if needed: sfn /images/case-001 --dino --sscd
```

CSV reports are written to `/app/` inside the container by default. To save them
on the host, redirect the path:

```bash
docker compose run --rm sfn-web sfn --dino --sscd --report /images/report.csv
```

**5. Open the web UI:**

```
http://localhost:8080
```

**GPU passthrough:**

Uncomment the relevant block in `docker-compose.yml`:

- **NVIDIA** — requires `nvidia-container-toolkit` on the host; uncomment the
  `deploy.resources.reservations.devices` block.
- **AMD ROCm** — uncomment the `devices` (`/dev/kfd`, `/dev/dri`) and
  `group_add` blocks.

Without GPU passthrough the app runs on CPU, which is significantly slower but
fully functional.

**Qdrant data persistence:**

Qdrant's storage is kept in the named Docker volume `qdrant_data`. It survives
`docker compose down` and image updates. To wipe it:

```bash
docker volume rm scalarforensic_qdrant_data
# Note: use the project-prefixed name shown by `docker volume ls`
```
