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
| `SFN_COLLECTION` | `sfn` | Qdrant collection name (holds all vector types) |
| `SFN_MODEL_DINO` | `facebook/dinov2-large` | DINOv2 model identifier |
| `SFN_MODEL_SSCD` | `models/sscd_disc_mixup.torchscript.pt` | Path to SSCD checkpoint |
| `SFN_NORMALIZE_SIZE` | `512` | DINOv2 resize dimension (N×N px) |
| `SFN_BATCH_SIZE` | `32` | Images per embedding batch |
| `SFN_DEVICE` | `auto` | Compute device: `auto` \| `cuda` \| `cpu` \| `mps` |
| `SFN_INPUT_DIR` | _(none)_ | Default input folder (can be passed as CLI argument instead) |
| `SFN_DUPLICATE_CHECK_MODE` | `hash` | Dedup strategy: `hash` \| `filepath` \| `both` |
| `SFN_EXTRACT_EXIF` | `false` | Store EXIF presence flags in the database |
| `SFN_REFERENCE_COLLECTION` | _(unset)_ | Qdrant collection holding external reference vectors. When set, reference material is indexed here instead of the main collection, keeping the two datasets separate. See [Reference collection](#reference-collection). |
| `SFN_ALLOW_ONLINE` | `false` | Allow HuggingFace Hub connections (for first-time model downloads only) |
| `SFN_VIDEO_FPS` | `1.0` | Frames to extract per second of video |
| `SFN_VIDEO_MAX_FRAMES` | `500` | Hard cap on frames extracted per video file (0 = no cap) |

## Remote deployment

ScalarForensic's web server binds to `0.0.0.0` by default, so it is reachable from any network interface without additional configuration. All data (indexed collections, reference images, thumbnails) lives on the server; only query images are uploaded from the remote client over HTTP and are held in a server-side temporary directory for the duration of the session.

**New env vars for remote deployments:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SFN_WEB_HOST` | `0.0.0.0` | Address the web server binds to |
| `SFN_WEB_PORT` | `8080` | Port the web server listens on |
| `SFN_MAX_UPLOAD_BYTES` | `2147483648` (2 GiB) | Maximum total upload size per analyze request. Set to `0` to disable. |
| `SFN_SESSION_TTL_SECONDS` | `3600` | Seconds of inactivity after which a session and its temp files are automatically deleted |
| `SFN_MAX_ACTIVE_SESSIONS` | `32` | Maximum number of concurrent sessions. New sessions beyond this limit receive HTTP 503. Set to `0` to disable. |

**TLS and authentication:** these are not provided by `sfn-web` and should be handled at the network layer. Use a reverse proxy (e.g. nginx, Caddy) for TLS termination and access control before exposing the service to untrusted networks.

**Qdrant:** when using Docker Compose, Qdrant is only reachable on the internal compose network — it is not published to the host. This prevents unintended vector-database exposure. To allow direct host access for debugging, add a local override file:

```yaml
# docker-compose.debug.yml
services:
  qdrant:
    ports:
      - "127.0.0.1:6333:6333"
```

```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml up -d
```

## Reference collection

The reference collection is an optional, separate Qdrant collection that holds vectors for externally labelled reference material. Keeping it separate from the case collection means your case vectors are never co-mingled with third-party reference material, and the reference collection can be reused across cases.

**Setup:**

1. Add `SFN_REFERENCE_COLLECTION=sfn_reference` to `.env` (or choose any name).
2. Index your reference material into that collection using the `--reference` flag:
   ```bash
   sfn /path/to/reference/images --dino --sscd --reference
   ```
   Points are stored in `sfn_reference` and tagged `is_reference=true` in their payload.
3. Start the web UI as normal. The reference collection is available as a lookup source in the tag triage workflow — query embeddings can be compared against it without touching the case collection.

**Notes:**

- `SFN_REFERENCE_COLLECTION` must be set before running `sfn --reference`. The indexer exits with an error if the variable is unset.
- The case collection (`SFN_COLLECTION`) is unaffected — `--reference` routes all writes to the reference collection only.
- The reference collection can be on the same Qdrant instance as the case collection; it is just a separate named collection.

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

## Face modality (optional)

The face modality is **disabled by default** and ships as an optional dependency group. It
detects faces in indexed media, quality-gates them, aligns and embeds them, and stores them as
searchable observations in a case-scoped sidecar collection. It is a **discovery aid, not an
identification system** — see `docs/face-matching-math.md` for the method chain and its stated
limits, and `docs/specs/face-pipeline.md` for the full design.

### 1. Install the dependencies

```bash
uv sync --group faces      # opencv-python-headless + onnxruntime
```

### 2. Fetch the detector (YuNet, MIT)

```bash
uv run python scripts/download_models.py --yunet
```

One-time online fetch of a ~230 KB ONNX file, checksum-verified against a pinned commit. After
this the face pipeline performs **zero network I/O**. Add the printed line to `.env`:

```bash
SFN_FACE_DETECTOR_MODEL=models/face_detection_yunet_2023mar.onnx
```

### 3. Supply a recognition model — and read this first

**ScalarForensic ships no face-recognition weights, and this is deliberate.**

The entire ArcFace / AdaFace / InsightFace weight family in common use is released for
**research purposes only**. No permissively-licensed face-recognition model of comparable
quality is known to us. That means:

- **Whether research-only weights may lawfully be used in your deployment is a legal question,
  not a technical one.** It depends on your jurisdiction, your organisation and your purpose.
- **"Non-commercial" is not automatically satisfied by government or law-enforcement use.**
  Public-sector operation is not the same thing as non-commercial use, and several licences in
  this family have been read narrowly.
- **The decision, and its legal review, belong to the operator — in writing.** We deliberately
  provide no "works out of the box" recognition path, because shipping one would make that
  decision silently on your behalf.
- If you obtain weights, their provenance and training data are your responsibility too.

Until an embedder is configured, the face modality reports itself unavailable and says why.

### 4. Describe the model with a manifest

Every embedder needs a JSON manifest next to it at `<model>.manifest.json`. It is validated
against the ONNX session's real inputs and outputs at load time; a mismatch is a hard error.
Example, using the InsightFace convention `(x − 127.5) / 128`:

```json
{
  "input_name": "input.1",
  "layout": "NCHW",
  "channel_order": "RGB",
  "dtype": "float32",
  "input_size": 112,
  "mean": 127.5,
  "scale": 128.0,
  "output_name": "683",
  "embedding_dim": 512
}
```

`input_size` must be 112 — the alignment template (`arcface-112-v1`) is fixed.

### 5. Enable it

```bash
SFN_FACES_ENABLED=true
SFN_EXAMINER_ID=your-examiner-id        # required; stamped on audit entries
SFN_FACE_DETECTOR_MODEL=models/face_detection_yunet_2023mar.onnx
SFN_FACE_EMBEDDER_MODEL=/path/to/your_embedder.onnx
```

Then index with faces:

```bash
./run.sh sfn <dir> --dino --faces     # alongside embeddings
./run.sh sfn <dir> --faces            # faces only
```

The **first activation prompts for a free-text authorization reference**, which is written into
the collection's enablement record and the audit log. Turning the feature on is a recorded act.

### Storage location

Face chips (aligned PNG, review JPEG, thumbnail) are written under `SFN_FACE_STORE_DIR`
(default `data/faces`). This accepts **any absolute path** — another drive is just another
path, another server is an NFS/SMB mount. The store is freely relocatable because the database
references chips by hash only, never by path. Leaving it unset disables chip storage entirely:
the pipeline still runs, and the UI reports "degraded-evidence mode" because review chips are
then unavailable.

> **Set `SFN_FACE_STORE_DIR` per case.** `SFN_FACE_COLLECTION` is already per case, but the
> chip store defaults to a single `data/faces` for all of them, and chips are content-addressed
> — two cases holding the same image share one file on disk. Purge's reference check scrolls
> only its own collection, so **purging case A can unlink a chip case B still references**. The
> code cannot enforce this the way `check_compat` refuses to mix biometric data across cases for
> vectors; it is an operator setting. Note also that purge assumes a **single writer**: it
> checks for references and then unlinks, so a concurrent index run can leave a dangling chip
> reference. Both situations are recoverable — chips are re-derivable from the source media —
> but neither should be discovered during an examination.

### Purging

```bash
uv run sfn-faces purge --media <sha256>   # one medium
uv run sfn-faces purge --all              # everything, with confirmation
```

Both delete the face points and their chip files, and append a `purge` event to the audit log.
The enablement record survives a purge — it is an auditable act, not routine data.

### Settings reference

| Variable | Default | Meaning |
|---|---|---|
| `SFN_FACES_ENABLED` | `false` | Master switch |
| `SFN_EXAMINER_ID` | — | Required while faces are enabled |
| `SFN_FACE_DETECTOR_MODEL` | — | Path to the YuNet ONNX |
| `SFN_FACE_EMBEDDER_MODEL` | — | Path to your recognition ONNX (needs a manifest) |
| `SFN_FACE_COLLECTION` | `{SFN_COLLECTION}_faces` | Case-scoped sidecar collection |
| `SFN_FACE_STORE_DIR` | `data/faces` | Chip store; unset disables chips |
| `SFN_FACE_THUMB_SIZE` | `256` | Browse thumbnail long side (px) |
| `SFN_FACE_DETECT_MAX_SIZE` | `1600` | Detector input cap (px, long side) |
| `SFN_FACE_MIN_CONF` | `0.8` | Minimum detector confidence to embed |
| `SFN_FACE_MIN_SIZE` | `64` | Minimum face size to embed (px) |
| `SFN_FACE_REVIEW_MIN_CONF` | `0.6` | Confidence floor for keeping a face for hand review |
| `SFN_FACE_REVIEW_MIN_SIZE` | `48` | bbox min side (detector-input px) for hand review |
| `SFN_FACE_MIN_SHARPNESS` | `25.0` | Minimum Laplacian variance |
| `SFN_FACE_MAX_CLIPPED` | `0.6` | Maximum clipped-pixel fraction |
| `SFN_FACE_MAX_POSE` | `0.35` | Maximum yaw proxy |
| `SFN_FACE_CROP_DILATION` | `0.15` | Review-chip context margin |

All gate thresholds are **bootstrap values**, not validated operating points. They are recorded
in every point's provenance so a later calibration can supersede them by name.

### Two gates: comparable and review-only

A detected face passes through two gates. The **review gate**
(`SFN_FACE_REVIEW_MIN_CONF` / `_SIZE`) decides whether the face is kept at all; the **embedding
gate** (`SFN_FACE_MIN_CONF` / `_SIZE` / `_SHARPNESS` / `_MAX_CLIPPED` / `_MAX_POSE`) decides
whether it is measured. A face that clears the first but not the second is kept as a
**review-only observation**: it has a review chip, it is croppable and examinable, and it is
never compared with any other face. The exclusion is structural — the point carries no vector —
not a filter a later query could forget to apply. The CLI summary and the audit record report
all three populations, and `detected = comparable + retained for review + rejected`.

The review thresholds are clamped so they can never exceed their embedding counterparts; if you
set them higher, the run prints a note and uses the clamped value. Both are bootstrap numbers
awaiting the same calibration as the rest.

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

## Video support

Video files are processed automatically — no extra setup required. PyAV (FFmpeg bundled as a wheel) is a mandatory dependency and is installed by `uv sync`.

Supported containers: `.mp4` `.avi` `.mov` `.mkv` `.wmv` `.flv` `.webm` `.m4v` `.mpg` `.mpeg` `.3gp` `.ts` `.mts`.

**How it works:** `sfn` extracts frames by uniform temporal sampling — one frame every `1/SFN_VIDEO_FPS` seconds (default: 1 fps). Within each video, identical frames (e.g. static title cards or freeze frames) are deduplicated by SHA-256 before embedding. Each unique frame is embedded and stored with full provenance: video file hash, source path, timecode, extraction fps, and PyAV version. Re-running `sfn` on the same video skips already-indexed frames.

**Tuning extraction:**

| Variable | Default | Notes |
|----------|---------|-------|
| `SFN_VIDEO_FPS` | `1.0` | Higher values extract more frames and increase indexing time proportionally |
| `SFN_VIDEO_MAX_FRAMES` | `500` | Caps frames per file regardless of duration; set to `0` to disable the cap |

**Web UI:** upload a video the same way you upload an image. The analysis pipeline extracts frames, embeds each one, and searches with each frame's embedding. Results are grouped by source video — the best-matching frame is shown as the card thumbnail, and a timeline bar in the detail panel marks all indexed and matched frame positions.

**Forensic reproducibility:** the same video file + the same `SFN_VIDEO_FPS` + `SFN_VIDEO_MAX_FRAMES` values always produces the same set of frames with identical SHA-256 hashes. This allows cross-run deduplication and makes results reproducible across re-indexes.

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
bash scripts/download_deps.sh --web        # CLI + web UI (recommended)
bash scripts/download_deps.sh --web --heif # add HEIC/HEIF support
```

This runs `uv export --frozen` to capture the locked dependency list, then downloads all
wheels into `vendor/` using `uv pip download`. The `--web` flag is required if you intend
to run `sfn-web` on the offline machine (FastAPI, Uvicorn, python-multipart). Without it
only the CLI indexer (`sfn`) will be available.

> **Platform requirement:** wheels are specific to OS, CPU architecture, and Python
> version. Run this script on a machine that matches the airgapped target (e.g.
> `linux/x86_64`, Python 3.12). Downloading on macOS or a different Python version and
> transferring to a Linux target will cause install failures on the offline machine.

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

> **Prerequisite:** `uv` must be installed on the airgapped machine. Install it
> once from the [uv releases page](https://github.com/astral-sh/uv/releases)
> (single static binary, no internet needed after download).

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

If Docker is available on both machines, the ScalarForensic environment —
Python interpreter, all wheels, and both models — can be baked into a Docker
image. The build script also pulls [Qdrant](https://qdrant.tech) (the vector
database backend, a separate application) and saves both images together into
one gzip'd tarball. A single `docker load` followed by `docker compose up`
replaces the seven steps above.

**When to prefer this over the wheel-based approach:**

- You want a single file to hand off, with no dependency on uv or Python being
  installed on the airgapped machine.
- You want Qdrant and the app to start together with one command and persist
  data automatically across restarts.
- You need to deploy to multiple airgapped machines from the same artifact.

#### On the internet-connected machine

**GPU backend selection (do this before building):**

The image bakes in a specific PyTorch backend. Choose before running the
build script:

- **NVIDIA CUDA (default)** — no changes needed; CUDA 12.8 is active in
  `pyproject.toml`.
- **AMD ROCm** — swap the active index blocks in `pyproject.toml` first
  (see [AMD ROCm](#amd-rocm) above), then run the build script. The resulting
  image will only work with ROCm GPU passthrough; it still runs on CPU if no
  GPU override is used.
- **CPU-only** — CUDA wheels run fine on CPU; no special build needed.

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
| `scalarforensic-<tag>.tar.gz` | Both Docker images: ScalarForensic + Qdrant |
| `docker-compose.yml` | Base service definitions (CPU, works everywhere) |
| `docker-compose.nvidia.yml` | NVIDIA GPU override |
| `docker-compose.amd.yml` | AMD ROCm GPU override |
| `.env.example` | Config template |

#### On the airgapped machine

**1. Load the image:**

```bash
docker load < scalarforensic-<tag>.tar.gz
```

**2. Create your config file and set your image directory:**

```bash
cp .env.example .env
```

Open `.env` and set `SFN_IMAGES_DIR` to the host path containing the images to
index.  Docker Compose reads this from `.env` for volume binding — setting it
here ensures it applies to **both** `docker compose up` and `docker compose run`
without having to export it in the shell each time:

```ini
SFN_IMAGES_DIR=/path/to/evidence/images
```

> **Why `.env` and not `export`?** Docker Compose re-evaluates the compose file
> on every `up` and `run` call.  A variable exported only in the shell is
> invisible to later `docker compose run` invocations in a new terminal or
> script.  Putting it in `.env` makes it persistent for all compose commands.

**3. Start Qdrant and the web UI:**

```bash
# CPU (default — works on any machine):
docker compose up -d

# NVIDIA GPU (requires nvidia-container-toolkit on the host):
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d

# AMD ROCm GPU (requires ROCm driver; image must be built with ROCm wheels):
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
```

> If you built with `--tag 1.0` (not `latest`), prefix any of the above with
> `SCALARFORENSIC_IMAGE=scalarforensic:1.0`.

If you need to change `SFN_IMAGES_DIR` after the stack is running, update `.env`
then restart:

```bash
docker compose down
# edit .env: SFN_IMAGES_DIR=/new/path
docker compose up -d
```

**4. Index images:**

`docker compose run` creates a fresh container and does **not** automatically
inherit override files from `up`. Pass the same `-f` flags you used in step 3:

```bash
# CPU:
docker compose run --rm sfn-web sfn --dino --sscd

# NVIDIA:
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm sfn-web sfn --dino --sscd

# AMD ROCm:
docker compose -f docker-compose.yml -f docker-compose.amd.yml run --rm sfn-web sfn --dino --sscd
```

`/images` is the default input directory inside the container (set by `SFN_INPUT_DIR`
in `docker-compose.yml`). Pass a subdirectory if needed:
`sfn /images/case-001 --dino --sscd`

If you need to index a directory that differs from the one the stack was started
with (e.g. a second evidence drive), override the volume for that one run:

```bash
docker compose run --rm \
  -v /path/to/other/images:/images:ro \
  sfn-web sfn /images --dino --sscd
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

GPU support is opt-in via override files — no file editing needed. The base
`docker-compose.yml` runs on CPU and works on any machine.

- **NVIDIA** — requires `nvidia-container-toolkit` on the host; use
  `docker-compose.nvidia.yml` as shown in step 3.
- **AMD ROCm** — requires the ROCm kernel driver (`/dev/kfd` present); use
  `docker-compose.amd.yml` as shown in step 3. The image must also have been
  built with the ROCm PyTorch index active in `pyproject.toml`.

Without a GPU override the app runs on CPU, which is significantly slower but
fully functional.

**Qdrant data persistence:**

Qdrant's storage is kept in the named Docker volume `qdrant_data`. It survives
`docker compose down` and image updates. To wipe it:

```bash
docker volume rm scalarforensic_qdrant_data
# Note: use the project-prefixed name shown by `docker volume ls`
```
