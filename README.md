# ScalarForensic

Large-scale image vector indexer for forensic use. Embeds images into a Qdrant vector database for similarity search and deduplication — designed for evidence rooms where you need to find exact copies, edited versions, and visually related images across tens of thousands of files.

<img width="1895" height="1272" alt="image" src="https://github.com/user-attachments/assets/cd51803e-cce1-4501-a4f5-d70833ae1afd" />

<img width="804" height="1201" alt="image" src="https://github.com/user-attachments/assets/fd925dd6-3c98-4cca-88a4-bc481ab801ac" />


**→ [Installation & setup guide](INSTALL.md)**

**→ [Docker bundle for airgapped deployment](INSTALL.md#docker-bundle-single-file-transfer)**

**→ [Performance assessment](docs/performance.md)**

---

## What it does

ScalarForensic indexes a directory of images and videos into a Qdrant vector database using two complementary neural models, then lets you query that database through a web UI to find matches.

### Indexing (CLI)

```bash
uv run sfn <media-dir> --dino --sscd
```

Scans a directory recursively, hashes each file to skip duplicates, and upserts embeddings into Qdrant. Both models can be run in a single pass — files are read and decoded once. Images are embedded directly; videos are temporally sampled (default: 1 frame/second, capped at 500 frames) and each frame is passed through the same pipeline.

### Querying (Web UI)

```bash
uv run sfn-web   # → http://localhost:8080
```

Upload one or more query images or videos, select search modes, and explore results with threshold and result-count sliders. Hit images are streamed from the server filesystem on demand. Video hits are grouped by source file — the best-matching frame is shown as the thumbnail, with a timeline bar marking all indexed and matched frame positions.

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

**Supported video formats:** `.mp4` `.avi` `.mov` `.mkv` `.wmv` `.flv` `.webm` `.m4v` `.mpg` `.mpeg` `.3gp` `.ts` `.mts` — decoded via [PyAV](https://pyav.org) (FFmpeg, bundled in the wheel, no system dependency). Frame extraction rate and per-file frame cap are configurable via `SFN_VIDEO_FPS` and `SFN_VIDEO_MAX_FRAMES`.

**Deduplication modes** (controlled by `SFN_DUPLICATE_CHECK_MODE`):

| Mode | Behaviour |
|------|-----------|
| `hash` | Skip images whose SHA-256 is already in the collection _(default)_ |
| `filepath` | Skip images whose absolute path is already stored — faster for large files |
| `both` | Skip if either hash or path is already indexed |

**EXIF metadata extraction** (`SFN_EXTRACT_EXIF=true`): stores two boolean fields on every indexed point — `exif` (any EXIF data present) and `exif_geo_data` (GPS coordinates present).

**Configuration stability after indexing:** `SFN_SSCD_N_CROPS`, `SFN_NORMALIZE_SIZE`, and the model paths (`SFN_MODEL_DINO`, `SFN_MODEL_SSCD`) are baked into every embedding and must not change after a collection has been indexed. Every Phase-2 query is re-embedded with the *current* values and cosine-compared against vectors produced under the *previous* ones — any drift puts query and corpus vectors in different embedding spaces and silently invalidates the calibrated similarity thresholds. Both `sfn` and `sfn-web` refuse to start on mismatch; pass `--ignore-config-mismatch` to either for read-only inspection (results will be silently wrong — never use for forensic conclusions). The check also samples up to 1 024 existing points per vector type and rejects collections that already contain *multiple* embedding configurations (the symptom of a previous `--ignore-config-mismatch` indexing run); such collections are unrecoverable and must be re-indexed from scratch — no .env change can make queries against them meaningful.

**Offline / airgapped by default:** the HuggingFace SDK is blocked from making any network requests at runtime. Connections to Qdrant and any configured remote embedder endpoint are the only outward traffic. Pass `--allow-online` (or set `SFN_ALLOW_ONLINE=true`) for first-time model downloads; omit it for all subsequent runs. Models and Python wheels can also be pre-bundled for fully airgapped deployment. See [INSTALL.md § Network policy](INSTALL.md#network-policy) and [INSTALL.md § Offline deployment](INSTALL.md#offline--airgapped-deployment).

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

# Download models once (requires internet — set SFN_MODEL_DINO=models/dinov2-large in .env afterwards)
uv run python scripts/download_models.py

# Index a folder (offline by default — no flag needed after models are local)
uv run sfn <image-dir> --dino --sscd

# Start the web UI
uv run sfn-web   # → http://localhost:8080
```

For full setup instructions, GPU configuration, offline deployment, and configuration
reference, see **[INSTALL.md](INSTALL.md)**.

---

## Third-party licenses

Bundled assets and their licenses are listed in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). All are compatible with GPL-3.
