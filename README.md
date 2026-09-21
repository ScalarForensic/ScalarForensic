# ScalarForensic

Large-scale image vector indexer for forensic use. Embeds images into a Qdrant vector database for similarity search and deduplication — designed for evidence rooms where you need to find exact copies, edited versions, and visually related images across tens of thousands of files.

<img width="1895" height="1272" alt="image" src="https://github.com/user-attachments/assets/cd51803e-cce1-4501-a4f5-d70833ae1afd" />

<img width="804" height="1201" alt="image" src="https://github.com/user-attachments/assets/fd925dd6-3c98-4cca-88a4-bc481ab801ac" />


**→ [Installation & setup guide](INSTALL.md)**

**→ [Docker bundle for airgapped deployment](INSTALL.md#docker-bundle-single-file-transfer)**

**→ [Performance assessment](docs/performance.md)**

**→ [Deployment topologies (single machine / distributed LAN)](docs/deployment.md)**

**→ [Optional image and video parser isolation](docs/deployment.md#optional-media-parser-isolation)**

**→ [Face modality setup (optional, disabled by default)](INSTALL.md#face-modality-optional)**

**→ [Transcoded video playback: decisions and limits](docs/specs/video-playback-transcode.md)**

---

## What it does

ScalarForensic indexes images and videos into Qdrant using two complementary neural models, with an optional face modality. The web UI lets you find matches, review source media and play compatible viewing copies of video.

### Running commands with `./run.sh`

Use `./run.sh <command>` for the examples below. The wrapper makes the virtualenv's
NVIDIA CUDA libraries available before calling `uv run` with your arguments. If those
libraries are absent, it simply calls `uv run`.

Run the CLI and web UI from the repository directory: `.env` and relative state paths
are resolved from the working directory. Use the same configuration and case state
for both; the [deployment guide](docs/deployment.md) describes the storage settings.

### Indexing (CLI)

```bash
./run.sh sfn <media-dir> --dino --sscd
```

Scans a directory recursively, hashes each file to skip duplicates, and upserts embeddings into Qdrant. Both image models share preprocessing in a single indexing pass. Videos are temporally sampled (default: 1 frame/second, capped at 500 frames per video), and each frame goes through the same embedding pipeline.

Add `--faces` to include [face observations](#face-modality-optional). For labelled image
reference material, `--reference` sends the image vectors to a separately configured
[reference collection](INSTALL.md#reference-collection). Each run writes a CSV ingestion
report under `SFN_REPORT_DIR` (default `data/reports`). Cached thumbnails and sampled
video frames support browsing when source media is no longer mounted.

### Querying (Web UI)

```bash
./run.sh sfn-web   # → http://localhost:8080
```

Upload one or more query images or videos, select search modes, and explore results with threshold and result-count sliders. Hit images are streamed from the server filesystem on demand. Video hits are grouped by source file, with a best-match thumbnail, a timeline of indexed and matched frames, and browser playback through a bounded viewing-copy cache.

### Maintenance (CLI)

```bash
./run.sh sfn-faces purge --media <sha256>   # remove face observations and unused chips
./run.sh sfn-video purge --media <sha256>   # remove managed viewing copies
./run.sh sfn-video render --path <file> --at <seconds>
./run.sh sfn-video render --path <file> --full
```

Both purge commands also accept `--all` and append to their audit logs.
`sfn-video render` prints the recorded rendering command and provenance. When no
record exists, it prints a clearly labelled reproduction recipe based on current
settings. See the [playback specification](docs/specs/video-playback-transcode.md#72-the-rendering-must-be-reproducible).

---

## Search modes

| Mode | What it finds | Model |
|------|--------------|-------|
| **Exact** | Byte-identical files (SHA-256 match) | — |
| **Altered** | Cropped, recolored, or lightly modified copies | SSCD ResNet-50 (Meta), 512-dim |
| **Semantic** | Same scene, subject, or visual style | DINOv2 ViT-L (Meta), 1024-dim |

Modes are automatically disabled in the UI if the corresponding collection has not been indexed yet. All three can be combined in a single query.

When the face modality is enabled, select a detected query face to search the face
collection or compare it with another observation. See [Face modality](#face-modality-optional).

---

## Features

**Supported image formats:** `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp` `.gif` `.jp2` `.ico` `.psd` — plus `.heic`/`.heif` via an optional dependency group.

**Supported video formats:** `.mp4` `.avi` `.mov` `.mkv` `.wmv` `.flv` `.webm` `.m4v` `.mpg` `.mpeg` `.3gp` `.ts` `.mts` — frame extraction uses [PyAV](https://pyav.org), whose wheels bundle FFmpeg libraries. Transcoded browser playback additionally requires the system `ffmpeg` executable with `zscale`/`tonemap` support (see [INSTALL.md](INSTALL.md)). Frame extraction rate and per-file frame cap are configurable via `SFN_VIDEO_FPS` and `SFN_VIDEO_MAX_FRAMES`.

**Deduplication modes** (controlled by `SFN_DUPLICATE_CHECK_MODE`):

| Mode | Behaviour |
|------|-----------|
| `hash` | Skip images whose SHA-256 is already in the collection _(default)_ |
| `filepath` | Skip images whose absolute path is already stored — faster for large files |
| `both` | Skip if either hash or path is already indexed |

**EXIF metadata extraction** (`SFN_EXTRACT_EXIF=true`): stores two boolean fields on every indexed point — `exif` (any EXIF data present) and `exif_geo_data` (GPS coordinates present).

**Configuration stability after indexing:** model weights and preprocessing settings, including `SFN_NORMALIZE_SIZE` and `SFN_SSCD_N_CROPS`, must match the collection's recorded configuration. Ingestion and similarity queries reject incompatible configurations; mixed configurations or missing provenance require a rebuild. `sfn-web --ignore-config-mismatch` permits metadata and original-file inspection while incompatible similarity analysis remains blocked. See [collection compatibility and rebuilding](docs/deployment.md#collection-compatibility-and-face-identity-rebuild).

**Offline / airgapped by default:** the HuggingFace SDK is blocked from making any network requests at runtime. Connections to Qdrant and any configured remote embedder endpoint are the only outward traffic. Pass `--allow-online` (or set `SFN_ALLOW_ONLINE=true`) for first-time model downloads; omit it for all subsequent runs. Models and Python wheels can also be pre-bundled for fully airgapped deployment. See [INSTALL.md § Network policy](INSTALL.md#network-policy) and [INSTALL.md § Offline deployment](INSTALL.md#offline--airgapped-deployment).

**Video playback in the browser:** when a source cannot play directly, the application can rewrap supported streams without re-encoding or create transcoded viewing copies on demand. Transcodes can resize the image and tone-map HDR; the cache is bounded, and rendering records disclose the processing applied. *Download original* serves the unchanged source bytes with their verified SHA-256. Full-copy requests are subject to size limits. See [playback setup](INSTALL.md#transcoded-playback) and the [decisions and limits](docs/specs/video-playback-transcode.md).

**GPU acceleration:** NVIDIA CUDA (default) and AMD ROCm are both supported, including RDNA 4 (RX 9070 / 9070 XT) natively on ROCm 6.4. See [INSTALL.md § GPU setup](INSTALL.md#gpu--hardware-acceleration).

---

## Face modality (optional)

Face processing is **disabled by default**. It stores observations in a separate
collection for the case, with native review crops and quality-gated embeddings.
Use it to find candidates for examination; the raw similarity scores are
**uncalibrated** and do not establish identity.

```bash
uv sync --locked --no-default-groups --group web --group faces
./run.sh python scripts/download_models.py --yunet
```

The download command fetches the YuNet detector and verifies its checksum. Recognition
weights are supplied by the operator, together with a validated JSON manifest.
Configure the model paths, set `SFN_FACES_ENABLED=true` and `SFN_EXAMINER_ID`, and follow
the [face setup guide](INSTALL.md#face-modality-optional) for storage, model licensing
and activation. The first collection activation prompts for an authorization reference
and records examiner attribution.

Faces that pass the review gate but fail the embedding gate remain available as
**review-only observations**, without vectors and therefore excluded from similarity
search. Native review crops never upscale; aligned embedding crops may resample and
are labelled accordingly. Quality exclusions and model provenance remain visible.

See the [method and limits](docs/face-matching-math.md) and the
[face pipeline specification](docs/specs/face-pipeline.md) for details.

---

## Quick start

```bash
git clone https://github.com/ScalarForensic/ScalarForensic
cd ScalarForensic
uv sync --locked --no-default-groups --group web
cp .env.example .env
# Add --group faces or --group heif to the install command if needed.

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant:v1.19.0

# Download models once (requires internet — set SFN_MODEL_DINO=models/dinov2-large in .env afterwards)
./run.sh python scripts/download_models.py

# Index a folder (offline by default — no flag needed after models are local)
./run.sh sfn <image-dir> --dino --sscd

# Start the web UI
./run.sh sfn-web   # → http://localhost:8080
```

For full setup instructions, GPU configuration, offline deployment, and configuration
reference, see **[INSTALL.md](INSTALL.md)**.

---

## Third-party licenses

Bundled assets and their licenses are listed in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). All are compatible with GPL-3.

The [current operations guide](docs/deployment.md) covers durable state, backup and
restore, compatibility inspection, and release inventories. Model weights are not
tracked in this repository; airgap bundles include the artifacts copied into them
and require artifact-specific redistribution review.

Further documentation: [specifications](docs/specs/), [benchmarks](docs/benchmarks/),
and [audit and release evidence](docs/audit/).
