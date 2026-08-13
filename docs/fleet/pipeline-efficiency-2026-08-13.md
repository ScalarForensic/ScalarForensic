# Ingestion pipeline efficiency audit — 2026-08-13

Pre-run audit for the iPhone test campaign. Corpus:
`/media/user01/SAM_870_SATA/Gitea_Backup/input_scalar` (read-only originals),
8,216 files / 20 GB: 3,098 HEIC · 2,329 PNG · 2,182 JPG · 478 MOV · 31 MP4 ·
20 WEBP · 2 GIF · 75 AAE + 1 CSV (unsupported, correctly skipped).
Hardware: RTX 4060 Ti 16 GB, Samsung 870 SATA SSD.
Method: 43-image sample (20 HEIC / 10 JPG / 10 PNG / 3 WEBP, seed 42) + 2 MOV
profiled per stage with `perf_counter`; all 509 videos duration-probed;
GPU sampled via `nvidia-smi` at 150 ms during embed. Raw numbers:
scratchpad `profile_results.json` (profiler: `profile_pipeline.py`, same dir).
Config as profiled: `SFN_NORMALIZE_SIZE=512`, `SFN_SSCD_N_CROPS=5`,
YuNet @ `SFN_FACE_DETECT_MAX_SIZE=1600`, SFace embedder.

## 1. Is each file read and decoded once?

No — by design it is close, with one deliberate exception (faces):

| Step | Disk read | Decode | Shared with |
|---|---|---|---|
| Hash pass (parallel ×32) | read #1 (streamed) | – | sha256 → HashCache; video sha256+md5 |
| Batch loop Phase A | read #2 (`read_bytes`) | – | md5, EXIF parsed from the in-memory bytes |
| `preprocess_batch` | – | **decode #1** (capped to 512 px short side) | one PIL image feeds DINO **and** SSCD **and** the thumbnail |
| Face pass | read #3 | **decode #2** (full native res) | YuNet detect + SFace embed |

- DINO/SSCD/thumbnail sharing is real and correct: one decode, per-model
  normalize only (`cli.py:815-836`).
- The face pass is a separate sequential pass over every medium
  (`cli.py:1564-1602`) — deliberate, because the batch loop only visits
  not-yet-embedded media and faces need their own idempotency (markers). It
  re-reads and re-decodes at native resolution (4032×3024 for HEIC), which
  faces genuinely need for small-face detection — but it means HEIC pays the
  expensive decode twice.
- **There is no perceptual hash in the codebase.** Exact sha256 + md5 only;
  near-duplicate detection is what DINO/SSCD are for. The operator's
  "perceptual hash" stage does not exist and therefore costs nothing.
- md5 is computed from bytes already in memory (no extra read); sha256 comes
  from the hash pass via lookup. Read #2 is unavoidable at current design
  (bytes are not kept across passes) but is cheap (SSD, ≤0.5 ms/file warm).
- Footgun found while profiling: HEIC decodability is an **import side effect
  of `scanner`** (`scanner.py:39` registers the HEIF opener). Every real
  entry point imports scanner, but any future tool that decodes without it
  will silently fail on HEIC bytes.

## 2. Measured per-stage wall time (mean s/file, single-threaded)

| Stage | HEIC | JPG | PNG | WEBP |
|---|---|---|---|---|
| read (warm) | 0.0004 | 0.0005 | 0.0005 | 0.0000 |
| sha256 | 0.0008 | 0.0008 | 0.0007 | 0.0000 |
| md5 | 0.0017 | 0.0018 | 0.0017 | 0.0001 |
| decode shared (cap 512) | **0.171** | 0.019 | 0.039 | 0.011 |
| thumbnail write | 0.0018 | 0.0016 | 0.0014 | 0.0013 |
| face decode (native res) | **0.103** | 0.032 | 0.020 | 0.005 |
| face detect (YuNet) | 0.035 | 0.026 | 0.020 | 0.010 |
| face align+embed (when faces) | 0.017 | 0.017 | 0.017 | 0.015 |
| mean file size (MB) | 2.0 | 2.1 | 2.0 | 0.09 |

HEIC decode is ~9× JPEG: the JPEG `draft()` downscale-during-decode shortcut
(`embedder.py:368-369`) has no HEIF equivalent, so every HEIC decodes at
12 MP and is Lanczos-resized down (~0.10 s decode + ~0.07 s resize).

GPU embed (batch of 43, warm, per-image):

| Model | s/img | img/s | GPU util mean/max | VRAM max |
|---|---|---|---|---|
| DINOv2-large @512 fp16 | 0.0278 | 36 | 35 % / 100 % | 4.1 GB |
| SSCD 5-crop | 0.0155 | 64 | 55 % / 100 % | 4.4 GB |

GPU util is bursty because the HF processor's tensor preprocessing runs on
CPU inside `embed_images`; the pipelined batch loop (depth 2) overlaps this
with Phase A, so it is not the bottleneck. VRAM headroom at the configured
`SFN_BATCH_SIZE=128` is ample (≈4.4 GB at 43 images; 16 GB card; previous
campaigns validated 128).

Cold start: DINO first batch +9 s (CUDA/compile warm-up), model loads ~1 s.

## 3. Videos — what the current extraction actually samples

Probe of all 509 videos (0.5 s, container-open only, zero errors):
total 4,856 s of footage, median 2.5 s (Live-Photo-style clips), mean 9.5 s,
max 318.5 s. Extraction is `SFN_VIDEO_FPS=1.0` (1 frame/s of footage) capped
at `SFN_VIDEO_MAX_FRAMES=500` per video; **the cap never binds on this corpus**
(longest video ⇒ ~319 frames). Yield: **≈4,856 frames**, stored as JPEG q85 in
`SFN_FRAME_STORE_DIR`, then treated exactly like images (same dedup, embed,
faces). Measured extraction wall rate: 27 frames/s (1080p HEVC decode-bound)
⇒ ≈180 s slicing + ~15 ms/frame JPEG encode ⇒ **≈4–5 min**, plus the frames'
share of the image pipeline (counted below).

**(b) Answer: no frame-rate/frame-count reduction is needed.** 1 fps adds
~4.9 k items (+39 % of item count) but only ~8 min of end-to-end time.

## 4. Extrapolation to the full corpus (7,631 images + 4,856 frames = 12,487 items)

| Stage | Single-thread s | Effective wall (parallelism) |
|---|---|---|
| Hash pass (20 GB) | ~50 (cold SATA ~530 MB/s) | ≤ 1 min (×32 threads) |
| Video slicing + JPEG encode | ~250 | ~4–5 min (sequential by design) |
| Shared decode (HEIC 530 + PNG 91 + JPG 41 + frames 92) | 754 | ~2–3 min (thread pool, overlapped with GPU) |
| DINO embed | 347 | ~6 min |
| SSCD embed | 194 | ~3 min |
| Thumbnails + upserts | ~40 | ~1 min |
| **Face pass (sequential, 1 thread)** | decode 591 + detect 338 + embed ~110 | **~17–18 min** |

**(a) Projected total: ~35 min wall, upper bound ~45 min. Trivially feasible
on this single GPU** — the GPU is busy for ~9 of those minutes; the campaign
is CPU-bound, not GPU-bound. Re-runs after interruption are much cheaper
(HashCache + Qdrant dedup + face markers all skip finished work).

**(c) Dominant stage: the face pass (~47 % of projected wall), runner-up HEIC
decode (~9 min of single-thread CPU inside the shared pool).** Top fixes:

1. **Parallelize the face pass** (thread pool over media like the hash pass;
   needs per-thread YuNet instances — OpenCV's detector is not thread-safe).
   Buys ~12–13 min (17 → ~4–5 min on 8 threads). Worth doing, not blocking.
2. **Single decode for faces + embeddings** (decode native once, downscale for
   the embed path). Buys ~6 min more but restructures two passes with
   different idempotency models — poor value at this corpus size.

Neither is required pre-run at a 35-minute projection.

## 5. Correctness findings that outrank efficiency (found while profiling)

1. **SFace multi-face crash — blocks the campaign's face modality.** The SFace
   ONNX declares input `(1, 3, 112, 112)`; `OnnxFaceEmbedder.embed` stacks all
   of an image's embeddable crops into one call (`indexing.py:226`), so every
   medium with ≥2 embeddable faces throws `INVALID_ARGUMENT`, fails
   `process_image`, gets no marker, and is retried-and-refailed forever.
   7/43 sample media (16 %) hit this; a family-photo corpus will hit it
   constantly. Fix (chunk to declared batch dim) + regression tests: **PR #111**.
2. **Suite hermeticity vs the campaign `.env`.** `Settings()` folds the CWD
   `.env` into the process env, so the newly written campaign config
   (`SFN_FACES_ENABLED=true`) failed 7 default-asserting tests locally while
   CI (no `.env`) stayed green. Fixed suite-wide in the same PR
   (`tests/conftest.py`); with it, 561 passed / 5 skipped with the campaign
   `.env` in place.

## Review

- Numbers were measured twice: the first profiler run silently skipped all 20
  HEIC (missing HEIF-opener registration, §1 footgun) and was discarded; the
  quoted run decodes all 43/43 sample images.
- Read timings are page-cache-warm; the cold-read bound is computed from the
  drive's interface limit, not measured, and is immaterial (≤1 min).
- Face align+embed mean excludes the 7 multi-face media that crash unfixed
  (§5.1) — with PR #111 merged, their cost is the same per-face rate.
- The SFace batch-1 behaviour was verified directly against the production
  model (`(1,3,112,112)` declared; 2 crops → INVALID_ARGUMENT; fixed build:
  3 crops → (3, 128)).
- Extrapolations assume sample-mean file cost is corpus-representative;
  HEIC/JPG/PNG sizes in the corpus are uniform iPhone output, so the risk is
  low. Suite bar quoted at `561 passed / 5 skipped` on branch
  `fix/sface-multi-face-batch` (porcelain clean except g1's
  `docs/CTO_LEDGER.md`, excluded per fleet note).
