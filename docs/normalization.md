# Image and Video Frame Normalization

This document describes how images and video frames are normalized before embedding in ScalarForensic. Understanding this pipeline is important for interpreting similarity scores, diagnosing unexpected mismatches, and configuring the system correctly for a given hardware budget and accuracy requirement.

---

## Pipeline Overview

All four ingestion paths share the same two-stage pipeline:

```
Input
  │
  ▼
[Stage 1: Shared Preprocessing]
  _open_rgb (images only)     EXIF orientation, RGB conversion (ICC profiles ignored)
  _cap_short_side             proportional downscale to effective cap
  │
  ▼
[Stage 2: Per-Model Normalization]
  DINOv2   AutoImageProcessor resize + center-crop to normalize_size × normalize_size
  SSCD     _sscd_resize → _sscd_crops → embed → optional multi-crop average
  Remote   passed through (server-side preprocessing assumed)
  │
  ▼
Embedding vector (unit-norm)
```

### The Four Cases

| | **Phase 1 (CLI / `sfn index`)** | **Phase 2 (Web upload / query)** |
|---|---|---|
| **Still images** | Bytes read from disk → `preprocess_batch` | Bytes from HTTP upload → `preprocess_batch` |
| **Video frames** | PyAV PIL frames → `preprocess_pil_batch` | PyAV PIL frames → `preprocess_pil_batch` |

The only structural difference: still images go through `_open_rgb` (bytes → PIL, with EXIF orientation correction), while video frames arrive from PyAV already decoded as RGB PIL images with correct geometry, so `_open_rgb` is skipped for them.

---

## Stage 1: Shared Preprocessing

### `_open_rgb` — Still images only

Two corrections are applied in sequence to every still image:

#### 1. EXIF Orientation Transpose

Phone cameras and some digital cameras store images with a rotation EXIF tag (`Orientation`) rather than physically rotating the pixel data. A portrait photo stored as landscape pixels with `Orientation=6` will be displayed correctly in any viewer, but without the transpose the raw pixel grid is 90° rotated from what the image depicts.

`ImageOps.exif_transpose()` applies the EXIF rotation/flip before the pixel data is fed to the model. Without this correction, two forensically identical photographs — one transposed in-camera and one rotated by software — would produce completely different embeddings and fail to match.

**Forensic implication**: phone evidence, body-camera stills, and any image whose orientation was set by the capture device rather than baked into the pixel data requires this correction for reliable matching.

#### 2. RGB Conversion

Strips alpha channels, palette modes, and any remaining colour metadata so all downstream code sees a plain `RGB` image.

> **Note on ICC colour profiles:** embedded ICC profiles (AdobeRGB, ProPhoto, etc.) are currently ignored — images are treated as sRGB regardless of any embedded profile. Wide-gamut sources produce slightly shifted pixel values relative to a colour-managed pipeline, but the semantic embedding models are robust to small colour shifts, and the per-image transform rebuild required for correct LCMS conversion was found to dominate Phase 1 preprocessing time in practice.

---

### `_cap_short_side(img, cap)` — Applied to all images and video frames

After `_open_rgb` (or directly for video frames), the image is proportionally downscaled if its short side exceeds `cap`. The aspect ratio is always preserved. No upscaling occurs here.

The effective cap is computed as `max(_SSCD_SCALE, normalize_size)`:

| `SFN_NORMALIZE_SIZE` | SSCD requirement | Effective cap |
|---|---|---|
| 224 (default) | 331 px | **331 px** |
| 448 | 331 px | **448 px** |
| 512 | 331 px | **512 px** |

This ensures DINOv2 receives images at its configured resolution while SSCD always gets at least 331 px on the short side.

---

## Stage 2: Per-Model Normalization

### DINOv2

DINOv2 was trained at 518×518 resolution, not the 224×224 commonly assumed from ImageNet conventions. The `AutoImageProcessor` is configured with `SFN_NORMALIZE_SIZE` (default 224) for both `size` and `crop_size`:

```
Input (short side ≤ effective cap)
  → resize to shortest_edge = normalize_size
  → center-crop to normalize_size × normalize_size
  → ImageNet mean/std normalisation
  → model input
```

#### Resolution vs. Compute Tradeoff

DINOv2 uses a Vision Transformer (ViT) architecture. Attention cost scales as O(n²) in the number of patches, and the patch count scales as (resolution / patch_size)²:

| `SFN_NORMALIZE_SIZE` | ViT-L/14 patch grid | Patch count | Relative attention cost |
|---|---|---|---|
| 224 | 16 × 16 | 256 | 1× (ImageNet default) |
| 336 | 24 × 24 | 576 | ~5× |
| 448 | 32 × 32 | 1024 | ~16× |
| 512 | ~36 × 36 | ~1296 | ~26× |
| 518 (training) | 37 × 37 | 1369 | ~28× |

The default `SFN_NORMALIZE_SIZE=224` matches the ImageNet convention and offers the best throughput. Raising it toward DINOv2's training resolution of 518 px can improve embedding fidelity for large, detail-rich images but increases inference cost quadratically — set `SFN_NORMALIZE_SIZE=512` if quality is the primary constraint and hardware permits.

---

### SSCD

```
Input (short side ≤ effective cap)
  → _sscd_resize: proportional resize to 331 px short side (upscales if needed)
  → _sscd_crops:  generate n_crops crops of 288 × 288
  → model(flat_batch)
  → if n_crops > 1: L2-normalise per crop → average → L2-normalise
  → unit-norm embedding vector
```

#### Crop Strategies (`SFN_SSCD_N_CROPS`)

**`SFN_SSCD_N_CROPS=1` (default)** — center crop only. The 288×288 center of the 331 px scaled image. This is the standard SSCD preprocessing and matches the model's training protocol for near-duplicate detection.

**`SFN_SSCD_N_CROPS=5`** — center crop + four corner crops. Each crop is 288×288; all five are embedded in a single forward pass and their L2-normalised embeddings are averaged and re-normalised. The resulting vector represents the full spatial extent of the image rather than just the center.

**When to use n_crops=5:**

Forensic evidence frequently contains images where the semantically relevant content is not centered:
- Surveillance stills with the subject near an edge
- Images that have been padded, letterboxed, or watermarked at the edges
- Composited images where content appears in a corner

The center crop of such an image will embed predominantly background or padding, while a corner crop may capture the relevant content. The five-crop ensemble ensures coverage of the full image regardless of subject position.

**Performance implications of n_crops=5:**

- GPU compute: ~5× the SSCD inference cost per batch
- Memory: the batch fed to the model is 5× larger
- Recommended: if using `SFN_SSCD_N_CROPS=5`, halve `SFN_BATCH_SIZE` to avoid OOM errors, or rely on the automatic batch-size calibration (which measures the actual forward-pass cost and will account for the expanded batch)
- The calibration system (`sfn calibrate`) must be run with the intended `SFN_SSCD_N_CROPS` value to produce an accurate batch-size recommendation

**Note for extreme aspect ratios**: an image with a very large aspect ratio (e.g., a 1920×200 banner) scaled to 331 px short side becomes 331×3170. Both the center crop and all four corner crops only see 288 px of the 3170 px long axis. A multi-crop strategy does not fully address this case; the image would need to be tiled along the long axis for complete coverage, which is outside the current scope.

---

### Remote Embedder

Images are PNG-encoded (lossless) before base64 transmission. This prevents DCT block artefacts on screenshots, rendered documents, and digital evidence that contains sharp edges or synthetic content, and eliminates double-JPEG compression if the source material was already JPEG. Preprocessing beyond the shared cap is assumed to be handled server-side.

---

## Breaking Changes

All five improvements introduced in this pipeline revision alter embedding values:

| Change | Affected images | Effect |
|---|---|---|
| EXIF orientation correction | Any image with non-trivial EXIF Orientation tag | Embeddings now match the display orientation |
| ICC profile conversion | Images with non-sRGB ICC profiles | Colour values now correspond to sRGB appearance |
| DINOv2 resolution wired in | All images (processor now respects SFN_NORMALIZE_SIZE; default 224 px) | Embeddings now match configured resolution |
| SSCD crop restructuring | No change for n_crops=1; new capability for n_crops=5 | Multi-crop embeddings are averaged |
| Remote PNG encoding | Images sent to remote endpoints | No re-compression artefacts |

**Any existing Qdrant collection indexed before this revision must be fully re-indexed** to maintain embedding consistency. Mixing old and new embeddings in the same collection will produce unreliable similarity scores because the vector space has changed.

Changing `SFN_SSCD_N_CROPS` between indexing runs also requires a full re-index of the SSCD collection, since center-crop-only embeddings are not comparable to five-crop ensemble embeddings.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `SFN_NORMALIZE_SIZE` | `224` | DINOv2 input resolution (px). Also sets the preprocessing cap floor above SSCD's 331 px requirement. Set to 512 for higher embedding fidelity at the cost of ~26× more ViT attention compute. |
| `SFN_SSCD_N_CROPS` | `1` | SSCD spatial crops per image. `1` = center only; `5` = center + 4 corners (recommended for forensic use if hardware permits). |
| `SFN_BATCH_SIZE` | auto | Override automatic batch-size calibration. Halve this value if using `SFN_SSCD_N_CROPS=5`. |
