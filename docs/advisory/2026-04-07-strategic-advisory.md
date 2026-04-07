# Strategic Advisory: Multi-Modal Media Identification System
**Date:** 2026-04-07
**Purpose:** Strategic advisory — tool evaluation, architecture recommendations, and pipeline design. Implementation decisions remain with the receiving technical team.

---

## 1. Executive Summary

### Mission Context
This system supports person identification and collection clustering. It ingests a new media file and answers four questions:
- Is this file already known?
- Is this person known?
- Is this scene/location known?
- Is this voice known?

### Scale Assumption
Baseline library pool: ~500,000 videos, ~2,000,000 images. All recommendations are validated against this scale.

### Precision-Over-Recall Mandate
The system is explicitly tuned to minimize false positives, even at the cost of missing degraded matches. A missed match is recoverable through other means. This principle governs every threshold and tool recommendation in this document.

### Open-Source Constraint
All recommended tools are freely available, locally hostable, and have no dependency on paid cloud APIs.

### Hardware Envelope
Recommendations are tiered across three hardware profiles:

| Tier | Hardware | VRAM |
|------|----------|------|
| Tier 1 | CPU-only workstation (16+ cores, 64GB RAM, NVMe) | — |
| Tier 2 | RTX 4060 Ti | 16GB |
| Tier 3 | RTX 6000 | 96GB |

No multi-GPU or cloud compute is assumed. Tier 3 is the maximum envelope.

### Operational Modes
The system operates in one of two explicitly selected modes per ingestion session:
- **Search & Forget** — new file is queried against the library pool, results are returned, file and its embeddings are discarded after the session. The library pool is unchanged.
- **Search & Enrich** — new file is queried, results are returned, and upon investigator approval, the file and its embeddings are added to the library pool.

### Human-in-the-Loop
All results — regardless of confidence level, including cryptographic hash matches — are surfaced. The system makes no autonomous decisions. Confidence tiers exist to prioritize investigator attention, not to trigger actions.

### Document Purpose
This is a strategic advisory document. It evaluates competing tools with evidence and recommends approaches per hardware tier.

---

## 2. Shared Infrastructure

All four pipelines share a common infrastructure layer. Components are deployed once and accessed by all pipelines via CLI or API calls. No pipeline bundles its own model loading.

### 2.1 Vector Database

The central store for all embeddings across all pipelines.

**Requirements at scale:**
- Handle hundreds of millions of vectors (faces, bodies, scenes, voices, file fingerprints)
- HNSW indexing for highest similarity search accuracy
- Quantization to reduce memory footprint while maintaining precision
- Single-node deployment (no distributed cluster required at this scale)
- Mature Python client

**Evaluated candidates:** Milvus, Qdrant, Weaviate

**Recommendation: Qdrant**

Qdrant is the clear choice for single-node deployment at 500M-vector scale. Its mmap-based hybrid strategy — quantized vectors resident in RAM, original vectors on disk (`on_disk=True`, `always_ram=True` for quantized index) — is the only architecture among the three that makes 400–500M vectors operationally viable on a single node with 64–96 GB RAM. This is a supported and documented production pattern, not an extrapolation.

**Quantization approach:** Qdrant's scalar quantization (float32 → int8) delivers 4× memory compression with 99%+ recall accuracy and SIMD-optimized query speed. This outperforms Product Quantization (PQ) for this use case: PQ compounds approximation errors across sub-vectors and is not SIMD-friendly, leading to speed penalties and higher recall loss — a poor tradeoff when high recall preservation is required. Qdrant's 1.5-bit, 2-bit, and asymmetric quantization options (added v1.15, 2025) offer additional flexibility if memory pressure increases beyond 500M vectors.

**Operational simplicity:** Qdrant ships as a single Rust binary with no external dependencies (no etcd, no MinIO, no message queues). A single Docker container is all that is required. This is substantially simpler than Milvus Standalone (three co-located containers, 500+ configuration parameters, etcd requiring NVMe SSDs with <10ms p99 fsync) and Weaviate (manageable for smaller deployments but resource-intensive at scale).

**Runner-up:** Milvus — viable if the deployment environment already has Docker Swarm infrastructure and the team can absorb etcd/MinIO operational complexity; Milvus offers richer index options (HNSW_PRQ) and higher indexing throughput at unconstrained hardware.

**Rejected:** Weaviate — the HNSW graph for 500M vectors at typical connection settings is estimated at 300+ GB and must remain fully in RAM, making single-node deployment at this scale extremely hardware-intensive. Teams report resource pressure above 100M vectors. Weaviate delivers 4× lower RPS than Qdrant at equivalent recall in published benchmarks (source: qdrant.tech/benchmarks — treat as vendor claim pending independent corroboration).

**Security note:** Before production deployment, access control, audit logging, and backup/restore capabilities must be assessed separately. Qdrant provides role-based access control, TLS, and API key authentication. 

**Sources:** qdrant.tech/articles/memory-consumption, qdrant.tech/articles/scalar-quantization, qdrant.tech/benchmarks, milvus.io/docs/benchmark.md

### 2.2 Model Serving Layer

A local inference API that exposes embedding generation as a callable endpoint. All pipelines call this layer — models are loaded once into VRAM and remain resident for the duration of a processing session.

**Evaluated candidates:**
- **NVIDIA Triton Inference Server** — production-grade, supports multiple frameworks, higher operational complexity
- **FastAPI + model loader** — simpler, sufficient for single-agency deployment, easier to maintain

**Tier guidance:**
- Tier 1: FastAPI wrapper around CPU-optimized (ONNX quantized) models
- Tier 2: FastAPI with GPU-accelerated models, VRAM budget managed explicitly (16GB constrains co-residency)
- Tier 3: Triton or FastAPI viable; multiple large models can co-reside in 96GB VRAM

### 2.3 Ingestion CLI / Orchestration

A command-line entry point that accepts a file path, auto-detects media type (image / video / audio-only), selects operational mode (Search & Forget / Search & Enrich), and dispatches to the cascade pipeline sequence.

This is the single integration point for connecting from existing file management systems. The cascade logic lives here.

In Search & Enrich mode, an explicit approval gate is required before any embedding is written to the vector DB.

### 2.4 FFmpeg

Used across all pipelines for media decoding, audio extraction, scene-change detection, and format normalization. Universally available, CPU-based, no licensing restrictions.

---

## 3. Pipeline 1: File Deduplication

The fastest and cheapest pipeline. A confirmed match here stops all further processing immediately (subject to review).

### 3.1 Stage 1 — Exact Match (Cryptographic Hashing)

**Tools:** MD5 / SHA-256

Detects byte-identical files. Advisory recommends augmenting with **block-level hashing** — splitting files into fixed-size chunks and hashing independently — to catch files that have been losslessly repackaged or container-swapped without content change.

Compute cost: negligible. Viable at all tiers.

### 3.2 Stage 2 — Near-Exact Match (Perceptual Hashing)

Detects re-encoded, compressed, or lightly cropped variants of known files.

**Evaluated candidates:**
- **pHash / dHash** — extremely fast, CPU-viable, brittle to moderate transformations
- **PDQ (Meta)** — open-source equivalent of PhotoDNA, robust to JPEG compression and minor crops, already used by NGO partners in this domain
- **SSCD (Self-Supervised Copy Detection, Meta)** — more robust than PDQ, requires GPU for practical throughput

**Recommendations:**
- Tier 1 & 2: **PDQ** — CPU-viable, battle-tested, interoperable
- Tier 3: **SSCD** — GPU throughput justifies the accuracy gain over PDQ

### 3.3 Stage 3 — Semantic Deduplication (Altered Matches)

Detects heavily cropped, color-shifted, or re-framed variants using dense visual embeddings compared against the vector DB.

**Evaluated candidates:** DINOv2 ViT-L, CLIP ViT-L/14, ImageBind

**Evaluation benchmark:** DISC21 (Image Similarity Challenge 2021) — the standard benchmark for transformation robustness in copy detection contexts. Primary metric: micro-average precision (µAP).

**Recommendation: SSCD ResNet50 (sscd_disc_mixup checkpoint) — all tiers**

SSCD is the only model in this evaluation explicitly designed and benchmarked for image copy detection under the DISC21 protocol:

| Model | DISC21 µAP | Copy detection design | License |
|-------|-----------|----------------------|---------|
| SSCD ResNet50 (sscd_disc_mixup) | **0.724** (official ISC21 evaluation) | Yes — contrastive training, entropy regularization, MixUp augmentation | MIT |
| DINOv2 ViT-L | ~0.64 (informal retrieval proxy, not official µAP) | No — general visual encoder | Apache 2.0 |
| CLIP ViT-L/14 | ~0.28 (informal retrieval proxy) | No — semantic cross-modal alignment | MIT (use OpenCLIP weights) |
| ImageBind | Not benchmarked on DISC21 | No — multi-modal joint embedding | CC-BY-NC 4.0 (disqualified) |

**Key advantages of SSCD:**
- **Precision calibration:** Cosine similarity threshold 0.75 = 90% precision on DISC21 pairs. This calibrated operating point aligns directly with the precision-over-recall mandate. CLIP and DINOv2 lack this.
- **Transformation robustness:** Training augmentations explicitly address JPEG compression (various quality factors), heavy cropping (MixUp/CutMix simulate partial copies), color shift (color jitter), and blur (Gaussian blur). The other models address these incidentally.
- **Hardware efficiency:** SSCD ResNet50 requires <1GB VRAM and achieves 50–150 img/s on CPU (16-core). ViT-L models achieve 3–8 img/s on CPU — inadequate for bulk workloads.
- **License:** MIT — no restrictions for deployment. ImageBind's CC-BY-NC 4.0 is disqualifying.

**ImageBind rejection:** In addition to the license disqualification, ImageBind is not benchmarked on any copy detection task, is optimized for cross-modal retrieval (not copy detection), and at ~1.2B parameters is the most VRAM-intensive candidate without any demonstrated benefit for this task.

**Per-tier configuration:**

| Tier | Model | Throughput (est.) | Notes |
|------|-------|------------------|-------|
| Tier 1 | SSCD ResNet50 (sscd_disc_mixup) | ~50–150 img/s | Only viable model at CPU speed; combine with PDQ as pre-filter for large corpora |
| Tier 2 | SSCD ResNet50 (sscd_disc_mixup) | ~800–2,000 img/s | Fits in <1GB VRAM; leaves full headroom for other pipeline models |
| Tier 3 | SSCD ResNet50 (primary) + DINOv2 ViT-L (optional secondary) | ~2,000–5,000+ img/s | DINOv2 (Apache 2.0) may be added as a secondary semantic layer for near-duplicate clustering where broader similarity is desired |

**Sources:** SSCD CVPR 2022 (github.com/facebookresearch/sscd-copy-detection), ISC 2021 official results (proceedings.mlr.press/v176/papakipos22a), DISC21 dataset (ai.meta.com/datasets/disc21-dataset)

### 3.4 Video Handling

FFmpeg scene-change detection extracts one representative frame per shot before any embedding is generated. This reduces per-video compute dramatically without sacrificing deduplication accuracy.

---

## 4. Pipeline 2: Person Re-ID

The most computationally expensive pipeline. Structured as a strict cascade — each stage only runs if the previous stage produces a positive signal.

### 4.1 Stage 1 — Motion Filtering

**Tools:** OpenCV background subtraction (MOG2 or KNN)

CPU-based pixel-difference analysis identifies video segments containing motion. Static segments are discarded before any GPU work begins. Critical at Tier 1 and Tier 2 where compute budget is constrained.

### 4.2 Stage 2 — Person Detection

Run at 2–3 FPS on motion segments only.

**Recommendations:**
- Tier 1: YOLOv8 (quantized ONNX variant, CPU-viable)
- Tier 2: YOLOv8 or YOLOv11
- Tier 3: YOLOv12 or RT-DETR

### 4.3 Stage 3 — Multi-Object Tracking

Links detections across frames into per-person tracklets.

**Evaluated candidates:** ByteTrack, StrongSORT, BoT-SORT

**Key evaluation criterion:** ID-switch rate — switches create false tracklets and pollute the database. Lower ID-switch rate is prioritized over raw tracking speed.

**Recommendation: StrongSORT++ (deployed via BoxMOT)**

Benchmark on MOT17 and MOT20 test sets, controlled comparison using YOLOX detection backbone (BoT-SORT paper, arXiv:2206.14651):

| Tracker | MOTA (MOT17) | IDF1 (MOT17) | ID Switches (MOT17) | ID Switches (MOT20) |
|---------|-------------|-------------|--------------------|--------------------|
| ByteTrack | 80.3 | 77.3 | 2,196 | 1,223 |
| StrongSORT++ | 79.6 | **79.5** | **1,194** | **770** |
| BoT-SORT | **80.6** | 79.5 | 1,257 | 1,212 |
| BoT-SORT-ReID | 80.5 | **80.2** | 1,212 | 1,257 |

StrongSORT++ achieves **45% fewer ID switches than ByteTrack on MOT17** (1,194 vs 2,196) and **37% fewer on MOT20** (770 vs 1,223). This is the decisive metric for this system — ID switches create false tracklet associations that pollute the database and generate spurious leads.

StrongSORT's advantages:
- **GSI (Gaussian-Smoothed Interpolation):** Fills missing detections in frame-drop or low-FPS video — the expected condition for archival surveillance footage.
- **AFLink post-processing:** Reduces fragmentation from occlusion recovery, improving long-tracklet identity consistency.
- **Re-ID embeddings:** Enables identity recovery after occlusion — critical when a person exits and re-enters frame.

**BoT-SORT rejection:** While BoT-SORT-ReID ranks first overall on MOTChallenge, its camera-motion-compensation (CMC) module accounts for 97.1% of per-frame compute time — prohibitive for CPU-only Tier 1 deployment. ID switch performance does not exceed StrongSORT++.

**Tier 1 (CPU) guidance:** StrongSORT with a lightweight OSNet-x0.25 Re-ID embedding model at detection confidence ≥0.5 is viable for non-real-time archival processing. If CPU throughput is insufficient, deploy ByteTrack for initial processing and apply Re-ID matching as a post-processing step on the resulting tracklet database.

**Integration framework:** Deploy via **BoxMOT** (github.com/mikel-brostrom/boxmot) — the actively maintained unified CLI and Python API for StrongSORT, ByteTrack, BoT-SORT, and others. Fast-ReID (unmaintained since May 2021) and Torchreid (inactive since August 2022) are not recommended as standalone frameworks.

**Sources:** BoT-SORT paper (ar5iv.labs.arxiv.org/html/2206.14651), StrongSORT paper (arxiv.org/pdf/2202.13514), BoxMOT (github.com/mikel-brostrom/boxmot)

### 4.4 Stage 4 — Body / Appearance Re-ID

Extracts appearance features from tracklets (clothing, body shape).

**Integration framework: BoxMOT with OSNet embedding backbone**

Fast-ReID (last release May 2021, 4+ years unmaintained) and Torchreid (inactive since 2022) are not recommended. BoxMOT provides the same tracker ensemble with maintained dependencies and bundles compatible Re-ID embedding models.

**Re-ID embedding model per tier:**
- Tier 1 (CPU): **OSNet-x0.25** — extremely lightweight, benchmarked at real-time speeds on modest CPU hardware
- Tier 2/3 (GPU): **OSNet-x1.0** — full model, higher accuracy, fits comfortably in GPU VRAM

If existing Fast-ReID pre-trained weights (ResNet-based) are already in use, they can be converted to ONNX and used as an embedding backbone within BoxMOT's pluggable Re-ID interface without retraining.

**Clothing-independent Re-ID: Not yet mature for primary operational deployment**

Cloth-changing person Re-ID (CC-ReID) is an active research area but not operationally ready as of April 2026:

| Model | PRCC Rank-1 | LTCC Rank-1 | Pre-trained weights available? |
|-------|------------|------------|-------------------------------|
| DM-ReID (2025) | 64.6% | N/A | No |
| TryHarder (ACM MM 2025) | SOTA +8.9% over previous | SOTA | No |
| ACD-Net | ~58% | ~40% | Training code only |

No CC-ReID model provides a ready-to-use inference pipeline with downloadable pre-trained weights. Even the best PRCC Rank-1 (~65%) implies a 35% miss rate under clothing change — operationally significant for a precision-first system. LTCC mAP (~38%) is insufficient as a primary identity signal.

**Recommended mitigation strategy:**
1. Deploy standard appearance-based Re-ID (BoxMOT + OSNet) as the primary matcher.
2. Augment with body-shape/silhouette signals where video quality permits (height-weight ratio from camera calibration, gait if multi-frame sequences exist).
3. Flag tracklets with large appearance-distance gaps as "potential clothing change" events for human analyst review rather than automated discard.
4. Re-evaluate when TryHarder or an equivalent 2025 CC-ReID model releases downloadable weights.

### 4.5 Stage 5 — Facial Biometrics (Conditional)

**Trigger condition:** Face bounding box width exceeds minimum resolution threshold (60px). Below this threshold, facial matching is not attempted.

**Evaluated candidates:**
- **DeepFace** — wrapper library supporting ArcFace, Facenet512, AdaFace weights
- **InsightFace** — direct ArcFace implementation, strong benchmark performance, widely used in research

**Thresholds (precision-over-recall):**
- Cosine similarity threshold: **> 0.92** before registering a hit
- Temporal voting: match must be confirmed across **≥ 5 distinct frames** within a tracklet

### 4.6 Stage 6 — VLM Edge-Case Arbitration (Tier 3 Only)

When similarity score falls in the grey zone (0.85–0.92), both candidate images are passed to a locally-hosted Vision-Language Model with a binary confirmation prompt.

**Evaluated candidates:** Qwen2-VL, LLaVA-1.6, InternVL2

**Tier restriction:** Only viable at Tier 3 given VRAM requirements of large VLMs.

---

## 5. Pipeline 3: Scene Matching

Identifies the static environment — room, location, setting — independently of the people or objects present. Goal: cluster material filmed in the same location across different sessions.

### 5.1 Core Challenge

Standard image embeddings (CLIP, DINOv2) encode people and background together. Scene matching requires isolating static background structure. This advisory evaluated two approaches: semantic segmentation to mask dynamic objects before embedding, and Visual Place Recognition (VPR) models designed to ignore dynamic content through architectural design or training regime.

### 5.2 Stage 1 — Semantic Segmentation (Preprocessing)

**Evaluated candidates:** SegFormer (lightweight, CPU-viable), Mask2Former (higher accuracy, GPU-required)

**Recommendation: Skip — not recommended as default preprocessing**

The research literature does not provide benchmark evidence that applying SegFormer or Mask2Former masking of dynamic objects (people, furniture) before indoor VPR retrieval improves recall on standard benchmarks (InLoc, 7Scenes, Baidu Mall). Findings:

- Semantic segmentation and dynamic object removal demonstrably improve visual odometry and pose estimation in dynamic environments, but this improvement does not directly transfer to VPR retrieval recall in the indoor scenario.
- Hierarchical VPR approaches that show accuracy gains from semantic cues integrate semantics architecturally at training time — not as a preprocessing mask applied at query time.
- Indoor VPR has fewer dynamic object problems than outdoor VPR (no cars, buses occupying large portions of frame); the proportion of moving objects relative to static architectural structure is lower.
- Added overhead: ~1–2GB VRAM, additional inference latency, significant engineering complexity.

**Revisit condition:** If evaluation data shows false-match patterns attributable to persons visible in query frames (e.g., same room, different person in foreground), add SegFormer at Tier 2/3 as a targeted mitigation. The recommended VPR model (MegaLoc) and fallback (AnyLoc) use patch-level DINOv2 feature aggregation that provides partial occlusion robustness by design.

If segmentation is used at Tier 3, SAM (Segment Anything Model) provides the highest accuracy.

### 5.3 Stage 2 — Visual Place Recognition Embedding

**Evaluated candidates:** NetVLAD, AnyLoc, EigenPlaces, CosPlace

**Critical evaluation context:** Most VPR models are trained on outdoor/urban environments. Their transfer to indoor environments — the most operationally relevant scenario for this system — was explicitly evaluated.

**Indoor benchmark results:**

| Model | Training data | Indoor R@1 (relative) | Transfer quality |
|-------|--------------|----------------------|-----------------|
| NetVLAD | Pitts-250k, MSLS (outdoor urban only) | Lowest | Poor — requires Patch-NetVLAD+ extension for indoor use |
| CosPlace | Outdoor urban (classification-based) | ~20% lower R@1 than AnyLoc | Moderate |
| EigenPlaces | Outdoor urban (viewpoint-robust) | High — close to ViT-based methods | Good |
| AnyLoc-VLAD-DINOv2 | None (zero-shot DINOv2 features) | +5% over MixVPR, +20% over CosPlace R@1 | Excellent — no domain shift by design |

**2024/2025 models evaluated:**
- **SALAD** (CVPR 2024): DINOv2-base + Optimal Transport aggregation; outperforms AnyLoc on several benchmarks
- **MegaLoc** (CVPR 2025 Workshop): DINOv2-base + SALAD aggregation, trained on ScanNet (indoor RGB-D) + outdoor datasets; **87.7% Recall@1 on Baidu Mall (indoor)** vs 75.6% for next best — the strongest indoor VPR result found in this evaluation

**Recommendation: MegaLoc at Tier 2/3; EigenPlaces at Tier 1**

| Tier | Hardware | Recommended VPR model | VRAM | Rationale |
|------|----------|-----------------------|------|-----------|
| Tier 1 | CPU | EigenPlaces (ResNet-50, indoor variant) | ~2–4 GB | Only quality option viable on CPU; ResNet-50 runs at acceptable throughput |
| Tier 2 | RTX 4060 Ti 16 GB | MegaLoc (DINOv2-base + SALAD) | ~4–5 GB | Best indoor R@1 (87.7%); ScanNet training provides genuine indoor coverage; fits easily in 16GB budget |
| Tier 3 | RTX 6000 Ada 96 GB | MegaLoc (DINOv2-base + SALAD) | ~4–5 GB | Same model as Tier 2; remaining VRAM used by other resident pipeline models |

**Fallback (ScanNet overlap concern):** If deployment locations include environments closely resembling ScanNet's RGB-D scans (academic/research spaces), use AnyLoc-VLAD-DINOv2 — zero training data, no domain overlap risk, still +20% R@1 over CosPlace indoors.

**Rejected:** NetVLAD (severe outdoor-to-indoor transfer gap; requires dedicated indoor fine-tuning), CosPlace (worst indoor performer of the four original candidates).

**Sources:** AnyLoc (IEEE RA-L 2023, arxiv.org/abs/2308.00688), MegaLoc (CVPR 2025W, github.com/gmberton/MegaLoc), EigenPlaces (ICCV 2023, github.com/gmberton/EigenPlaces), VPR comparative study (March 2026, arxiv.org/html/2603.13917)

### 5.4 Similarity Thresholds and Confidence Treatment

Scene matching is inherently more ambiguous than facial matching. Generic environments (plain walls, standard furniture, corridor-type spaces) produce false clusters regardless of model quality.

**Recommended approach:**
- Cosine similarity threshold: **≥ 0.85** — stricter than the spec's 0.80 baseline; warranted by the precision-over-recall mandate and the prevalence of visually repetitive indoor architectural elements (identical corridors, standardized hotel rooms)
- Clusters: Clusters form series of same location / involved persons. Combining files into a cluster should be a strict process to avoid false positives and have either strong similarity or multiple overlaps (for example two different categories like voice similarity + location)
- Scene matches are treated as **leads only**, not confirmed hits, regardless of similarity score
- Threshold calibration against own test set is mandatory before operational deployment (see Section 8.3)

---

## 6. Pipeline 4: Audio / Voice Profiling

Extracts and indexes speaker identity and audio fingerprints. Complements visual pipelines — a person may be heard but not clearly seen, or audio may survive transformations that destroy visual quality.

### 6.1 Stage 1 — Audio Extraction & Preprocessing

**Tool:** FFmpeg (audio track extraction, format normalization)

Noise reduction applied before further processing.

**Recommendations:**
- Tier 1: **RNNoise** — lightweight, CPU-viable
- Tier 2 & 3: **DeepFilterNet** — stronger noise reduction, GPU-accelerated, meaningful improvement on low-quality recordings

### 6.2 Stage 2 — Audio Fingerprinting (File-Level)

Detects exact or near-exact audio matches — clips ripped from longer recordings, re-encoded audio.

**Evaluated candidates:** Chromaprint/AcoustID, audfprint, dejavu

**Recommendation: audfprint**

Chromaprint is built on chroma features — 12-bin pitch-class representations mapping spectral energy to musical semitones. This design is fundamentally unsuitable for speech-dominant recordings:

- Speech does not concentrate energy in stable harmonic pitch-class patterns
- Unvoiced consonants (fricatives, stops) produce no chroma signal
- Speech identity relies on formant transitions and prosodic patterns, not melodic/harmonic content
- Chromaprint's own documentation explicitly targets music identification

| Tool | Speech match accuracy | Noise robustness | Re-encoding robustness | CPU viable |
|------|----------------------|------------------|----------------------|------------|
| Chromaprint | Poor — wrong feature domain | N/A for speech | Good (music only) | Yes |
| audfprint | Moderate — spectral peaks are domain-agnostic | Degrades above moderate noise | Moderate (≥128kbps) | Yes |
| dejavu | Moderate — similar to audfprint | Similar to audfprint | Moderate | Yes |

**Winner: audfprint** — Shazam-style spectral peak-pair hashing on spectrogram local maxima is domain-agnostic, capturing energy concentrations in speech audio. Compact index (19MB per 2,000 references / 74 hours of audio). Partial clip matching supported via time-offset voting.

**Operational caveat:** All three tools are classical fingerprinting algorithms. At high distortion or heavy re-encoding (<64kbps), all degrade. For heavily processed recordings, supplement audfprint exact-match results with the speaker embedding similarity pipeline (Section 6.5) for near-duplicate detection.

**dejavu** is a viable fallback if audfprint integration is problematic — functionally equivalent, but less actively maintained.

**Sources:** audfprint (github.com/dpwe/audfprint), BAF benchmark evaluation (ISMIR 2022), Chromaprint documentation (acoustid.org/chromaprint)

### 6.3 Stage 3 — Voice Activity Detection

**Recommendations:**
- Tier 1: **Silero VAD** — lightweight, CPU-viable, strong accuracy
- Tier 2 & 3: **pyannote.audio VAD** — higher accuracy, integrates directly with the diarization stack

### 6.4 Stage 4 — Speaker Diarization

**Tool:** pyannote.audio (SpeakerDiarization 3.1 pipeline) — current state of the art in open-source diarization. Critical for files containing multiple speakers — correct "who spoke when" separation is prerequisite to useful speaker embedding extraction.

### 6.5 Stage 5 — Speaker Embedding / Voice Re-ID

Extracts a speaker identity vector for each diarized segment, indexed in the vector DB.

**Evaluated candidates:** SpeechBrain ECAPA-TDNN, WeSpeaker ResNet34, TitaNet (NVIDIA NeMo)

**Recommendation: ECAPA-TDNN (SpeechBrain) — primary at all tiers**

| Model | EER (VoxCeleb1-O-clean) | VRAM | CPU viable (Tier 1) | Weights |
|-------|------------------------|------|---------------------|---------|
| ECAPA-TDNN C=1024 (SpeechBrain) | 0.728% | ~0.5 GB | **Yes — 70–300ms/utterance** | speechbrain/spkrec-ecapa-voxceleb (HuggingFace) |
| WeSpeaker ResNet34 | 0.723% | ~1–2 GB | Possible but slower | wenet-e2e/wespeaker (GitHub) |
| TitaNet Large (NVIDIA NeMo) | ~1.71–1.91%* | <4 GB | Possible | nvidia/speakerverification_en_titanet_large |

*TitaNet EER measured on a curated comparative dataset — not directly comparable to VoxCeleb1-O-clean figures.

**ECAPA-TDNN advantages:**
- Forensically validated: explicitly evaluated for forensic automatic speaker recognition in a 2024 peer-reviewed study (ScienceDirect)
- CPU-viable: TDNN architecture achieves 70–300ms per utterance on CPU, enabling Tier 1 deployment without GPU
- Quantizable: 4×–8× model size reduction via weight quantization with <0.16% EER increase
- Minimal dependencies: SpeechBrain with HuggingFace weights; no proprietary framework required
- Accuracy equivalent to WeSpeaker ResNet34 (0.728% vs 0.723% EER — within measurement noise)

**TitaNet rejection:** Requires full NVIDIA NeMo framework (heavyweight dependency); higher EER than ECAPA or ResNet34 on standard benchmarks; not recommended for standalone deployment.

**Per-tier configuration:**

| Tier | Hardware | Model | Configuration |
|------|----------|-------|---------------|
| Tier 1 | CPU | ECAPA-TDNN (SpeechBrain) | C=512 variant; optional quantized weights for lower footprint |
| Tier 2 | RTX 4060 Ti 16 GB | ECAPA-TDNN (SpeechBrain) | C=1024 full model; GPU inference <10ms/utterance |
| Tier 3 | RTX 6000 Ada 96 GB | WeSpeaker ResNet34 or ECAPA-TDNN C=1024 | Either viable; WeSpeaker adds AS-Norm score calibration for higher-confidence speaker decisions |

**Threshold note:** Voice embeddings are more susceptible to degradation than face embeddings (noise, channel variation, short utterance length). Similarity thresholds must be calibrated separately from facial biometrics. Voice matches carry lower standalone evidentiary weight; corroboration from visual pipelines is strongly recommended before a lead is elevated.

**Sources:** speechbrain/spkrec-ecapa-voxceleb (HuggingFace), WeSpeaker baselines for VoxSRC2023 (arxiv.org/pdf/2306.15161), ECAPA-TDNN forensic validation (ScienceDirect 2024, doi.org/10.1016/j.specom.2024.103078), MDPI 2024 speaker model comparison

### 6.6 Stage 6 — Transcription / Keyword Detection

**Tool per tier:**

| Tier | Whisper variant | VRAM | WER (LibriSpeech test-other) | Notes |
|------|----------------|------|------------------------------|-------|
| Tier 1 | whisper-small (Faster-Whisper/CTranslate2) | ~2 GB | ~8.7% | Faster-Whisper is 12.5× faster than reference implementation on CPU |
| Tier 2 | **whisper-large-v3-turbo** | ~6 GB | ~6.9% | 8× faster than large-v3; near-identical WER; fits within 16GB VRAM budget |
| Tier 3 | whisper-large-v3 | ~10 GB | ~6.7% | Best available WER; 10GB is negligible at 96GB capacity |

**Tier 2 rationale:** whisper-large-v3-turbo (distilled from large-v3, 809M parameters) fits within the Tier 2 VRAM budget (alongside MegaLoc ~5GB, ECAPA-TDNN ~0.5GB, leaving ~4–6GB allocation for Whisper). At 8× the throughput of large-v3 with near-identical WER, turbo is the optimal Tier 2 configuration. If turbo weights are unavailable in a specific deployment environment, whisper-medium (5GB, ~7.5% test-other WER) is the fallback.

**Framing:** Transcription is an aid only — not a matching signal. Transcription errors on noisy/overlapping speech are common and would generate false leads if treated as evidence. Keyword hits are surfaced as "notes" (lowest confidence tier), not as leads.

**Sources:** Whisper model sizes (openwhispr.com/blog/whisper-model-sizes-explained), Whisper VRAM (huggingface.co/openai/whisper-large-v3/discussions/83), Tom's Hardware GPU benchmark, Faster-Whisper (github.com/SYSTRAN/faster-whisper)

---

## 7. Integration & Cascade Logic

### 7.1 Ingestion Entry Point

The CLI/API accepts a file path. Media type is auto-detected (image / video / audio-only). Operational mode is explicitly selected (Search & Forget / Search & Enrich). All pipeline calls route through the shared model serving layer and vector DB.

### 7.2 Cascade Sequence

Ordered strictly by compute cost, cheapest first. A confirmed match at any stage surfaces to the investigator and stops further processing pending their decision. Tool names reflect research-validated recommendations from this document.

```
New file ingested → mode selected (Search & Forget / Search & Enrich)
│
├─ 1. Cryptographic hash (MD5/SHA-256 + block-level hashing)
│      → match found? → surface for decision, halt cascade
│
├─ 2. Perceptual hash
│      Tier 1/2: PDQ | Tier 3: SSCD
│      → match found? → surface for decision, halt cascade
│
├─ 3. Semantic embedding — file-level copy detection
│      All tiers: SSCD ResNet50 (sscd_disc_mixup), threshold 0.75 → 90% precision
│      → altered match? → log as lead, continue
│
├─ 4. Audio fingerprint (if audio present)
│      All tiers: audfprint (spectral peak hashing)
│      → match found? → log as lead, continue
│
├─ 5. Person detection (2–3 FPS on motion segments)
│      Tier 1: YOLOv8 ONNX | Tier 2: YOLOv8/YOLOv11 | Tier 3: YOLOv12/RT-DETR
│      → persons found?
│      ├─ YES →
│      │    Tracking: StrongSORT++ via BoxMOT (OSNet-x0.25 Tier 1; OSNet-x1.0 Tier 2/3)
│      │    Body Re-ID: BoxMOT + OSNet appearance features → log tracklet hits as leads
│      │    Face matching (if face ≥60px width):
│      │       DeepFace (ArcFace) or InsightFace
│      │       Threshold: cosine > 0.92, temporal vote ≥ 5 frames
│      │       → face match? → log as strong lead
│      │    Tier 3 only: VLM arbitration (Qwen2-VL / LLaVA-1.6 / InternVL2)
│      │       for grey-zone scores 0.85–0.92
│      └─ NO → skip stages 5
│
├─ 6. Scene matching
│      Tier 1: EigenPlaces (ResNet-50, indoor variant)
│      Tier 2/3: MegaLoc (DINOv2-base + SALAD), threshold ≥ 0.85
│      Segmentation preprocessing: not applied by default (no indoor VPR evidence)
│      → location match (cluster ≥ 3 files)? → log as lead
│
└─ 7. Voice profiling (if speech detected via VAD)
       Noise reduction: RNNoise (Tier 1) | DeepFilterNet (Tier 2/3)
       VAD: Silero VAD (Tier 1) | pyannote.audio VAD (Tier 2/3)
       Diarization: pyannote.audio SpeakerDiarization 3.1
       Speaker embedding: ECAPA-TDNN C=512 (Tier 1) | ECAPA-TDNN C=1024 (Tier 2) |
                          WeSpeaker ResNet34 or ECAPA-TDNN C=1024 (Tier 3)
       → speaker match? → log as lead
       Transcription (all tiers):
          Tier 1: whisper-small (Faster-Whisper) | Tier 2: whisper-large-v3-turbo |
          Tier 3: whisper-large-v3
          → keyword detection → log as investigative note
```

### 7.3 Result Aggregation & Confidence Tiers

All results are surfaced to an investigator. The system takes no autonomous action at any confidence level.

| Tier | Basis | Investigator action required |
|------|-------|------------------------------|
| Confirmed hit | Cryptographic or perceptual hash match | Review and verify |
| Strong lead | Similarity above threshold (face cosine >0.92 + ≥5 frames; SSCD >0.75; speaker embedding above calibrated threshold) | Review and assess |
| Weak lead | Similarity in grey zone; scene cluster <3 files; voice match without visual corroboration | Review and assess |
| Investigative note | Transcription keywords; weak scene signal; single-file scene match | Contextual review |

### 7.4 Operational Modes & Database Growth

**Search & Forget:** The library pool is unchanged after ingestion. File embeddings are discarded after the session. No approval required for database state.

**Search & Enrich:** After results are reviewed, explicitly approved or rejected addition of the new file to the library pool. Only on approval are embeddings written to the vector DB. The ingestion CLI enforces this approval gate — it cannot be bypassed programmatically.

**Embedding store maintenance:** As the library pool grows, periodic deduplication of the vector DB is recommended to prevent bloat. Qdrant's payload filtering can be used to identify and remove superseded embeddings (e.g., when a higher-quality version of a file replaces an earlier lower-quality version).

---

## 8. Benchmarking & Validation Recommendations

### 8.1 Synthetic Test Set Construction

Before operational deployment, a controlled test set from known library material with verified ground truth labels is recommended. The test set must include:

- Exact duplicates (cryptographic match expected)
- Re-encoded / compressed variants (perceptual hash expected)
- Heavily cropped or color-shifted variants (SSCD semantic embedding expected, cosine >0.75)
- Multi-person videos with known identities (Re-ID expected)
- Recordings from the same location across different sessions (scene matching expected)
- Audio recordings with known speaker identities (voice profiling expected)

**The test set must never enter the operational library pool.**

### 8.2 Metrics Per Pipeline

Each pipeline is evaluated independently:

| Metric | Description | Priority |
|--------|-------------|----------|
| Precision | Of all matches flagged, how many are correct | Primary |
| Recall | Of all true matches, how many were found | Secondary |
| False Positive Rate | Incorrect matches surfaced to investigators | Explicit tracking required |
| Throughput | Files processed per hour per hardware tier | Operational planning |

### 8.3 Threshold Calibration

Similarity thresholds recommended throughout this document are starting points derived from benchmark literature:
- SSCD: 0.75 (→ 90% precision on DISC21)
- Face matching: cosine >0.92 (precision-first, conservative)
- Scene matching: cosine ≥0.85 (conservative for repetitive indoor environments)
- Speaker embedding: to be calibrated per deployment (voice quality, recording conditions vary)

Each agency must calibrate thresholds against their own test set and investigator tolerance for false positives before operational deployment. No threshold in this document should be treated as final without local validation.

### 8.4 Regression Testing

When updating any model or infrastructure component, the full test set must be re-run. Benchmarking gates every system update, not just initial deployment.

### 8.5 Hardware Tier Validation

Benchmarking must be run independently on each hardware tier the agency intends to deploy. Throughput figures from Tier 3 do not extrapolate to Tier 1 or Tier 2.

---

## Appendix: Tool Summary by Pipeline

| Pipeline | Stage | Tier 1 | Tier 2 | Tier 3 |
|----------|-------|--------|--------|--------|
| File Dedup | Crypto hash | MD5/SHA-256 + block | MD5/SHA-256 + block | MD5/SHA-256 + block |
| File Dedup | Perceptual hash | PDQ | PDQ | SSCD |
| File Dedup | Semantic embedding | SSCD ResNet50 | SSCD ResNet50 | SSCD ResNet50 (+ DINOv2 optional) |
| Person Re-ID | Motion filter | OpenCV MOG2/KNN | OpenCV MOG2/KNN | OpenCV MOG2/KNN |
| Person Re-ID | Detection | YOLOv8 ONNX | YOLOv8/YOLOv11 | YOLOv12/RT-DETR |
| Person Re-ID | Tracking | StrongSORT++ (BoxMOT) | StrongSORT++ (BoxMOT) | StrongSORT++ (BoxMOT) |
| Person Re-ID | Body Re-ID | OSNet-x0.25 | OSNet-x1.0 | OSNet-x1.0 |
| Person Re-ID | Face | DeepFace/InsightFace | DeepFace/InsightFace | DeepFace/InsightFace + VLM arbitration |
| Scene Matching | Segmentation | Skip | Skip | Skip (revisit if needed) |
| Scene Matching | VPR | EigenPlaces (ResNet-50) | MegaLoc | MegaLoc |
| Audio/Voice | Noise reduction | RNNoise | DeepFilterNet | DeepFilterNet |
| Audio/Voice | Audio fingerprint | audfprint | audfprint | audfprint |
| Audio/Voice | VAD | Silero VAD | pyannote.audio VAD | pyannote.audio VAD |
| Audio/Voice | Diarization | pyannote.audio 3.1 | pyannote.audio 3.1 | pyannote.audio 3.1 |
| Audio/Voice | Speaker embedding | ECAPA-TDNN C=512 | ECAPA-TDNN C=1024 | WeSpeaker ResNet34 |
| Audio/Voice | Transcription | whisper-small (FW) | whisper-large-v3-turbo | whisper-large-v3 |
| Shared Infra | Vector DB | Qdrant | Qdrant | Qdrant |
| Shared Infra | Model serving | FastAPI + ONNX | FastAPI + GPU | Triton or FastAPI |
