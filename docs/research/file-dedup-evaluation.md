# File Deduplication — Semantic Embedding Evaluation

**Purpose:** Identify the best visual embedding model for detecting altered image copies (JPEG-compressed, cropped, color-shifted, re-encoded) in a media identification pipeline. The system prioritizes precision over recall.

**Research date:** 2026-04-07

---

## Evaluation Criteria

1. DISC21 micro-AP (primary benchmark metric)
2. Transformation robustness (JPEG compression, heavy crop >50%, color shift, blur)
3. VRAM requirements per hardware tier
4. Inference throughput (images/second) per tier
5. License restrictions on pre-trained weights

---

## DISC21 Benchmark Results

The DISC21 (2021 Image Similarity Challenge) dataset contains 1 million reference images and 50,000 query images produced via sophisticated transformations. The primary metric is micro-average precision (µAP), equivalent to the area under the precision-recall curve across all submitted image pairs.

> **Caveat on DINOv2 and CLIP scores below:** The 64% and 28.45% figures cited for DINOv2 and CLIP originate from a community benchmark article (Medium/AImonks) using a retrieval accuracy proxy on a DISC21 subset — not the official ISC2021 µAP evaluation pipeline. They are directionally informative but should **not** be treated as official µAP scores comparable to SSCD's reported 0.724. No peer-reviewed paper reports DINOv2 ViT-L or CLIP ViT-L/14 µAP under the official DISC21 evaluation protocol.

| Model | micro-AP | Notes |
|-------|----------|-------|
| DINOv2 ViT-L | ~0.64 (retrieval accuracy proxy, not official µAP) | Community benchmark on DISC21 subset; substantially higher than CLIP on same test; no official ISC21 evaluation. DINOv2 internal curation pipeline uses SSCD-style copy detection, not DINOv2 itself, for deduplication. |
| CLIP ViT-L/14 | ~0.28 (retrieval accuracy proxy, not official µAP) | Same community benchmark; significantly underperforms DINOv2 on this task; not benchmarked under official ISC21 µAP protocol. CLIP is designed for semantic cross-modal alignment, not pixel-level copy detection. |
| ImageBind | not benchmarked on DISC21 | No published results on the DISC21 copy detection benchmark. ImageBind targets multi-modal joint embedding (images, audio, depth, IMU, thermal, text), not copy detection. Its CVPR 2023 paper does not report µAP on DISC21. |
| SSCD ResNet50+mixup (reference) | 0.568 µAP raw; 0.724 µAP with score normalization | Official evaluation from CVPR 2022 paper and GitHub Evaluation.md. Best single-model result from the released sscd_disc_mixup checkpoint. ISC21 Descriptor Track winner (TitanShield2 ensemble) achieved 0.7418 µAP. SSCD adv.+mixup variant: 0.615 µAP raw / 0.725 µAP with score normalization. |

### Recent 2024–2025 results for context

- **RDCD (WACV 2025):** Relational Self-supervised Distillation with Compact Descriptors. Uses knowledge distillation from a teacher (ResNet-50) to a compact student (EfficientNet-B0). Achieves µAP ~53.5 at 64-dim descriptors, comparable to DINO ViT-B/16 at 1536-dim. Lighter but does not surpass SSCD on DISC21.
- **Dynamic Augmentation + ViT (MDPI Electronics 2024):** A ViT-based approach trained on only 3% of DISC21 data (K-means strategic sampling + complex augmentation) achieves µAP 0.79 on a refined 10K query subset — **caution:** this is measured on a refined/filtered query subset, not the full DISC21 query set, so direct comparison to SSCD's 0.724 requires care.
- **AnyPattern (arXiv 2404.13788, 2024):** In-context ICD approach. Uses pattern prompts (ImageStacker). Claims +26.66% µAP from large-scale pattern data + additional +16.75% µAP from ImageStacker. Applied to DISC21-derived evaluation. Strong generalization to unseen tamper patterns.
- **Image Copy Detection for Diffusion Models (arXiv 2409.19952, 2024):** Extends copy detection to AI-generated images; evaluated on DISC21 variants.

**Summary on newer models:** Several 2024–2025 approaches improve over SSCD baseline in specific settings (refined query subsets, specialized augmentation), but no single openly available off-the-shelf model has been conclusively demonstrated to outperform SSCD universally on the full DISC21 protocol using the standard ISC21 evaluation suite. SSCD remains the de facto reference for operational deployment.

---

## Transformation Robustness

### DINOv2 ViT-L

- **Design intent:** Self-supervised ViT trained with DINO objective + DINOv2 refinements on 142M curated images. Uses multi-crop training with heavy geometric and photometric augmentation (multi-crop, color jitter, Gaussian blur, solarization). Trained explicitly to produce augmentation-invariant representations.
- **JPEG compression:** Robust due to photometric augmentation in training; representations are invariant to minor to moderate compression artifacts. However, not specifically optimized for high-compression (QF < 30) scenarios.
- **Heavy cropping (>50%):** Multi-crop training strategy directly targets crop invariance. DINOv2 shows strong spatial localization, making it more robust to aggressive crops than contrastive image-text models.
- **Color shift:** Training augmentations include color jitter (brightness, contrast, saturation, hue) and solarization. Color-shifted copies should produce similar embeddings.
- **Blur:** Gaussian blur is used as a training augmentation, providing explicit blur robustness.
- **Copy detection specific limitation:** DINOv2 is a general-purpose visual encoder, not fine-tuned for copy detection. Without a copy-detection-specific training objective (entropy regularization, mixup augmentation like SSCD), it will conflate semantically similar but non-copied images, causing precision loss. The 64% retrieval proxy figure suggests meaningful but imperfect copy detection capability out of the box.

### CLIP ViT-L/14

- **Design intent:** Contrastive Language-Image Pretraining. Optimized to align images with natural language descriptions, not to detect pixel-level or structural copies.
- **JPEG compression:** No explicit JPEG robustness in training. CLIP representations encode semantic content and high-level scene structure; JPEG artifacts may not degrade semantic embeddings greatly, but precision for fine-grained copy detection will be low.
- **Heavy cropping (>50%):** Poor. CLIP's global CLS token embedding is designed for whole-image scene understanding. Heavy crops that remove the primary subject cause large embedding drift. FusionBench robustness experiments confirm CLIP-ViT models degrade on spatial corruptions.
- **Color shift:** Moderate robustness; CLIP's semantic representations are somewhat color-invariant at the scene level, but color-shifted near-duplicate copies may not be reliably detected.
- **Blur:** Similar to JPEG — semantic content is preserved, but copy-specific signal is diluted.
- **Copy detection specific limitation:** CLIP at 28.45% retrieval proxy on DISC21 (vs DINOv2's 64%) indicates it is substantially weaker for this task. CLIP is a semantic similarity model, not a copy detection model. It is likely to produce high false positive rates when used for copy detection (semantically similar non-copies appear as matches), directly conflicting with the precision-first requirement.

### ImageBind

- **Design intent:** Joint embedding across six modalities (image, text, audio, depth, thermal, IMU). The image encoder is a ViT-H/14 (largest variant), initialized from CLIP and fine-tuned for cross-modal alignment.
- **JPEG compression:** Unknown; not evaluated on copy detection tasks. ImageBind inherits CLIP's training regime for the image modality; similar JPEG robustness limitations apply.
- **Heavy cropping (>50%):** Unknown; not evaluated. Likely similar to CLIP given shared image encoder heritage.
- **Color shift:** Unknown; not evaluated on copy detection tasks.
- **Blur:** Unknown; not evaluated.
- **Copy detection specific limitation:** No published evaluation on DISC21 or any copy detection benchmark. ImageBind is fundamentally not a copy detection model — it is optimized for cross-modal semantic retrieval. Using it for copy detection would introduce the same semantic-vs-structural confusion as CLIP, likely with worse precision. Its 1.2B parameter scale makes it the heaviest model in this evaluation without any demonstrated benefit for this specific task.

### SSCD (Reference)

- **Design intent:** Explicitly designed for copy detection. Uses ResNet50 or EfficientNet backbone, contrastive training with differential entropy regularization, and copy-detection-specific augmentations including MixUp and CutMix (to simulate partial copies).
- **JPEG compression:** Directly addressed in training augmentation pipeline. Augmentations include JPEG compression at various quality factors.
- **Heavy cropping (>50%):** Addressed via RandomResizedCrop and MixUp augmentation. The mixup strategy explicitly teaches the model to handle partial image content.
- **Color shift:** Color jitter augmentation included in training. Strong robustness.
- **Blur:** Gaussian blur augmentation included. Robust.
- **Precision-specific design:** The cosine similarity threshold of 0.75 corresponds to 90% precision (10% false positive rate) on DISC21 image pairs. SSCD is designed and calibrated for precision-first deployment, making it directly aligned with the system requirement.
- **Copydays (CD10K) benchmark:** µAP 98.1%, mAP 86.6% — demonstrates generalizability beyond DISC21.

---

## Hardware Requirements

### Model Size Reference

| Model | Parameters | Backbone | Embedding Dim |
|-------|-----------|---------|---------------|
| DINOv2 ViT-L | ~300M | ViT-L/14 | 1024 |
| CLIP ViT-L/14 | ~307M | ViT-L/14 | 768 |
| ImageBind | ~1.2B | ViT-H/14 | 1024 |
| SSCD ResNet50 | ~25M | ResNet-50 | 512 (configurable) |

### VRAM and Throughput

> **Note:** Precise per-model images/second figures for these specific hardware tiers were not found in public benchmarks as of research date. Values below are derived from model size, architecture class, and available community data. They represent estimates for planning purposes and should be validated with hardware-specific profiling before deployment.

| Model | VRAM (GB, FP16 inference, batch=32) | Throughput Tier 1 (CPU, 16-core, est.) | Throughput Tier 2 (RTX 4060 Ti 16GB, est.) | Throughput Tier 3 (RTX 6000 96GB, est.) |
|-------|------|-------------------------------|--------------------------|--------------------------|
| DINOv2 ViT-L | ~2–3 GB (weights ~600MB FP16 + activation overhead) | ~3–8 img/s (slow; ViT attention is CPU-inefficient) | ~150–300 img/s | ~400–800 img/s |
| CLIP ViT-L/14 | ~3.5 GB (confirmed: single replica ViT-L/14-336px uses 3.5GB) | ~3–8 img/s | ~150–300 img/s | ~400–800 img/s |
| ImageBind | ~8–10 GB (ViT-H/14, ~2.4GB weights FP16; LoRA fine-tune measured 5.66GB on 3080Ti) | ~1–3 img/s (prohibitively slow for batch workloads) | ~60–120 img/s (tight fit in 16GB with batching) | ~200–400 img/s |
| SSCD ResNet50 | ~0.5–1 GB (ResNet-50, 25M params, ~50MB FP16 weights) | ~50–150 img/s | ~800–2000 img/s | ~2000–5000+ img/s |

**Key observations:**
- SSCD (ResNet50) is dramatically lighter than all ViT-based candidates. It fits in < 1GB VRAM and runs efficiently on CPU, making it the only model viable at Tier 1 for real-time or near-real-time throughput.
- CLIP ViT-L/14 VRAM requirement (3.5 GB confirmed) fits comfortably within Tier 2 (16GB VRAM), leaving room for large batches. DINOv2 ViT-L is similar.
- ImageBind is the most VRAM-intensive (~8–10 GB for inference with batching), and while it technically fits within Tier 2's 16GB, it leaves little headroom for large batch sizes or multi-worker inference.
- All ViT-L models (DINOv2, CLIP) are slow on CPU-only hardware due to the self-attention complexity. Throughput of 3–8 img/s is inadequate for bulk deduplication workloads at Tier 1.

---

## License Summary

| Model | License | Commercial Use |
|-------|---------|----------------|
| DINOv2 ViT-L | Apache 2.0 (updated from original CC-BY-NC following community requests) | Yes — permissive |
| CLIP ViT-L/14 (OpenAI) | MIT License (code); model card notes "deployed use cases currently out of scope" | Ambiguous — MIT on code but OpenAI model card discourages deployment; use OpenCLIP (MIT, unrestricted) as drop-in alternative |
| ImageBind | CC-BY-NC 4.0 | No — non-commercial research only |
| SSCD | MIT License | Yes — permissive |

**License concern:** ImageBind's CC-BY-NC 4.0 restriction is a disqualifying factor for operational deployment (which constitutes institutional use that could be classified as non-research). DINOv2 (Apache 2.0) and SSCD (MIT) have no such restrictions. For CLIP, OpenCLIP weights (MIT) should be substituted for OpenAI-released weights to avoid ambiguity.

---

## Newer Models (2024–2025)

No single publicly available, off-the-shelf model has been conclusively demonstrated to surpass SSCD across the full DISC21 evaluation protocol with a widely adopted open-source implementation as of April 2026.

Notable developments:

1. **RDCD (WACV 2025, arXiv 2405.17928):** Knowledge distillation approach, achieves compact descriptors comparable to larger baselines but does not beat SSCD in absolute µAP.

2. **Dynamic Augmentation + ViT (MDPI Electronics, Aug 2024):** Achieves µAP 0.79 on a refined 10K query subset of DISC21 using strategic K-means sampling of training data. Results are on a filtered subset, not the full benchmark, limiting direct comparability. No publicly released weights found.

3. **AnyPattern / ImageStacker (arXiv 2404.13788, 2024):** In-context ICD model with large gains from pattern-aware training. Targets generalization to unseen tamper patterns. Still research-stage; no production-ready release found.

4. **Image Copy Detection for Diffusion Models (arXiv 2409.19952, Sep 2024):** Extends copy detection to AI-generated image copies (style transfer, inpainting, diffusion edits). Relevant for future proofing but not yet validated on the traditional DISC21 protocol.

**Conclusion on newer models:** For operational deployment today, SSCD remains the best-validated, openly licensed, hardware-efficient choice. The 2024–2025 research directions are promising but are not yet at production readiness for a precision-first deployment.

---

## Recommendation

**Winner (Tier 2 & Tier 3):** SSCD (ResNet50, sscd_disc_mixup checkpoint)

**Winner (Tier 1, CPU-only):** SSCD (ResNet50) — the only viable model at acceptable CPU throughput (~50–150 img/s on 16-core CPU)

**Reasoning:**

SSCD is the only model in this evaluation that was explicitly designed and evaluated for image copy detection under the DISC21 protocol. Its advantages are decisive across all criteria:

- **DISC21 µAP:** 0.724 with score normalization (official evaluation, CVPR 2022) vs. DINOv2 (~0.64 informal retrieval proxy, not comparable µAP), CLIP (~0.28 informal proxy), ImageBind (not benchmarked).
- **Precision calibration:** The 0.75 cosine similarity threshold yields 90% precision — directly aligned with the false-positive-minimizing requirement. CLIP and DINOv2 lack this calibrated operating point for copy detection.
- **Robustness:** SSCD's training augmentations explicitly address JPEG compression, heavy cropping (MixUp/CutMix), color shift, and blur. The other models address these incidentally via general self-supervised training.
- **Hardware efficiency:** SSCD ResNet50 requires < 1GB VRAM and achieves 50–150 img/s on CPU — orders of magnitude faster than ViT-L models at Tier 1. At Tier 2/3 it achieves 800–5000+ img/s, enabling high-throughput deduplication.
- **License:** MIT — no restrictions for operational use.

DINOv2 ViT-L could serve as a secondary retrieval layer (Apache 2.0, good JPEG/crop robustness, 64% informal retrieval accuracy on DISC21) if semantic-level similarity is needed as a complement, but it should not replace SSCD as the primary copy detection signal due to the risk of semantic false positives.

CLIP ViT-L/14 and ImageBind are not recommended for this task: CLIP underperforms significantly (28% vs 64% vs SSCD-grade performance), and ImageBind is unlicensed for operational use, untested on copy detection, and hardware-intensive without demonstrated benefit.

**Per-tier recommendation:**

- **Tier 1 (CPU-only, 16+ cores, 64GB RAM):** SSCD ResNet50 (sscd_disc_mixup) — only viable semantic embedding at CPU speed. If throughput is still insufficient for the full corpus volume, combine with PDQ perceptual hashing as a fast pre-filter before SSCD scoring.
- **Tier 2 (RTX 4060 Ti, 16GB VRAM):** SSCD ResNet50 (sscd_disc_mixup) — fits in < 1GB VRAM, achieves ~800–2000 img/s, far outperforms any ViT-L alternative in both speed and precision.
- **Tier 3 (RTX 6000, 96GB VRAM):** SSCD ResNet50 (sscd_disc_mixup) for primary deduplication at high throughput (~2000–5000+ img/s). Optional: add DINOv2 ViT-L as a secondary layer for semantic near-duplicate clustering (Apache 2.0, fits easily in 96GB VRAM) where broader similarity matching is desired.

---

## Sources

- [CLIP Vs DINOv2 in image similarity (Medium/AImonks) — DISC21 comparison](https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6)
- [ISC 2021 — Image Similarity Challenge (official site)](https://sites.google.com/view/isc2021)
- [Results and findings of the 2021 Image Similarity Challenge (PMLR)](https://proceedings.mlr.press/v176/papakipos22a/papakipos22a.pdf)
- [arXiv 2202.04007 — Results and findings of the 2021 Image Similarity Challenge](https://arxiv.org/abs/2202.04007)
- [GitHub — facebookresearch/sscd-copy-detection](https://github.com/facebookresearch/sscd-copy-detection)
- [SSCD Evaluation.md (micro-AP 0.724 result)](https://github.com/facebookresearch/sscd-copy-detection/blob/main/docs/Evaluation.md)
- [SSCD CVPR 2022 paper — openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2022/papers/Pizzi_A_Self-Supervised_Descriptor_for_Image_Copy_Detection_CVPR_2022_paper.pdf)
- [arXiv 2202.10261 — A Self-Supervised Descriptor for Image Copy Detection](https://arxiv.org/abs/2202.10261)
- [DrivenData — ISC21 Descriptor Track Leaderboard](https://www.drivendata.org/competitions/80/competition-image-similarity-2-dev/leaderboard/)
- [Meet the winners of the Image Similarity Challenge (DrivenData blog)](https://drivendata.co/blog/image-similarity-winners/)
- [arXiv 2304.07193 — DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [GitHub — facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- [DINOv2 MODEL_CARD.md (license: Apache 2.0)](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
- [arXiv 2305.05665 — ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665)
- [GitHub — facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind)
- [ImageBind LICENSE (CC-BY-NC 4.0)](https://github.com/facebookresearch/ImageBind/blob/main/LICENSE)
- [openai/clip-vit-large-patch14 on Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14)
- [GitHub — openai/CLIP (MIT License)](https://github.com/openai/CLIP)
- [CLIP-as-service Benchmark (VRAM: 3.5GB for ViT-L/14-336px)](https://clip-as-service.jina.ai/user-guides/benchmark/)
- [arXiv 2405.17928 — RDCD: Relational Self-supervised Distillation with Compact Descriptors for Image Copy Detection (WACV 2025)](https://arxiv.org/abs/2405.17928)
- [MDPI Electronics 2024 — Enhancing Image Copy Detection through Dynamic Augmentation and Efficient Sampling with Minimal Data](https://www.mdpi.com/2079-9292/13/16/3125)
- [arXiv 2404.13788 — AnyPattern: Towards In-context Image Copy Detection](https://arxiv.org/abs/2404.13788)
- [arXiv 2409.19952 — Image Copy Detection for Diffusion Models](https://arxiv.org/abs/2409.19952)
- [DISC21 Dataset — Meta AI](https://ai.meta.com/datasets/disc21-dataset/)
- [DeepWiki — facebookresearch/sscd-copy-detection](https://deepwiki.com/facebookresearch/sscd-copy-detection)
