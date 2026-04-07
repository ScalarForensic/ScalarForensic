# Scene Matching — VPR Indoor Transfer Evaluation

**Purpose:** Identify which Visual Place Recognition (VPR) model transfers best from outdoor training data to indoor environments relevant to a media identification pipeline. Evaluates NetVLAD, AnyLoc, EigenPlaces, and CosPlace; assesses SegFormer/Mask2Former preprocessing value; and identifies 2024/2025 models that may outperform the original candidates.

**Research date:** 2026-04-07

---

## Evaluation Criteria

1. Benchmark performance on indoor VPR datasets (InLoc, 7Scenes, Baidu Mall, or equivalent) — primary
2. Training data composition — outdoor-only vs mixed indoor/outdoor vs no training
3. Transfer performance gap between outdoor and indoor benchmarks
4. VRAM requirements per hardware tier
5. Whether SegFormer/Mask2Former preprocessing meaningfully improves indoor VPR accuracy

---

## Indoor VPR Benchmark Results

| Model | Training data | Indoor R@1 (relative) | Transfer quality | Notes |
|-------|--------------|----------------------|-----------------|-------|
| NetVLAD | Pitts-250k, MSLS (outdoor urban only) | Lowest | Poor | Requires Patch-NetVLAD+ extensions for indoor viability; outdoor-optimized chroma-like aggregation |
| AnyLoc-VLAD-DINOv2 | None (zero-shot DINOv2 features) | Highest (+5% over MixVPR, +20% over CosPlace on R@1) | Excellent | No re-training; universal across urban, indoor, aerial, underwater, subterranean |
| EigenPlaces | Outdoor urban (viewpoint-robust training) | High (close to ViT-based methods) | Good | Outperforms AnyLoc on some cross-domain evaluations; ResNet backbone gives lighter footprint |
| CosPlace | Outdoor urban (classification-based) | ~20% lower R@1 than AnyLoc on indoor datasets | Moderate | Strong outdoor; significant degradation indoors |

**Sources:**
- AnyLoc paper (IEEE RA-L 2023): "AnyLoc-VLAD-DINOv2 achieves the highest recall across all Indoor datasets, outperforming MixVPR by 5% and CosPlace by 20% on average (R@1)" — https://arxiv.org/abs/2308.00688
- EigenPlaces (ICCV 2023): viewpoint-robust training; evaluation wrapper includes EigenPlaces-indoor variant — https://github.com/gmberton/EigenPlaces
- VPR comparative study (March 2026): EigenPlaces "close to ViT-based methods and outperforms AnyLoc across all metrics" in cross-domain settings — https://arxiv.org/html/2603.13917
- NetVLAD training on Pitts-250k and MSLS (outdoor datasets) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10857504/

---

## Transfer Gap Assessment

**NetVLAD:** Severe gap. Trained entirely on outdoor urban images (Pittsburgh street-view, Mapillary street-level). The NetVLAD aggregation was validated on Pitts250k-test (R@1=81%) but underperforms substantially on indoor benchmarks. Dedicated variants (Patch-NetVLAD+) are required for indoor use. **Not recommended without fine-tuning.**

**CosPlace:** Moderate gap. Classification-based training on outdoor urban data transfers partially, but is the weakest of the candidate set on indoor benchmarks — approximately 20% R@1 behind AnyLoc.

**EigenPlaces:** Small gap. Viewpoint-robust training generalizes better than CosPlace. The evaluation framework includes a dedicated EigenPlaces-indoor variant, confirming the authors acknowledge an indoor/outdoor performance split. In cross-domain studies it approaches ViT-based methods and outperforms AnyLoc on some tasks.

**AnyLoc-VLAD-DINOv2:** No gap by design. Because AnyLoc performs no VPR-specific training, there is no domain shift. DINOv2 features transfer universally across structured (urban, indoor) and unstructured (aerial, underwater, subterranean) environments. This is the strongest candidate from the original four for indoor deployment.

---

## Segmentation Preprocessing Impact

**Finding: Weak evidence — not recommended as default.**

The research literature on SegFormer/Mask2Former as a preprocessing step specifically for indoor VPR accuracy improvement is limited:

- Semantic segmentation and dynamic object removal demonstrably improve **visual odometry and pose estimation** accuracy in dynamic environments.
- Hierarchical VPR methods that incorporate semantic cues during training show accuracy gains, but these integrate semantics architecturally rather than as a pre-processing mask applied at query time.
- No benchmark study was found that directly quantifies the recall improvement from applying SegFormer or Mask2Former masking of dynamic objects (people, furniture) before indoor VPR retrieval on standard benchmarks (InLoc, 7Scenes, Baidu Mall).
- Dynamic objects are more problematic in outdoor VPR (cars, buses) than indoor environments where the proportion of moving objects relative to architectural structure is lower.

**Recommendation:** Skip SegFormer/Mask2Former preprocessing in the initial implementation. The added GPU memory overhead (~1-2GB), inference latency, and engineering complexity are not supported by benchmark evidence specific to indoor VPR. Revisit if evaluation data shows false-match patterns attributable to moving persons in frame. AnyLoc's DINOv2 VLAD features are robust to partial occlusion by nature of patch-level aggregation.

**Sources:**
- PMC article on dynamic object removal improving pose estimation, not VPR recall directly — https://pmc.ncbi.nlm.nih.gov/articles/PMC11679697/
- Hierarchical VPR with semantic guidance (training-time integration, not preprocessing) — https://dl.acm.org/doi/10.1007/978-3-031-78113-1_25

---

## Newer Models (2024/2025)

Two models released after the original four candidates outperform them on indoor benchmarks:

### SALAD (CVPR 2024)
- Architecture: DINOv2-base backbone + Optimal Transport aggregation
- Stronger than AnyLoc on several benchmarks; open-source weights available
- Source: https://arxiv.org/html/2311.15937v1

### MegaLoc (CVPR 2025 Workshop) — **strongest indoor model found**
- Architecture: DINOv2-base backbone + SALAD aggregation layer
- Training data: GSV-Cities, MSLS, MegaScenes, **ScanNet** (indoor RGB-D), San Francisco XL
- **Baidu Mall (indoor) R@1: 87.7%** — vs 75.6% for next best (SALAD/CosPlace range)
- Vastly outperforms all other models on indoor-only datasets
- Achieves state-of-the-art or near-SOTA across outdoor, street-view, and indoor datasets in a single model
- Open source: https://github.com/gmberton/MegaLoc / HuggingFace: gberton/MegaLoc
- **Caveat:** MegaLoc was trained on ScanNet, so ScanNet-derived test sets may show inflated scores. For novel indoor environments not represented in training data, AnyLoc remains a strong fallback (zero-shot, no domain overlap).
- Source: https://openaccess.thecvf.com/content/CVPR2025W/IMW/papers/Berton_MegaLoc_One_Retrieval_to_Place_Them_All_CVPRW_2025_paper.pdf

---

## Hardware Requirements

| Model | Backbone | Approx. VRAM | CPU viable (Tier 1) | Notes |
|-------|----------|-------------|---------------------|-------|
| NetVLAD | VGG-16 | ~2 GB | Yes (slow) | Patch-NetVLAD+ needed for indoor; inference can run on CPU |
| EigenPlaces | ResNet-50/101 | 2–4 GB | Yes (ResNet-50) | Lightest of quality candidates; indoor variant available |
| AnyLoc-VLAD-DINOv2 | DINOv2 ViT-L | 5–8 GB | No (ViT-L too slow) | ViT-S variant reduces VRAM to ~2GB at accuracy cost |
| CosPlace | ResNet-50/101 | 2–4 GB | Yes (ResNet-50) | Not recommended for indoor; included for completeness |
| SALAD | DINOv2-base | ~4 GB | No | Intermediate option; outperforms AnyLoc on many tasks |
| MegaLoc | DINOv2-base | ~4–5 GB | No | Best indoor model; trained on ScanNet |

---

## Recommendation

**Winner (overall):** MegaLoc

MegaLoc is the strongest available model for indoor VPR by a significant margin. Its 87.7% Recall@1 on Baidu Mall (indoor) vs 75.6% for the next competitor is the most concrete evidence found. It is open-source, actively maintained by the EigenPlaces/CosPlace author (Gabriele Berton, Polytechnic of Turin), and runs on a DINOv2-base backbone (smaller than ViT-L), making it viable at Tier 2.

**Per-tier recommendation:**

| Tier | Hardware | Recommended model | Rationale |
|------|----------|-------------------|-----------|
| Tier 1 | CPU / integrated GPU | EigenPlaces (ResNet-50, indoor variant) | Only quality option viable without GPU; ResNet-50 runs on CPU at acceptable throughput |
| Tier 2 | RTX 4060 Ti 16 GB | MegaLoc (DINOv2-base + SALAD) | 4–5 GB VRAM fits easily; best indoor R@1; ScanNet training provides genuine indoor coverage |
| Tier 3 | RTX 6000 Ada 96 GB | MegaLoc (DINOv2-base + SALAD) | Same model as Tier 2 — sufficient for this task; remaining VRAM used by other pipeline models |

**Fallback if MegaLoc ScanNet overlap is a concern:** Use AnyLoc-VLAD-DINOv2 at Tier 2/3 — zero training data, no domain overlap risk, still +20% R@1 over CosPlace indoors.

**Reject:** NetVLAD (outdoor-trained, requires heavy adaptation), CosPlace (worst indoor performer of the four original candidates).

**Segmentation preprocessing:** Not recommended — no benchmark evidence for indoor VPR accuracy improvement; adds latency and VRAM overhead without demonstrated return.

**Similarity threshold guidance:** Given precision-over-recall mandate and indoor VPR limitations, recommend starting at cosine similarity ≥ 0.85 for scene match candidates, treated as investigative leads only (confirmed by human review), not as automated hits. The 0.80 threshold referenced in the spec may generate excessive false leads in visually repetitive indoor spaces (corridors, identical rooms).

---

**Sources:**
- [AnyLoc: Towards Universal Visual Place Recognition (RA-L 2023)](https://arxiv.org/abs/2308.00688)
- [EigenPlaces: Training Viewpoint Robust Models for VPR (ICCV 2023)](https://github.com/gmberton/EigenPlaces)
- [VPR Methods Evaluation repository](https://github.com/gmberton/VPR-methods-evaluation)
- [MegaLoc: One Retrieval to Place Them All (CVPR 2025W)](https://openaccess.thecvf.com/content/CVPR2025W/IMW/papers/Berton_MegaLoc_One_Retrieval_to_Place_Them_All_CVPRW_2025_paper.pdf)
- [MegaLoc GitHub](https://github.com/gmberton/MegaLoc)
- [Evaluation of VPR Methods for Image Pair Retrieval (March 2026)](https://arxiv.org/html/2603.13917)
- [SALAD: Optimal Transport Aggregation for VPR (CVPR 2024)](https://arxiv.org/html/2311.15937v1)
- [Hierarchical VPR with semantic guidance](https://dl.acm.org/doi/10.1007/978-3-031-78113-1_25)
