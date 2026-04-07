# Person Re-ID — Tracker and Re-ID Model Evaluation

## Multi-Object Tracker Evaluation

### Evaluation Criteria
1. ID-switch rate (primary — lowest is best)
2. MOTA score
3. IDF1 score
4. Compute overhead at Tier 1 (CPU viable?)
5. Performance on low-quality / low-resolution video

### Benchmark Results (MOT17 / MOT20)

Data sourced from the BoT-SORT paper (arXiv:2206.14651), which uses the same detection backbone (YOLOX) for all trackers, making it a controlled apples-to-apples comparison.

#### MOT17 Test Set

| Tracker | MOTA | IDF1 | HOTA | ID Switches | Notes |
|---------|------|------|------|-------------|-------|
| ByteTrack | 80.3 | 77.3 | 63.1 | **2196** | Motion-only; no Re-ID; fastest |
| StrongSORT++ | 79.6 | **79.5** | **64.4** | **1194** | Best IDF1 on MOT17; uses AFLink + GSI post-processing |
| BoT-SORT | **80.6** | 79.5 | 64.6 | 1257 | No Re-ID module; camera motion compensation |
| BoT-SORT-ReID | 80.5 | **80.2** | **65.0** | 1212 | Re-ID enabled; top-ranked overall on MOTChallenge |

#### MOT20 Test Set (denser / more crowded scenes)

| Tracker | MOTA | IDF1 | HOTA | ID Switches | Notes |
|---------|------|------|------|-------------|-------|
| ByteTrack | **77.8** | 75.2 | 61.3 | **1223** | Higher ID switches in crowds |
| StrongSORT++ | 73.8 | **77.0** | **62.6** | **770** | Lowest ID switches by large margin |
| BoT-SORT | 77.7 | 76.3 | 62.6 | 1212 | Similar IDs to ByteTrack on MOT20 |
| BoT-SORT-ReID | **77.8** | **77.5** | **63.3** | 1257 | Slightly more IDs than BoT-SORT on MOT20 |

Sources: [BoT-SORT paper (arXiv:2206.14651)](https://ar5iv.labs.arxiv.org/html/2206.14651), [StrongSORT paper (arXiv:2202.13514)](https://arxiv.org/pdf/2202.13514)

**Key observation:** StrongSORT++ is the clear winner on ID switches — 45% fewer switches than ByteTrack on MOT17 (1194 vs. 2196) and 37% fewer on MOT20 (770 vs. 1223). This is the metric that matters most for this system.

---

### Low-Quality Video Notes

**ByteTrack:**
ByteTrack is a motion-only tracker (Kalman filter + IoU association). It produces no appearance embeddings, which means it cannot recover identity after occlusion or re-entry. In low-quality surveillance video — where detections may drop frames, bounding boxes are noisy, and frames may be sub-15 FPS — ByteTrack's lack of Re-ID leads to aggressive ID fragmentation. Its association relies entirely on spatial overlap, which degrades when detections are intermittent.

**StrongSORT:**
StrongSORT incorporates deep appearance features (Re-ID embeddings), Gaussian-smoothed interpolation (GSI) to fill missing detections, and an appearance-free link (AFLink) post-processing module. The GSI component is specifically designed to handle frame-drop scenarios, making it more robust to degraded / low-FPS video. The Re-ID embeddings help re-associate tracks after occlusion, which is common in surveillance camera footage. StrongSORT was published in TMM 2023 and is actively maintained via the BoxMOT library.

**BoT-SORT:**
BoT-SORT includes a camera-motion compensation (CMC) module using optical flow / EGO-motion estimation. This is highly relevant for PTZ surveillance cameras or footage with camera shake. However, CMC is computationally expensive — a C++ benchmark shows it accounts for 97.1% of per-frame processing time. BoT-SORT-ReID adds appearance features on top of CMC. The original BoT-SORT GitHub repo (NirAharon/BoT-SORT) has not been actively maintained since 2022; the BoxMOT library provides an updated, maintained implementation.

Sources: [BoT-SORT GitHub](https://github.com/NirAharon/BoT-SORT), [BoTSORT-cpp performance report](https://github.com/viplix3/BoTSORT-cpp/blob/main/docs/PerformanceReport.md), [StrongSORT GitHub](https://github.com/dyhBUPT/StrongSORT), [BoxMOT library](https://github.com/mikel-brostrom/boxmot)

---

### Recommendation

**Winner: StrongSORT++ (via BoxMOT)**

**Reasoning:**
- StrongSORT++ achieves the lowest ID switches in both MOT17 (1,194) and MOT20 (770) — 45% fewer than ByteTrack on MOT17, which is the most consequential metric for this pipeline. Precision over recall demands that a tracklet, once assigned, stays correct.
- The GSI (Gaussian-Smoothed Interpolation) module explicitly handles frame-drop and low-FPS video, which is the expected condition for archival surveillance footage.
- The AFLink post-processing module reduces fragmentation from occlusion recovery, improving long-tracklet identity consistency.
- StrongSORT++ trades ~0.7 MOTA points against ByteTrack on MOT17 (79.6 vs. 80.3), an acceptable cost for a 45% reduction in ID switches.
- BoT-SORT-ReID ranks first on MOTChallenge overall, but its ID switch performance is not better than StrongSORT++ and its camera-motion-compensation (CMC) overhead at 97% of compute time makes it impractical for CPU-only Tier 1 deployment.

**CPU viability (Tier 1):** Conditional yes for StrongSORT.
- ByteTrack is the most CPU-friendly — no embedding extraction needed. On a 16-core CPU, ByteTrack easily sustains real-time speeds.
- StrongSORT's Re-ID embedding extraction (typically OSNet or ResNet-based) adds per-detection GPU/CPU inference cost. On Tier 1 (CPU only), this will limit throughput. Recommended approach: use a lightweight OSNet-x0.25 embedding model and limit tracking to detections above 0.5 confidence to reduce embedding calls. At non-real-time speeds (processing archived video), this is viable.
- BoT-SORT's CMC step is the most CPU-intensive; not recommended for Tier 1 without disabling CMC (which eliminates its key advantage over ByteTrack).
- **Tier 1 fallback:** If CPU throughput becomes a bottleneck, deploy ByteTrack for initial processing and apply Re-ID matching as a post-processing step on the resulting tracklet database.

---

## Clothing-Independent Re-ID Evaluation

### Evaluation Criteria
1. PRCC mAP / LTCC mAP (standard cloth-changing benchmarks)
2. Operational maturity (maintained repo, pre-trained weights available without custom training)
3. Performance vs. standard Re-ID (is the accuracy gap acceptable?)

---

### Maturity Assessment

Cloth-changing person Re-ID (CC-ReID) is an **active research area but not yet operationally mature** as of April 2026.

The gap between published results and deployable tooling is significant:

- The research literature (2024–2025) contains numerous papers with strong benchmark results on PRCC and LTCC.
- However, pre-trained weights with documented inference pipelines ready for production integration are rare. Most repos release training code only.
- Models achieving the best PRCC/LTCC results rely on body silhouettes, contour parsing, gait cues, or multi-modal fusion (sketch + text attributes), which require additional preprocessing pipelines not included in the Re-ID model itself.
- Standard Re-ID models (appearance-based, same clothing assumed) remain significantly more mature: stable frameworks, pre-trained weights, ONNX export, and integration with trackers.

**Key limitation for operational deployment:** The performance gap between CC-ReID and same-clothes Re-ID is still large enough to affect operational precision. PRCC Rank-1 of ~64% (best 2025 results) for the cloth-changing test split vs. ~99% same-clothes split illustrates the fundamental difficulty.

---

### Benchmark Results

PRCC benchmark: 221 identities, cloth-changing test split vs. same-clothes reference gallery.
LTCC benchmark: 152 identities, 3.1 clothing changes per identity on average.

| Model | PRCC Rank-1 | PRCC mAP | LTCC Rank-1 | LTCC mAP | Weights available? | Actively maintained? |
|-------|------------|---------|------------|---------|-------------------|---------------------|
| AE-Net (2024) | 60.4% | ~55% est. | 42.9% | ~25% est. | No public weights found | No |
| DM-ReID (2025) | 64.6% | **63.1%** | N/A | N/A | No public weights | No |
| MIPL (2024) | N/A | N/A | **74.8%** Rank-1 | **38.1%** | No public weights | No |
| DIFFER (2025) | ~64% (+3.4% over baseline) | N/A | ~43% (+3.6%) | N/A | No public weights | No |
| ACD-Net | ~58% | ~52% | ~40% | ~23% | [GitHub](https://github.com/jk-love-ge/ACDNet) (training code) | Low |
| TryHarder (2025, ACM MM) | SOTA +8.9% Rank-1 on PRCC | SOTA +11.5% mAP | SOTA on LTCC | SOTA | [GitHub](https://github.com/undooo/TryHarder-ACMMM25) | Recent |

Sources: [DM-ReID results](https://link.springer.com/article/10.1007/s10462-025-11250-6), [MIPL](https://arxiv.org/html/2411.00330), [DIFFER](https://arxiv.org/html/2503.22912), [Try Harder](https://arxiv.org/html/2507.11119v1), [Cloth Change Paper List](https://github.com/wangxiao5791509/Cloth_Change_Person_reID_Paper_List)

**Note on "weights available":** None of the top-performing CC-ReID models (2024–2025) provide pre-trained inference weights as a downloadable artifact. ACD-Net and TryHarder provide training code and may yield weights after training on PRCC/LTCC, but that requires supervised training on these datasets — not zero-shot deployment.

---

### Recommendation

**Not yet mature for primary operational deployment — recommend standard Re-ID with targeted mitigation.**

Evidence:
1. No CC-ReID model provides a clean, ready-to-use inference pipeline with pre-trained weights as of April 2026. All published results require either custom training or a multi-stage preprocessing pipeline (silhouette extraction, contour parsing).
2. Even the best published PRCC Rank-1 (~65%) is low for a precision-focused system. A 35% miss rate under clothing change is operationally significant.
3. LTCC mAP numbers (~38%) are not sufficient for use as a primary identity signal.

**Recommended mitigation strategy (standard Re-ID with clothing-change awareness):**
- Deploy a standard Re-ID model (see Fast-ReID / BoxMOT section) as the primary identity matcher.
- Augment with body-shape / silhouette-based signals: gait analysis (if video permits) or body proportion features (height/weight ratio from camera calibration) provide clothing-independent secondary signals.
- Flag tracklets with large appearance-distance gaps as "potential clothing change" events rather than new identities — human analyst review rather than automated discard.
- When CC-ReID weights from TryHarder or a similar 2025 model become available for download, run a parallel evaluation against this system's specific footage conditions before replacing the primary matcher.

---

## Fast-ReID / Torchreid Current Status

### Fast-ReID (JDAI-CV/fast-reid)

**Maintenance status: Effectively unmaintained.**
- Last GitHub release: v1.3.0, dated May 31, 2021 (4+ years ago as of April 2026).
- Total releases: 4 (v0.1 May 2019 → v1.3.0 May 2021).
- PyPI package has received no new versions in 12+ months (per Snyk package health analysis).
- No significant pull request activity or issue resolution detected in 2024–2025.
- The paper was published at ACM Multimedia 2023 (describing the framework), but the codebase itself has not been updated to match modern PyTorch versions or new architectures.
- Known integration: BoT-SORT uses Fast-ReID as its Re-ID backend, which is why the BoT-SORT original repo is similarly dated.

**What Fast-ReID still offers:** A large model zoo with pre-trained weights (ResNet-50, ResNet-101, SE-ResNet, ViT-based) evaluated on Market-1501, MSMT17, DukeMTMC — all standard same-clothes benchmarks. These weights are still usable for appearance-based Re-ID if PyTorch compatibility issues are managed.

Sources: [Fast-ReID GitHub releases](https://github.com/JDAI-CV/fast-reid/releases), [Fast-ReID model zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md), [Snyk package health](https://snyk.io/advisor/python/fastreid)

### Torchreid (KaiyangZhou/deep-person-reid)

**Maintenance status: Inactive.**
- Last meaningful update: August 2022 (ONNX/OpenVINO/TFLite export support added).
- Latest PyPI version: 1.4.0 — no new releases in 2+ years.
- The project is classified as inactive by package health monitors.
- OSNet (the lightweight backbone it popularized) remains competitive for CPU-constrained Tier 1 deployment and is available via BoxMOT and Hugging Face.

Sources: [Torchreid GitHub](https://github.com/KaiyangZhou/deep-person-reid), [Torchreid PyPI](https://pypi.org/project/torchreid/)

### BoxMOT — Active Alternative

**BoxMOT (mikel-brostrom/boxmot) is the recommended actively-maintained framework** for combining tracking + Re-ID in 2025–2026.
- 3,764+ commits on master; continuous development.
- Provides a unified CLI and Python API for ByteTrack, StrongSORT, BoT-SORT, OC-SORT, DeepOCSORT, HybridSORT, BoostTrack.
- Bundles Re-ID model support (OSNet, ResNet) with tracker configuration.
- Python 3.9–3.12 compatible; AGPL-3.0 licensed.
- Handles both AABB and OBB detection formats.
- Integrates with YOLO family detectors natively.

Sources: [BoxMOT GitHub](https://github.com/mikel-brostrom/boxmot), [BoxMOT PyPI](https://pypi.org/project/boxmot/)

---

### Recommendation

**Switch from Fast-ReID to BoxMOT as the primary tracking + Re-ID integration framework.**

Reasoning:
- Fast-ReID has been unmaintained for 4+ years; PyTorch version compatibility issues will compound over time and are already a maintenance burden.
- Torchreid is similarly inactive.
- BoxMOT provides the same tracker ensemble (including StrongSORT, the recommended tracker) with maintained dependencies, and bundles compatible Re-ID embedding models.
- For the Re-ID embedding model specifically: use **OSNet-x1.0** (available via BoxMOT / Torchreid weights on Hugging Face at `kadirnar/osnet_x0_25_imagenet`) for Tier 2/3 GPU deployments, and **OSNet-x0.25** for Tier 1 CPU deployments. OSNet-x0.25 is extremely lightweight and has been benchmarked at real-time speeds on modest CPU hardware.
- If Fast-ReID pre-trained weights (ResNet-based) are already in use, they can be loaded independently via the `fastreid` package or converted to ONNX and used as an embedding backbone within BoxMOT's pluggable Re-ID interface — this preserves any accuracy investments without requiring full retraining.

---

**Sources:**
- [BoT-SORT paper (ar5iv)](https://ar5iv.labs.arxiv.org/html/2206.14651)
- [BoT-SORT GitHub](https://github.com/NirAharon/BoT-SORT)
- [ByteTrack paper](https://arxiv.org/abs/2110.06864)
- [StrongSORT paper](https://arxiv.org/pdf/2202.13514)
- [StrongSORT GitHub](https://github.com/dyhBUPT/StrongSORT)
- [BoxMOT GitHub](https://github.com/mikel-brostrom/boxmot)
- [BoxMOT PyPI](https://pypi.org/project/boxmot/)
- [BoTSORT-cpp Performance Report](https://github.com/viplix3/BoTSORT-cpp/blob/main/docs/PerformanceReport.md)
- [Fast-ReID GitHub](https://github.com/JDAI-CV/fast-reid)
- [Fast-ReID releases](https://github.com/JDAI-CV/fast-reid/releases)
- [Fast-ReID model zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)
- [Snyk fast-reid package health](https://snyk.io/advisor/python/fastreid)
- [Torchreid GitHub](https://github.com/KaiyangZhou/deep-person-reid)
- [Torchreid PyPI](https://pypi.org/project/torchreid/)
- [OSNet Hugging Face weights](https://huggingface.co/kadirnar/osnet_x0_25_imagenet)
- [Cloth Change ReID Paper List](https://github.com/wangxiao5791509/Cloth_Change_Person_reID_Paper_List)
- [DIFFER (2025)](https://arxiv.org/html/2503.22912)
- [Try Harder CC-ReID (2025)](https://arxiv.org/html/2507.11119v1)
- [MIPL cloth-changing (2024)](https://arxiv.org/html/2411.00330)
- [DM-ReID (2025)](https://link.springer.com/article/10.1007/s10462-025-11250-6)
- [ACD-Net GitHub](https://github.com/jk-love-ge/ACDNet)
- [TryHarder GitHub](https://github.com/undooo/TryHarder-ACMMM25)
- [IET Computer Vision MOT review (2025)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.70010)
- [Nature Scientific Reports attention-enhanced StrongSORT](https://www.nature.com/articles/s41598-025-99524-5)
- [Veroke real-world tracker comparison](https://www.veroke.com/insights/how-top-ai-multi-object-trackers-perform-in-real-world-scenarios/)
