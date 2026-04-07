# Face-Based Person Identification — Strategy

## Use Case

Build a system that ingests images and videos, identifies every person whose face appears with sufficient quality, and stores them in a searchable database. Given a new image or video, the system extracts faces and finds matches against the existing database.

**Key properties:**
- **Discovery mode, not enrollment.** No labeled gallery. The database grows organically as new files are ingested. Identities emerge as clusters.
- **Faces are the only identity signal.** Clothing, body shape, gait, and contextual cues are explicitly out of scope. Face embeddings are the only thing that survives across arbitrary videos, lighting conditions, time gaps, and outfit changes.
- **Precision over recall.** False positives (wrongly merging two different people, or wrongly matching a query to the database) are the primary failure to avoid. Missing a low-quality face is acceptable.
- **Data model: persons ↔ files is many-to-many.** A person points to all files they appear in; a file points to all persons it contains.
- **No tracking objective.** Within-video tracking is only useful as a tool to group face detections from the same person in the same video before embedding — it is not a product feature.

---

## Why This Is a Face Recognition Problem, Not a Re-ID Problem

Person Re-Identification (Re-ID) systems — StrongSORT, BoT-SORT, Fast-ReID, OSNet, and the rest of the MOTChallenge ecosystem — are built around **appearance embeddings**: clothing color and texture, body shape, pose, accessories. They are designed for short-horizon tracking across overlapping camera views, where the assumption "same outfit, same day, similar lighting" holds.

This system violates every one of those assumptions. Files may be captured days, months, or years apart. Clothing changes are guaranteed across files. Lighting and camera characteristics vary arbitrarily.

The cloth-changing Re-ID literature confirms this is unsolved at operationally useful precision: the best published PRCC Rank-1 as of 2025 is around 65%, and no production-ready weights are publicly available. A 35% miss rate under clothing change is incompatible with a precision-focused system.

**The signal that survives across all these conditions is the face.** Modern face recognition models (ArcFace, AdaFace, and successors) achieve 99%+ verification accuracy on LFW and 98%+ on harder benchmarks like IJB-C, with embeddings that are robust to pose, age, and lighting variation within reasonable bounds. This is the right tool for the job.

---

## Pipeline Architecture

```
File ingest
   │
   ▼
Frame sampling (videos only)
   │
   ▼
Face detection
   │
   ▼
Face quality filtering   ←── primary precision lever
   │
   ▼
Face alignment + embedding (ArcFace / AdaFace)
   │
   ▼
Within-file clustering   ←── group embeddings of the same person in the same file
   │
   ▼
Cross-file matching against vector index   ←── strict threshold = second precision lever
   │
   ├── match found  → link existing person ID to this file
   └── no match     → create new person ID, add embeddings to index
```

### Stage 1: Frame Sampling (videos only)

Processing every frame is wasteful and produces near-duplicate embeddings. Sample at a fixed rate (e.g., 2–5 FPS) or use scene-change detection to sample more densely around cuts. For short videos, sample more aggressively; for long surveillance-style footage, sparser sampling is fine because the quality filter will discard most frames anyway.

Images are processed as a single "frame."

### Stage 2: Face Detection

Run a face detector on each sampled frame. Recommended options:

| Detector | Strengths | Notes |
|---|---|---|
| **SCRFD** | Fast, accurate, ONNX-friendly | Recommended default. InsightFace project. |
| **RetinaFace** | Very accurate, returns 5-point landmarks | Slightly heavier than SCRFD. |
| **YOLOv8-face** | Easy integration if YOLO is already in the stack | Slightly less accurate on small/occluded faces. |

The detector should return bounding boxes plus 5-point facial landmarks (eyes, nose, mouth corners). Landmarks are required for the alignment step before embedding.

### Stage 3: Face Quality Filtering — The Primary Precision Lever

**This is the single most important stage for false-positive control.** A low-quality face crop produces an unreliable embedding, which is the root cause of nearly all spurious matches downstream. The strategy is to be aggressive about discarding faces that aren't good enough, on the assumption that any person who appears meaningfully in a file will have *some* frames where their face is clean enough to use.

Filter on the following criteria, all of which should be tunable thresholds:

- **Resolution.** Reject faces smaller than ~80×80 pixels in the bounding box. Embeddings degrade rapidly below ~64×64.
- **Sharpness / blur.** Compute a Laplacian variance score on the face crop and reject below threshold. Motion-blurred frames are common in video and produce noisy embeddings.
- **Pose (yaw, pitch, roll).** Reject faces beyond ~±35° yaw or ~±25° pitch. Profile faces are unreliable for cross-video matching. Many face detectors return pose estimates directly, or a separate lightweight head-pose model can be used.
- **Occlusion.** Reject faces where landmarks indicate significant occlusion (e.g., a hand over the mouth, heavy sunglasses covering both eyes). A landmark-confidence check is usually sufficient.
- **Illumination.** Reject extreme over/underexposure, where most pixels are clipped to 0 or 255 in the face region.
- **Detector confidence.** Reject below ~0.85 detector confidence as a baseline floor.

A face must pass **all** filters to proceed. In practice, this typically discards 70–95% of detected faces in surveillance-style footage and is the right behavior. The goal is not to embed every face — it is to embed only the faces that will produce trustworthy matches.

> **Design note:** It is much better to fail to enroll a person from a marginal file than to enroll them with a bad embedding that later causes false matches against unrelated people. The quality filter should be tuned conservatively and reviewed periodically against false-positive incidents.

### Stage 4: Face Alignment and Embedding

Align each surviving face crop to a canonical position using the 5-point landmarks (standard 112×112 ArcFace alignment). Then extract an embedding using a face recognition model.

Recommended models:

| Model | Embedding dim | Notes |
|---|---|---|
| **ArcFace (R100, MS1MV3)** | 512 | Industry standard. Strong baseline. Available via InsightFace. |
| **AdaFace (R100)** | 512 | Improvement over ArcFace, particularly on low-quality faces. Recommended if low-quality robustness matters even after the filter. |
| **CurricularFace, MagFace** | 512 | Alternatives in the same family; comparable performance. |

All of these are available as pre-trained ONNX or PyTorch weights through the InsightFace project or the original author repos. None require custom training for general-purpose face recognition.

For this system, **AdaFace is the recommended default**, because its training objective explicitly handles quality variation and it provides a small but meaningful margin over ArcFace on harder benchmarks (IJB-B, IJB-C). The cost is identical at inference time.

### Stage 5: Within-File Clustering

A single file (especially a video) will produce many embeddings from the same person. Before writing to the database, group embeddings within a file by identity:

- **For videos:** A lightweight IoU-based tracker (even ByteTrack-style motion-only association is fine here) can link face detections across consecutive frames into per-person tracklets. This is the *only* place tracker logic earns its keep in this system. Each tracklet collapses into one or a few representative embeddings.
- **For images:** Multiple faces in one image are simply treated as separate per-person observations.
- **Fallback (no tracker):** Cluster the per-frame embeddings within the file using a similarity threshold (e.g., cosine ≥ 0.5). This works but is slightly noisier than tracker-assisted grouping.

For each within-file person cluster, keep the **top-K highest-quality embeddings** (e.g., K=5–10) rather than averaging. Storing multiple high-quality views gives the cross-file matcher more to work with than a single centroid and is more robust to pose variation. Disk cost is negligible — 10 × 512 floats × 4 bytes is 20 KB per person per file.

### Stage 6: Cross-File Matching — The Second Precision Lever

For each within-file person cluster, query the vector index for nearest neighbors among all stored embeddings.

- **Index:** FAISS (local, simple, fast), Qdrant, or Milvus. For up to a few million embeddings, FAISS with an `IndexFlatIP` (exact cosine) is sufficient and avoids approximate-search false positives. Move to IVF or HNSW only when scale demands it.
- **Similarity metric:** Cosine similarity on L2-normalized embeddings.
- **Match decision:** Aggregate over the top-K embeddings of both the query cluster and the candidate person in the database. A reasonable rule:
  - Compute pairwise cosine similarities between query top-K and candidate top-K.
  - Take the **mean of the top-N highest similarities** (e.g., top 3 of 25 pairs).
  - If this score exceeds a strict threshold (e.g., ≥ 0.55 for ArcFace, tune empirically), declare a match.
  - If multiple candidate persons exceed the threshold, take the highest **only if** its margin over the second-best exceeds a separation threshold (e.g., 0.05). Otherwise, flag for manual review rather than auto-merging.

This dual threshold (absolute score + separation margin) is the second main precision lever and dramatically reduces false merges in ambiguous cases.

If no candidate exceeds the threshold, create a new person ID and add the cluster's embeddings to the index.

---

## Data Model

```
File
  - file_id
  - source path
  - media type (image / video)
  - ingest timestamp
  - frame sample rate (videos)

Person
  - person_id
  - created_at
  - embeddings: list of (vector, source_file_id, source_frame_or_timestamp, quality_score)
  - canonical_thumbnail (optional, for UI review)

PersonFileLink (many-to-many)
  - person_id
  - file_id
  - first_appearance_timestamp (videos)
  - last_appearance_timestamp (videos)
  - num_observations
  - best_quality_score
```

A file can contain many persons; a person can appear in many files. Both directions are queryable.

---

## False-Positive Control: Summary of Levers

The system has four independent knobs that all push toward fewer false positives. They should be tuned together against a held-out evaluation set:

1. **Quality filter strictness** (Stage 3) — discards bad inputs before they ever become embeddings. Most important.
2. **Embedding model choice** (Stage 4) — AdaFace is more robust than ArcFace on imperfect crops.
3. **Top-K aggregation in matching** (Stage 6) — averaging over multiple high-quality views per person is more reliable than single-vector matching.
4. **Match threshold + separation margin** (Stage 6) — the absolute similarity threshold sets the false-positive floor; the separation margin handles ambiguous cases.

A useful operational pattern is to run the system in **shadow mode** initially: produce match candidates but require human confirmation for any merge. Log the rate of confirmed vs. rejected merges as a function of similarity score, then set the auto-merge threshold at the point where the human-confirmed rate exceeds 99% (or whatever the precision target is).

---

## Recommended Stack

| Component | Recommendation | Rationale |
|---|---|---|
| Face detector | **SCRFD** (InsightFace) | Fast, accurate, ONNX-exportable, returns landmarks. |
| Quality filter | Custom: resolution + Laplacian blur + pose + landmark confidence + exposure | No off-the-shelf single tool; assemble from standard components. |
| Face embedding | **AdaFace R100** | SOTA on low-quality faces; drop-in replacement for ArcFace. |
| Within-file grouping | ByteTrack (motion-only IoU tracker) on face boxes | Lightweight; this is the only place a tracker is needed. |
| Vector index | **FAISS** (`IndexFlatIP`) initially; migrate to Qdrant or Milvus at scale | Exact search avoids ANN-induced false positives at small/medium scale. |
| Storage | SQLite or Postgres for metadata; FAISS index file or Qdrant collection for vectors | Separate metadata from vectors; rebuild index from metadata if needed. |

InsightFace ([github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)) bundles SCRFD, ArcFace, and several alignment utilities in one project and is the most operationally mature option for the detection + embedding stages. AdaFace ([github.com/mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)) provides pre-trained weights compatible with InsightFace's preprocessing.

---

## Out of Scope (and Why)

- **Person Re-ID models (StrongSORT, Fast-ReID, OSNet, etc.).** These rely on clothing and body appearance and do not generalize across files captured at different times. They were the right answer for the previous version of this strategy and are the wrong answer for this one.
- **Cloth-changing Re-ID (CC-ReID).** Not yet operationally mature; best published results are below the precision floor this system requires.
- **Gait, body shape, or clothing as identity signals.** Excluded by scope. Faces only.
- **Cross-camera multi-object tracking.** Not a product goal. Within-file tracking exists only as a tool for grouping face embeddings before storage.

---

## Open Questions for the Next Iteration

- **Threshold calibration data.** What labeled data is available (or can be assembled) to tune the quality filter and match thresholds against the actual files this system will ingest? Public benchmarks (LFW, IJB-C) are useful starting points but rarely match deployment conditions.
- **Review UI.** Discovery mode produces clusters with no human-readable labels. A minimal review interface (show person thumbnail + list of files + merge/split controls) is essential for catching false merges and splits early.
- **Scale planning.** How many files, how many persons, expected query rate? Determines whether FAISS-flat is sufficient or whether an ANN index and a managed vector DB are needed from day one.
- **Re-embedding strategy.** When the embedding model is upgraded, all stored embeddings become incomparable to new ones. Plan for periodic re-embedding by keeping the original face crops (or at least the source frame + bbox) alongside the embeddings.