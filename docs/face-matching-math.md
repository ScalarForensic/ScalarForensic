# Face matching: the method chain

Companion to `docs/specs/face-pipeline.md`. This document is the method chain an examiner or
reviewer follows to understand how a stored face observation was produced, and — just as
importantly — what it does **not** claim.

Status: Phase 1 (detect / gate / align / embed / store, browse only). There is deliberately no
cross-file face search yet; that is Phase 1b and is gated on a face-calibration record.

---

## 1. Detection

- Detector: **YuNet** (`cv2.FaceDetectorYN`), MIT-licensed code and weights, loaded from a local
  ONNX file. No network I/O at any point.
- The image is decoded at **full oriented resolution** (EXIF orientation applied), *not* through
  the embedding path's 331 px draft decode — that would destroy small faces.
- The detector input is capped at `SFN_FACE_DETECT_MAX_SIZE` (default 1600 px long side).
  Coordinates are scaled back to oriented source pixels, and `detect_scale` records the factor.
- Every detection carries a confidence in [0, 1] and five landmarks.

**Canonical landmark order** everywhere in the codebase:
`[left eye, right eye, nose tip, left mouth corner, right mouth corner]`, where "left" means
**image-left** (the subject's right side). This was verified empirically against the real
2023mar YuNet model on real photographs; a swapped eye pair would mirror every alignment and
silently degrade matching, so `assert_canonical_landmarks()` re-checks the invariant on every
emitted face and drops (and counts) any face that violates it.

## 2. Quality gate — two stages

The gate is the primary false-positive lever. Rejected faces are **never persisted**; the
reason is counted and reported.

**Pre-alignment** (cheap, on the detection):

| Check | Setting | Default | Rejection reason |
|---|---|---|---|
| Detector confidence | `SFN_FACE_MIN_CONF` | 0.8 | `confidence` |
| Face size (min bbox side, detector-input px) | `SFN_FACE_MIN_SIZE` | 64 | `size` |
| Pose (yaw proxy) | `SFN_FACE_MAX_POSE` | 0.35 | `pose` |

The pose proxy is the horizontal offset of the nose tip from the eye midpoint, divided by the
eye span: 0.0 is frontal, growing toward ±1 in profile. It is coarse by design — its job is
rejecting strong profiles, not estimating angles.

**Post-alignment** (on the **native-resolution source crop**, not the 112×112 resample, whose
Laplacian variance mostly re-encodes the resize factor):

| Check | Setting | Default | Rejection reason |
|---|---|---|---|
| Sharpness (Laplacian variance) | `SFN_FACE_MIN_SHARPNESS` | 25.0 | `sharpness` |
| Clipped-pixel fraction | `SFN_FACE_MAX_CLIPPED` | 0.6 | `exposure` |

Exposure is checked **before** sharpness: a clipped strobe frame can have a huge Laplacian
variance and would otherwise pass as "sharp".

> **Every threshold above is a bootstrap value.** None of them is derived from a dataset or a
> validation protocol. They are starting points that a per-deployment face-calibration record
> (Phase 1b) supersedes by name. Do not present them as validated operating points.

## 3. Alignment

- Similarity transform (rotation + uniform scale + translation, **no reflection**) estimated by
  the Umeyama least-squares method, mapping the five detected landmarks onto a fixed reference
  template.
- Reference points are the standard ArcFace 112×112 five-point template, verified against
  `insightface.utils.face_align.arcface_dst`:

  ```
  (38.2946, 51.6963)   left eye
  (73.5318, 51.5014)   right eye
  (56.0252, 71.7366)   nose tip
  (41.5493, 92.3655)   left mouth corner
  (70.7299, 92.2041)   right mouth corner
  ```

- Output: a 112×112 RGB crop, warped bilinearly with a constant black border.
- The alignment is versioned as **`arcface-112-v1`**, is not configurable, and is part of both
  the stored provenance and the face point ID — a future `v2` re-index coexists with v1 points
  rather than overwriting them.
- Implementation follows the Umeyama (1991) paper, not any secondary source.

## 4. Embedding

- The recognition model is **operator-supplied ONNX** (ScalarForensic ships no recognition
  weights — see `INSTALL.md` for the licensing reasons).
- A JSON **manifest** next to the model declares input tensor name, layout (NCHW/NHWC), channel
  order (RGB/BGR), dtype, input size (must be 112), normalisation mean/scale, output tensor name
  and embedding dimension. The manifest is validated against the ONNX session's actual I/O at
  load time; a mismatch is a hard, actionable error.
- Normalisation is affine, `(x − mean) / scale`, recorded as a `normalization_id` such as
  `affine-127.5-128.0`.
- Execution is **CPU-only** in Phase 1, with ORT thread counts pinned to avoid OpenMP
  oversubscription against torch in the same process.
- The output vector is **L2-normalised** by us and stored as float32; the pre-normalisation norm
  is stored alongside as `embedding_norm` (a useful low-quality signal).
- Similarity is therefore **cosine** on unit vectors.

## 5. What is stored

Per face observation, one Qdrant point in a case-scoped sidecar collection
(`{SFN_COLLECTION}_faces` by default), carrying:

- **Location**: `image_hash`, `image_path`, and for video frames `video_hash`, `video_path`,
  `frame_timecode_ms`.
- **Geometry**: `bbox` (ints), `landmarks`, `det_conf`, `detect_scale`.
- **Quality**: every subscore (`quality_confidence`, `quality_size`, `quality_pose`,
  `quality_sharpness`, `quality_exposure`) plus a composite `quality` in [0, 1] used only for
  browse ranking. The composite normalises each subscore against its own gate threshold, so 0
  means "at the gate boundary". It is an aid, not evidence.
- **Identity**: `observation_key` — the model-independent coordinates
  (`image_hash:timecode:x:y:w:h`), deliberately *without* the alignment version, so human labels
  attached to an observation survive a re-alignment or re-detection.
- **Provenance**: detector id and model hash, embedder model and manifest hashes, embedding
  dimension, alignment version, normalisation id, **every gate threshold in force**, library
  versions, and a `pipeline_config_hash` over all of it except the library versions.
- **Artefacts**: `chip_hash`, addressing a lossless aligned PNG (the exact tensor the embedder
  saw) and a source-resolution review JPEG. A browse thumbnail is derived, non-evidentiary and
  regenerable, and is excluded from the hash.

Comparability is hard-gated: embedder model hash, manifest hash, embedding dimension, alignment
version and normalisation id must match what the collection recorded, or the run refuses.
Detector and gate-threshold differences warn instead. Fields absent from an older collection's
metadata are treated as *unknown*, not as a mismatch.

Use is logged separately from provenance: an append-only JSONL audit log records enablement,
index runs and purges. Provenance says how the data was produced; the audit log says that and
how the system was used.

## 6. What is NOT claimed

- **This is not an identification.** The system produces investigative leads: candidate face
  observations ranked by similarity. Any assertion that two observations depict the same person
  is a human conclusion, made by a named examiner, and must be recorded as such.
- **No enrollment gallery.** This is discovery, not targeted matching against a known-identity
  database.
- **No accuracy claim.** ScalarForensic states no false-match or false-non-match rate for this
  pipeline. Any such figure must come from a per-deployment calibration on representative data,
  with stated uncertainty, and arrives with the Phase 1b calibration record.
- **Twins, siblings and close relatives** are a known and unmitigated limitation of face
  embeddings; so is heavy occlusion, extreme pose, motion blur and very low resolution — the
  quality gate reduces but does not remove these failure modes.
- **Differential performance across demographic groups is documented in the literature for face
  recognition systems generally.** ScalarForensic performs no demographic estimation and makes
  no claim that its error rates are uniform across groups. A deployment that needs to reason
  about this must measure it on its own data.
- **No claim about the operator's recognition weights.** Their provenance, training data and
  legal status are the operator's responsibility.
- **No calibrated threshold, and no error rate, is claimed for face search in this deployment.**
  The cosine shown beneath a matched face is the raw model output. The number that would tell an
  examiner what it means — a face-calibration record with stated pair counts and confidence
  intervals (spec §10) — does not exist here. SFace's published 0.363 same/different figure is
  the model authors' measurement on their material, not a measurement on this operator's.

A per-deployment method annex — the document that could carry validated operating points —
arrives only with a calibration record. Until then, this document describes the *method*, not
its measured performance.
