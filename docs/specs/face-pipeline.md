# Spec: Face Pipeline — optional identity modality for ScalarForensic

Status: **draft v2, post-review** · 2026-08-11
Sources: `docs/research/person-reid-evaluation_reevaluation.md` (our prior strategy),
`research/distilled/overview.md` (critical digest of the Gemini corpus `research/1.md`–`5.md`),
targeted Q&A with the research-librarian session, and two independent design reviews (Codex,
Opus) whose accepted findings are integrated throughout.

---

## 1. What this feature is

Answer the investigative question **"who appears in this evidence set, and where else do they
appear?"** ScalarForensic today matches *images* (exact hash, DINOv2 semantic, SSCD copy-detect).
This feature adds a *face* modality: detect faces in indexed images and video frames, embed them,
store them as first-class searchable points, and let an examiner query "find other occurrences of
this face" across the corpus — with the same offline, provenance-recorded, court-legible
discipline as the existing modalities.

Explicitly **discovery, not enrollment**: no labeled gallery. Identity hypotheses emerge from
similarity; humans confirm them. Machine output is an investigative lead, never an identification.
(The existing `SFN_REFERENCE_COLLECTION` mechanism is an enrollment-gallery pattern; face
reference collections are deliberately out of scope for this spec — they change the legal
character of the system from discovery to targeted identification and would need their own
assessment.)

### Non-goals

- Person re-identification via clothing/body/gait (rejected with reasons in our prior strategy doc).
- Demographic estimation, emotion, or any soft-biometric inference — including at runtime for
  calibration cohorts (§10).
- Automatic identity assertions in reports. The tool ranks and scores; the examiner decides.
- Real-time / live-camera operation. This is a post-hoc archive tool.
- The Gemini corpus's GPU-cluster pipeline (VPF/CV-CUDA/NVENC zero-copy). Wrong scale premise for
  us, CUDA-locks a codebase with a live ROCm switch, and `av` already decodes fast enough for
  archive processing.

---

## 2. How the research was used (and where it was overruled)

The Gemini corpus is four documents (1.md ≡ 2.md byte-identical), sharing one reference pool —
agreement between them is one source quoted four times, not corroboration. The digest
(`research/distilled/overview.md` §3) lists 19 credibility flags. Decisions taken:

| Corpus claim | Our decision | Why |
|---|---|---|
| pgvector over Qdrant (docs 4/5) | **Rejected.** Second Qdrant collection. | Verdict derived from a "personal photo collection" premise; doc 4 itself concedes Qdrant wins latency, quantization, and in-graph filtering — the one portable insight (post- vs in-graph filtering) favors Qdrant. Adding Postgres for one feature contradicts our architecture. |
| SCRFD/YOLOv12-Face detectors | **Rejected as defaults.** YuNet default; others pluggable. | SCRFD weights are research-only. The YOLO lineage is rejected on operational grounds: `ultralytics` phones home at import time (offline violation), the YOLOv12-Face WIDER numbers trace to a single community repo, and its AGPL-3.0 terms — while compatible with our GPL-3.0 — add a compliance duty for downstream redistributors we don't need to take on. YuNet is MIT (code and weights), ships in OpenCV, loads from a local ONNX file, zero network I/O. |
| ArcFace/AdaFace pretrained weights as the embedder | **Deferred to operator.** Pluggable ONNX embedder, no bundled weights. | The corpus proposes *no* permissively-licensed recognition model (researcher confirmed after full grep — licensing is mentioned once in four documents, then dismissed). Weight choice is a legal decision per deployment; see §6.3. |
| Chinese Whispers because "HDBSCAN is O(n²), hours on 100k" | **Premise unsupported**, decision open. | The O(n²) claim is unsupported as stated, but actual HDBSCAN cost at 512 dimensions (where space-partitioning trees degrade) is an open empirical question — to be measured on our data before Phase 3 commits (§14.3). Dimensionality reduction (e.g. PCA to ~64 d) is the standard mitigation to evaluate alongside. |
| Numeric thresholds (cosine 0.55–0.65, FQA > 0.65, buffer = 50, conf 0.90) | **All treated as placeholders.** | None carries a dataset, protocol, or derivation. Every operating threshold must come from our own calibration (§10) and be recorded in the face-calibration record. |
| Umeyama alignment math, 112×112 reference points | **Adopted — from primary sources.** | The five reference coordinates check out against InsightFace's `arcface_dst`. But every formula was stripped from the doc exports (empty lines where equations were), so implementation follows the Umeyama 1991 paper / `skimage.transform.SimilarityTransform`, not the corpus. |
| Track → one best crop per track; crop dilation 10–20 % + boundary clamp | **Idea adopted, mechanism replaced.** | Within-file grouping is done by embedding similarity, not motion tracking (§6.5 — IoU association is unsound at our 1 fps sampling). Dilation applies to the *review chip*, not the aligned embedder input, which is fully determined by landmarks (§7.3). |
| Doc 5 schema entities (faces ↔ media ↔ segments, labels split from hypotheses) | **Adopted as Qdrant payload + sidecar collections**, not SQL. | The shape is good; the storage engine argument is not. |
| Shadow-buffer incremental clustering (doc 4) | **Adopted as Phase 3 architecture; every parameter re-derived.** | The only design idea in the corpus we take structurally unchanged. |

Our own `docs/research/person-reid-evaluation_reevaluation.md` remains the primary design source:
quality-filter-first, multiple-high-quality-views-per-identity, and shadow-mode rollout all come
from there. Its top-K aggregation and separation-margin machinery is **deferred until an identity
grouping exists** (Phase 2/3) — review found that applying it per-file in Phase 1 pools different
people from group photos into one candidate and inverts the margin's purpose (§8).

---

## 3. Hard constraints

1. **Offline.** Isolated LAN, no internet at runtime. All models are local files configured by
   path, loaded with zero network I/O. The `insightface` Python package is banned as a dependency
   (auto-downloads to `~/.insightface` on first call); `ultralytics` likewise (update checks at
   import). `cv2.FaceDetectorYN` and `onnxruntime` take local paths and never phone home.
2. **No bundled or downloaded model weights.** ScalarForensic ships face *infrastructure* only.
   The detector model (YuNet, MIT) is small and permissively licensed but is still
   operator-fetched via the existing `scripts/download_models.py` pattern, not vendored. The
   recognition embedder is operator-supplied (§6.3) — **until the operator provides one, the face
   modality is unavailable and says so**; there is deliberately no "works out of the box"
   embedding path, because no permissively-licensed recognition model is known (§6.3).
3. **Forensically legible.** Every persisted artefact must be explainable to a reviewer: bbox,
   landmarks, detector confidence, quality subscores, model hashes, alignment version, pipeline
   configuration hash, score, threshold provenance. Anything decorative gets cut (project
   precedent).
4. **Comparability safeguards.** A face collection records full pipeline provenance and hard-gates
   on embedding-comparability parameters, exactly as image collections gate on model hashes and
   `SFN_NORMALIZE_SIZE` / `SFN_SSCD_N_CROPS` today (§7.2).
5. **Two maintainers.** No speculative generality. "Plugin" means *optional, isolated,
   removable* — not a plugin framework.

---

## 4. Plugin shape: optional modality, not a framework

A generic plugin API for a two-person project is overengineering. The "plugin thing" is realised
as an **optional modality** with four properties:

1. **Optional dependencies.** New PEP 735 dependency group `faces` in `pyproject.toml`
   (the project uses `[dependency-groups]`, not extras): `opencv-python-headless>=4.10`
   (YuNet + `warpAffine`), `onnxruntime>=1.18` (embedder). Core install unchanged.
   `THIRD_PARTY_LICENSES.md` gains entries for both. Known risk to note in the implementation:
   OpenCV wheels bundle their own FFmpeg, coexisting with our `av>=12` — the faces package must
   never use `cv2` video I/O, only its image ops (`FaceDetectorYN`, `warpAffine`, `Laplacian`).
2. **Isolated package.** All logic in `src/scalar_forensic/faces/` (decode, detect, quality,
   align, embed, group, store, provenance). Web endpoints in `web/routes/faces.py` per existing
   convention; frontend code in a new part file `web/static/js/faces.js` (computed getters in
   `computed.js`), merged by the existing property-descriptor mechanism. Core modules never
   import `faces` at module level.
3. **Graceful absence.** If the dependency group, the detector model, or the embedder model is
   missing, the web UI receives a distinct `faces_available: false` capability flag (this is
   **not** a fourth entry in `get_available_modes` — those are case-collection query modes wired
   into `_MODE_PRIORITY` and hit merging; faces are a separate flow over a separate collection),
   the CLI flag errors with an actionable message in the established style, and everything else
   runs untouched.
4. **Own collection.** Face points live in a collection derived from the case collection:
   default `{SFN_COLLECTION}_faces` (overridable via `SFN_FACE_COLLECTION`). Deriving the name
   keeps face observations case-scoped — a bare fixed name would silently mix biometric data
   across cases when only `SFN_COLLECTION` changes. The collection records its
   `case_collection` in a meta point; a mismatch at startup is a hard error.

Deleting the feature = deleting the package, the routes file, the part file, the dependency
group, and the collections. Nothing else changes.

---

## 5. Pipeline overview

```
corpus indexing — CLI only (`sfn <dir> --faces`), like all indexing today
  image ─ face decode path (§6.1, full resolution) ──┐
  video ─ existing frame slicing (frames stored as   │
          full-res JPEG q=85, indexed as images) ────┤
                                                     ▼
                detect faces (YuNet default) → bbox + 5 landmarks (canonical order) + conf
                                                     ▼
                pre-align gate: confidence, size, pose-from-landmarks (§6.2)
                                                     ▼
                align: Umeyama 5-pt similarity transform → 112×112 aligned crop (§6.4)
                                                     ▼
                post-align gate: sharpness, exposure (§6.2)
                                                     ▼
                embed: operator-configured ONNX model → 512-d L2-normalised (§6.3)
                                                     ▼
                upsert to {collection}_faces (§7.1); aligned PNG + review chip to face store (§7.3)
                                                     ▼
                per-media processed marker: counts kept/rejected-by-reason (§7.4)

query — web UI, ephemeral session, never writes to case or face collections
  probe = a stored face observation (click in results), or Phase 2: uploaded probe image
  (processed identically in session scope, never persisted)
                                                     ▼
                kNN in {collection}_faces (exact search by default, §8) + payload filters
                                                     ▼
                per-observation ranked list, duplicate-media collapsed (§8)
                                                     ▼
                examiner review; adjudications to labels sidecar (§9); query log appended (§11)
```

Phases (§12): 1 = index + face browser. 1b = calibration + cross-file search. 2 = video
grouping + probe upload. 3 = corpus clustering.

---

## 6. Components

### 6.1 Decode and detection — `faces/decode.py`, `faces/detect.py`

**Decode.** Face detection gets its own decode path, *not* `_open_rgb`: `_open_rgb` deliberately
uses JPEG `draft()` scaling against the 331 px embedding cap, which would destroy small faces.
The face decode path is EXIF-orientation-corrected, full-resolution, RGB. Video frames enter as
the stored full-resolution JPEG q=85 frames the CLI already writes — the known cost of that
re-encode on small-face recall is a calibration question (§14.2), not a new mechanism.

**Detection input scaling.** Running YuNet at native resolution of large sources is neither
tested territory for its anchors nor affordable. Detection runs at a capped size:
`SFN_FACE_DETECT_MAX_SIZE` (long side, default 1600) with aspect preserved;
`FaceDetectorYN.setInputSize` is set per image. All bboxes/landmarks are scaled back into
**oriented source-pixel coordinates** before leaving the detector; the applied scale factor is
persisted per observation. Minimum face size (§6.2) is defined in detector-input pixels, with
the source-pixel equivalent derived and stored.

**Detector contract.** `FaceDetection`: `bbox` (x, y, w, h, oriented source px), `landmarks`
(5×2 float, oriented source px, **canonical order: left eye, right eye, nose tip, left mouth
corner, right mouth corner — "left" meaning image-left, i.e. the subject's right side**),
`confidence`, `detect_scale`. Every detector adapter must reorder its native output into this
order (YuNet's native order differs); a unit test with an asymmetric synthetic landmark set must
fail if any pair is swapped — a swapped eye pair yields a plausible-looking but mirrored
alignment that silently degrades matching, the worst failure class for this tool. Channel order
is part of the contract at every boundary: decode produces RGB; the YuNet adapter converts to
BGR internally; the aligned crop is RGB; the embedder manifest declares what it consumes (§6.3).

**Default: YuNet** via `cv2.FaceDetectorYN`, model path `SFN_FACE_DETECTOR_MODEL` (no default
download). A `FaceDetector` protocol (`detect(img) -> list[FaceDetection]`, `model_hash`,
`detector_id`) leaves a seam for an operator-supplied ONNX SCRFD adapter later
(`SFN_FACE_DETECTOR=scrfd`); the seam existing is Phase 1 scope, the adapter is not.

### 6.2 Quality gate — `faces/quality.py`

The primary false-positive lever (our prior doc's central design point). Two explicit stages,
because the signals live at different pipeline points:

**Pre-alignment gate** (cheap, raw detection):
| Check | Signal | Bootstrap default (env; superseded by calibration record, §10) |
|---|---|---|
| Detector confidence | from detector | ≥ 0.8 (`SFN_FACE_MIN_CONF`) |
| Resolution | bbox min side, detector-input px | ≥ 64 (`SFN_FACE_MIN_SIZE`) |
| Pose | yaw/roll proxy from the 5 canonical landmarks (inter-eye vs eye-nose geometry) | reject strong profiles; calibrate |

**Post-alignment gate** (on the aligned crop, before embedding):
| Check | Signal | Default |
|---|---|---|
| Sharpness | Laplacian variance measured on the **native-resolution source crop** (not the 112×112 resample, whose Laplacian mostly re-encodes the resize factor) | calibrate |
| Exposure | fraction of clipped pixels in the crop | calibrate |

**Two gates, three outcomes.** The checks above decide *embedding*. A separate, lower **review
gate** decides *retention*, and runs first:

| Check | Signal | Bootstrap default |
|---|---|---|
| Detector confidence | from detector | ≥ 0.6 (`SFN_FACE_REVIEW_MIN_CONF`) |
| Resolution | bbox min side, detector-input px | ≥ 48 (`SFN_FACE_REVIEW_MIN_SIZE`) |

The review gate deliberately does **not** check pose: a face turned away from the camera is
poor material for a vector and perfectly good material for a human examiner. Both review
thresholds are clamped to never exceed their embedding counterparts (a review bar above the
embedding bar would discard faces the pipeline was about to embed); the clamp is reported at
startup, not silently applied.

Every detection therefore lands in exactly one of three outcomes:

1. **Rejected** — below the review gate. Not persisted; counted by reason on the marker.
2. **Review-only** — clears the review gate, fails the embedding gate. Persisted as a
   **vectorless point** with `embedding_status: "review_only"` and the failing check in
   `embedding_exclusion_reason`, plus a review chip. It is what an examiner looks at and is
   never compared with anything. The exclusion is structural — the point carries no vector, so
   a similarity search cannot return it — not a payload filter that a later query could forget.
3. **Embedded** — clears both. Aligned, embedded, comparable.

Subscores the embedding path never measured are persisted as `null`, never `0.0`: a review-only
face has no pose, sharpness or exposure score, and a stored `0.0` pose would read as
"perfectly frontal" rather than "never measured". The composite `quality` is likewise `null`.

The per-media processed marker (§7.4) records counts for all three, which is the auditable
trail ("this file had 14 detections, 3 embedded, 5 retained for review, 6 rejected: 4 pose,
2 exposure").

The **embedding-norm quality proxy** (pre-normalisation L2 norm, meaningful for
ArcFace/AdaFace-family models) is computed after embedding and stored as an *annotation*
(`embedding_norm`), never as a gate — it cannot gate a stage that runs before it exists. It
serves as a ranking signal (§6.5) and calibration input. MagFace as a separate FQA model stays
rejected (one passing mention in the corpus, no license/weights/method; the norm gives the same
signal free).

### 6.3 Embedding — `faces/embed.py`

- **Operator-supplied ONNX model** described by a **model manifest** (JSON next to the model
  file, path via `SFN_FACE_EMBEDDER_MODEL`, manifest at `<model>.manifest.json`): input tensor
  name, layout (NCHW/NHWC), channel order (RGB/BGR), dtype, input size (must be 112×112, §6.4),
  normalisation (mean/scale; InsightFace convention `(x−127.5)/128` as the reference example),
  output tensor name, embedding dim, batch policy. The manifest is validated against the ONNX
  session's actual I/O at load; mismatch is a hard, actionable error. This is what makes
  "arbitrary operator model" a contract instead of a hope.
- Output is L2-normalised by us; stored as float32.
- **No weights ship, and the docs must say why**: the entire ArcFace/AdaFace/InsightFace weight
  family is research-only; the research corpus offers no permissive alternative (confirmed by
  full-corpus grep). Whether research-only weights are acceptable is a per-deployment legal
  call — that decision belongs to the operator, in writing, not silently to us. `INSTALL.md`
  gains a section listing known weight sources and their license terms, including that
  "non-commercial" restrictions are *not* automatically satisfied by government use. (AuraFace
  appears in the corpus only as a bibliography entry; believed CC-BY-NC — verify before listing.)
- Model identity = SHA-256 of the ONNX file + SHA-256 of the manifest; both recorded per point
  and hard-gated by safeguards (§7.2).
- **Execution: CPU EP only in Phase 1.** `SFN_DEVICE` does **not** apply to the face embedder
  (it resolves torch device strings, not ORT execution providers; there is no PyPI ROCm ORT
  wheel, and promising GPU parity here would fork the hardware story the same way CV-CUDA was
  rejected for). A future `SFN_FACE_ORT_PROVIDERS` may expose providers for operators who build
  their own runtime. ORT intra/inter-op thread counts are pinned explicitly to avoid OpenMP
  oversubscription against torch in the same process. 112×112 CPU inference is acceptable at
  Phase 1 archive scale; measure before optimising.

### 6.4 Alignment — `faces/align.py`

- Umeyama similarity transform (rotation + isotropic scale + translation, reflection-corrected)
  from the 5 canonical landmarks onto the InsightFace 112×112 reference points
  (`(38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655),
  (70.7299, 92.2041)` — verified against `insightface.utils.face_align.arcface_dst`).
- Implemented in ~25 lines of numpy (SVD) + `cv2.warpAffine` with **pinned parameters**
  (bilinear interpolation, `BORDER_CONSTANT` black fill) — implementation reference is the
  Umeyama 1991 paper / scikit-image's `SimilarityTransform`, not the research corpus (its
  equations were stripped in export).
- Crop size is **fixed at 112×112 and not configurable** — `arcface_dst` is defined for 112×112
  and resizing it is a silent comparability break. `alignment_version: "arcface-112-v1"` is part
  of the hard comparability tuple (§7.2).
- Validation: a golden fixture generated once, offline (landmarks JSON + reference crop PNG in
  `tests/fixtures/faces/`), asserting the 2×3 matrix matches to stated tolerance and the warped
  crop matches within `max |Δ| ≤ 1` uint8 (bit-exactness across OpenCV builds is not a stable
  claim). No face test may require a network or an unshipped model; detector-dependent tests
  skip when `SFN_FACE_DETECTOR_MODEL` is absent — suite stays hermetic.

### 6.5 Within-file grouping and representatives — `faces/group.py` (Phase 2)

- Purpose is **within-file dedup and evidence quality only**, never a product feature.
- **Not a motion tracker.** At the default 1 fps sampling (and `extract_frames`' irregular,
  dedup-skipping cadence) consecutive detections of a moving subject have no bbox overlap — IoU
  association is unsound there, and Kalman prediction doesn't rescue it. Instead: agglomerative
  grouping on the **face embeddings already computed**, constrained by a maximum time gap —
  which is our prior doc's own fallback path, needs no new dependency, and is more defensible
  than motion heuristics. Payload field `group_id` (namespaced per media file:
  `{video_hash}:{n}`), plus `ts_first_ms`, `ts_last_ms`, `n_observations`.
- **Open question (Phase 2, deliberately unresolved):** how review-only observations participate
  in within-file grouping and group counts. Grouping is agglomerative over embeddings, which
  review-only faces do not have, so they can neither join a group nor be ranked against its
  members — yet excluding them silently would make a group's `n_observations` disagree with what
  an examiner sees in the same file. Resolve this when Phase 2 lands, not before.
- Per group keep **top-K representatives** (K default 5, `SFN_FACE_TOPK_PER_GROUP`) ranked by a
  monotone ranking proxy — face size × sharpness, or `embedding_norm` where the model family
  supports it — *not* by the min-composite gate score, which is a pass/fail construct dominated
  by its harshest subscore. Non-representative group members are recorded in the processed
  marker's counts, not persisted.

### 6.6 Pipeline explainer view — web UI (Phase 1)

A read-only "how this observation was produced" view, openable from any face in the browser.
Purpose is **evidentiary legibility**: the examiner — and anyone the examiner must explain the
tool to — can *see* what the pipeline did instead of trusting a prose description. This is
squarely inside the "legible to reviewers/courts" mission (contrast the removed 3-D background
viz, which explained nothing); it is also how new users learn what the tool actually does.

The view is assembled **entirely from persisted data** — no pipeline step is re-run:

1. **Source** — the original image / stored frame, with the detection bbox and the five
   landmarks drawn on it (client-side overlay from payload coordinates; the media itself is
   served by the existing file/frame endpoints, never modified).
2. **Detection** — detector id + confidence vs the `SFN_FACE_MIN_CONF` threshold *in force at
   index time* (read from `pipeline_config`, not current env — the view must describe what
   happened, not what would happen today).
3. **Pre-align gate** — size and pose subscores vs their recorded thresholds, shown pass-style.
4. **Review crop** — the dilated source-resolution chip (what a human reviews).
5. **Alignment** — the 112×112 aligned crop with the five ArcFace reference points overlaid,
   next to the landmark-annotated source: the warp becomes visually obvious.
6. **Post-align gate** — sharpness / exposure subscores vs recorded thresholds.
7. **Embedding** — stated as metadata: model name + hash, 512-d, L2-normalised, cosine
   comparison. No vector visualisation — there is nothing honest to draw.

Each step carries one plain-language sentence of explanation, fixed in the UI (same review bar
as all court-facing copy; "similar faces" language rules apply). Rejected faces cannot be
replayed (they are not persisted — §6.2); the view therefore also shows the source file's
rejection counts by reason, which is the honest statement of what was filtered out.

For a **review-only** observation the same view describes a shorter history, and every step's
pass flag is derived from the stored payload rather than assumed: the step whose check failed
is marked not-passed, and so is every step after it, because those measurements were never
taken. The embedding step is replaced by a statement that the face was not embedded, which
check excluded it, and that it is never compared with other faces. No aligned-chip link is
offered where no aligned chip exists — a URL that is a guaranteed 404 would read as a lost file
rather than an artefact that was never produced.

Implementation note: this is a frontend feature over existing endpoints (payload + chips +
media) plus at most one convenience endpoint bundling an observation's explainer data. It lives
in the `faces.js` part file like the rest of the face UI.

---

## 7. Storage

### 7.1 Collection `{SFN_COLLECTION}_faces`

One Qdrant point per persisted face observation. Vector: `face` (512-d, cosine).

**Point identity (idempotency).** Point ID = `uuid5(NAMESPACE_URL,
f"face:{image_hash}:{frame_timecode_ms or ''}:{bbox_rounded}:{alignment_version}")` — derived
from provenance, not pixels, following the deterministic-ID convention of `indexer.py` /
`tags.py`, so re-runs are idempotent and landmark jitter cannot spawn duplicates.

Payload (mirrors existing schema conventions — there is no `media_hash` in this codebase):

```
image_hash          sha256 of the source image / stored frame JPEG (joins to case collection)
image_path          as indexed
is_face             true (marker + payload index, per is_tag / is_video convention)
video_hash          sha256 of source video, when applicable
video_path          original video file as indexed, when applicable (mirrors case collection)
frame_timecode_ms   when applicable
group_id            "{video_hash}:{n}", Phase 2, else null
observation_key     "{image_hash}:{frame_timecode_ms or ''}:{bbox_rounded}" — the durable,
                    model-independent identity labels reference (§9)
bbox                [x, y, w, h] oriented source px
landmarks           5×[x, y] oriented source px, canonical order
det_conf, detect_scale
quality_*           per-check subscores; embedding_norm annotation
chip_hash           sha256 of the aligned crop's raw RGB bytes (dimension-prefixed, per
                    _frame_pixel_hash convention) — chip-store lookup key only, not identity
pipeline_config     hash + inline copy of: detector_id, detector_model_hash, embedder_model_hash,
                    embedder_model_name, manifest_hash, embedder_dim, alignment_version,
                    normalization_id, gate thresholds in force, detect max size,
                    library versions (cv2, onnxruntime), sfn_version
indexed_at
```

Payload indexes: `image_hash`, `image_path`, `video_hash`, `video_path`, `is_face`,
`group_id` (keyword); `quality` (float); `frame_timecode_ms` (integer). This mirrors the case
collection's convention (`indexer.py` indexes both hashes *and* paths as keywords) so the
examiner-facing lookups are direct filtered scrolls with no hash round-trip: "all faces from
`/evidence/cam1.mp4`" (`video_path`), "faces in this video between 00:10 and 00:30"
(`video_hash` + `frame_timecode_ms` range), "faces from this image" (`image_path` or hash).
Keyword indexes are exact full-path matches; *substring* filename search stays where it
already lives — resolve the file via the case collection's existing search, then filter faces
by the resolved hash. In-graph filtered search is exactly the Qdrant capability doc 4 conceded
pgvector lacks.

**Scale budget.** Working assumption: ~0.5–2 faces kept per corpus image after gating. At
512-d float32 (2 KB/vector + payload), 10⁵ faces ≈ 0.3 GB, 10⁶ ≈ 3 GB, 10⁷ ≈ 30 GB RAM in
default Qdrant. `{}_faces` is created with on-disk payload and scalar quantization enabled
(rescoring happens in full precision — the digest's one durable numerics point, and Qdrant's
default behaviour); revisit only if measured recall at exact-search parity degrades.

### 7.2 Safeguards

Extend the safeguard mechanism (`safeguards.py`) to the face collection, split by role:

- **Hard comparability tuple** (mismatch ⇒ refuse, mirroring `check_collection_compat`):
  `embedder_model_hash`, `manifest_hash`, `embedder_dim`, `alignment_version`,
  `normalization_id`.
- **Recorded, non-gating provenance** (mismatch ⇒ warning + display): `detector_id`,
  `detector_model_hash`, `detect_scale` cap, gate thresholds. Detector identity affects recall,
  not embedding-space comparability — hard-gating it would forbid detector upgrades and the §6.1
  pluggable seam.
- Absent fields are treated as "unknown, not mismatch", following `safeguards.py`'s existing
  stance.

### 7.3 Face store

The artefact hierarchy per kept face, from source to model input:

```
original media — never modified; referenced via image_hash / video_hash + frame_timecode_ms
  ├─ review chip     unwarped source-resolution crop, bbox dilated, high-quality JPEG   (evidence review)
  │    └─ thumbnail  downscaled copy of the review chip                                 (browse only, derived)
  └─ aligned crop    112×112 lossless PNG — the exact embedder input                    (reproducibility)
```

**Two hash domains, not one.** `chip_hash` is split into `aligned_chip_hash` (the 112×112 RGB
tensor) and `review_chip_hash` (the unwarped source crop), each computed under its own domain
prefix that also covers dtype and full array shape. The aligned PNG is addressed in the aligned
domain; *every* review artefact — JPEG and thumbnail, for embedded and review-only observations
alike — in the review domain. A review-only observation has `aligned_chip_hash: null`, so a
consumer must choose the path builder from `embedding_status` rather than assuming both exist.

One consequence is intended: because review artefacts are content-addressed on the source crop,
an embedded face and a review-only face with byte-identical crops share one JPEG on disk. That
is why purge must check whether a surviving observation still references a chip before
unlinking it (§7.5).

Three artefacts per **embedded** face and two per **review-only** face (there is no aligned
crop where there was no alignment), under `SFN_FACE_STORE_DIR` (default `data/faces/`), sharded
by the first two hex chars of the chip hash (frame-store precedent):

1. **Aligned crop, lossless PNG** of the exact 112×112 RGB tensor fed to the embedder
   (pre-normalisation). `chip_hash` is computed over these exact raw bytes, so the stored file
   authenticates the model input. This is the reproducibility artefact — nobody reviews it;
   it exists to prove what the model saw.
2. **Review chip, JPEG quality 95**: the *unwarped* source-resolution crop, bbox dilated by
   `SFN_FACE_CROP_DILATION` (default 0.25) and clamped to image bounds — hairline/ears/context
   preserved, resolution inherited from the source. This is what the examiner sees; a warped
   112×112 crop is poor review evidence. (This is where the corpus's dilation prescription
   lands — it has no effect on the aligned crop, which is fully determined by landmarks.)
   Should a proceeding require pixel-lossless review material, the payload's bbox + landmarks
   in oriented source coordinates permit an exact re-crop from the original media at any time.
3. **Browse thumbnail, JPEG** (`<chip_hash>.thumb.jpg`): the review chip downscaled to
   `SFN_FACE_THUMB_SIZE` (default 256) px on the long side. Explicitly **derived and
   non-evidentiary**: regenerable from the review chip at any time, not hashed, never cited in
   reports or logs. It exists so grids of hundreds of faces load fast in the browser; the chip
   endpoint regenerates a missing thumbnail lazily from the review chip.

**Location and relocatability.** `SFN_FACE_STORE_DIR` accepts any absolute path (same pattern
as `SFN_FRAME_STORE_DIR`): a different drive is just a different path; a different server is a
network mount (NFS/SMB) at that path — the store always logically belongs to the application
host, and clients only ever reach chips through the app host's HTTP endpoints, never by file
path or server IP. The database stores `chip_hash` only, never an absolute chip file path, and
filenames are derived from the hash — so the entire store is **relocatable**: move the
directory, update the env var, every reference still resolves, and nothing in payloads or the
audit log goes stale. (Storing per-file server/path locations is rejected for exactly that
reason: it would turn every storage migration into an evidence-payload rewrite.) For the
operational record, the meta point stores `face_store_dir` as configured plus the app host's
hostname at enablement time — informational context for case handover, explicitly not an
integrity-relevant field (it does not join the comparability tuple, and changing it later is
an audit-logged update, not an error).

**Moving the original media** is a different, app-wide concern (the case collection has the
same property): payloads store `image_path`/`video_path` as absolute paths at index time, and
the web serving layer resolves them directly. Those stored paths are *provenance* — "where the
file was when indexed" remains historically true and is never rewritten. If evidence media is
later relocated, the correct mechanism is a serve-time path-prefix remap in **core** (e.g.
`SFN_MEDIA_PATH_MAP=/old/root=/new/root`, applied in `routes/files.py`/`video.py` when
resolving stored paths for display; multiple mappings comma-separated; remapped roots join the
allowed-roots check). Payloads stay untouched; identity is carried by hashes throughout, so a
stale path degrades only preview convenience, never evidence integrity. This is deliberately
**not** part of the faces feature — faces inherit it for free once core has it — and is noted
here only so the design intent (hash = identity, path = provenance + convenience, remap =
operator-declared, config-level, never per-file) binds face artefacts too.

Setting `SFN_FACE_STORE_DIR=` (empty) disables the store; because review then depends on source
media being mounted, this is a **documented degraded-evidence mode** and every query result
produced in it is marked as such.

### 7.4 Processed markers and incremental indexing

Per-media payload-only marker point in the face collection (deterministic ID from
`image_hash`/`video_hash` + `"faces_processed"`): `faces_processed_at`, the pipeline-config
hash, `n_detected`, `n_kept`, `n_rejected` broken down by rejection reason, `n_review_only`,
`review_only_reasons` broken down by failing check, and `n_dropped_noncanonical`.

The invariant the marker must satisfy:

```
n_detected == n_kept + n_review_only + sum(n_rejected.values())
```

`n_dropped_noncanonical` counts detections discarded before any gate saw them, because the
detector returned a row that is not a canonical 5-landmark face. It is subtracted from
`n_detected` at the detector, so it is *outside* the equation above — and it must be persisted
rather than merely incremented, because a non-zero value on ordinary material is the signal for
a wrong YuNet landmark-column map. Video rollups carry the same fields summed across frames.

This makes
"legitimately zero faces" distinguishable from "never processed", gives `--faces` runs the same
skip-already-done behaviour the image pipeline gets from `get_all_indexed_hashes()`, and is the
per-file audit trail for gate decisions.

### 7.5 Deletion and retention

Face observations are **derived biometric data** and must be purgeable independently of the
evidence workflow: a CLI operation (`sfn faces purge --media <hash>` / `--all`) deletes the
observations, marker points, chip-store files, and labels referencing them, and appends the
purge to the query/event log (§11). Case closure and data-subject erasure both need this path;
it ships in Phase 1, not later.

Because chips are content-addressed and therefore shared (§7.3), purge filters the freed hashes
through a reference check before unlinking, so it cannot remove a chip that authenticates a
surviving observation. Three limits of that check are deployment properties, stated rather than
engineered away:

1. **The check is collection-scoped; the chip store is not.** `SFN_FACE_COLLECTION` is per case
   but `SFN_FACE_STORE_DIR` defaults to one `data/faces` for every case, and content-addressing
   does not stop at a collection boundary. Purging case A can unlink a chip case B still
   references. **Set `SFN_FACE_STORE_DIR` per case** — the same cross-case rule `check_compat`
   already enforces for vectors, which the chip store cannot enforce for itself.
2. **Check-then-unlink.** A concurrent index run can write a referencing point between the check
   and the `unlink()`; the file then already exists, so the writer will not recreate it and the
   reference dangles. Purge assumes a single writer.
3. Both are mitigated but not cured by chips being re-derivable from the source media.

**Stale observations on re-index.** Point IDs derive from `image_hash` + timecode + rounded bbox
+ alignment version, never from a threshold, so changing a gate rewrites a point in place and
leaves nothing behind. Two changes do leave something: a face that drops below the *review* gate
produces no point at all, and a detector change shifts bboxes onto new IDs. In both the old
point survives with its old provenance, and if it was embedded it is still in the search space
under thresholds the current run no longer applies — while the medium's marker reports counts
that disagree with it.

The face pass therefore reconciles per medium: before writing this run's points it collects any
`is_face` point for that `image_hash` the run did not produce, and at the end of the pass
**shows the operator what was found** — how many, of which kind, under which config hash — and
deletes only on explicit confirmation. Deletion routes through the same reference check as purge
so shared chips survive. Declining is recorded, not silent: `n_stale_detected` and
`n_stale_removed` are separate fields in the `index_run` record, along with the affected
observation keys, so a run where the operator said no is distinguishable from one where nothing
was stale. A non-interactive run never deletes.

Adjudications (§9) reference `observation_key`, not point IDs, so deleting a stale point cannot
destroy an examiner's decision — but the deleted observation's key stops resolving to a point.
When Phase 1b lands, reconciliation must check the labels collection and say so before deleting.

---

## 8. Query semantics (Phase 1b+)

- Probe = a stored face observation (click a face in the browser/results). Phase 2 adds uploaded
  probe images, processed identically in session scope and never persisted (the web analyze path
  is ephemeral by design and never writes to collections — indexing is CLI-only).
- **Search is exact by default** (`SearchParams(exact=True)`): at Phase 1 scale exact kNN is
  affordable, and "no further occurrences found" from an approximate index is not defensible
  testimony. ANN mode is an explicit opt-in for interactive exploration, and every logged query
  records which mode produced it.
- **Ranking unit is the individual face observation.** Phase 1b performs **no aggregation**:
  one hit per observation, ranked by raw cosine, each with full provenance. Aggregation over
  identity groups (mean of top-N pairs, the prior doc's design) enters only when an identity
  grouping exists (per-file groups in Phase 2, clusters in Phase 3) — per-file aggregation
  without grouping pools different people from group photos into one candidate.
- **Duplicate collapse**: at most one hit per distinct `image_hash`; media that are
  SSCD-near-duplicates of each other are flagged as a duplicate set in the result, so "12 hits"
  visibly reads as "1 photo, 12 copies" where that is the truth. This reuses the tool's core
  competence and is a genuine differentiator — a face matcher that knows about copy detection.
- **Thresholds**: the display threshold T applies to the pairwise cosine — the exact statistic
  the calibration record calibrates (§10). The separation margin M from the prior doc is
  **deferred to Phase 3**, where candidate *identities* exist for it to separate; per-file
  runner-up margins measure occurrence duplication, not identity ambiguity, and are not
  computed before then.
- Results display score, threshold, and calibration provenance ("cosine 0.71; display threshold
  0.62 [model abc123…, face-calibration fc-2026-09, FMR bounded ≤ 1e-3 (CI)]"). UI language is
  fixed by spec: "similar faces", never "identified person"; raw cosine under a named model,
  never percentages or "confidence".

---

## 9. Human adjudications

Examiner decisions get their own sidecar collection `{SFN_COLLECTION}_face_labels` — following
the `sfn_tags` *pattern* (payload-only points), not reusing its collection or its `Tag` model
(name-keyed positive/negative lists don't express pairwise decisions, and reuse would pollute
the triage tag UI). Adjudication record:

```
decision_id (uuid), subject_a, subject_b   observation_keys (§7.1) — durable across re-indexing
decision      same_person | different_person | uncertain
examiner_id   from SFN_EXAMINER_ID — self-asserted, not authenticated (see below)
decided_at, rationale (free text)
supersedes    decision_id | null — decisions are append-only; corrections supersede, never delete
```

Adjudications are **human assertions of record** — authored, timestamped, immutable — not
mechanically-verified ground truth; the docs use the former language. They survive re-indexing
because they reference `observation_key`, not point IDs; the resolution path from key to current
point is part of the store API. Phase 3 clustering consumes them as must-link / cannot-link
constraints, taking the latest non-superseded decision per pair.

**Examiner identity**: `SFN_EXAMINER_ID` is required whenever `SFN_FACES_ENABLED=true`; it is
stamped on every adjudication and every logged query. The application has no authentication
(intentional LAN architecture), so this is **self-asserted identity** — the docs and UI say so
plainly, and binding it to a real person is the deploying organisation's duty at the network/OS
layer. The spec claims no more than that.

---

## 10. Calibration and validation (release gate for search)

No number from the research corpus is quotable — the 0.55–0.65 corridor has a blog-tier citation
and no protocol. Face *search* (Phase 1b) is gated on a **face-calibration record** existing
(named `face_calibration` throughout — `calibration.py` already means batch-size calibration in
this codebase and must not be confused):

1. Labelled evaluation data: operator-local imagery they may lawfully use, evaluated **through
   our exact pipeline** (our decode, detector, gate, alignment, stored-frame JPEG path).
   Public sets (e.g. IJB-C) require licenses and internet and don't transfer to operator
   imagery; they are optional cross-checks, never the basis of the record.
2. **Primary evaluation is open-set identification, matching what the tool does**: FPIR/FNIR (or
   TPIR@FPIR) at stated gallery sizes, with unknown probes, under the actual search parameters
   (exact mode, duplicate collapse). Pairwise mated/non-mated DET is computed as a supplementary
   diagnostic. The calibrated statistic must be **the statistic the UI thresholds** (§8: pairwise
   cosine in 1b; aggregate statistics get their own calibration when they ship).
3. **Statistical honesty is mandatory**: the record stores pair/probe counts and confidence
   intervals; a claimable FMR/FPIR tier requires the pair counts to support it (observing
   1e-4 needs on the order of 10⁵ non-mated pairs). Where the local set cannot support the
   target, the record states the rate as **bounded, not measured**, and the UI prints the CI
   alongside T. Since the operator supplies the weights, ScalarForensic itself makes **no
   accuracy claim**; `docs/face-matching-math.md` explains the method chain generically, and a
   per-deployment method annex is generated from the active face-calibration record.
4. **Record schema and precedence**: JSON under `data/face_calibration/<id>.json` — id, date,
   dataset descriptor, pipeline-config hash it was measured under, thresholds (T, gate values),
   curves, counts, CIs. Env-var thresholds (§13) are bootstrap defaults only; when a record is
   active, **the record wins**, and any env/record disagreement is logged on every query. A
   record whose pipeline-config hash mismatches the collection is refused.
5. Shadow-mode rollout (prior doc): initially every match candidate requires examiner
   adjudication; confirmed/rejected rates as a function of score feed threshold revision.

Known, documented limits (docs + UI method notes): identical twins and close relatives are not
separable at operating thresholds (the corpus's single most important buried sentence);
demographic error-rate differentials exist in all face models — the method annex cites the
operator-chosen model's published demographic evaluation (e.g. NIST FRVT) where one exists, and
states plainly when the local calibration data cannot measure cohort effects. Cohort labels come
from dataset annotation only; the tool never infers demographics at runtime (§1 non-goal).

**DEPLOYMENT DIVERGENCE — 2026-08-12, by the maintainer's ruling.** This deployment ships
Phase 1b face search **without** an active face-calibration record, and **displays the raw
pairwise cosine**. The gate above was not met and was deliberately waived; the ruling is
recorded verbatim in `docs/specs/face-query-ux.md`. What holds instead:

- The score is presented as a raw model output, labelled as having **no confidence interval and
  no calibrated threshold**. No FMR/FPIR tier is claimed anywhere in the UI or the docs.
- The model authors' published reference figure (SFace, 0.363) may be shown, always attributed
  to the model authors and never as this deployment's threshold. It is not a default of any
  control and is applied nowhere in the search path.
- The "uncalibrated — not evidential" banner and the §11 opt-in enablement gate remain in force.
- §10.4's precedence rule is unchanged: when a record exists, **the record wins**, and the UI
  copy above is replaced by the record's T, CI and provenance.
- Nothing in §6.2's exclusion guarantee is relaxed: review-only observations remain vectorless
  and unreachable by search, on the index side and on the query side alike.

**Also diverging from §8/§12:** uploaded probes, listed as Phase 2, are pulled into Phase 1b
because the maintainer's query flow is upload-driven. The §8 constraint they carry is kept in
full — query-side faces are session-scoped, written to no collection and to no chip store.

---

## 11. Legal position, logging, and enablement

This section states **assumptions, not legal conclusions**: the feature is a post-hoc biometric
search aid operated by the deploying organisation (the deployer role); depending on jurisdiction
and use, EU AI Act high-risk obligations for post-remote biometric identification, GDPR/LED
duties (DPIA, Art. 9/10 processing), and national law (incl. authorisation regimes for
law-enforcement use) may apply, with obligations phasing in on the Act's timeline. **The
controls below support such compliance; they do not establish it.** Jurisdiction-specific legal
review before enablement is an operator duty, and `docs/deployment.md` says so. (The research
corpus invoked GDPR on every page and never once mentioned the AI Act — the gap that prompted
this section.)

Technical controls in this spec that the legal story rests on:

- **Event/query log (Phase 1 deliverable)** — append-only JSONL (`data/face_audit.log`): every
  face query (probe observation key or session-probe hash, examiner_id, timestamp, collection,
  pipeline-config hash, face-calibration id, search mode exact/ANN, thresholds, result count,
  top scores), every purge (§7.5), every enablement. Artefact provenance (§7.1) records how data
  was produced; this log records **that and how the system was used** — the thing an oversight
  body actually asks for.
- **Enablement record**: first activation with `SFN_FACES_ENABLED=true` writes a record
  (examiner_id, date, free-text authorisation reference prompted at CLI) into the collection
  meta and the event log; shown in the UI's method notes. Turning the feature on is thereby a
  recorded act, not a config drift.
- Human adjudication workflow (§9), calibration records with stated uncertainty (§10),
  provenance on every artefact (§7), purge path (§7.5), disabled-by-default shipping.

---

## 12. Phasing

- **Phase 1 — index + browse (no cross-file search):** package skeleton, decode path, YuNet
  detection, two-stage gate, alignment + golden fixtures, ONNX embedder + manifest validation,
  `{}_faces` collection + safeguards split, deterministic IDs, processed markers, chip/review/
  thumbnail store (§7.3), purge path, event log, CLI `--faces`, `faces_available` capability
  flag, face browser UI (faces of a selected result; no similarity search), pipeline explainer
  view (§6.6), docs (install/licensing/method skeleton). Behind `SFN_FACES_ENABLED` +
  enablement record. **Frontend UX is developed iteratively with the maintainer testing in the
  running UI** — backend contracts land TDD-first as usual; panel layout and interaction polish
  are explicitly expected to go through hands-on feedback rounds rather than being finalised
  from the plan alone.
- **Phase 1b — calibrate, then search:** face-calibration tooling and record schema, first
  operator calibration, cross-file "find similar faces" gated on an active record, duplicate
  collapse, shadow-mode adjudication workflow, query logging of searches.
- **Phase 2 — video grouping + probes:** embedding-based within-file grouping, top-K
  representatives, uploaded probes (session-scoped), stored-frame JPEG recall measurement
  (§14.2), group-level aggregation with its own calibrated statistic.
- **Phase 3 — corpus clustering:** shadow-buffer incremental design from doc 4 (centroids in a
  sidecar, novelty test, buffer re-clustering, merge/split with *derived* parameters, fixed
  seeds + `algorithm_version`), adjudication constraints, cluster review UI, separation-margin
  semantics over identities. Clusters are investigative aids and labelled as such.

Each phase lands via the normal workflow: TDD, hermetic tests (no network, no unshipped models;
fixture-driven), ruff, suite runs without Qdrant.

## 13. Config surface (new `SFN_FACE_*`)

All parsed eagerly in `Settings.__init__` via the existing `_parse_*` helpers, failing at
construction with actionable messages (e.g. `SFN_FACES_ENABLED=true` with a missing/unreadable
model path is a startup error, not a first-detection surprise):

`SFN_FACES_ENABLED` (bool, false) · `SFN_FACE_DETECTOR` (enum, `yunet`) ·
`SFN_FACE_DETECTOR_MODEL` (path) · `SFN_FACE_EMBEDDER_MODEL` (path; manifest expected beside
it) · `SFN_FACE_COLLECTION` (default derived `{SFN_COLLECTION}_faces`) · `SFN_FACE_STORE_DIR`
(path, `data/faces`, empty disables ⇒ degraded-evidence mode) · `SFN_FACE_DETECT_MAX_SIZE`
(int > 0, 1600) · `SFN_FACE_MIN_CONF` (float 0–1, 0.8) · `SFN_FACE_MIN_SIZE` (int > 0, 64) ·
`SFN_FACE_REVIEW_MIN_CONF` (float 0–1, 0.6; clamped to `SFN_FACE_MIN_CONF`) ·
`SFN_FACE_REVIEW_MIN_SIZE` (int > 0, 48; clamped to `SFN_FACE_MIN_SIZE`) ·
`SFN_FACE_CROP_DILATION` (float 0–0.5, 0.25; review chip only) · `SFN_FACE_THUMB_SIZE`
(int > 0, 256; browse thumbnail long side, non-evidentiary — §7.3) · `SFN_FACE_TOPK_PER_GROUP`
(int > 0, 5) · `SFN_EXAMINER_ID` (string, required when faces enabled). Gate thresholds beyond
these bootstrap values live in the face-calibration record, which supersedes env values when
active (§10.4).

## 14. Open decisions (deliberately not resolved here)

1. **Which embedder weights the first real deployment uses** — operator/legal decision; the spec
   guarantees the seam (manifest contract) and the documentation of the trade-off, nothing more.
2. Whether the stored-frame **JPEG q=85 re-encode** costs small-face recall or embedding
   fidelity (the frames are full-resolution; `SFN_FRAME_STORE_SIZE` exists in config but is
   currently applied nowhere — decide during Phase 2 whether to wire it up or delete it, and
   measure the recompression effect in calibration).
3. Phase 3 clustering algorithm (HDBSCAN vs Chinese Whispers for buffer/cold-start) — decide on
   measured behaviour on our data at 512-d (or after PCA); the corpus's complexity argument for
   CW was unsupported, so the question is genuinely open.
