# Design — Splitting the face quality gate by purpose

**Date:** 2026-08-12 · **Branch:** `feat/face-pipeline-phase1`
**Amends:** `docs/specs/face-pipeline.md` §6.2, §6.6, §7.3, §7.4, §13
**Status:** revised after independent review (Opus and Codex, 2026-08-12); awaiting maintainer approval

> **Revision note.** The first draft of this document was reviewed against the source by two
> independent reviewers. Both returned *proceed with changes*. Three of its claims were wrong
> and are corrected here: that only `align_face` resamples, that a chip-hash collision would be
> harmless, and that a `Settings` ordering check could simply raise. Its test plan gave the
> design's load-bearing guarantee no coverage at all. Those corrections are folded in below;
> the split-by-purpose decision itself survived both reviews unchanged.

## Problem

The face modality serves two distinct maintainer use cases:

1. **Hand identification.** Find faces in a medium and crop them from the original, with a
   reference back to the source, so an examiner can look at them. This wants *recall*.
2. **Embedding similarity.** Align faces to a fixed 112×112 template, embed them, and later
   compare them. This wants *precision*: upsampling a small face to fill the template
   fabricates pixels and yields an embedding dominated by interpolation.

Phase 1 serves both through a single `pre_align_gate` running before both paths. A face failing
any check is discarded outright — no observation, no chip, only a tally in the processed marker.
Protecting the embedding population therefore also destroys the review crop, which is the
artefact the first use case exists to produce.

This is not hypothetical. On `analysis_test/danny1.jpeg` the detector finds three faces at
confidence 0.930 / 0.923 / 0.912 with bbox minimum sides of 40.1 / 46.9 / 40.8 px, all with
pose far inside `max_pose=0.35`. Against the `SFN_FACE_MIN_SIZE` default of 64 all three are
rejected for size and the medium yields zero observations, despite three confident detections a
human could usefully examine. Both reviewers reproduced these figures independently against the
checked-in YuNet model.

Crucially, **the review path never upscales.** `decode.py:17` decodes at full oriented
resolution with no cap, `detect.py:82` scales bbox and landmarks back to source pixels, and
`chips.py:83` crops `source_rgb` at those native coordinates. `write_thumbnail` downsamples only
when the image exceeds the target (`chips.py:58`). Resampling does occur elsewhere in the
pipeline — the detector resizes its own input (`detect.py:72`) and thumbnails downsample — but
no step on the review path invents pixels. Only `align_face` does, and it exists to feed the
embedder. The gate is thus positioned to protect a path that does not need protecting, at the
cost of one that gains nothing from it.

## Decision

Split the gate by purpose. Detections cross two independent bars:

| | Review path | Embedding path |
|---|---|---|
| Purpose | hand identification | similarity comparison |
| Checks | confidence, size | confidence, size, pose, sharpness, exposure |
| Bootstrap defaults | `SFN_FACE_REVIEW_MIN_CONF=0.6`, `SFN_FACE_REVIEW_MIN_SIZE=48` | unchanged: 0.8 / 64 / 0.35 / 25.0 / 0.6 |
| Artefacts | review JPEG + thumbnail, native resolution | aligned 112×112 PNG + vector |
| Fabricates pixels | never | yes, by design |
| Comparable | no, and the UI says so | yes |

A detection clearing the review bar but not the embedding bar is persisted as a **review-only
observation**: browsable, croppable, explainable, and structurally excluded from similarity. A
detection failing the review bar is discarded exactly as today.

The embedding population becomes *more* conservative under this design, not less: its
thresholds are untouched and nothing new enters the vector space. What changes is that faces
excluded from comparison stop being invisible.

### Why the review floor defaults to 48, not 24

A 24 px face is roughly 31 px after dilation — too little for a human to identify from, which is
close to why the detector carries its own 0.5 confidence floor. Persisting such crops as
browsable "observations" with a per-face explainer would lend them an evidentiary presentation
their pixel content cannot support. 48 px retains all three `danny1` faces, so it costs nothing
on the actual validation set, and the maintainer can lower it deliberately after inspecting real
crops. The shipped default is what ends up in reports; it should be the defensible one.

### Threshold ordering

Review bars must be no stricter than embedding bars, or the embedding path would receive faces
the review path rejected. **This is enforced by clamping, not by raising.**

`Settings.__init__` parses the whole `SFN_FACE_*` block unconditionally — `faces_enabled` is
only consulted later, in `face_startup_error()` (`config.py:159`, `config.py:309`) — and
`Settings()` is constructed per request across the app (`routes/faces.py:53, 89, 111, 245, 278`).
A `ValueError` here would therefore 500 every route, including non-face ones, for an operator
who set `SFN_FACE_MIN_CONF=0.5` and never touched the review variable: the *default* review
value 0.6 would exceed their explicit 0.5. **A default must never invalidate a user's explicit
value.** So:

- Effective review thresholds are `min(review, embedding)` for both conf and size.
- When clamping occurs, `face_startup_error()` reports it — the established channel for face
  misconfiguration. No `warnings.warn` and no logging in `Settings.__init__`: it runs per
  request and there is no warning channel there (`grep -n warn src/scalar_forensic/config.py`
  returns nothing).
- The *clamped* values are what enter `PipelineConfig`, so provenance records what actually ran.

The detector's own `score_threshold` (0.5, `detect.py:53`, never wired to `Settings` —
`indexing.py:62` constructs the detector with the default) is a hard floor beneath both bars. A
review confidence below 0.5 has no effect; `face_startup_error()` says so rather than failing.

## Design

### Structural exclusion from search

The face collection declares a **named** vector (`store.py:95`), and vectorless points
(`vector={}`) are an established pattern — meta, marker and video-rollup points all use it
(`store.py:131`, `:198`, `:227`). Those three have never executed, though; the stronger
precedent is `indexer.py:282`, which writes vectorless points into the **case** collection,
which also has a named-vector config and has been running in production.

Review-only observations are written as points carrying no named vector. A search over
`FACE_VECTOR_NAME` cannot return them. The exclusion is a property of the storage shape, not a
payload filter a future author must remember to apply — which matters because the consumer of
that guarantee (cross-file search) does not exist yet.

Codex verified this against the pinned stack — qdrant-client 1.18.0 (`uv.lock:876`), Qdrant
server v1.17.1 (`docker-compose.yml:42`) — confirming with a direct in-memory test that a
`vector={}` point is masked out of a named-vector query. **This is verified for freshly written
points only.** See "Demotion on re-index" below for the case that is not.

Payload gains `embedding_status: "embedded" | "review_only"` for display and filtering. It is an
annotation; the vector's absence is the mechanism.

### Demotion on re-index — the case the first draft missed

`face_point_id` (`store.py:75`) derives from image hash, timecode, bbox and alignment version —
**not** `config_hash`. But `processed_hashes` (`store.py:242`) keys idempotency on an exact
`config_hash` match, so a threshold change reprocesses every medium and re-upserts **at the same
point IDs**.

Concretely: index with `min_sharpness=25`, then raise it to 40. A previously embedded face
becomes review-only, and the pipeline upserts `vector={}` over a point ID that currently holds a
vector. If that does not clear the stored vector, the result is a point whose payload reads
`review_only` and which similarity search still returns — silently, and precisely the failure
this design exists to prevent.

Qdrant's separate `update_vectors` / `delete_vectors` API implies `upsert` replaces the point
wholesale, but a load-bearing forensic guarantee does not run on an implication. **Requirement:
demotion is explicit.** When writing a review-only point whose ID may already exist, the store
calls `delete_vectors` for `FACE_VECTOR_NAME` on that ID after the upsert. The call is
idempotent and harmless when no vector was present, so it needs no read-before-write.

The reverse (review-only → embedded) is safe: the upsert supplies the vector.

### Chip identity

`chip_hash()` hashes the aligned array (`chips.py:24`) and all three filenames derive from it
(`chips.py:31`). Review-only faces have no aligned array and need their own basis. The first
draft proposed reusing the same `"{h}x{w}:"` prefix scheme in the same directory and dismissed
collisions as harmless. **That reasoning was wrong.**

The hazard is not a hash collision but a **cross-domain alias**: the same dimension-prefixed RGB
array can legitimately arise once as an aligned crop and once as a native review crop. Because
paths are selected by hash plus suffix alone, and the aligned-chip endpoint (`routes/faces.py:244`)
never loads the point, a review-only observation could be served *another* observation's aligned
PNG. Idempotent writes (`chips.py:80`) keep whichever artefact arrived first.

Requirements:

- **Domain-separate the hash inputs**: `b"aligned-rgb-v1\0" + "{h}x{w}:"` versus
  `b"review-source-rgb-v1\0" + "{h}x{w}:"`. Different domains can no longer alias.
- **Two payload fields, not one plus a discriminator**: `aligned_chip_hash` and
  `review_chip_hash`. Embedded observations carry both; review-only carry only the latter. This
  drops the overloaded `chip_hash` + `chip_kind` scheme of the first draft. `chip_kind` is not
  introduced.
- `write_chips` splits into `write_aligned_chips` (three artefacts, unchanged behaviour) and
  `write_review_chips` (review JPEG + thumbnail only).
- The aligned-chip endpoint returns 404 with an explanatory detail when an observation has no
  `aligned_chip_hash`, and must resolve the observation rather than trusting the path.
- A pre-existing wrinkle is now visible and should be recorded: review JPEG filenames currently
  derive from the *aligned* hash, though distinct source crops can align identically. Under the
  new scheme review chips key on their own content, which fixes this incidentally.

### Chip reference safety and orphans

Two chip-lifecycle problems, one pre-existing and one created by this design:

- **Shared hashes (pre-existing, widened).** `_purge_by_filter` (`store.py:300`) collects chip
  hashes from payloads and `cli.py:1743` unlinks them unconditionally. Two observations sharing
  a chip hash — far more likely for review crops, which are byte-identical across exact-duplicate
  files — means purging one breaks the other. Purge must unlink a chip only when no surviving
  observation references it.
- **Orphans on re-index (new).** Point IDs are stable while chip hashes are kind-dependent, so a
  face flipping embedded ↔ review-only overwrites its payload with different hashes, leaving the
  previous files referenced by nothing and unreachable by purge. Face imagery of real people
  surviving `purge --all` is not an acceptable failure mode. Demotion and promotion must unlink
  the chip artefacts they orphan, in the same operation that rewrites the payload.

**Review-only points carry `is_face: True`.** `purge_all` (`store.py:325`) filters on
`is_face` and `is_face_marker`; a distinct flag would let review-only observations escape both
purge and `list_faces` (`store.py:280`). This is an explicit, tested requirement.

Separately, `purge_all` omits `is_face_video_rollup` (written at `store.py:229`), so rollup
points survive a purge today. **Pre-existing bug, out of scope here, fix separately** — recorded
so the new rollup fields do not compound it.

### Pipeline flow

`process_image` (`indexing.py:128`) currently gates, aligns, then batch-embeds `kept`, consuming
`embeddings[i]` and `embedding_norms[i]` index-aligned with that list (`indexing.py:171-215`).
Revised:

1. Apply the review gate. Failures increment `rejected[reason]` and are dropped.
2. For survivors, apply the embedding gate: confidence, size, pose; then align; then sharpness
   and exposure on the native-resolution source crop.
3. Partition into `embeddable` and `review_only`. Step-2 failures are **not** counted in
   `rejected` — they are retentions, not rejections — but in `n_review_only`, with a per-reason
   breakdown, and each observation persists an `embedding_exclusion_reason`.
4. Batch-embed `embeddable` only.
5. Write chips and points for both partitions.

Two hazards both reviewers raised independently:

- **Embeddings must be paired strictly over `embeddable`.** The embeddings array has one row per
  embeddable face; enumerating any combined list against it misassigns vectors or indexes out of
  bounds. A misassignment would attach one person's vector to another's observation — the worst
  failure this modality can produce, and silent. Tests must include an interleaved-outcome
  sentinel case (review-only and embeddable alternating) asserting each vector lands on its own
  observation.
- **The early return at `indexing.py:168` must gate on both partitions being empty**, or an
  all-review-only image silently yields nothing — the exact scenario `danny1` produces.

**Degenerate crops.** `indexing.py:154` counts an empty crop as a `size` rejection after the
pre-gate passes, and `chips.py:84` silently skips the review JPEG when the dilated bbox clamps to
zero area. A review-only observation whose crop does not exist is useless — its entire reason to
exist is that crop, and the endpoint would 404. Specify: a review-path face with an empty crop is
a **rejection**, not a retention, and `write_review_chips` signals failure rather than returning a
hash for files it did not write.

Subscores already persisted on kept faces are persisted for review-only observations too, with
the post-align pair absent where alignment never ran.

### Counts, CLI and audit

`marker_point` and `video_rollup_point` gain `n_review_only: int` and
`review_only_reasons: dict[str, int]`. `n_kept` keeps its meaning — faces embedded — so nothing
already written is silently redefined. The invariant
`n_detected == n_kept + n_review_only + sum(rejected.values())` is asserted in tests.

Two consumers the first draft missed:

- **CLI summary** (`cli.py:1638`) prints `{kept} kept / {detected} detected ({n} rejected)`.
  Under the split those no longer sum, so the line would show faces vanishing. Needs
  `n_review_only`.
- **Audit record** (`cli.py:1645`) writes the `index_run` event with `n_kept` / `n_rejected`
  only. That record is spec §7.4's auditable account of the run; as-is it would **understate how
  many biometric crops were written to disk**. For a modality whose defensibility rests on its
  audit trail this is not a detail. Needs `n_review_only`.

`n_detected` is `len(detections)` (`indexing.py:139`), and `detect.py:90` drops rows failing the
canonical-landmark check *before* returning, incrementing `n_dropped_noncanonical` — a counter
persisted nowhere and consumed by nothing. The invariant holds arithmetically, but "14
detections" already excludes those drops. A document staking its case on the honest account of a
medium should persist that counter; doing so is in scope here.

### UI and explainer

The two populations must be distinguishable visibly, not only in a tooltip. Review-only
observations carry a persistent label and never appear in a context implying comparability. Copy
follows the existing review bar: face *observations*, never identifications, and review-only ones
additionally *not comparable*.

**Display resolution.** `style.css:2243` renders every chip at `width:72px; height:72px;
object-fit: cover`. The stored review JPEG is native resolution, but the browser upscales *and*
centre-crops it — worst for exactly the sub-64px faces this design adds, hiding part of the crop
and fabricating display pixels. Claiming "never upscales" while shipping that rendering is the
kind of gap a competent cross-examination finds. Review-only chips render without upscaling
(`object-fit: contain`, no forced dimensions above native size) and the full crop must remain
visible.

**The explainer hardcodes `"passed": True` on all five steps** (`routes/faces.py:128`) — possible
only because today just kept faces are stored — and emits a `chips.aligned` URL whenever a chip
hash is set (`routes/faces.py:226`), which for a review-only face points at a guaranteed 404.
Both become payload-driven. The step list gains a terminal state for review-only faces: the
embedding step shown as not performed, naming the failing check and the threshold in force at
index time.

**`/by-image`** (`store.py:280`) will now return both populations. It needs a defined contract:
embedded observations first, then review-only, each group by descending quality, with
`embedding_status` present on every entry so the UI never has to infer it from a missing field.

Explainer *visuals* — the source overlay and aligned-crop comparison of spec §6.6 items 1, 4 and
5 — remain **out of scope here** and follow as the next workstream, per the maintainer's
sequencing decision. Building the split first means the explainer handles review-only
observations from its first version rather than being retrofitted.

### Phase 1b constraint (recorded now, cheap now)

Cross-file face search must operate strictly over embedded observations. The named-vector shape
enforces this for vector search. Any future payload-scan path — duplicate collapse, grouping,
adjudication sidecars — must filter `embedding_status == "embedded"` explicitly. A review-only
face entering a comparison population is exactly the fabricated-pixel failure this design exists
to prevent.

Spec §6.5 (Phase 2 within-file grouping) additionally needs to state how review-only
observations participate in grouping and in group counts. Recorded as an open question for that
phase, not resolved here.

## Config surface

| Variable | Default | Meaning |
|---|---|---|
| `SFN_FACE_REVIEW_MIN_CONF` | `0.6` | Detector confidence floor for retaining a face for hand review |
| `SFN_FACE_REVIEW_MIN_SIZE` | `48` | bbox min side, detector-input px, for hand review |

Both parse eagerly in `Settings.__init__` per spec §13, clamped as described above, and both
enter `PipelineConfig` and therefore `config_hash`.

### Why they belong in `config_hash`

Both reviewers recommended excluding them, reasoning that `config_hash` gates reprocessing
(`store.py:242`) and that retuning a review floor should not re-run detection and embedding for a
whole case. **That recommendation is rejected, because it produces a silent no-op.**
`processed_hashes` matches markers on exact hash equality. If the review thresholds are outside
the hash, lowering the floor from 48 to 32 leaves every already-indexed medium matching its old
marker, so it is skipped and the newly admitted faces are never picked up — the setting appears
to do nothing, with no signal that it did nothing. That is worse than slow reprocessing,
particularly since retuning the floor is the loop the maintainer will actually run.

This also matches existing behaviour: `min_size`, `max_pose` and `min_sharpness` are admission
thresholds already inside the hash, so changing one already forces reprocessing today.

The review thresholds are, however, added to `_SOFT_FIELDS` (`store.py:43`) — the mechanism that
already distinguishes fields changing *which* faces get in from fields changing what the vectors
*mean*. Without this, `check_compat` (`store.py:142`) reports a stale account of the collection's
admission criteria and may raise a hard incompatibility for a change that cannot affect a single
vector.

`config_hash` changes value. It is a comparability field, and no medium has ever been indexed
with faces, so nothing is invalidated.

## Migration and preflight

The first draft asserted "no Qdrant instance has run against this branch" as a premise. That is
an operational claim about external state, not a code fact, and neither reviewer could verify it.
It becomes a **preflight check**: before implementation lands, confirm the face collection does
not exist or is empty. If it is populated, the schema changes here need a migration path and this
section must be rewritten.

Subject to that check passing, there is no migration. Every schema change — `embedding_status`,
`aligned_chip_hash` / `review_chip_hash`, the new marker counts, `embedding_exclusion_reason`,
`n_dropped_noncanonical` — is free precisely now.

The explainer workstream needs one further addition (oriented source width/height, so the client
can scale bbox and landmark coordinates onto a displayed image without depending on the browser's
EXIF handling). It is specified with that workstream but shares this window: landing both before
the first real index run avoids the second paying a migration the first did not.

## Testing

Hermetic by default — no network, no unshipped models — with one marked exception.

- Review gate: pass/fail at each boundary; clamping applied and reported when review exceeds
  embedding; an explicit embedding threshold below a review *default* does not raise.
- Partitioning: a detection set spanning all three outcomes yields the right partition, counts,
  and the count invariant.
- **Interleaved-outcome sentinel**: alternating review-only and embeddable detections, asserting
  each embedding lands on its own observation. Guards the misassignment hazard.
- The embedder is called with the embeddable subset only — a review-only face never reaches it.
- All-review-only image produces observations (guards the early-return regression).
- Degenerate crop is rejected, not retained; `write_review_chips` returns no hash for files it
  did not write.
- `review_chip_hash` and `aligned_chip_hash` are domain-separated: identical pixel arrays in the
  two domains produce different hashes.
- Aligned-chip endpoint 404s for a review-only observation; review and thumb serve both kinds.
- Purge: review-only points are deleted by `purge_all`; a chip referenced by a surviving
  observation is not unlinked; demotion unlinks its orphaned artefacts.
- Thumbnails never upscale — asserted for a sub-thumb-size review crop, since the no-upscale
  property is load-bearing for the review path's honesty.

**Integration test, opt-in and marked** (skipped by default, like the real-YuNet test):
against a live Qdrant v1.17.1, write a point with a vector, re-upsert it as review-only, and
assert a named-vector search does not return it. The first draft's hermetic test inspected a
`PointStruct` constructor, which cannot observe storage or search behaviour — leaving the
design's central guarantee with no coverage. This test is the only thing that can catch a
demotion regression.

## Out of scope

Cross-file matching and any similarity claim between media (Phase 1b, gated on a calibration
record). Explainer visuals (§6.6 items 1, 4, 5) — next workstream. The `purge_all` video-rollup
omission — pre-existing, fix separately. Recognition weights and their licensing position —
unchanged, operator's decision, `INSTALL.md` §3.

## Validation after implementation

A real `--faces` run has never executed. After this lands, index `analysis_test/danny*` against a
running Qdrant using a throwaway non-evidential embedder ONNX, and confirm: `danny2`'s 148 px face
embeds; `danny1`'s three ~40 px faces are retained as review-only with `size` recorded as the
excluding check; review crops are legible at native resolution in the UI without upscaling; the
two populations are labelled distinctly; and the audit record's counts reconcile with what is on
disk. The maintainer then judges from the actual crops whether 48 is the right review floor —
which is the evidence the Phase 1b calibration record needs and cannot be obtained by choosing a
constant now.

Note that Qdrant currently runs unpublished to the host (`docker-compose.yml:44`, deliberate);
the validation run needs a local ports override, which is itself a decision to record.
