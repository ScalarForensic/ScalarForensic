# Design — Splitting the face quality gate by purpose

**Date:** 2026-08-12 · **Branch:** `feat/face-pipeline-phase1`
**Amends:** `docs/specs/face-pipeline.md` §6.2, §6.5, §7.3, §7.4, §13
**Status:** proposed, awaiting maintainer review

## Problem

The face modality serves two distinct maintainer use cases:

1. **Hand identification.** Find faces in a medium and crop them from the original, with a
   reference back to the source, so an examiner can look at them. This wants *recall*.
2. **Embedding similarity.** Align faces to a fixed 112×112 template, embed them, and later
   compare them. This wants *precision*, because upsampling a small face to fill the template
   fabricates pixels and produces an embedding dominated by interpolation.

Phase 1 serves both through a single `pre_align_gate` that runs before both paths. A face that
fails any check is discarded outright: no observation, no chip, only a tally entry in the
processed marker. The consequence is that protecting the embedding population also destroys the
review crop, which is the artefact the first use case actually needs.

This is not hypothetical. On `analysis_test/danny1.jpeg` the detector finds three faces at
confidence 0.93 / 0.92 / 0.91 with bbox minimum sides of 40 / 47 / 41 px. Against the
`SFN_FACE_MIN_SIZE` default of 64 all three are rejected for size, and the medium yields zero
observations — despite three confident detections a human could usefully examine.

Critically, the review artefacts do **not** fabricate pixels. `write_chips` derives the review
JPEG from the unwarped, dilated source crop at native resolution, and `write_thumbnail` never
upscales (`chips.py:51`). Only `align_face` resamples, and it exists solely to feed the
embedder. The gate is therefore positioned to protect a path that does not need protecting at
the cost of one that does not benefit from it.

## Decision

Split the gate by purpose. Detections cross two independent bars:

| | Review path | Embedding path |
|---|---|---|
| Purpose | hand identification | similarity comparison |
| Checks | confidence, size | confidence, size, pose, sharpness, exposure |
| Bootstrap defaults | `SFN_FACE_REVIEW_MIN_CONF=0.6`, `SFN_FACE_REVIEW_MIN_SIZE=24` | unchanged: 0.8 / 64 / 0.35 / 25.0 / 0.6 |
| Artefacts | review JPEG + thumbnail (native resolution) | aligned 112×112 PNG + vector |
| Fabricates pixels | never | yes, by design |
| Comparable | no, and the UI says so | yes |

A detection that clears the review bar but not the embedding bar is persisted as a
**review-only observation**: browsable, croppable, explainable, and structurally excluded from
similarity. A detection failing the review bar is discarded exactly as today.

The embedding population becomes *more* conservative under this design, not less: its
thresholds are untouched, and nothing new enters the vector space. What changes is that faces
excluded from comparison stop being invisible.

### Threshold ordering constraint

The review bars must be no stricter than the embedding bars, or the embedding path would
receive faces the review path rejected. `Settings.__init__` validates
`face_review_min_conf <= face_min_conf` and `face_review_min_size <= face_min_size`, raising
`ValueError` in the established style. Both review values remain bootstrap defaults in the
sense of spec §10 — recorded in provenance, superseded by a calibration record, never to be
quoted as validated operating points.

The detector's own `score_threshold` (0.5, `detect.py:52`) is a hard floor beneath both bars;
`SFN_FACE_REVIEW_MIN_CONF` below 0.5 has no effect. `Settings` warns rather than errors, since
the floor is a detector property and may change with a different adapter.

## Design

### Structural exclusion from search

The face collection already declares a **named** vector (`FACE_VECTOR_NAME`, `store.py:95`),
and vectorless points (`vector={}`) are an established pattern there — processed markers and
the collection meta point both use it (`store.py:131`, `store.py:198`, `store.py:227`).

Review-only observations are therefore written as points carrying no named vector. A Qdrant
search over `FACE_VECTOR_NAME` cannot return them. The exclusion is a property of the storage
shape rather than a payload filter someone must remember to apply — which matters because the
consumer of that guarantee (cross-file search) does not exist yet and will be written by
someone reading this document rather than the code.

Payload gains `embedding_status: "embedded" | "review_only"` for display and filtering. It is
an annotation; the vector's absence is the mechanism.

### Chip identity

`chip_hash()` hashes the aligned array (`chips.py:24`), and all three chip filenames derive
from it (`chip_paths`). Review-only faces have no aligned array, so they need a distinct
identity basis: `review_chip_hash(source_crop_rgb)`, hashing the dilated source crop with the
same `"{h}x{w}:"` prefix scheme.

Consequences:

- `write_chips` splits into `write_aligned_chips` (all three artefacts, unchanged) and
  `write_review_chips` (review JPEG + thumbnail only).
- Payload keeps a single `chip_hash` field, plus `chip_kind: "aligned" | "review_only"`, so
  endpoints know which artefacts exist without probing the filesystem.
- `/api/faces/chip/{hash}` (the aligned PNG) returns 404 with an explanatory detail for a
  review-only chip hash. `/review` and `/thumb` work for both.
- The two hash spaces are disjoint in practice but not by construction. They share a directory
  and a sharding scheme; a collision would mean identical bytes, so reuse is correct either
  way.

### Pipeline flow

`process_image` (`indexing.py:128`) currently gates, aligns, then batch-embeds `kept`. Revised:

1. Apply the review gate. Failures increment `rejected[reason]` and are dropped.
2. For survivors, apply the embedding gate: confidence, size and pose as today, then align, then
   sharpness and exposure on the native-resolution source crop.
3. Partition into `embeddable` and `review_only`. Failures at step 2 are **not** counted in
   `rejected` — they are retained observations, not rejections. They are counted separately in
   `n_review_only` with a per-reason breakdown.
4. Batch-embed `embeddable` only, preserving the existing single-batch call.
5. Write chips and points for both partitions.

The subscore fields already persisted on kept faces (`quality_confidence`, `quality_size`,
`quality_pose`, and the post-align pair) are persisted for review-only observations too, with
the post-align pair absent where alignment never ran. The explainer reads them to state exactly
which check excluded the face from comparison.

### Marker and rollup counts

`marker_point` and `video_rollup_point` gain `n_review_only: int` and
`review_only_reasons: dict[str, int]`. `n_kept` retains its meaning — faces embedded — so
existing readers are not silently redefined. `n_detected == n_kept + n_review_only +
sum(rejected.values())` becomes an invariant worth asserting in tests.

This changes the honest account of a medium from "14 detections, 11 rejected: 6 size, 4 pose,
1 exposure" to "14 detections, 3 comparable, 10 retained for review, 1 rejected" — a
strictly better statement, because the ten are now inspectable rather than asserted.

### UI implications

The face panel and explainer must distinguish the two populations visibly, not merely in a
tooltip. Review-only observations carry a persistent label and are never presented in any
context implying comparability. Copy follows the existing review bar: these remain face
*observations*, never identifications, and review-only ones are additionally *not comparable*.

The explainer's step list gains a terminal state for review-only faces: the embedding step is
shown as not performed, with the specific failing check and the threshold in force at index
time. This is a better artefact than the current design can produce — it shows the examiner
what was excluded from comparison and why, per face.

Detailed explainer layout, including the source overlay and aligned-crop visuals of spec §6.6,
is deliberately **out of scope here** and follows as the next workstream, per the maintainer's
sequencing decision. Building the split first means the explainer handles review-only
observations from its first version rather than being retrofitted.

### Phase 1b constraint (recorded now, cheap now)

Cross-file face search must operate strictly over embedded observations. The named-vector shape
enforces this for vector search. Any future payload-scan path — duplicate collapse, grouping,
adjudication sidecars — must filter `embedding_status == "embedded"` explicitly. A
review-only face entering a comparison population would be exactly the fabricated-pixel
failure this design exists to prevent.

## Config surface

| Variable | Default | Meaning |
|---|---|---|
| `SFN_FACE_REVIEW_MIN_CONF` | `0.6` | Detector confidence floor for retaining a face for hand review |
| `SFN_FACE_REVIEW_MIN_SIZE` | `24` | bbox min side, detector-input px, for hand review |

Both parse eagerly in `Settings.__init__` with actionable messages, per spec §13. Both enter
`PipelineConfig` and therefore `config_hash()`.

**`config_hash()` changes value.** It is a comparability field. Since no medium has ever been
indexed with faces, there is no existing data to invalidate — this is free today and a
migration after the first real run.

## Migration

None. The face collection has never been populated: no Qdrant instance has run against this
branch, and `.env` carries no `SFN_FACE_*` entries. Every schema change here —
`embedding_status`, `chip_kind`, the new marker counts — is free precisely now. This window
closes at the first real index run.

The explainer workstream needs one further schema addition (oriented source width/height, so
the client can scale bbox and landmark coordinates onto a displayed image without relying on
the browser's EXIF handling). It is specified with that workstream, but it shares this window:
if the two land separately, the second one pays a migration cost the first did not. Landing
both before the first real index run avoids that.

## Testing

Hermetic, no Qdrant, no unshipped models, per the existing suite's constraints.

- Review gate: pass/fail at each boundary; ordering validation in `Settings` raises.
- Partitioning: a detection set spanning all three outcomes (rejected, review-only, embedded)
  produces the right partition, the right counts, and the count invariant holds.
- Embedder is called with the embeddable subset only — a review-only face must never reach it.
- Review-only points carry no named vector; embedded points do.
- `review_chip_hash` is stable and dimension-sensitive; `write_review_chips` writes exactly two
  artefacts and no PNG.
- Aligned-chip endpoint returns 404 for a review-only chip hash; review and thumb endpoints
  serve both kinds.
- Thumbnails never upscale — asserted directly for a sub-thumb-size review crop, since the
  no-upscale property is load-bearing for the review path's honesty.

## Out of scope

Cross-file matching and any similarity claim between media (Phase 1b, gated on a calibration
record). Explainer visuals (§6.6 items 1, 4, 5) — next workstream. Recognition weights and
their licensing position — unchanged, operator's decision, `INSTALL.md` §3.

## Validation after implementation

A real `--faces` run has never executed. After this lands, index `analysis_test/danny*` against
a running Qdrant using a throwaway non-evidential embedder ONNX, and confirm: danny2's 148 px
face embeds; danny1's three ~40 px faces are retained as review-only with `size` recorded as
the excluding check; the review crops are legible at native resolution; and the UI labels the
two populations distinctly. The maintainer then judges from the actual crops whether 24 is the
right review floor — which is the evidence the Phase 1b calibration record needs, and cannot be
obtained by choosing a constant now.
