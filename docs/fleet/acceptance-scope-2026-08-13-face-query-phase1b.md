# Acceptance scope — face query flow (Phase 1b, stages S3–S7)

**Written 2026-08-12 by `scalarforensic-com-m3`, BEFORE the acceptance pass was run.**
This file exists so that the scope of a passing acceptance cannot be overstated later. It is
the record; the runbook carries the narrative, this carries the boundary.

## The decision this file attaches to

`SFN_FACE_MIN_SIZE` — the **embed** floor — **stays at 64**. This was taken as a
test-material choice, not an evidential one: it changes no gate and admits no face that was
not already admitted. The separate question of **lowering** the floor is the maintainer's and
remains open (`~/.claude/cx/cto.md`, ScalarForensic decision 1).

The acceptance pass therefore uses **`danny2.jpeg`** as the query image.

## What the acceptance pass covers

- Query-side detection, gating, alignment and embedding of a face **at 147.8 px**.
- Selection, the green border, FACES as a fourth hit mode, the raw cosine under the matched
  face, the face query controls, the uncalibrated banner, the FACE audit/dist-stats pair.
- The exclusion guarantee on the query side: a review-only face is vectorless and is refused
  as a probe.

## What the acceptance pass does NOT cover — read this before citing it

**A green acceptance on `danny2.jpeg` exercises only material the 64 px embed floor already
admits. It says nothing about the small-face cohort.**

Measured (runbook, 2026-08-12):

| image | face | detection size | embed gate at `min_size` 64 |
|---|---|---|---|
| danny2.jpeg | 0 | **147.8 px** | passes → embedded, searchable |
| danny1.jpeg | 0 | 40.1 px | **excluded — size** |
| danny1.jpeg | 1 | 46.9 px | **excluded — size** |
| danny1.jpeg | 2 | 40.8 px | **excluded — size** |

So, in one sentence, and this is the sentence to quote:

> **danny2 only; the 40–47 px cohort is untested because the embed floor excludes it.**

`danny1.jpeg` appears in the acceptance pass in one role only: to demonstrate that
review-only chips are labelled not-searchable and cannot be selected. It produces **zero
searchable probes**. That is correct behaviour at the shipped floor, not a defect, and it is
also not coverage.

## Two things that must not be inferred from a green here

1. **Not evidence the pipeline handles this case's imagery.** The imagery at issue in this
   case is the 40–47 px cohort. The pass never touches it.
2. **Not a justification for lowering the floor.** "It worked on danny2" proves nothing about
   40 px. The two are deliberately not bundled: if the maintainer lowers
   `SFN_FACE_MIN_SIZE`, that decision must stand on its own measurement — including that the
   46.9 px danny1 face scores **0.6097** against danny2's, and the other two danny1 faces
   score **−0.0142** — and it needs its own acceptance pass over the newly admitted cohort.

## Standing requirement

Any report, commit message or handoff that cites this acceptance pass must carry the
one-sentence exclusion above alongside the result. A green whose scope is unstated is not a
result.

---

## UPDATE 2026-08-12 (same day): the floor was lowered, and the cohort is STILL untested

The maintainer directed the floors lowered. Shipped in `b91fd61`:

- `SFN_FACE_MIN_SIZE` (**embed** floor): **64 → 40** — the largest value admitting the
  measured 40.1 / 46.9 / 40.8 px cohort.
- `SFN_FACE_REVIEW_MIN_SIZE` (**review**/retention floor): **48 → 24**.

**The review floor was 48 in code, never 36.** `36` never existed in `config.py` (checked with
`git log -S`; the default has been 48 since `f91a5e2`). It survived only as an ad-hoc export
during an indexing run and as a MagicMock fixture, and from there it was written into the
handoff, the plan and this project's notes as though it were the deployed default. The same
pattern produced the collection name (see below).

### The lowering is CONFIG-ONLY so far. It has NOT changed any stored data.

Measured against `faces_danny_validation` before **and after** a `--faces` re-index attempt —
**identical**:

| | observations | with vector | vectorless |
|---|---|---|---|
| before | 17 | **2** | 15 |
| after | 17 | **2** | 15 |

The three danny1 faces are **still vectorless and still unsearchable**. Root cause: the
ingestion loop skips media that are already indexed *before* the face pipeline runs, so
`--faces` only processes **new** media. Lowering a gate does not retroactively vectorise
stored observations — this is precisely the case `docs/specs/stale-observation-purge.md` was
written for, and that fork is still an open maintainer decision.

**So the exclusion sentence still stands, unchanged, and for a second reason now:**

> **danny2 only; the 40–47 px cohort is untested because the embed floor excludes it.**

At the new floor the cohort would be *admitted* — but nothing has re-detected it, so no
observation of it carries a vector today. **Do not read the lowered config as evidence the
cohort is now searchable.** It is not, until the face collection is rebuilt.

### To make it real (not run — deliberately left to a fresh session)

```bash
uv run sfn-faces purge --all        # destructive; face collection only
SFN_QDRANT_URL=http://172.20.0.2:6333 SFN_COLLECTION=danny_validation \
  SFN_FACE_COLLECTION=faces_danny_validation SFN_FACES_ENABLED=true \
  SFN_EXAMINER_ID=m4v1 \
  SFN_FACE_DETECTOR_MODEL=models/face_detection_yunet_2023mar.onnx \
  SFN_FACE_EMBEDDER_MODEL=models/face_recognition_sface_2021dec.onnx \
  ./run.sh sfn analysis_test --faces
```

Then re-measure the table above; the expected result is **5 with vector, 12 vectorless**, and
a fresh acceptance pass over the newly admitted cohort — which is the pass that would actually
say something about this case's imagery.

**`SFN_FACE_COLLECTION=faces_danny_validation` is load-bearing.** `Settings` derives
`{SFN_COLLECTION}_faces` = `danny_validation_faces`, which **does not exist**. The live data is
in `faces_danny_validation`, created by another ad-hoc export. Omitting it silently creates a
second, empty collection and splits the data — the first re-index attempt aborted on the
"first face-collection activation" prompt for exactly this reason.
