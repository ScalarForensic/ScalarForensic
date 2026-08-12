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
