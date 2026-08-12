# Feature request: face-driven query flow (Phase 1b UX)

**Raised by the maintainer, 2026-08-12.** Not designed, not planned, not started.
This document records the intent precisely enough that a fresh session can spec it
without re-deriving it from conversation.

## The maintainer's intended logic, as stated

1. The server holds indexed collections for semantic, hash and altered, **including faces**.
2. A picture is uploaded; it is checked against the server for semantic, hash, altered
   **and faces**.
3. On the **left** (query side): the query image **plus the faces detected in it**.
4. On the **right**: hits, labelled by mode — EXACT, SEMANTIC, ALTERED, **FACES**.
5. The examiner can **filter by mode** — e.g. search only for pictures containing
   similar faces.

Consequences the maintainer drew out:

- Faces must be **selectable on the query side**; only selected faces are searched.
- There must be **query controls for the face parameters**, alongside the existing
  TOP K / DINOv2 threshold controls.
- Selected query faces get a **green border**; on the match side, the matched face also
  gets a green border **with a score beneath it**.

## What already exists (verified 2026-08-12, at `b008cf9`)

- Detection, gating, alignment, embedding, chips, audit, purge, per-image face browse.
- `GET /api/faces/by-image/{image_hash}` returns the observations for an indexed image —
  this is the "faces of a hit" panel, working, with review-only observations clearly
  labelled and excluded from search.
- `GET /api/faces/explain/{point_id}` — the per-face explainer.
- Face vectors live in `{SFN_COLLECTION}_faces` under a named vector; review-only
  observations are vectorless and therefore unreachable by search (demonstrated live).

## What does not exist and is required

- **No cross-file face search of any kind.** There is no endpoint that takes a face
  vector and returns other images containing similar faces. Step 5 above is the whole
  of it, and it is unbuilt.
- **No faces on the query side.** The upload path never runs detection on the query
  image; `by-image` only serves *indexed* media. Step 3 requires detecting and embedding
  faces in an uploaded image at query time, session-scoped, not written to the collection.
- **No FACES mode** in the mode selector, the hit list, or the scoring display.
- **ALTERED is not usable in the current validation setup** — it needs an SSCD index
  pass; `danny_validation` was built with `--dino` only.

## The blocking constraint — read before planning

`docs/specs/face-pipeline.md` §10 and §12 place cross-file face search in **Phase 1b**,
and gate it on an **active face-calibration record**. The spec's position is that a
similarity score between two faces must not be shown to an examiner until the operator
has calibrated what that number means on their own data. This feature request *is*
Phase 1b, and the calibration tooling is itself unbuilt.

Specifically, the "score beneath the matched face" in step 5 is the part that cannot
simply be implemented: displaying a raw cosine as if it were meaningful is the thing
the spec forbids. SFace ships a reference same/different threshold of **0.363**, which
is a starting point and not a calibration on this operator's material.

**This is the first thing the next session must put to the maintainer:** build the
calibration record first (spec §10.4) and gate the UI on it, or build the UI first
behind an explicit "uncalibrated — not evidential" banner. That is a decision about
evidential posture, not about implementation order.

## Groundwork already measured that a plan should reuse

- SFace, Apache 2.0, 112×112 → 128-d, manifest at
  `models/face_recognition_sface_2021dec.onnx.manifest.json`
  (`RGB`, mean 0.0, scale 1.0 — determined empirically against OpenCV's own
  `FaceRecognizerSF.feature`, cosine 1.000000).
- Same face across a JPEG q=60 re-encode: **0.9898**. Different people: **0.1042**.
- A 46.9 px face matched its 147.8 px counterpart at **0.6097**, which is why
  `SFN_FACE_REVIEW_MIN_SIZE` was lowered from 48 to 36.
- Review-crop framing now `SFN_FACE_CROP_DILATION=0.25` (1.50× the detection box).

Full measurement trail: `docs/fleet/runbook.md`, entries of 2026-08-12.

---

## RULING — 2026-08-12, the maintainer, on the blocking constraint above

*"We show it without calibration. Just give us the data and then we can work with it later.
no need for hiding this, the number is just showing what the tool generated."*

**The blocking question above is answered and is not to be re-raised.** The per-face score
**is displayed**, uncalibrated. §11 makes the pre-enablement legal judgement the operator's,
and this is that judgement.

What follows from it, for whoever plans this:

1. The score is shown as a **raw model output**, labelled as having **no confidence interval
   and no calibrated threshold** (§10.3's language, inverted: the record that would supply
   them does not exist). SFace's 0.363 may appear as *the model authors' reference
   threshold*, never as this deployment's.
2. The "uncalibrated — not evidential" banner and the opt-in enablement gate **remain** —
   they were never the disputed part. The ruling is about hiding versus showing the number,
   not about dropping the surrounding honesty.
3. `docs/specs/face-pipeline.md` §10 gates search on an active calibration record. The
   deployment now diverges from that gate deliberately. **Record the divergence in §10
   itself**; do not leave the code silently contradicting the spec. §10.4's precedence rule
   still stands for the day a record exists: the record wins.
4. Nothing about the search path changes. Review-only observations remain vectorless and
   therefore unreachable by search — a structural guarantee, not a display choice.

## Second request, same message — per-model explainer buttons

*"Currently we have an 'Audit' and 'Dist Stats' button on query image. These are for DINOv2.
We should name them with DINO: Dist Stats | Audit and then do the same for our new model,
similar explaining and data stuff so it is forensically sound and explaining."*

The existing pair does not say which model it explains — tolerable with one modality,
ambiguous with two. Rename to name the model, then give the face model an equivalent pair:
the same shape of explanation and underlying data, so a reviewer can ask "what did this
model do, and on what distribution" of either modality. Independent of the ruling above:
this surface reports what the pipeline *did*, not what a score *means*.
