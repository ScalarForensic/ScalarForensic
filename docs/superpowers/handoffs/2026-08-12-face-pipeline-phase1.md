# Handoff — Face Pipeline Phase 1

**Date:** 2026-08-12 · **Branch:** `feat/face-pipeline-phase1` (20 commits ahead of `main`, not pushed)
**Plan:** `docs/superpowers/plans/2026-08-11-face-pipeline-phase1.md` (execution-status block at its top)
**Spec:** `docs/specs/face-pipeline.md`

## State in one line

All 16 plan tasks are implemented and committed, one commit per task; the suite is green
(**398 passed, 0 skipped**, up from 300) and lint is clean. What remains is **human judgement on
the UI**, not unfinished code.

## Verify before you touch anything

```bash
uv run pytest -q                                  # expect 398 passed
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
```

If the face tests suddenly *skip*, the YuNet model is missing — see "Local setup" below. That
skip is not a failure, but it silently removes the only check that can catch a wrong landmark
map.

## What exists now

- `src/scalar_forensic/faces/` — isolated package: `types, decode, detect, quality, align, embed,
  provenance, chips, store, audit, indexing`. Core modules never import it at module level.
- **CLI**: `--faces` flag on `sfn` (valid on its own — the `--dino/--sscd` guard was relaxed), and
  a second console script `sfn-faces purge --media <sha256> | --all`.
- **Web**: `web/routes/faces.py` — `/api/faces/availability`, `/by-image/{hash}`,
  `/chip/{hash}[/review|/thumb]`, `/explain/{point_id}`. Registered in `app.py`; lifespan logs a
  startup warning when faces are enabled but unusable.
- **Frontend**: `web/static/js/faces.js` (registers via `window.__sfnParts`), state fields in
  `state.js`, an init call in `lifecycle.js`, panel + explainer markup in `index.html`, styles
  appended to `style.css`.
- **Docs**: INSTALL.md "Face modality (optional)", `docs/face-matching-math.md` (method chain +
  "what is NOT claimed"), THIRD_PARTY_LICENSES entries, deployment legal assumptions, two CLAUDE.md
  lines.

## Open items — start here

1. **Hands-on UI testing (the actual next step).** The plan's UX ground rule (Tasks 14 and 16) is
   deliberately half-done: first working versions landed, *no* maintainer testing yet. Expect
   layout, interaction and copy iteration from your feedback. Known gap: the explainer has **no
   bbox/landmark canvas overlay** — spec §6.6 wants the source image with the box and five points
   drawn from payload coordinates, and the aligned crop with the ArcFace reference points over it.
   The endpoint already returns everything needed (`geometry.bbox`, `geometry.landmarks`).
2. **Never run end-to-end.** A real `--faces` index pass has not executed once: it needs a running
   Qdrant *and* an operator-supplied recognition ONNX (the project ships none by design). Every
   seam is unit-tested with fakes, but no real data has moved through the pipeline. First real run
   is the highest-value validation left, and the first-activation authorization prompt has never
   been exercised interactively.
3. **Composite `quality` formula is a bootstrap.** So is every gate threshold. They are recorded in
   provenance so a Phase 1b calibration record supersedes them by name — do not let them be quoted
   as validated operating points.

## Local setup for face work

```bash
uv sync --group faces                                   # already in the dev group; CI installs it
uv run python scripts/download_models.py --yunet        # ~230 KB, checksum-verified
```

`models/` is gitignored, so the YuNet ONNX does not travel with the branch. `.env` currently has
**no** `SFN_FACE_*` entries; to actually run faces you need at minimum:

```
SFN_FACES_ENABLED=true
SFN_EXAMINER_ID=<id>
SFN_FACE_DETECTOR_MODEL=models/face_detection_yunet_2023mar.onnx
SFN_FACE_EMBEDDER_MODEL=<your .onnx, plus a sibling .onnx.manifest.json>
```

Full variable table and the recognition-weights licensing position are in INSTALL.md.

## Gotchas found this session (not obvious from the code)

- **opencv-python-headless resolved to 5.0**, not 4.x. `cv2.FaceDetectorYN.create` has the same
  signature there, verified — but the `>=4.10` floor allows a major-version jump, so re-check if
  detection behaviour ever looks odd after a sync.
- **opencv_zoo serves models via git-lfs.** `raw.githubusercontent.com` returns a ~130-byte LFS
  pointer; the fetch uses `media.githubusercontent.com` pinned to a commit, with the sha256
  verified after download. Do not "simplify" that URL.
- **Typer collapses a single-command app** into the top level, which would have made the
  invocation `sfn-faces --media …` instead of `sfn-faces purge --media …`. A no-op
  `@faces_app.callback()` keeps it a command group — that callback is load-bearing.
- **The ONNX test fixture is exported with `external_data=False`** so weights stay inside the
  `.onnx` and remain covered by `hash_file()`. A sidecar `.data` file would sit outside provenance.
- **The plan's inline code blocks are not always current.** Task 9's block contradicted its own
  interface contract (2-tuple/q=92 vs. the amended 3-artefact/q=95); Task 10's `_cfg` test helper
  omitted four required `PipelineConfig` fields. Both were corrected. Read the **Interfaces**
  section and the Amendments list as authoritative when they disagree with a snippet.
- The YuNet landmark map question is **settled**: the identity map is correct, confirmed on 10 real
  faces. `assert_canonical_landmarks` still guards it at runtime and counts drops.

## Deliberately out of scope (do not start without a decision)

Phase 1b: cross-file face search, calibration record and precedence, adjudications sidecar
(`{}_face_labels`), duplicate collapse. Phase 2: within-file grouping and
`SFN_FACE_TOPK_PER_GROUP`. Face *reference collections* are out of scope by spec — they change the
system's legal character from discovery to targeted identification.

## Merge posture

Local-only workflow: nothing pushed, no PR opened. When the UI work settles, the branch is a
straightforward merge to `main` — it touches `cli.py`, `config.py`, `web/app.py` and four static
files; everything else is new.
