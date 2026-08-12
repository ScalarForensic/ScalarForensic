# Fleet runbook — ScalarForensic

## 2026-08-12 — survey window (manager `scalarforensic-com-m1`)

Survey only; no behaviour changed, no worker spawned. Report:
`docs/survey-2026-08-12-open-work.md`, committed `21f65f1`.

Measured at HEAD `b270d47` (2026-08-12T15:22:54+02:00): tree clean — 0 tracked-dirty
files, 0 untracked files (`git status --porcelain` counted both ways). `uv run pytest -q`
→ 480 collected, 475 passed, 5 skipped (all `tests/faces/test_store_integration.py`,
`SFN_TEST_QDRANT_URL not set`), exit 0. `ruff check` and `ruff format --check` on
`src tests scripts` → exit 0, 85 files formatted. No Makefile/justfile/nox/tox; the only
CI file is `.github/workflows/ci.yml`. Qdrant not running (`curl` exit 7).

Branch `feat/face-pipeline-phase1` is 42 commits ahead of `main`, 0 behind, no upstream.

Doc/code disagreements found: CLAUDE.md says 300 tests (measured 480); CLAUDE.md says the
YuNet real-model test skips (it runs — the ONNX is in `models/`); the 2026-08-11 phase-1
plan has 61 unticked boxes for work that all landed as commits.

**5 decisions escalated to the user** (ranked in §4 of the report): embedder choice for
the `danny*` validation run; merge to `main` after validation; Phase 1b vs Phase 2 next;
`purge_all()` missing video rollup points; standalone stale-observation purge command.

### Decision 1 — RESOLVED by the maintainer, 2026-08-12

**Real recognition weights, not the throwaway random-weight embedder.** Overrides the
survey's recommendation (report §4.1). Consequence: the `danny*` run becomes a real
calibration exercise, so the 48 px review-floor judgement and the `danny1.jpeg`
review-only outcome are now evidential findings rather than plumbing checks — and the
manifest's `channel_order`/`mean`/`scale` must match the model exactly, since a wrong
value produces plausible but silently wrong vectors (`faces/embed.py` validates only
shape and `input_size`, not semantics).

Blocked on the maintainer supplying `models/<model>.onnx` + `<model>.onnx.manifest.json`.
RAN `sed -n '20,45p' src/scalar_forensic/faces/embed.py`: nine required manifest fields
(`input_name, layout, channel_order, dtype, input_size, mean, scale, output_name,
embedding_dim`); `input_size` **must be 112** (`embed.py:67` raises otherwise —
arcface-112-v1 is the pinned alignment template) and `embedding_dim` is checked against
the model's real output shape (`embed.py:101`). Decisions 2–5 remain open.

### Embedder licensing research — 2026-08-12, findings

RAN web research at the maintainer's request. Material enough to change decision 1.

- **InsightFace `w600k_r50` (buffalo_l): non-commercial research only.** Verbatim from the
  project README: *"The pretrained models we provided with this library are available for
  non-commercial research purposes only."* The MIT license commonly cited covers the
  **code**, not the weights.
- **ArcFace ResNet100 (ONNX Model Zoo): Apache 2.0 label, questionable provenance.**
  OpenVINO's model docs and the HF card both say Apache 2.0 (65.1M params, 24.2 GFLOPs,
  BGR, 112×112, 512-d, LFW 99.68%). But the weights originate at deepinsight/insightface,
  whose weights are research-only — the label may be the rehosting repo's license rather
  than a valid grant. **I reported this as clean earlier in the day; that was wrong.**
- **SFace (`face_recognition_sface_2021dec.onnx`, OpenCV Zoo): Apache 2.0, per-model
  LICENSE file in the model's own directory.** MobileFaceNet + SFace loss, 112×112,
  128-d output, quoted accuracy 0.9940. Same zoo as the YuNet detector already in
  `models/`, and the recognizer OpenCV pairs YuNet with. Training data not stated —
  `INSTALL.md:239`'s provenance caveat still applies. Ruled out: Intel
  `face-reidentification-retail-0095` (128×128) and FaceNet (160×160), both refused by
  `embed.py:67`'s hard 112 requirement.

`INSTALL.md:227-229` ("no permissively-licensed model **of comparable quality**") is NOT
contradicted by the SFace finding — SFace is permissive but mobile-grade, ~1M params
against ResNet100's 65M. Doc left unchanged.

**Note on the "include both, switch by config" proposal:** already supported, zero code —
`SFN_FACE_EMBEDDER_MODEL` is a path and `embedder_model_hash` is a hard field
(`store.py:35-41`), so each model needs its own `SFN_FACE_COLLECTION` and mixing raises.
Bundling weights in-repo was declined: GPL-3.0 repo, and `INSTALL.md:236-238` deliberately
refuses a works-out-of-the-box recognition path.

### SFace adopted and verified — 2026-08-12

Maintainer chose SFace as the real candidate; a private, unpublished research fork will
compare against research-only weights separately (licences left intact, no redistribution).

**Downloaded** `models/face_recognition_sface_2021dec.onnx` (38 696 353 bytes, sha256
`0ba9fbfa…c34e79`) plus its Apache-2.0 LICENSE from the OpenCV Zoo HF mirror. `models/` is
gitignored — no weights entered the repo.

**Preprocessing determined empirically, not guessed.** RAN a probe that aligns a real face
with OpenCV's own `FaceRecognizerSF.alignCrop`, takes `FaceRecognizerSF.feature` as ground
truth, and scores 8 candidate manifests against it by cosine:

| cosine vs OpenCV reference | channel_order | mean | scale |
|---|---|---|---|
| **1.000000** | **RGB** | **0.0** | **1.0** |
| 0.956842 | BGR | 0.0 | 1.0 |
| 0.121230 | RGB | 0.0 | 255.0 |
| 0.058843 | RGB | 127.5 | 128.0 |

The 0.9568 row is the hazard in the flesh: the wrong channel order still looks plausible.
And `INSTALL.md:247`'s example manifest (the InsightFace `(x−127.5)/128` convention) scores
**0.0588** here — copying it would have produced silently wrong vectors. ONNX io read from
the file: input `data` `[1,3,112,112]`, output `fc1` `[1,128]`.

Manifest written to `models/face_recognition_sface_2021dec.onnx.manifest.json`;
`normalization_id` = `affine-0.0-1.0`, `embedding_dim` 128.

**Pipeline verified through the repo's own modules** (`load_for_detection` → `YuNetDetector`
→ `align_face` → `OnnxFaceEmbedder`):

- Same face, JPEG re-encoded at q=60: cosine **0.9898**.
- Different person (`orig_04_*.jpg`): cosine **0.1042** — SFace's own same/different
  threshold is 0.363.
- **`n_dropped_noncanonical` = 0** over both danny images: the YuNet landmark-column map is
  correct. That was the designated stop-and-investigate signal.

**Gate outcomes, measured in detector-input px** (`quality.review_gate`, defaults
review 48 px/0.6, embed 64 px/0.8; `detect_scale` = 1.000 for both images, so source px
and input px coincide here):

| file | face | size | conf | review(48) | embed(64) |
|---|---|---|---|---|---|
| danny2.jpeg | 0 | 147.8 px | 0.945 | PASS | **PASS → embedded** |
| danny1.jpeg | 0 | 40.1 px | 0.930 | **size** | size |
| danny1.jpeg | 1 | 46.9 px | 0.923 | **size** | size |
| danny1.jpeg | 2 | 40.8 px | 0.912 | **size** | size |

**The plan's prediction is wrong, and this is the calibration finding it asked for.** The
plan expected danny1's faces to be *retained for review* with reason `size`; all three are
**rejected outright** — they fall below the 48 px review floor, not merely below the 64 px
embedding floor. The plan anticipated exactly this possibility and called it the finding.

**The floor is discarding an identifiable face.** danny1 face 1 (46.9 px, thrown away by
the 48 px floor) scores cosine **0.6097** against danny2's embedded face — well above
SFace's 0.363 same-person threshold. Faces 0 and 2 score −0.0142 and 0.1125, so the model
is not scoring everything high. Caveat: identity is inferred from the score, not
independently confirmed, and this is one image with three faces.

**NOT done:** no Qdrant run, no collection, no audit record, no UI check. Those need Qdrant
started and the maintainer at the activation prompt.
