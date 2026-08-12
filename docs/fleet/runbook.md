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

### Validation run EXECUTED — 2026-08-12, SFace + `analysis_test`

First end-to-end run of the face pipeline. Qdrant was already up (container
`scalarforensic-qdrant-1`, qdrant/qdrant:v1.17.1) but publishes no host port; reached
directly on the container IP via `SFN_QDRANT_URL=http://172.20.0.2:6333`, so **no
`docker-compose.override.yml` was created** and nothing was written into the working tree.

Config: examiner `m4v1`, `SFN_FACE_REVIEW_MIN_SIZE=36` (lowered from 48 on the maintainer's
authorisation, after the 48 px floor was measured to reject all three danny1 faces),
embed floor unchanged at 64/0.8, collection `faces_danny_validation`, store
`data/faces/danny_validation`.

**CLI summary:** `6 detected │ 2 comparable │ 3 retained for review (3 size) │ 1 rejected:
1 confidence`. Reconciles: 6 = 2 + 3 + 1.

**Audit record** (`data/faces/face_audit.log`): `enablement` written once with examiner id
and the authorization reference; one `index_run` with `n_detected=6, n_kept=2,
n_review_only=3, rejected{confidence:1}, review_only_reasons{size:3},
n_dropped_noncanonical=0, n_stale_detected=0`. Reconciles against the CLI summary.

**Qdrant census** (18 points): 1 meta + 12 markers + 5 faces = **2 face points WITH a
vector, 3 vectorless**. The vectorless three are exactly the review-only observations.

**On disk:** 2 aligned PNGs (one per embedded face) and 10 JPEGs (review crop + thumbnail
for each of the 5 retained faces). Counts match the point census exactly.

**Payload spot-check** on a review-only point: `embedding_exclusion_reason="size"`,
`quality_size=46.89`, `aligned_chip_hash=null`, `embedding_norm=null`, full provenance
present (`embedder_model_hash`, `manifest_hash`, `normalization_id="affine-0.0-1.0"`,
`review_min_size=36`). (The field is `embedding_exclusion_reason`, not
`exclusion_reason` — an earlier guess at the name was wrong; there is no defect.)

**Exclusion guarantee demonstrated live.** Querying the collection with danny2's own
embedded vector returns **2 points, both `embedded`** (self at 1.0000, `orig_04` at
0.1042). Zero review-only points appear, while all 3 remain retrievable by id. The
strength of this test: the 46.9 px review-only face scores **0.6097** against that exact
probe — far above SFace's 0.363 same-person threshold — so it is a face that would
certainly have matched had it been searchable. It is retained and invisible to search,
which is the design guarantee.

**Test suite with a live server:** `SFN_TEST_QDRANT_URL=http://172.20.0.2:6333 uv run
pytest -q` → **480 passed, 0 skipped**. This morning's bar (475 passed / 5 skipped) was
skip-limited only by the absent server.

**Still outstanding:** Task 10 Step 4, the visual UI check — review crops legible at
native resolution, the two populations distinguishable, explainer naming the failing
check. Needs a human looking at `./run.sh sfn-web`. The production review floor is still
an open judgement; the measured input is that 48 px discards an identifiable face (46.9 px
→ 0.6097) and 36 px retains all three.

### UI check performed — 2026-08-12. One real defect found.

Task 10 Step 4 (the manual UI check the plan could not automate) was carried out against
`./run.sh sfn-web`. Two prerequisites the plan does not mention: the case collection must
actually be indexed (`--faces` alone writes no image vectors, so there is no hit to select
— `danny_validation` had to be built with `--dino`), and **`SFN_INPUT_DIR` must be set**
or `/api/metadata` returns 403 and the hit panel shows no hashes.

**DEFECT: the face panel never loads for the initially-selected hit.** `index.html:933`
wires the panel as `x-init="$watch('selectedHit', h => loadFacesForHit(h?.image_hash))"`.
The watcher fires only on *change*; `selectedHit` is already set when the panel mounts, so
in the ordinary flow no `/api/faces/by-image/` request is ever issued and the panel reads
**"0 in this image"** for every image. Evidence, all RAN in the live page:

- `list_network_requests` over two full page loads: **zero** `/api/faces/by-image/` calls,
  including after clicking the hit row.
- In-page state at that moment: `facesAvailable=true`,
  `selectedHit.image_hash="ca4bed…"` (correct), `facesForHit.length=0`.
- Calling `loadFacesForHit(selectedHit.image_hash)` by hand returns **3** observations,
  all `review_only/size` — so API, payload and rendering are all correct.
- Forcing the watcher (`selectedHit = null`, then restore) loads **1** face on danny2.
  The watcher works; it simply never sees the initial value.

Impact: with a single hit — the common case — the face browser is unreachable through the
UI. Task 10's substituted static checks (`node --check`, TestClient smoke, static wiring
assertions) could not have caught this; only the manual check could, exactly as the
handoff anticipated. **Not fixed — this window changes no code.** Likely one line: invoke
`loadFacesForHit` once in `x-init` alongside registering the watcher.

**Once loaded the panel is correct:** "3 in this image", each observation labelled
`review only — not comparable` / `size below threshold` with its confidence, thumbnails
from the review hash domain, full-resolution review crop on click, an explainer button per
face, and the standing disclaimer "Machine-detected face observations — an investigative
lead, not an identification." Both populations are distinguishable.

### Crop framing — `SFN_FACE_CROP_DILATION` 0.15 → 0.25

Maintainer observed from the rendered panel that the crops clip the tops of heads. RAN a
re-index at 0.25: review crops go from **1.30× to 1.50×** the detection box (e.g. the
47×62 bbox: 61×80 → 71×92 px). Visual comparison confirms hair and chin are no longer cut.
`crop_dilation` is inside `PipelineConfig`, so it is part of `config_hash`
(`provenance.py:37-43`) — the markers changed and all 12 media were reprocessed, no trap.
Counts unchanged: `6 detected │ 2 comparable │ 3 review (3 size) │ 1 rejected`.
Aligned PNGs still 2 and byte-identical in hash, confirming the dilation is review-only and
no vector moved. The 10 superseded review JPEGs are now unreferenced on disk — that is what
`unreferenced_chip_hashes` exists to reclaim on purge.

**Open decision this raises:** whether 0.25 becomes the default (currently 0.15 in
`config.py`).

### Panel defect FIXED — 2026-08-12, `b008cf9`

Test-first: added `test_faces_panel_loads_for_the_already_selected_hit` to
`tests/faces/test_static_wiring.py`, confirmed it FAILED against the old markup, then
changed `index.html:935` to `x-init="loadFacesForHit(selectedHit?.image_hash); $watch(...)"`.
Suite **476 passed, 5 skipped** (no live Qdrant in that run); ruff check + format rc=0.

**Diagnosis gotcha worth keeping.** After the fix the panel still read "0 in this image",
and I nearly re-opened the diagnosis. The page was serving **cached HTML**: reading the
live attribute showed `x-init="$watch('selectedHit', …)"` — the old markup — while
`curl http://localhost:8080/` returned the new one. `ignoreCache` on the navigation did
not defeat it; a query-string cache-bust (`/?cachebust=1`) did, and the panel then loaded
on first render with no click. The app serves no cache headers on `/`. Anyone testing UI
changes here must cache-bust or they will debug a stale page.

### Crop dilation recorded in config — 2026-08-12

`SFN_FACE_CROP_DILATION=0.25` appended to `.env` (gitignored, local config) with a comment
carrying the measurement (47×62 box → 61×80 at 0.15, → 71×92 at 0.25) and the warning that
the value is inside the pipeline config hash, so changing it reprocesses every medium.
**Still open:** whether 0.25 replaces the 0.15 default in `config.py`.

### New feature request — face-driven query flow

Maintainer specified a face-search UX: query-side face detection with selectable faces,
FACES as a first-class hit mode alongside EXACT/SEMANTIC/ALTERED, mode filtering, face
query controls, green borders on selected and matched faces with a score beneath.
Written up as `docs/specs/face-query-ux.md`. **This is Phase 1b**, which spec §10/§12 gate
on an active calibration record — the "score beneath the matched face" is precisely what
the spec forbids showing uncalibrated. First question for the next session is that
evidential-posture decision, not implementation order.
