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

## 2026-08-12 — manager window OPEN (`scalarforensic-com-m2`, successor to `m1`)

Read-only so far: handoff `docs/handoffs/scalarforensic-com-m1-20260812-194005.md`, this
runbook's 2026-08-12 entries, `docs/specs/face-query-ux.md`, survey §4, spec §10. No code
touched, no worker spawned, no number re-measured (nothing below is published as mine
except the two file facts named). Ownership claimed: `docs/fleet/runbook.md` only.

State check RAN (`cx s`): `scalarforensic-com-m1` is still live at **249.5k, busy** — past
the 230k wrap-up rung, one rung below the 250k hard stop, and its row is **not** `sent`.
Its handoff is written and committed (`e5d14d5`), so nothing is lost if the ladder closes
it. Not killed by me while it is busy and unreported.

### THE BLOCKING QUESTION — put to the maintainer, ranked #1

The face-query UX request is Phase 1b. Spec §10 gates face *search* on an active
`face_calibration` record; §10.4 fixes its schema and gives it precedence over env
thresholds. The record does not exist, and the "score beneath the matched face" in the
request (`docs/specs/face-query-ux.md`, step 5) is exactly the display §10 forbids
uncalibrated. Two postures, and the choice is evidential, not technical:

- **(A) Calibration first, UI gated on the record.**
- **(B) UI first, behind an explicit "uncalibrated — not evidential" banner.**

**Recommended default: (B), with the number suppressed.** Reasons, from what is already
measured here — not re-derived:

1. §10.1 requires labelled operator-local imagery evaluated through our exact pipeline.
   The whole corpus on hand is `analysis_test` — 2 images, 6 detected faces, 2 embedded —
   and its one identity claim (46.9 px ↔ 147.8 px at 0.6097) is *inferred from the score,
   not independently confirmed*. §10.3 makes counts and CIs mandatory; observing 1e-4 needs
   ~1e5 non-mated pairs. (A) therefore blocks the feature on data acquisition the
   maintainer must do regardless, with no delivery in between.
2. §10.4 refuses a record whose pipeline-config hash mismatches the collection.
   `crop_dilation` moved 0.15 → 0.25 today and `review_min_size` 48 → 36; both sit inside
   that hash. Calibrating now would produce a record invalidated by the next config
   judgement — i.e. (A) is also *sequenced wrong* until those two settle.
3. The 1b search path is the instrument the calibration run needs. FPIR/FNIR at a stated
   gallery size (§10.2) is measured *through* cross-file face search, which is unbuilt.
   (B) builds the measuring device first; (A) asks for the measurement without it.

The recommendation's price, stated plainly: (B) puts a face-match UI in front of an
examiner before anyone has calibrated it, which is precisely the risk §10 exists to
prevent. It is only acceptable if the banner is not the sole guard. Proposed shape —
the maintainer may reject any part:

- Display **rank order and the green border, no number on the panel.** The raw cosine
  stays reachable in `/api/faces/explain/` and the audit record, labelled raw and
  uncalibrated. This is the one deviation from the request as written, and it is the
  deviation that keeps (B) inside the spec's intent rather than merely disclaiming it.
- Gate the whole path on an explicit opt-in env flag plus the examiner id already
  required for faces, so the posture is enforced by config, not by a paragraph.
- When a record later goes active, the record wins (§10.4) and the number appears with
  its CI. Nothing built under (B) is thrown away.

If the maintainer wants the number visible now, that is their call to make and it is
recorded as theirs; the banner alone then carries it.

### The other open decisions, ranked, with defaults

Survey §4.1 (embedder) is **CLOSED** — SFace, adopted and verified. Six remain open:

1. **Production `SFN_FACE_REVIEW_MIN_SIZE`?** *Default: keep 36.* Measured: 48 discards a
   face that scores 0.6097 against SFace's 0.363 threshold; 36 retains all three danny1
   faces. This is a **retention** floor, not a search floor — a review-only face is
   vectorless and cannot produce a machine match, so a lower floor adds human review, not
   false hits. Ranked first because it is an input to any later calibration record.
2. **Make `SFN_FACE_CROP_DILATION=0.25` the `config.py` default?** *Default: yes.* It is
   already in `.env`; leaving 0.15 in code means the next case silently reverts. It is in
   `config_hash`, so it reprocesses existing collections — only `danny_validation` exists,
   local, so the cost is now and near zero. Ranked second: it must settle before §10.4.
3. **`purge_all()` skips `is_face_video_rollup` points — fix before 1b?** *Default: yes.*
   A retention promise; "purge --all" leaving biometric-derived points is a courtroom
   question.
4. **Build the standalone stale-observation inspect/purge command?** *Default: yes, with 3.*
5. **Merge `feat/face-pipeline-phase1` into `main` now that validation passed?**
   *Default: yes, locally, no push.* 42 commits, 0 behind.
6. **Phase 1b or Phase 2 next?** *Default: 1b* — the maintainer's own request already
   answers this in practice; it is listed only so the ranking is complete.

Escalated to `wallet-recovery-com-g1` via `~/.claude/cx/cto.md` inbox. Nothing proceeds on
Phase 1b until the posture answer comes back; the six above are independently actionable
and item 2 is the smallest first move if the maintainer wants motion meanwhile.

### Maintainer questions, 2026-08-12 — the two gates, and why small crops look worse

**Q1 "size below threshold — reduce it for further testing?"** The label they are reading is
the **embedding** gate, not the review gate. Two different settings, and only one was
lowered today:

- `SFN_FACE_REVIEW_MIN_SIZE` — 48 → **36**, already done. It decides *retention*. All three
  danny1 faces (40.1 / 46.9 / 40.8 px) now pass it; that is why they appear in the panel.
- `SFN_FACE_MIN_SIZE` — still the **64** default (`config.py:184`). It decides *embedding*,
  i.e. whether a face gets a vector and enters search. 40–47 px is below 64, so all three
  are `review only — not comparable / size below threshold`.

To make them comparable in testing, lower `SFN_FACE_MIN_SIZE` (to ~36–40), not the review
floor. Two constraints, both verified in code, not remembered:
`config.py:205` clamps `review_min_size` to ≤ `face_min_size`, so the embed floor can never
sit below the review floor; and `min_size` is a hashed field of `PipelineConfig`
(`provenance.py:25`, `_VERSION_INFO_FIELDS` excludes only the version fields), so changing
it changes `config_hash` and reprocesses every medium — same trap as `crop_dilation`.

**This is the gate with evidential consequence.** Lowering the review floor only adds human
review; a review-only point is vectorless and cannot produce a machine match. Lowering the
embed floor puts ~40 px faces *into search*. The one measurement pointing at it is the
46.9 px face scoring 0.6097 vs SFace's 0.363 — and the other two danny1 faces score −0.0142
and 0.1125 against the same probe, so the model is not indiscriminate at that size. One
image, three faces, identity inferred from score: enough to justify a test run, not enough
to set a production floor. Recommend lowering it in a **separate throwaway collection**
(`SFN_FACE_COLLECTION`), leaving `faces_danny_validation` intact as the 480/0 evidence.

**Q2 "the observation faces look worse quality than the faces in the real pictures — why?"**
They carry fewer pixels, and nothing in the pipeline degrades them. RAN `PIL` over the chip
store:

| observation | review crop on disk |
|---|---|
| danny2 face 0 (147.8 px detection) | **222 × 315** |
| danny1 faces (40–47 px detection) | **53 × 64** and **61 × 80** (0.15-era), **71 × 92** at 0.25 |

A ~4× linear difference, straight from the source image. The chips are JPEG **quality 95**
(`chips.py:27`), and `style.css:2254` deliberately refuses to upscale a review-only chip
(`width:auto; max-width:72px; object-fit:contain; image-rendering:pixelated`) — the comment
above it says why: upscaling would make the spec's "native resolution" claim false. So the
panel is showing the truth. Opening a 61×80 crop full-screen in a new tab is the browser
upscaling it; that softness is the browser's, not the store's. Embedded chips take
`object-fit:cover` into a fixed 72×72, so they are *down*-scaled — which is the other half
of why the two populations look different side by side.

Nothing changed in this window; both answers are read-only findings.

### Decision triage, 2026-08-12 — three answered at manager level, two are the user's

CTO (`wallet-recovery-com-g1`) verified the §10 premise independently rather than relaying
it, and returned one **refinement I adopt**: §10 does not *literally* forbid displaying a
per-face score — it forbids displaying one **with no CI and no defensible threshold**,
which before calibration is the same thing in practice. The precise wording survives
challenge; the loose one does not. The runbook entry above should be read with that
correction. Also from the CTO: evidential posture is the **user's** call, not the CTO's
(§11 makes jurisdiction-specific legal review an operator duty), and the standing argument
against my default (B) is that **a banner is a weaker control than a gate — a banner is
what gets cropped out of a screenshot**. My answer to that is why (B) as I specified it is
not banner-only: the number is suppressed and the path is behind an opt-in env flag, and
§10.5's shadow mode (every candidate examiner-adjudicated before thresholds are trusted) is
the spec's own precedent for human-adjudicated use ahead of trusted thresholds.

**Answered by me (manager), not sent up:**

- **(2) `SFN_FACE_CROP_DILATION` default → 0.25.** The maintainer already authorised the
  value and it is in `.env`; aligning `config.py` is hygiene, not a new judgement. Assigned.
- **(3) `purge_all()` missing `is_face_video_rollup`.** A bug against a stated retention
  promise, not a scope question — fix now, before 1b. Verified again at HEAD:
  `store.py:474-486` matches `is_face` and `is_face_marker` only; the flag is set at
  `store.py:251`. Assigned.
- **(6) Phase 1b vs Phase 2.** Closed — the maintainer's own feature request answers it.

**Deferred by me, deliberately:** (4) the standalone stale-observation inspect/purge command
is the least specified of the six and adds a user-facing CLI surface; it is not batched with
(2) and (3) until its shape is written down.

**Still the user's, and only theirs:** the Phase 1b posture (ranked #1 above);
**(1) production `SFN_FACE_REVIEW_MIN_SIZE`** — a retention policy, default keep 36; and
**(5) merging `feat/face-pipeline-phase1` into `main`** — 42 commits, 0 behind, local merge
only, never a push.

**Worker spawned:** `scalarforensic-com-c1` (first worker in this project), owning
`config.py`, `faces/store.py`, `tests/faces/test_store.py`, `tests/test_config.py`, tasked
with (2) and (3) test-first and explicitly fenced off from Phase 1b, `face_min_size` and
`face_review_min_size`. It reports its own measured pytest line; I do not hand it mine.

### `scalarforensic-com-c1` DONE — both items landed; one misattribution corrected

Worker reported and I re-derived rather than relayed. Verified at HEAD `8dc2b65`:

- `bb39d0e` **fix(faces): purge --all now matches video rollup points** — one line at
  `store.py:484` adding the `is_face_video_rollup` should-clause, plus 14 test lines
  (`test_purge_all_matches_video_rollup_points`). `is_face_meta` correctly still excluded:
  the enablement record must survive a routine purge.
- `479d046` **default `SFN_FACE_CROP_DILATION` 0.15 → 0.25** — `config.py:224`, plus
  `docs/specs/face-pipeline.md:460,740`. Worth keeping: the worker found the **old pin
  could not fail** — `.env` puts 0.25 in the process env and a missing env file makes
  `load_dotenv` fall back to `find_dotenv()`, so the new test
  (`tests/faces/test_config.py::test_crop_dilation_default_is_pinned_without_env`) passes an
  empty env file explicitly. A default-pinning test that reads the operator's `.env` pins
  nothing.

**Bar re-measured by me, not quoted from the worker:** `uv run pytest -q` →
**478 passed, 5 skipped**, 14 warnings, 7.87 s. `ruff check` → `All checks passed!` rc=0;
`ruff format --check` → `85 files already formatted` rc=0. `git status --porcelain` → 0.
Consistent with the worker's own line.

**Correction to a CTO instruction.** The CTO ordered c1 pulled off the purge item on the
grounds that "m1 landed it while your window was open" and that c1's test might collide with
the 14 lines already in `tests/faces/test_store.py`. The code state it read was real; the
attribution was not. RAN `git log --format='%h %aI %s'` and `git show bb39d0e`: that commit
is **c1's own**, 21:46:57, carrying the `[scalarforensic-com-c1]` trailer, and the 14 test
lines are the same 14 lines. `m1` landed nothing here. There was no duplicate to prevent and
no re-scope to perform — c1 had finished both items and was `idle` before the order arrived.

The CTO's general rule is still worth keeping and is the reason this was checked rather than
obeyed: **a re-verification is only as fresh as its read**, and in a shared tree that is
minutes. The corollary this episode adds: when a fact goes stale, check *who* changed it
before acting — the freshest reading of a shared tree still needs authorship to interpret.

c1 left alive at 57.6k, warm, no work assigned; item (4) stays deferred and unassigned.

### c1 re-tasked rather than retired — item (4), design note only

CTO accepted the correction (`docs/CTO_LEDGER.md` `dd7e418`, under its own name) and named
the underlying defect precisely enough to be worth quoting: *it verified the code state and
**inferred** the author, then built an order on the inferred half — and the attribution was
the part the order rested on.* Verifying the load-bearing half of a claim and inferring the
rest reads exactly like having verified it. Two standing rules out of this window:

1. **When a fact goes stale, check who changed it before acting** — `git log -1 --format=%an`
   plus the commit trailer. A CTO order is not evidence.
2. **A default-pinning test that reads the operator's `.env` pins nothing** —
   `load_dotenv` falls back to `find_dotenv()`, so the green is satisfied by the wrong
   condition. Supply an empty env file. Config-shaped, and probably not unique to the one
   test c1 found; nobody has swept the rest.

**c1 not parked idle and not retired.** Given item (4) as a **design note, no code**:
`docs/specs/stale-observation-purge.md`. The reason it goes to c1 rather than to a fresh
session is context shape — c1 has just been inside `purge_all` and the chip-hash reclaim
path for `bb39d0e`, which is exactly the code this command sits on. The reason it is a doc
and not an implementation is the same reason I deferred it: it adds user-facing CLI surface
nobody has approved, and the honest unblocking step is a concrete proposal the maintainer
can reject in one read. It is fenced off Phase 1b and both size gates, and told to stop at
the doc.

**Still with the user, nothing proceeding on them:** Phase 1b posture; production
`SFN_FACE_REVIEW_MIN_SIZE` (default keep 36); merge to `main` (local, no push).

### `.env` pin sweep — 0 tests flip, and my first discriminator was broken

Ran the sweep the CTO sized. Its bound (≤6 candidates from an 8-file superset) was not used
as the answer; the set was derived from the mechanism instead.

**Mechanism, read at `config.py:33-36`:** `Settings.__init__` resolves `Path(".env")` and,
only if that misses, calls `load_dotenv(None)` — whose python-dotenv fallback walks up from
`config.py`'s own directory and reaches the same operator `.env` anyway. `override=False`,
so a test that sets the var explicitly is unaffected. Only the **13 keys present in `.env`**
can leak at all.

**My first discriminator blocked nothing, and the run that "passed" proved nothing.** It
disabled only the `path is None` fallback — but pytest runs from the repo root, where the
direct `Path(".env")` branch hits first, so `load_dotenv` was still reading the operator's
file. The 478/5 it returned was indistinguishable from a real result. Caught by a positive
control, not by the suite: `Settings()` returned `batch_size 128` under both arms. **A
discriminator that has not been shown to move something is not evidence.** Corrected version
blocks the repo `.env` *by resolved path*, leaving a test-supplied env file working; the
control then moves — `batch_size 128 → None`, `normalize_size 512 → 224`.

**Result with the working discriminator:** `478 passed, 5 skipped, 14 warnings` — identical
to the baseline. **Zero tests flip.** Nothing to fix; nothing was fixed. c1's `479d046` was
the only live instance.

**The blind spot this method has, stated because the green does not state it.** A verdict
flip only appears where the operator's value differs from the code default. RAN a full
`vars(Settings())` diff across both arms: **5 attrs move** (`model_dino`, `model_sscd`,
`normalize_size`, `batch_size`, `input_dir`); the other **8** `.env` keys currently *equal*
their code defaults — `SFN_COLLECTION_DINO`, `SFN_COLLECTION_SSCD`, `SFN_DEVICE`,
`SFN_DUPLICATE_CHECK_MODE`, `SFN_EXTRACT_EXIF`, `SFN_FACE_CROP_DILATION`, `SFN_QDRANT_URL`,
`SFN_THUMBNAIL_DIR`. A test asserting one of those defaults passes under both arms and still
pins nothing — it is the same defect, invisible to this sweep, and it is exactly how the
dilation test survived (0.25 in `.env` and 0.25 in code after `479d046`). The durable fix is
c1's shape: supply an empty env file, do not rely on the value differing.

Scratch plugin (not in the repo): `scratchpad/noenvplug.py`.
