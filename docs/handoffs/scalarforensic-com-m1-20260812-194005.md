# Handoff — scalarforensic-com-m1 → successor manager

**Written:** 2026-08-12 19:40:05 UTC. **Branch:** `feat/face-pipeline-phase1` (local only,
never pushed). **HEAD at handoff:** `bddd382`.

## 1. What I was asked to do

Survey the project for open work and surface the decisions needing the maintainer — which
then grew, at the maintainer's direction, into choosing and validating a face embedder,
executing the outstanding validation run, and fixing what the UI check found.

## 2. What is done

**Survey** — `docs/survey-2026-08-12-open-work.md` (`21f65f1`). Ground truth, doc/code
disagreements, six open work items, five ranked decisions.

**Embedder chosen and verified.** SFace (OpenCV Zoo, Apache 2.0, 112×112 → 128-d) at
`models/face_recognition_sface_2021dec.onnx` (gitignored). Manifest determined
**empirically** against OpenCV's own `FaceRecognizerSF.feature`: `RGB`, mean 0.0,
scale 1.0 → cosine **1.000000**. Wrong channel order still scores 0.9568 and
`INSTALL.md:247`'s example convention scores 0.0588 — do not copy a manifest, measure it.

**Validation run executed** — the last outstanding item of the gate-split plan.
`6 detected │ 2 comparable │ 3 review (3 size) │ 1 rejected: confidence`, reconciling
against the audit record and disk. `n_dropped_noncanonical = 0`. Exclusion guarantee shown
live: search with danny2's own vector returns only the 2 embedded points, while the 46.9 px
review-only face that scores **0.6097** against that probe stays retrievable by id and
invisible to search. With a live server the suite is **480 passed, 0 skipped**.

**Calibration finding.** The 48 px review floor rejected all three danny1 faces (40.1 /
46.9 / 40.8 px). Lowered to **36** on the maintainer's authorisation. The discarded 46.9 px
face is identifiable (0.6097 vs SFace's 0.363 threshold) — that is the evidence for the
floor, and the production value is still an open judgement.

**UI defect found and fixed** (`b008cf9`, test-first). `index.html` registered the face
panel loader only as `$watch('selectedHit', …)`; the value is already set when the panel
mounts, so no `/api/faces/by-image/` request ever fired and every image read "0 in this
image". With a single hit the face browser was unreachable. Verified fixed in the live UI.

**Crop framing.** `SFN_FACE_CROP_DILATION=0.25` in `.env` with the measurement in a comment
(47×62 box → 61×80 at 0.15, → 71×92 at 0.25). Review chip only; aligned chips unchanged.

**Not done:** decisions 2–5 of the survey remain open; the production review floor is
unjudged; `purge_all()` still skips `is_face_video_rollup` points; the standalone
stale-observation purge command is unbuilt.

## 3. The next concrete step

Take the maintainer's face-query UX request — `docs/specs/face-query-ux.md` (`bddd382`) —
and **put its blocking question to them before planning anything**: what they described is
Phase 1b, which spec §10/§12 gate on a calibration record that does not exist, and the
per-face score they want displayed is precisely what the spec forbids showing uncalibrated.
Calibration first and gate the UI on it, or UI first behind an explicit "uncalibrated — not
evidential" banner. That is an evidential-posture decision, theirs alone.

## 4. Written down nowhere else

- **The app sends no cache headers on `/`, and `ignoreCache` does not defeat the browser
  cache.** After fixing the panel I read "0 in this image" from a stale page and nearly
  re-opened a closed diagnosis; the live DOM had the old `x-init` while `curl` returned the
  new one. Use `/?cachebust=N`. This will bite the next person testing UI changes.
- **Two prerequisites the validation runbook omits:** `--faces` alone writes no image
  vectors, so there is no hit to select and the face panel is unreachable — the case
  collection needs a real `--dino` pass; and `SFN_INPUT_DIR` must be set or `/api/metadata`
  returns 403 and hits show no hashes.
- **Qdrant is up but publishes no host port.** Reach it at `http://172.20.0.2:6333`
  (container `scalarforensic-qdrant-1`); no `docker-compose.override.yml` is needed and none
  was created.
- **Licensing, settled by measurement not memory.** InsightFace `w600k_r50` is
  research-only (weights, not code). ArcFace R100's Apache-2.0 label is likely a rehosting
  artefact over insightface weights — **I reported it as clean earlier and was wrong.**
  SFace's grant comes from the author contributing it into the zoo, which is stronger, but
  its upstream repo carries no license file at all. `INSTALL.md:227-229` survives because of
  its "of comparable quality" qualifier; leave it alone.
- **`CLAUDE.md` says 300 tests; 480 are collected.** It also says the YuNet real-model test
  skips — it runs, the ONNX is in `models/`. Both stale, both unfixed.
- The maintainer plans a **private, unpublished research fork** to compare research-only
  weights. Licences stay intact, nothing is redistributed, nothing is rebranded.

## 5. Ownership map, live agents, open escalations

- **Ownership: none.** I claimed no `cx o` rows; `/home/user01/Schreibtisch/gitea/ScalarForensic`
  reports `unowned; unlocked`. Nothing to release.
- **Live agents under me: none.** I spawned no workers in this window.
- **Open escalations:** five decisions in survey §4, plus three added since — make 0.25 the
  `SFN_FACE_CROP_DILATION` default? what production review floor? and the Phase 1b
  posture question in §3 above, which ranks first. All are recorded in
  `~/.claude/cx/cto.md`'s inbox for `wallet-recovery-com-g1`.
  **CORRECTION (appended 2026-08-12, m2 flagged it and is right):** this section originally
  said the project had no `docs/CTO_LEDGER.md` and that the CTO should write one. It exists —
  seeded from the survey at `f2de36c`. That was true when the survey ran and stale by the
  time I wrote this handoff. Read the ledger, not this sentence's original claim.
- **Full measurement trail:** `docs/fleet/runbook.md`, all entries dated 2026-08-12.
