# CTO ledger — ScalarForensic

Written by the CTO; rewrite in place, keep it tight, commit every fold.
`git log -S` on this file is the project's decision history.

Seeded 2026-08-12 by cto8 from `docs/survey-2026-08-12-open-work.md` (`21f65f1`),
the first survey window. Manager `scalarforensic-com-m1`, no worker, no code
touched.

## what this is

Forensic image-similarity tool: FastAPI UI + CLI over a Qdrant vector store,
embedding with DINOv2 and SSCD, matching by cosine similarity plus exact and
perceptual hashing. An env-gated **face modality** is being added as a second
identity axis. **Design rule from `CLAUDE.md`: a feature must have forensic
value and be legible to a court; decorative features get removed.** Target
deployment is a distributed isolated LAN.

## current state

- **`m1` RETIRED at ~250k** (handoff `docs/handoffs/scalarforensic-com-m1-20260812-194005.md`,
  `276a944`); **`m2` is live** and holds Phase 1b. `m1` went well beyond its
  survey brief on the user's direct instruction: **SFace chosen and verified,
  the validation run executed, a UI panel defect fixed test-first (`b008cf9`)**.
- **`m1` reported the live bar as 480 passed / 0 skipped** — the 5 Qdrant skips
  ran for the first time. **I could NOT reproduce it: Qdrant is unreachable
  again** (`curl` → 000). **That number belongs to the window Qdrant was up in,
  and is not re-derivable on demand.** Do not quote it as current.
- **Tree has one in-flight file, `tests/faces/test_config.py` (`m2`).** `m1`'s
  handoff named `tests/faces/test_store.py` — **wrong file.** A relayed
  observation, stale by one commit. Re-derive dirty state; never inherit it.
- **Earlier bar, superseded but still the only reproducible one:** `475 passed,
  5 skipped` of 480, ruff check + format rc=0, at `b270d47` — RAN by cto8. The 5
  skips are the Qdrant-backed tests, and **they are precisely the ones that could
  observe the review-only exclusion guarantee against a real store**, so a green
  suite says least about the part that matters most **whenever Qdrant is down.**
- **There is NO check/verify script** (no Makefile/justfile/noxfile/tox; only
  `.github/workflows/ci.yml`). **pytest + ruff check + ruff format ARE the gate.**
- **The face pipeline is UNMERGED** on local-only `feat/face-pipeline-phase1`,
  0 behind `main`, no upstream.

## standing rules

- **Forensic value is the acceptance test, not feature completeness.** Anything
  that cannot be explained to a court comes out. This outranks convenience.
- **`purge_all` must never `delete_collection`** — the enablement record is an
  auditable act and has to survive routine purges. The docstring says so; keep
  it that way.
- **`SFN_FACE_STORE_DIR` is set PER CASE.** The maintainer chose operator
  discipline (Option A) over a per-case default subdirectory (Option B). **This
  is a decision, not an invariant** — purging case A can still unlink a chip
  case B references. Do not "fix" it without re-opening the decision.
- Commit **explicit paths**, never a broad `git add`.

## verified findings

- **`purge_all()` rollup gap — FOUND, then FIXED at `bb39d0e`. VERIFIED BY
  cto8:** `is_face_video_rollup` is now in the `should` filter (1 line + 14 test
  lines), and **`is_face_meta` is still correctly excluded** — the enablement
  record must survive routine purges. `sfn-faces purge --all` now purges
  everything biometric-derived. This was decision 4 and it is CLOSED.
- **Docs disagree with the code in three places, all doc-side:** `CLAUDE.md`
  says "300 tests" (measured: **480 collected**); it says the face real-model
  test skips without a YuNet ONNX (the YuNet **is** present at `models/`, so it
  runs); and the phase-1 plan has **61 unchecked boxes for work that all landed**
  as commits `d3b96e0`…`c9fbd38`. **That plan is a stale document, not open
  work** — the most valuable thing the survey found, because unticked boxes read
  as a backlog.
- `SFN_FRAME_STORE_SIZE` is parsed at `config.py:74` and **never read**. The
  spec's claim that it is applied nowhere is correct.

## open threads

- **The `danny*` validation run is DONE** (`m1`, SFace chosen and verified).
  **Qdrant must be up to reproduce anything it showed.**
- Task 13 residual: a declined stale-observation prompt is never re-offered
  (`processed_hashes` skips the medium on re-index). Considered, not built.
- Phases 1b (calibration record + cross-file face search), 2 (video grouping)
  and 3 (corpus clustering) are **specified and unstarted**; no `group.py`.

## pending user decisions

**THE SURVEY'S FIVE ARE SUPERSEDED. `m2` reports EIGHT open, ranked with
defaults in `docs/fleet/runbook.md`.** Of the original five, **1 is answered**
(SFace chosen, validation run executed) and **4 is CLOSED** (`bb39d0e`). **I
have not re-derived the current eight and am deliberately not restating them
here from memory** — a stale decision list is the same defect as a stale plan
with 61 unticked boxes, which is the worst thing this project's survey found.
Read the runbook, or ask `m2`.

**THE ONE I HOLD, escalated to the user and blocking `m2`:**
**Phase 1b evidential posture** — spec §10 gates face search on a
`face_calibration` record that does not exist. **(A)** calibrate first, gate the
UI; **(B)** UI first behind an "uncalibrated — not evidential" banner with the
**number suppressed** (rank + green border; raw cosine confined to
`/api/faces/explain/` and the audit record; whole path behind an opt-in env
flag). *Default: (B).*
**I verified §10 myself:** search IS gated on the record; §10.2 requires the
calibrated statistic to BE the statistic the UI thresholds; §10.3 makes
statistical honesty mandatory and prints the CI alongside T. **§10 does not
literally forbid a per-face score — it forbids one with no CI and no defensible
threshold**, which pre-calibration is the same thing.
**The argument against (B), which must be weighed and not skipped: a banner is a
weaker control than a gate, because a banner is what gets cropped out of a
screenshot.** §10.5's shadow mode is (B)'s best support. **§11 makes
jurisdiction-specific legal review an OPERATOR duty — this is the user's call,
never the CTO's.**

**Unresolved and NOT a maintainer question yet:** the real embedder's licensing
status. Spec §14.1 names it an operator/legal decision and **no artefact in the
repo records a choice.** It gates evidential use, not the throwaway validation.
