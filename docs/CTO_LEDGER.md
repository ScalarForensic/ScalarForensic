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

- **Bar RAN BY cto8 and reproduced exactly** (`475 passed, 5 skipped` of 480
  collected, 7.71s) at branch `feat/face-pipeline-phase1`, HEAD `b270d47` —
  **the SHA the evidence belongs to, never "current HEAD", because a document
  moves the SHA it quotes.** `ruff check` and `ruff format --check` both rc=0.
  Tree clean: dirty 0 / untracked 0.
- **There is NO check/verify script.** No Makefile, justfile, noxfile or tox
  config; `.github/workflows/ci.yml` only. **Those four commands are the entire
  gate.**
- **The 5 skips need a live Qdrant** (`SFN_TEST_QDRANT_URL` unset), and Qdrant is
  **not running** (`curl` exit 7). Those five are precisely the tests that could
  observe the review-only exclusion guarantee against a real store, so **the
  suite being green says least about the part that matters most.**
- **42 commits sit on a local-only branch**, 0 behind `main`, no upstream. The
  whole face pipeline is unmerged.

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

- **Blocked on decision 1:** the `danny*` validation run — 6 unchecked boxes of
  75 in the gate-split plan, all validation steps. Prerequisites are **further
  along than the handoff says**: `danny1/2.jpeg` present, `cv2 5.0.0` +
  `onnxruntime 1.28.0` installed, YuNet in place. **The only missing pieces are
  the embedder ONNX + manifest, and a running Qdrant.**
- Task 13 residual: a declined stale-observation prompt is never re-offered
  (`processed_hashes` skips the medium on re-index). Considered, not built.
- Phases 1b (calibration record + cross-file face search), 2 (video grouping)
  and 3 (corpus clustering) are **specified and unstarted**; no `group.py`.

## pending user decisions

Ranked, with defaults. Full text in the survey §4.

1. **Validate with a throwaway random-weight embedder now, or wait for real
   recognition weights?** *Default: throwaway now; real weights before any
   evidential use.* Waiting blocks everything below.
2. **Merge `feat/face-pipeline-phase1` into `main` once validation passes?**
   *Default: yes, merge locally, no push.*
3. **Next after Phase 1 — 1b (calibrate + cross-file search) or 2 (video)?**
   *Default: 1b — face search stays disabled until a calibration record exists.*
4. **Is the `purge_all` rollup gap a pre-1b fix or a Phase 2 item?**
   *Default: fix before 1b — it is a deletion promise.*
5. **Build the standalone stale-observation inspect/purge command?**
   *Default: yes, small, alongside 4.*

**Unresolved and NOT a maintainer question yet:** the real embedder's licensing
status. Spec §14.1 names it an operator/legal decision and **no artefact in the
repo records a choice.** It gates evidential use, not the throwaway validation.
