# Survey — open work and decisions awaiting the maintainer

**Written:** 2026-08-12. Survey only; nothing in this window changed behaviour.
Every claim below is tagged RAN (a command was run, and it is quoted) or REASONED
(inference from what was read).

## 1. What this project is

RAN `ls src/scalar_forensic` and read `CLAUDE.md`: a forensic image-similarity tool —
FastAPI web UI + CLI over a Qdrant vector store, embedding images with DINOv2 and SSCD
and matching them by cosine similarity plus exact/perceptual hashing. REASONED from
`src/scalar_forensic/faces/` (12 modules) and `docs/specs/face-pipeline.md`: an optional,
env-gated face modality is being added as a second identity axis. Design rule, from
CLAUDE.md: features must have forensic value and be legible to a court; decorative
features get removed. Target deployment is a distributed isolated LAN.

## 2. Ground truth

| Fact | Value | Command (RAN) |
|---|---|---|
| HEAD | `b270d47361c8639e2a85cd067543eb94a83cb14e`, committed 2026-08-12T15:22:54+02:00 | `git rev-parse HEAD`; `git log -1 --format='%H %cI %s'` |
| Branch | `feat/face-pipeline-phase1`, **no upstream configured** | `git rev-parse --abbrev-ref HEAD`; `git rev-parse --abbrev-ref --symbolic-full-name @{u}` → fatal, no upstream |
| Divergence from `main` | **0 commits behind, 42 commits ahead** | `git rev-list --left-right --count main...HEAD` → `0 42` |
| Tracked-dirty files | **0 files** | `git status --porcelain \| grep -vc '^??'` |
| Untracked files | **0 files** | `git status --porcelain \| grep -c '^??'` |
| Test suite | **480 tests collected; 475 passed, 5 skipped, 14 warnings, 7.71 s, exit 0** | `uv run pytest -q` |
| The 5 skips | all `tests/faces/test_store_integration.py`, reason `SFN_TEST_QDRANT_URL not set` | `uv run pytest -q -rs` |
| Lint | `All checks passed!`, exit 0 | `uv run ruff check src tests scripts` |
| Format | `85 files already formatted`, exit 0 | `uv run ruff format --check src tests scripts` |
| Tracked files | 276 files | `git ls-files \| wc -l` |
| TODO/FIXME/XXX/HACK | **1 file** repo-wide (`docs/tag-triage.md`); **0 lines** in `src tests scripts` | `git grep -lIE 'TODO\|FIXME\|XXX\|HACK' -- .`; `git grep -nIE ... -- src tests scripts \| wc -l` |
| Qdrant reachable | **no** — `curl` exit 7 (connection refused) on `http://localhost:6333/collections` | `curl -s -m 3 http://localhost:6333/collections` |

**No dedicated check/verify script exists.** RAN `ls Makefile justfile noxfile.py tox.ini`
→ all absent; RAN `ls .github/workflows` → only `ci.yml`. The four commands above (pytest,
ruff check, ruff format) are the whole gate, and all four exited 0.

Note the binary-match trap: RAN `git ls-files | grep -E '__pycache__|\.pyc$|\.onnx$'` →
exactly 1 hit, `tests/fixtures/faces/tiny_face.onnx`, a test fixture. No tracked
`__pycache__`.

### Where the docs disagree with the code

- **CLAUDE.md says "full suite (300 tests)". RAN `pytest --collect-only` → 480 collected.**
  The number is stale by 180 tests.
- **CLAUDE.md says the face real-model test "skips unless a YuNet ONNX is present".**
  RAN `pytest -rs`: no such skip occurred — RAN `ls -la models/` shows
  `face_detection_yunet_2023mar.onnx` (232 589 bytes, mtime 2026-08-12 08:10). The
  YuNet **is** in place and that test now runs.
- **`docs/superpowers/plans/2026-08-11-face-pipeline-phase1.md` has 61 unchecked boxes
  and 0 checked** (RAN `grep -c '^\s*- \[ \]'`), yet RAN `git log --oneline main..HEAD`
  shows all 12 of its deliverables landed as commits (`d3b96e0`…`c9fbd38`). REASONED:
  the plan was executed without being ticked. It is a stale document, **not** open work.
- **Spec §14.2 claims `SFN_FRAME_STORE_SIZE` "is applied nowhere".** RAN
  `grep -rn frame_store_size src` → exactly 1 hit, `src/scalar_forensic/config.py:74`
  (parsed, never read). The doc is correct; the config key is dead.

## 3. Open work

1. **`danny*` validation run — the only outstanding item of the face gate-split plan.**
   RAN `grep -c '^\s*- \[ \]'` on `2026-08-12-face-review-embedding-split.md`: 6 unchecked
   of 75 boxes, and all 6 are the validation-run steps. **Its prerequisites are further
   along than the handoff says.** RAN `ls analysis_test`: `danny1.jpeg` (mtime 2026-08-12
   09:15) and `danny2.jpeg` (09:16) are present; RAN
   `uv run python -c "import cv2, onnxruntime"` → `cv2 5.0.0 ort 1.28.0`, so the `faces`
   dependency group is installed; YuNet is present (above). RAN `ls models/*.manifest.json`
   → no matches. **The single missing prerequisite is the embedder ONNX + manifest**, plus
   starting Qdrant (unreachable, above). The run also carries Task 10's visual UI check and
   the calibration judgement on the 48 px review floor.
2. **`purge_all()` does not delete video rollup points.** RAN `grep -n 'def purge_all' -A 12
   src/scalar_forensic/faces/store.py`: the filter matches `is_face` and `is_face_marker`
   only; RAN `grep -o 'is_face[a-z_]*' src -r | sort -u` → a fourth flag
   `is_face_video_rollup` exists (set at `store.py:251`). REASONED: `sfn-faces purge --all`
   leaves biometric-derived rollup points behind. Recorded as out of scope in the plan;
   still true at HEAD. Not started.
3. **Task 13 residual gap — declined stale points are never re-offered.** Read the handoff
   and confirmed the mechanism in code: re-index skips a medium via `processed_hashes`, so
   an operator who declines the stale-deletion prompt and re-runs with the same config hash
   is never asked again. A standalone inspect/purge command would close it. Considered and
   not built.
4. **Chip-store cross-case unlink hazard — decided, accepted, not eliminated.** The handoff
   records the maintainer choosing Option A (document that `SFN_FACE_STORE_DIR` is set per
   case) over Option B (per-case default subdirectory). REASONED: this is operator
   discipline, not an enforced invariant; purging case A can still unlink a chip case B
   references. Listed as a live risk, not as a reopened decision.
5. **Phases 1b, 2 and 3 are specified and unstarted.** Read `docs/specs/face-pipeline.md`
   §12: 1b = calibration record + cross-file face search; 2 = video grouping and probes;
   3 = corpus clustering. No code for any of them (RAN `ls src/scalar_forensic/faces/` —
   no `group.py`, which §6.5 assigns to Phase 2).
6. **42 commits sit on a local-only branch.** RAN `git rev-list --left-right --count`
   (above) and `git remote -v` (an `origin` exists; the branch has no upstream). The whole
   face pipeline is unmerged.

## 4. Decisions the user must make

Ranked by what is blocked behind them.

1. **Do we run the `danny*` validation with a throwaway random-weight embedder now, or wait
   for the real recognition weights?** — *Recommended: throwaway first (plan §Option B),
   real weights before any evidential use.* Waiting keeps the gate split unvalidated and
   blocks every item below it; the throwaway validates counts, gates, chips, audit record
   and UI but proves nothing about match quality.
2. **After the validation run passes, does `feat/face-pipeline-phase1` merge into `main`?**
   — *Recommended: yes, merge locally, no push.* Leaving 42 commits on a branch means every
   later phase either stacks on an unmerged base or diverges from `main`.
3. **Which comes next after Phase 1 closes — Phase 1b (calibrate, then cross-file face
   search) or Phase 2 (video grouping)?** — *Recommended: 1b.* Face search stays disabled
   until a calibration record exists, so 1b is what makes the modality actually usable;
   Phase 2 adds volume to a modality nobody can search yet.
4. **Is `purge_all()` missing video rollup points a bug to fix before Phase 1b, or a Phase 2
   item?** — *Recommended: fix before 1b.* It is a retention/deletion promise — "purge
   --all" leaving biometric-derived points behind is the kind of gap a court asks about.
5. **Do we build the standalone stale-observation inspect/purge command?** — *Recommended:
   yes, small, alongside decision 4.* Without it, an operator who declines the stale prompt
   can only recover the observation keys from the audit record by hand.

Answered here rather than sent up, per this window's brief: the stale phase-1 plan document
(61 unticked boxes for landed work) should be marked superseded, and CLAUDE.md's "300 tests"
corrected to the measured count — both are documentation hygiene, not maintainer calls, and
neither was changed in this survey window.

## 5. What could not be established

- **Whether the face pipeline behaves correctly end-to-end.** Nothing was executed against a
  live Qdrant (RAN `curl` → exit 7) and no embedder exists on disk (RAN `ls
  models/*.manifest.json` → no matches). The 5 skipped tests are precisely the ones that
  could observe the review-only exclusion guarantee against a real store.
- **Whether 48 px is the right review floor.** That is a calibration judgement the plan
  itself says cannot be made without indexed face data.
- **Whether `main` contains work absent from this branch.** RAN
  `git rev-list --left-right --count main...HEAD` → `0` behind, so the branch is a strict
  superset of `main` — this was established, and no other branches were surveyed.
- **The real embedder's licensing/legal status.** Spec §14.1 names it an operator/legal
  decision; no artefact in the repo records a choice.
