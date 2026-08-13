# CTO ledger — ScalarForensic

Written by the CTO; rewrite in place, keep it tight, commit every fold.
`git log -S` on this file is the project's decision history.

Seeded 2026-08-12 by cto8 (`21f65f1`). This fold 2026-08-13 by
`scalarforensic-cfm-g1` collapses the m1/m2/m3 inbox era (raw record:
`~/.claude/cx/cto.md` git history and `docs/fleet/runbook.md`).

## what this is

Forensic image-similarity tool: FastAPI UI + CLI over a Qdrant vector store,
embedding with DINOv2 and SSCD, matching by cosine similarity plus exact and
perceptual hashing, with an env-gated **face modality** as a second identity
axis. **Design rule from `CLAUDE.md`: a feature must have forensic value and be
legible to a court; decorative features get removed.** Target deployment is a
distributed isolated LAN.

## current state (2026-08-13)

- **The face pipeline era is MERGED AND SHIPPED.** Phase 1b (query-face strip,
  FACES fourth mode, face query controls, DINO/FACE audit modals) landed via m3;
  merge `2933a11`; since then the repo moved to GitHub with PR-only `main`
  (ruleset "standard", 2 required checks). `main` at `e50fb28`; bar per
  CLAUDE.md: **559 passed / 5 skipped at `32f3bf9`**, coverage 66.58% vs 65%
  floor.
- **Closed maintainer rulings — do NOT re-escalate:** uncalibrated per-face
  cosine IS displayed (labelled raw, no CI, 0.363 never a threshold;
  spec §10 divergence recorded in the spec itself); SFace is the embedder
  (Apache-2.0, OpenCV Zoo); floors lowered `SFN_FACE_MIN_SIZE` 64→40,
  `SFN_FACE_REVIEW_MIN_SIZE` 48→24 (`b91fd61`); merge done.
- **Still open from the m3 era:** the `stale-observation-purge.md` re-detect vs
  config-hash-diff fork (user-owned); the audit-button label shipped as the
  maintainer's string + subtitle (accepted by silence so far). The
  "purge --all + rebuild for the danny 40–47px cohort" item is SUPERSEDED by
  the fresh-start below.

## the iPhone test campaign (opened 2026-08-13, operator-directed)

Operator decisions, FINAL:
- Input (read-only originals, never write there):
  `/media/user01/SAM_870_SATA/Gitea_Backup/input_scalar` — 8216 files, 20G,
  HEIC-heavy, some HEIC+JPG pairs, videos included.
- All derived artifacts to
  `/media/user01/SAM_870_SATA/Gitea_Backup/created_by_scalar/`
  ({thumbnails,frames,faces,reports} + hash_cache.db) via `.env`
  (`SFN_THUMBNAIL_DIR`, `SFN_FRAME_STORE_DIR`, `SFN_FACE_STORE_DIR`,
  `SFN_HASH_CACHE_PATH`).
- Old test data (`data/images` 16G, unsplash zips, thumbnails, faces,
  hash_cache) is deleted; `data/models` and `data/sample_images` KEPT.
- DB starts fresh: drop image collections + `sfn-faces purge --all`
  (enablement record survives by design).
- **No ingestion run until the operator triggers it interactively.** First a
  MEASURED pipeline-efficiency audit (per-stage wall time on ~50 files incl.
  HEIC + video, GPU utilization, extrapolation to 8216 files, video frame-rate
  question) → `docs/fleet/pipeline-efficiency-2026-08-13.md`.
- Interactive test loop: operator drives a real Chrome
  (`--remote-debugging-port=9222`); a frontend-tester agent (spawn as `c` with
  the frontend-tester role file — `cx n` still refuses letter `f`) attaches via
  chrome-devtools MCP, reports defects to the manager.
- Known gap found at dispatch: **`pillow_heif` is not installed** — without it
  `scanner.py` silently classifies all HEIC as unsupported.

Manager `scalarforensic-cfm-m1` (this fleet's numbering restarted) spawned
2026-08-13 with the 5-item task + measurement addendum.

## standing rules

- **Forensic value is the acceptance test, not feature completeness.**
- **`purge_all` must never `delete_collection`** — the enablement record is an
  auditable act and survives routine purges.
- **`SFN_FACE_STORE_DIR` is set PER CASE** (operator discipline, a decision not
  an invariant).
- Commit **explicit paths**, never a broad `git add`.
- **A default-pinning test must supply its OWN empty env file** — `.env` leaks
  into the process env and `find_dotenv()` fallback pins nothing.
- A value that appears only in prose and a mock needs `git log -S` before it is
  trusted (the "review floor 36" that never existed in code).
- Never quote a test bar against a dirty tree; porcelain 0 first.

## how this fleet fails — kept instances

- **cto8 verified the code and INFERRED the author, then ordered on the
  inferred half.** Attribution is load-bearing; `git log -1 --format=%an` costs
  one command. Subordinates checking orders ("shaky clause") caught it.
- **m3 published 526/5 against a sha from a dirty-tree run** (parallel worker's
  uncommitted tests counted). Now a CLAUDE.md gotcha.
- **Ad-hoc env exports became "documented facts"** twice (review floor 36;
  `SFN_FACE_COLLECTION` load-bearing but recorded nowhere).

## pending user decisions

1. Pipeline pre-run speed fixes — waiting on the measured audit's DECIDE lines.
2. `stale-observation-purge.md`: re-detect vs config-hash diff (carried).
3. Audit-button label: shipped string+subtitle stands unless objected (carried).
