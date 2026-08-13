# CTO ledger — ScalarForensic

Written by the CTO; rewrite in place, keep it tight, commit every fold.
`git log -S` on this file is the project's decision history.

Seeded 2026-08-12 by cto8 (`21f65f1`). Final fold 2026-08-13 by
`scalarforensic-cfm-g1`. Raw record: `docs/fleet/runbook.md` and the
`~/.claude/cx/cto.md` git history.

> ## THE CTO ROLE IS RETIRED (2026-08-13, operator's decision)
>
> `scalarforensic-cfm-g1` folded and exited without a successor. **There is no
> CTO on this project.** The manager owns the work and **escalates directly to
> the operator**, not upward to a CTO session that does not exist. Nobody should
> spawn a replacement CTO; if the operator wants one later they will run
> `cx cto` themselves.
>
> This file remains the project's decision record. **The manager now writes it**
> — same discipline: rewrite in place, keep it tight, commit every fold.

## what this is

Forensic image-similarity tool: FastAPI UI + CLI over a Qdrant vector store,
embedding with DINOv2 and SSCD, matching by cosine similarity plus exact and
perceptual hashing, with an env-gated **face modality** as a second identity
axis. **Design rule from `CLAUDE.md`: a feature must have forensic value and be
legible to a court; decorative features get removed.** Target deployment is a
distributed isolated LAN, fully offline.

## current state (2026-08-13, late)

- `main` at **`0dbc1cd`**. Bar **710 passed / 5 skipped**, coverage 69.35%
  against the 65% floor, measured on a clean tree at `ebe3ed6` and unchanged by
  `#141`; `#142`/`#143` are docs-only. **`CLAUDE.md`'s 559/5 line is stale and
  should be corrected** — good first errand for a coder touching `CLAUDE.md`.
- **Ownership: all rows are clear.** `cx o -l` shows nothing for this project;
  the manager claims what it dispatches.
- Campaign complete: 8,137 files indexed. Cropper delivered as the standalone
  repo `portable_face_cropper` (8138→4537 crops, 0 failures, 35m01s).
- **Fleet is EMPTY.** Managers m1–m3, coders c2–c11 and the frontend-tester c4
  all retired or dead; their ownership rows were reaped this session. Work is
  currently driven directly from the CTO session.

### shipped and verified in the m3 window

UI items 1–4; frame display at stored resolution end-to-end (`#128`–`#130`);
HEIC serving (`#123`); compare/point-probes (`#124`); the CLI trio
(`#131`/`#132`/`#134`); error surfaces (`#133`); in-browser playback of source
video from a frame hit (`#136`); query-face chip generation race fixed (`#137`,
proven by a pixel-signature check); runbook (`#138`); run-progress display
restored (`#139`).

### landed this session

- `#140` — recovered m3's runbook entry, written but never committed before its
  window closed.
- `#141` — the 5 **true** CodeQL alerts: `ci.yml` had no `permissions:` block
  (default token scope), and three handlers interpolated exception text into
  client responses. The worst was `video.py`'s `get_video_info` returning
  `{"error": str(exc)}` straight to the browser via `/api/query-metadata`.
- `#142` — **spec: on-demand video transcoding** (`0dbc1cd`),
  `docs/specs/video-playback-transcode.md`. This is the current work queue; §15
  is the phase plan.
- `#143` — this ledger's previous fold.

## the video playback work — READ THE SPEC, IT IS THE PLAN

`docs/specs/video-playback-transcode.md` is **draft v2, post-review**. v1 proposed
a lazy-HLS segment engine; two independent adversarial reviews (Codex + Opus,
2026-08-13) both returned "do not merge". The operator then chose a simpler shape
— two single-encode MP4 artifacts, a 30 s chunk on play plus an explicit
background full-file job — which removes most of the blockers by construction
instead of answering them.

**Do not re-propose HLS/MSE/CMAF.** §4.1 records why it was rejected and what
evidence would be needed to revisit it. The reviews are the reason the spec is
trustworthy; their findings are already integrated, so the spec's §3 is careful
about what was measured and §3.4 lists what was not.

**Sequencing that matters:**

- Phases 1–3 (digest correctness via `HashCache`, download-original, codec
  detection) are unblocked and need no operator input. Start there.
- Phase 4 is **blocked on two things**: the HDR test-fixture decision (§14 — a
  gitignored `data/` means no committed sample; generate one or gate-and-skip like
  the YuNet test), and the real-hardware measurements §14 requires (4K rates,
  long-source seek, concurrent scaling, minimum hardware floor).
- The subsystem carve (§11) happens **between phases 3 and 4**, not first. It is
  not a pure move: `tests/test_video_endpoints.py` and `tests/test_video_playback.py`
  patch `scalar_forensic.web.routes.video.*` by name, and patch targets are
  per-module. `CLAUDE.md` must be updated in the *same* PR as the carve.
- §17 carries five open questions; 1, 3 and 5 are operator calls.

**One defect the spec found in shipped code**, worth fixing early regardless:
`_evict_cache` (`routes/video.py:327`) globs `*.mp4`, so it cannot see or protect
the new artifacts, and would match a CMAF-style `init.mp4` and delete it mid-play.

## closed rulings — do NOT re-escalate

- Uncalibrated per-face cosine IS displayed, labelled raw; 0.363 is the model
  authors' reference figure and never a threshold, filter or default
  (spec `face-pipeline.md` §10).
- SFace is the embedder (Apache-2.0, OpenCV Zoo). Floors lowered
  `SFN_FACE_MIN_SIZE` 64→40, `SFN_FACE_REVIEW_MIN_SIZE` 48→24 (`b91fd61`).
- `purge_all` must never `delete_collection` — the enablement record is an
  auditable act and survives routine purges.
- `sfn_tags` collection dropped. DBs are test-only: drop + reingest is free
  (~3 min for 10%).
- Restarts are **always** allowed (standing operator grant); canonical recipe
  (`setsid nohup`) is in the runbook.
- **Stored-data-first** is standing: never a UI workaround over missing data.
- **The ingestion run-progress display is an operator exemption from the
  decoration rule.** `#134` removed it; the operator reversed that and `#139`
  restored it. Do not re-remove. The Kalman ±1σ "calibrated" band stays gone —
  its Q/R were hand-picked constants, which is the same defect class as an
  uncalibrated face cosine.
- **17 `py/path-injection` CodeQL alerts are false positives**, verified
  independently this session, not inherited: every flagged path passes through
  `_check_allowed_path` (`routes/_shared.py:12`), which calls `resolve()` —
  normalising traversal and symlinks — *before* `relative_to()` against resolved
  roots, and fails closed when no root is configured. CodeQL does not model
  `Path.relative_to` as a sanitizer. Dismissal needs `security_events` scope, so
  it is an operator action; agents cannot do it.
- Cropper disk-only risk is **CLOSED**: `portable_face_cropper` has a GitHub
  remote and `main` (`8dc490c`) matches it. The `c58c58f` in older notes is
  merely an ancestor.

## standing rules

- **Forensic value is the acceptance test, not feature completeness.**
- **`SFN_FACE_STORE_DIR` is set PER CASE** (operator discipline, not an invariant).
- Commit **explicit paths**, never a broad `git add`.
- **A default-pinning test must supply its OWN empty env file** — `.env` leaks
  into the process env and `find_dotenv()` fallback pins nothing.
- A value that appears only in prose and a mock needs `git log -S` before it is
  trusted (the "review floor 36" that never existed in code).
- **Never quote a test bar against a dirty tree**; `git status --porcelain` must
  be empty first.
- **Verify an inherited verdict before acting on it.** Two claims carried in
  this ledger turned out stale within one session (the cropper remote, the test
  bar). Checking cost one command each.

## how this fleet fails — kept instances

- **cto8 verified the code and INFERRED the author, then ordered on the
  inferred half.** Attribution is load-bearing; `git log -1 --format=%an` costs
  one command. Subordinates checking orders caught it.
- **m3 published 526/5 from a dirty-tree run** (a parallel worker's uncommitted
  tests were counted). Now a `CLAUDE.md` gotcha.
- **Ad-hoc env exports became "documented facts"** twice (review floor 36;
  `SFN_FACE_COLLECTION` load-bearing but recorded nowhere).
- **An agent died holding uncommitted work.** m3 wrote a runbook entry and its
  window closed before it committed; only an ownership-row check found it.
  Recovered as `#140`. A dead agent's owned paths are worth `git diff`-ing
  before the row is reaped.

## pending user decisions

1. **CodeQL dismissals** — 17 verified false positives; one-liner supplied to
   the operator, needs their token scope.
2. **`stale-observation-purge.md`**: re-detect vs config-hash diff (carried).
3. **Audit-button label**: shipped string + subtitle stands unless objected (carried).
4. **Perceptual-hash modality** absent (carried).
5. **Cropper `.gif` inputs** — 2 files were skipped on the delivered run; nobody
   has ruled on whether to re-run including them (carried, low stakes).

Closed since the last fold: **HEVC remedy** — superseded by the `#142` spec. The
answer is on-demand segment transcoding keyed to what the analyst actually
watches, not pre-built viewing copies; the earlier NVENC-transcode
recommendation was measured and found to silently drop rotation metadata.
**Cropper remote** — was never actually open; the push had already happened.
