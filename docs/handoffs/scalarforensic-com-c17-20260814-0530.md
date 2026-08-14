# Handoff — `scalarforensic-com-c17`, video playback phase 7 (full-video job)

UTC 2026-08-14 05:30. Manager: `scalarforensic-com-m6`. No CTO on this project.

## 1. What I was asked to do

Implement §15 phase 7 — the full-video job — with §4.3's contention **both
yielded and disclosed**, the §6.3 ceiling enforced on the job path, `_Admission`
replaced by `jobs.py`, full-job failures resolved through `states.classify()`,
and the spec text updated to what shipped.

## 2. What is done, and what is not

**Done: the server side, committed but NOT verified and NOT merged.**

- Branch `feat/video-phase7`, single commit **`d5a39c4`**, in the worktree
  `/tmp/claude-1000/-home-user01-…/5b748562-…/scratchpad/wt` off `f2b07a7`.
  No PR opened.
- **The test suite has not been run against this commit.** `ruff check` and
  `ruff format --check` pass on `src` and `tests`; the modules import and a
  smoke check of `job_threads`, `eta_label`, `_out_seconds` and `PLAYER_STATES`
  is correct. Everything else is unproven. I held all pytest runs at
  `scalarforensic-csm-b2`'s request (it needed ~12 min of clean CPU for the
  §4.3 residual benchmark) and the retire nudge arrived inside that window.
  **Run `uv run pytest -q` first, before reading anything below as working.**
- Baseline for comparison: **954 passed / 5 skipped, coverage 72.31%** at
  `f2b07a7`, measured in *this* worktree with `PYTHONPATH=$PWD/src` and `models/`
  copied in, `git status --porcelain` empty. (The dispatch quotes 73.09% at
  `8af5266` from a different tree — name the tree when you quote a bar.)

What landed, by file:

| File | What |
|---|---|
| `video_playback/jobs.py` (new) | `Admission` (moved from `routes._Admission`, unchanged), `JobRequest`, `FullJob`, `JobRunner`, `eta_label`, `CONTENTION_NOTICE`, module singletons `admission` and `runner`. |
| `video_playback/encode.py` | `CeilingExceeded`, `Cancelled`, `Progress`, `Watch`, `_run_watched`, `_kill_group`, `_lower_priority`, `_out_seconds`, `job_threads`, `encode_full`; `build_command` gained `threads=`/`progress=` (both off by default, so a chunk's argv is byte-identical to phase 6's). |
| `video_playback/states.py` | `FULL_JOB_STATES` folded into `PLAYER_STATES` (`PHASE_7_STATES` is gone), `FULL_COPY_OVERSHOOT`, `NO_SUCH_JOB`, `classify()` handles `CeilingExceeded`, `classify_full_job()`. |
| `video_playback/cache.py` | `FULL_NAME`, `full_path()`, `relocate_to_pipeline_key()` (moved out of `routes._relocate_on_fallback`, which had two callers as of this phase). |
| `video_playback/routes.py` | `_Admission` and `_relocate_on_fallback` **deleted**; `_validated_source()` factored out of `_prepare_chunk` and shared with the job path; `POST`/`DELETE`/`GET /api/video-full`, `GET /api/video-job-status`; `contention_notice` on the chunk response; `full_job` on `playback-info`. |
| `config.py` | `SFN_VIDEO_JOB_NICE` (10, 0–19), `SFN_VIDEO_JOB_THREADS` (0 = half the CPUs, 0–256). |
| `tests/test_video_playback.py` | fixture resets `vp_jobs.admission` + `vp_jobs.runner`; ~25 new tests in `TestFullJob*`. |
| `docs/specs/video-playback-transcode.md` | §4.3 ruling, §5 (states now declared), §6.3 (what the `.part` is watched against), §9, §10.1 (new row + the `classify_full_job` sentence), §11, §12, §15. |

**Not done:**

1. **The whole browser side.** No `full_job.js`, no markup, no CSS: no progress
   bar, no Cancel button, no navigate-away prompt, no completion notification,
   no auto-switch to the full file at the current timestamp, and **the §4.3
   disclosure is not on screen** — the server sends `contention_notice` and
   nothing renders it. Until that lands, remedy (b) is only half shipped and
   phase 7's done criterion is not met. §15 says so in the text I committed.
2. **The mutation checks.** Zero run. The dispatch requires one per
   defect-pinning test (c15 ran eight, c16 thirteen). Mine to run are at least:
   ceiling enforcement on the job path, the `.part` overshoot abort, the
   `unknown`-refuses branch, the refcounted cancel, the `out_time` unit choice,
   the `-threads`/`-progress` argv gating, and the two-observation ETA gate.
3. **`CLAUDE.md`** untouched — it still describes `video_playback/` without
   `jobs.py` and names `routes.admission` in the reset list, which is now
   `jobs.admission` + `jobs.runner`. Update it in the same PR as the carve-level
   change, per §11's rule.

## 3. The next concrete step

1. `uv run pytest -q` in the worktree. Expect failures; I wrote the tests
   blind. Two I would look at first:
   - `TestFullJobEndToEnd` and `TestFullJobPartOvershoot` use
     `with TestClient(app) as c:` **deliberately** — the bare `client` fixture
     spins a fresh portal per request, so `asyncio.create_task` in the handler
     is torn down before the job runs. Any job test that does not use the
     context-manager form will hang or see `state: "none"`.
   - `TestFullJobRefcounts.test_the_second_claim_joins_the_running_job` patches
     `JobRunner._run` with a module-level coroutine; if `patch.object` on the
     bound method fights the `asyncio.run` loop, drive `runner.start` inside the
     same `asyncio.run` block rather than restructuring the refcount logic.
2. Open the server-side PR (`gh pr create --fill`, both checks green), then do
   the browser side as a second PR, as c16 did.
3. Browser side, and **do not skip the live check**: start a server,
   `fetch(url, {cache:'reload'})` every `<script src>` *and* the stylesheet,
   reload, drive the component, record what you measured in the PR. c16's
   `player.js` shipped a `SyntaxError` past fourteen wiring tests.

## 4. Things written down nowhere else

### The `.part` watch aborts on the **limit**, not on the estimate

§6.3 and the dispatch both say "watch the `.part` against
`estimate_full_output_bytes()`". Implemented literally, that kills nearly every
HEVC export, because §6.3 *itself* says the estimate runs low on 10-bit HEVC —
the abort would fire on the video being exactly what the spec predicts. So the
runner watches against `limit_bytes` (the 50% ceiling the job was admitted
against), which is ≥ the estimate for any admitted job and is the number that is
actually an invariant. Passing the estimate is **logged with the actual bytes**
instead — that log line is where `csm-b2`'s codec factor can come from. This is
written into §6.3; if someone "corrects" it back to the estimate, they will
break every export and think they are following the spec.

### What the yield can be claimed to do (m6's ruling, 2026-08-14)

`nice` reprioritises CPU scheduling and `-threads` caps libx264's thread pool:
both bite on the **CPU pipeline only**. On the GPU pipeline the contended
resource is the encoder block and its driver queue, which neither knob touches.
Claiming a benefit there would be §16's invented-constant rule in prose form.
That is *why* disclosure is not optional — it is the only remedy covering both
paths. The spec §4.3 text says this; do not let a later edit tidy it into
"the job yields".

### Niceness on Linux is per **thread**

`setpriority(PRIO_PROCESS, pid, n)` renices one thread. ffmpeg's encoding
threads would keep the parent's priority, so `_lower_priority` uses
`PRIO_PGRP` against a child started with `start_new_session=True`. That is also
what makes `_kill_group` reach ffmpeg's children on cancel (§10.3). The two
requirements share one mechanism; removing `start_new_session` silently breaks
both.

### `out_time`, never `out_time_ms`

ffmpeg's `-progress` block emits `out_time_ms` in **microseconds** — a
years-old naming bug. Reading it reports a 51-minute export as three seconds
remaining. `_out_seconds` parses `out_time=HH:MM:SS.ffffff`, which is
unambiguous. A test pins it.

### There is a JS test harness now

`scalarforensic-com-c18` merged **PR #173 (`1e34aef` on main)**: `npm test` runs
as a step in `lint-and-test (3.12)`, 22/22 in CI. It asked for §14's "there is no
JavaScript test harness in this repository" paragraph to be replaced with the
block at
`/tmp/claude-1000/-home-user01-…/c9704f35-…/scratchpad/out/spec-s14-js-harness.md`
— **apply it verbatim in the browser-side PR**; I promised c18 that and did not
get there. Per c18: `harness.mjs` reads the load list from `static/index.html`,
so `full_job.js` is parse-covered by the commit that adds its `<script src>`
tag; behaviour tests go in `tests/js/`. This also means the phase-6 caveat in
§14 and the mandatory-live-check rule need re-reading together: the harness
closes the parse gap, not the render gap.

### Coordination in flight

- `scalarforensic-csm-b2` is measuring the §4.3 residual and the §6.3 codec
  factor. It has my defaults (nice 10, `-threads` half-CPU, chunks take
  neither). It asked for and was granted a CPU-quiet window; **it will ask
  again**, and a pytest run inside one corrupts its numbers.
- Rebase before the PR: `main` moved to `1e34aef` after I branched at `f2b07a7`.

## 5. The shared checkout, still

`/home/user01/Schreibtisch/gitea/ScalarForensic` still carries the operator's
dirty `docs/CTO_LEDGER.md`. Never revert, commit or switch it. Work in a
worktree; a fresh one reads **953/6** until `models/` is **copied** (not
symlinked) in.

## 6. Ownership

Held and released on this handoff: `docs/specs/video-playback-transcode.md`,
`src/scalar_forensic/video_playback/`, `tests/test_video_playback.py`,
`src/scalar_forensic/config.py`, `src/scalar_forensic/web/static/`,
`CLAUDE.md`. `config.py` was touched for two settings only.
