# Handoff — `scalarforensic-com-c19`, phase 7 server side to merged and green

UTC 2026-08-14 07:16. Manager: `scalarforensic-com-m7` (took over from `m6`).
No CTO on this project.

## 1. What I was asked to do

Take phase 7's **server side** from c17's committed-but-never-verified `d5a39c4`
to merged and green: run the suite, run the mutation checks, bring `CLAUDE.md`
current, re-measure the bar naming the tree.

## 2. What is done, and what is not

**Done, merged to `main`:**

- **PR #179 → `1bb24b0`** — phase 7's server side, verified. c17's two commits
  rebased onto `main`, plus one commit of mine: one test fixed, three added, the
  spec's estimate-direction and yield claims corrected, `CLAUDE.md` updated.
- **PR #180 → `32d9da8`** — the bar sha in `CLAUDE.md` corrected to the merged
  commit. Same defect as `#151`, same fix.
- **The bar: 979 passed / 5 skipped, coverage 74.51%, at `1bb24b0`.** Measured on
  a clean tree in the worktree
  `/tmp/claude-1000/-home-user01-…/5b748562-…/scratchpad/wt` with `PYTHONPATH=$PWD/src`
  and `models/` **copied** in — without `models/` it reads 978/6. Baseline was
  954/5 at `8af5266`. `npm test` 22/22 locally and in CI.
- **Eleven mutation checks**, each reverted after. Eight went red immediately;
  three survived and are now caught by tests I wrote. The full table with the test
  that caught each is in #179's description — read it there rather than
  re-deriving it. The three survivors, because they were real gaps in c17's
  otherwise strong set:
  - `out_time` → `out_time_ms` **at the call site**. The existing test pinned
    `_out_seconds` (the parser); nothing pinned which field `_run_watched` hands
    it. Now `TestFullJobProgress::test_the_watcher_reads_out_time_and_not_out_time_ms`
    drives a real child emitting the same instant in both fields.
  - `PRIO_PGRP` → `PRIO_PROCESS`. Nothing observed the renice reaching ffmpeg's
    encoding threads. Now `TestFullJobYield::test_the_renice_reaches_every_
    encoding_thread_and_not_just_the_pid`.
  - `attach()` dropping its cancelled-before-`Popen` re-check. Now
    `TestFullJobRefcounts::test_a_cancel_that_races_the_encoder_still_stops_it`.
- **Three spec claims corrected** against `csm-b2`'s measurements (§4.3 and §6.3
  — see §4 below).

**Not done, and not mine:** the whole browser side. `full_job.js`, the progress
bar, Cancel, the navigate-away prompt, the completion notification, the
auto-switch at the current timestamp, and the on-screen rendering of
`contention_notice`. It goes to `scalarforensic-com-c18` as a second PR on top of
`1bb24b0`; §15's phase 7 entry says so in the text on `main`. Until it lands,
§4.3's remedy (b) is only half shipped — the server sends the sentence and nothing
displays it, which matters more than it did before, because the measurement in §4
below makes disclosure the *load-bearing* half.

## 3. The next concrete step

For `c18`: the browser side, and §14's harness-block replacement at
`/tmp/claude-1000/-home-user01-…/c9704f35-…/scratchpad/out/spec-s14-js-harness.md`,
applied verbatim in that PR. c17 promised c18 that and did not get there; I did
not either, because it is browser-side text.

For phase 8 (provenance and audit), what it inherits — see §5.

## 4. What phase 7 shipped that phase 8 and later must not undo

### The §6.3 over-estimate, as it now stands in the spec

`csm-b2`'s report (`docs/benchmarks/video-codec-factor-2026-08-14.md`) measured
`output_bytes / estimate_full_output_bytes()` and **falsified the direction claim
§6.3 had argued from.** Medians: HEVC 10-bit HDR **0.301** on CPU (over-reads on 8
of 8 samples, one by 8×) and 0.471 on GPU; HEVC 8-bit 0.886 on CPU and **1.409**
on GPU; H.264 1080p 0.309 on CPU. The error is **not uniformly low-side** — it is
frequently high-side on the CPU pipeline and crosses 1 mainly for GPU encodes of
lightly-compressed 8-bit sources, which is not the case the spec named. With n=3–8
per cell the **ordering** is the finding; these ratios do not support a single
multiplicative codec factor and §16 forbids inventing one from them.

**The consequence is a named, open limitation and not a bug to fix quietly.** An
estimate that over-reads by up to 8× can `refuse` a full-video job whose real
output would have fit — a false refusal §6.3 does not contemplate. Loosening a
forensic capacity gate is the **operator's** decision; `m6` escalated it on
2026-08-14 and it is open. §6.3 records it with a pointer to the evidence, and
phase 7 shipped without touching the gate. **Do not "fix" this by applying a codec
factor, widening the 50% limit, or downgrading `refused` to a warning** until that
decision comes back. If you are here because a 10-bit HDR export was refused and
you believe it would have fit: you are probably right, and that is the known
limitation, not a new discovery.

### The yield is measured and it is not the remedy

Chunk encode latency under a sustained full-copy competitor, CPU/libx264, n=6 per
arm: unniced median **18.31 s**, niced (`nice` 10, `-threads` 12) **16.83 s**.
**~9%** — real, free, keep the knobs — but *neither* arm is inside §4.2's 6–10 s
window, so §4.3 must not read as though the job "yields" and the contention is
handled. Disclosure is load-bearing. Two further facts that constrain what may be
written: the measured unniced 18.31 s is **worse** than §4.3's own former ~16.35 s
extrapolation from §3.5 (a projection, now replaced by the measurement), and the
**GPU path is unmeasured** — there the contended resource is the encoder block and
its driver queue, which neither `nice` nor `-threads` touches. Claiming a GPU
benefit is §16's invented-constant rule in prose form.

## 5. What phase 8 inherits from the job/admission layer

- **There is exactly one writer of a full copy**: `jobs.JobRunner._run` →
  `encode.encode_full` → `cache.relocate_to_pipeline_key`. §7.3's audit record for
  the full-job path has one place to hook — `_run`'s success arm and its
  `except BaseException` arm, both of which already set `finished_at` and a
  terminal `state`. Do not add a second write path; the chunk path's writer is
  `routes._prepare_chunk`, and those two are the complete set.
- **`result.pipeline` is the pipeline that ran; `request.pipeline` is the one that
  was selected.** They differ on a §8 GPU→CPU fallback, and the artifact is keyed
  by the former (`full_url()` already uses `result.pipeline.fingerprint()`). §7.2
  requires the label record the *actual* pipeline — **an audit record built from
  `request.pipeline` would name the encoder that was chosen and not the one that
  produced the bytes**, which is the precise false-label failure §7.2 exists to
  prevent. This is easy to get wrong because `request` is the object in scope for
  most of `_run`.
- **`sfn-video render` must record the thread count for a full copy.** A chunk's
  argv carries no `-threads`; a full job's carries `-threads job_threads(settings)`
  (default: half the online CPUs). libx264's output depends on its thread count, so
  a reproduction command that omits it will not reproduce the bytes on a box with a
  different core count. `build_command` is deterministic given its arguments —
  record `threads` alongside the pipeline, or the §7.2 promise is not kept for full
  copies even though it is for chunks.
- **One job can have many claimants.** `JobRunner` is refcounted: `waiters` counts
  analysts, and a DELETE from one drops a claim rather than killing the encode. An
  audit record is per *encode*, not per request, and `SFN_EXAMINER_ID` for a shared
  job is whoever started it — decide that deliberately rather than by accident.
- **The admission gate is shared, in one place**: `jobs.admission`, used by both the
  chunk path and the job path. `routes._Admission` is gone. Two pools would be two
  bounds and therefore no bound (§4.3).
- **`jobs.py` is process-wide state a fixture must reset**: `jobs.admission.reset()`
  and `jobs.runner.reset()` (the latter kills what is still running). Both are in
  `tests/test_video_playback.py`'s module-scope autouse `_clean_cache_state`, and
  `CLAUDE.md` names them. An admission counter left above zero turns a later chunk
  request into a spurious 503 that reads as a flake.

## 6. Things in `jobs.py` / `encode.py` that look like a bug and are not

1. **The `.part` is watched against `limit_bytes`, never `estimate_bytes`.** c17
   chose this because the estimate was believed to run low; §4 above shows it is
   wrong in *both* directions, which makes `limit_bytes` — the 50% ceiling the job
   was admitted against, and ≥ the estimate for any admitted job — the only number
   here that is actually an invariant. The reasoning is in the `CeilingExceeded`
   docstring precisely so the next reader does not "correct" it back to the spec's
   old literal wording and break every export.
2. **`_lower_priority` uses `PRIO_PGRP`, and the child is started with
   `start_new_session=True`.** Two requirements, one mechanism: niceness on Linux
   is per **thread**, so `PRIO_PROCESS` would renice only the thread that called
   `Popen`; and the new session is what lets `_kill_group` reach ffmpeg's children
   on cancel (§10.3). Removing `start_new_session` breaks both silently — and
   entertainingly: with it removed, `killpg` targets the test runner's own process
   group and `pytest` exits 137.
3. **`out_time`, never `out_time_ms`.** ffmpeg has emitted `out_time_ms` in
   *microseconds* under a millisecond name for years; reading it reports a
   51-minute export as three seconds remaining. Two tests pin this now — the parser
   and the field choice at the call site.
4. **`_run` catches `BaseException`.** Deliberate, and flagged `# noqa: BLE001`:
   every ending of this job is a reported state, `asyncio.CancelledError` included.
   Narrowing it to `Exception` leaves a cancelled task with `state:
   "full-job-running"` forever.
5. **A cancelled job's state is `needs-transcode`, not a fourth state.** Nothing is
   running, so the video is back where it was; `cancelled: true` says why. A
   cancelled encode also dies of SIGKILL, which `classify()` would otherwise read as
   the OOM killer — the `if job.cancelled` branch runs *before* `classify_full_job`
   for that reason, and the order is load-bearing.
6. **`Admission.enter` increments `admitted` before acquiring the semaphore.**
   `SFN_VIDEO_QUEUE_MAX` bounds *admitted* requests — running plus waiting — so the
   counter must include waiters or the queue is unbounded with a slower fuse.
7. **`eta_seconds` returns `None` until two observations exist**, via
   `_RateTracker.eta`'s `_n < 2` gate — a single block is not a rate, and an ETA
   drawn from one is a number with no content that an analyst will plan around
   (`#139`). It also returns `None` once `remaining <= 0`.
8. **`TestFullJobEndToEnd` and friends use `with TestClient(app) as c:` on
   purpose.** The bare `client` fixture spins a fresh portal per request, so the
   handler's `asyncio.create_task` is torn down before the job runs and the test
   hangs or sees `state: "none"`. If you add a job test, use the context-manager
   form; do not restructure the handler.

## 7. Two process notes

- **I merged #179 and #180 myself.** m7's rule — workers report the PR number and
  the manager merges — arrived after both had landed. Green CI is not the review.
  Follow the rule; I did not have it in time.
- **A re-measure is externally indistinguishable from a retyped sha.** m7 read
  #180 as a sha swap and asked me to reword; I had in fact checked out `1bb24b0`
  and re-run the suite there, and the reflog proved it. Say *in the commit message*
  that you re-ran and where, or the reviewer cannot tell — that is now in m7's
  runbook fold.

## 8. Ownership

Held and released on this handoff, all to `c18`:
`docs/specs/video-playback-transcode.md`, `src/scalar_forensic/video_playback/`,
`tests/test_video_playback.py`, `src/scalar_forensic/config.py`, `CLAUDE.md`.
`config.py` carries phase 7's two settings only (`SFN_VIDEO_JOB_NICE`,
`SFN_VIDEO_JOB_THREADS`).

## 9. Review

- **Verified:** full suite 979/5 at `1bb24b0` on a clean tree in the worktree named
  in §2; `ruff check` and `ruff format --check` on `src tests scripts`; `npm test`
  22/22; both required checks green on #179 and #180; eleven mutation checks, each
  reverted and the tree confirmed clean after.
- **Not verified:** nothing renders `contention_notice` — c18's PR. The GPU
  pipeline is unmeasured for both the yield residual and the estimate ratio. No
  live browser check was done or claimed; this work touches no `web/static/`.
- **Risk if wrong:** the highest-consequence line is the `.part` watch target
  (§6.1 above). If a later edit moves it to `estimate_bytes`, HEVC 8-bit GPU
  exports — measured at 1.409× the estimate — abort for being exactly what they
  are. Two places now say why they must not.
