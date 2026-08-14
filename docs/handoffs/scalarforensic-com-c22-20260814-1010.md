# Handoff — scalarforensic-com-c22, phase 8 (provenance and audit)

## 1. The task

Phase 8 of `docs/specs/video-playback-transcode.md` — the label records the
pipeline that actually ran (§7.2), every transcode writes an audit record via a
new `video_playback/audit.py` following `faces/audit.py`'s shape (§7.3), and
`sfn-video render --path X --at T` prints the exact reproducing invocation.

## 2. Done, and not

**PR #193 — merged as `03847d6`.** Deliverables 1 and 2.
Commits `1127c8e` (module + both writers + override + job view) and `69280f6`
(the GPU-fallback pins). Branch `feat/video-phase8`.

- `src/scalar_forensic/video_playback/audit.py` — new. `Rendering` (§7.2 record),
  `record_transcode` / `record_override` / `record_purge`, `audit_dir` /
  `audit_log`. It **imports `faces.audit.AuditLog`** rather than defining a second
  appender.
- Writers wired: `routes._prepare_chunk` (success and failure) and
  `jobs.JobRunner._run` (success arm and the `except BaseException` arm).
- `jobs.FullJob.view()` gained `"rendering"`; the chunk report's `"pipeline"` is
  now the full `Rendering.describe()` superset.
- `JobRequest.started_by` carries `SFN_EXAMINER_ID` as captured at the click.
- Bar: **1016 passed / 5 skipped at `69280f6`**, clean tree
  (`git status --porcelain` empty), worktree with `models/` copied in, coverage
  74.83%. 14 mutations, 14 caught — the log is at
  `…/da608504-…/scratchpad/mutations-pr1.txt` (tmpfs: treat as gone; the harness
  that produced it is `scratchpad/mutate.py`, same fate).

**PR #194 — open, green-pending, not merged.** Branch
`feat/video-render-command` off merged `main`, one commit. (An earlier WIP commit
`bfb2ac0` on `feat/video-phase8` is superseded — #193 was squash-merged, so that
branch no longer rebases; the delta was re-applied onto `origin/main` instead.)

- `audit.py`: `find_transcode`, `_record_for_another_digest`, `_rebuild_command`,
  `reproduction_report`, and the four notice constants `RENDER_NO_RECORD`,
  `RENDER_SOURCE_CHANGED`, `RENDER_PART_NOTE`, `RENDER_NO_PIPELINE`.
- `cli.py`: `sfn-video render --path X [--at T | --full]`, and `sfn-video purge`
  now calls `record_purge` (m8 asked for the caller to land in PR 2 — it has).
- **`record_purge` got its caller**: `cli.video_purge` files an `EVENT_PURGE`
  record, tested (`TestPurgeIsFiled`), and the log survives `purge --all` because
  it lives beside the cache directory.
- Bar on that branch: **1025 passed / 5 skipped**, clean tree, coverage 75.01%;
  re-run after rebasing onto `03847d6`, not carried over. 7 mutations, 7 caught.

**Not started, and deliberately not mine:** the spec §15 phase list and
`CLAUDE.md` — m8 holds that fold until all phase-8 code is on `main`. The browser
side went to `com-c23`: the §7.2 label renders three fields of what §7.2
requires, and #193 landed the payload for the rest.

## 3. The next concrete step

Watch #194 to green and merge it (m8 merges, not the author). Then the spec §15
phase list and `CLAUDE.md` fold, once `com-c23`'s browser work is also on `main` —
§15 still says "Done: phases 1–3 and the carve", which has been false since
phase 4, and `CLAUDE.md`'s `video_playback/` module list names neither
`capability.py`, `encode.py`, `jobs.py`, `states.py` nor `audit.py`.

`CLAUDE.md` also needs: the bar re-measured on merged `main` (1025/5 was measured
on #194's branch, and the file's current figure is 771/5 at `f6cef02`), and the
audit log's location — `data/video_audit.log`, beside the cache and outside it.

## 4. Written down nowhere else

- **The dispatch's reading list points at handoffs that are not on the shared
  checkout's `main`.** `/home/user01/Schreibtisch/gitea/ScalarForensic` is parked
  at `43a7222`; `origin/main` was `5ca812a` and carries phases 4–7. Branch off
  `origin/main`, never off the local `main`, or `video_playback/` has no
  `encode.py`, `jobs.py` or `capability.py` and the dispatch reads as fiction.
- **`_capability()` in the test module gives `ffmpeg_path="ffmpeg"`, but
  `build_command` uses `settings.ffmpeg_path`** — so a fake capability with a
  real `Settings()` still shells out to the real ffmpeg. That is what makes
  `TestProvenanceAfterAGpuFallback` work: select `h264_nvenc`, make every argv
  containing it raise from a patched `_run`/`_run_watched`, and the CPU retry
  encodes for real. It is the only way to exercise the §7.2 trap without a GPU.
- **`hevc_10bit_mov` is 0.6 s long** (6 frames at 10 fps). Any test using a
  timecode above that gets a 422 from `TIMECODE_OUT_OF_RANGE` and the audit
  assertion then fails for the wrong reason — I lost a cycle to `t=1.5`.
- **`job.settings` is the `Settings()` built by the route at start**, not a fresh
  one. So a test that changes `SFN_EXAMINER_ID` mid-job passes even without
  `started_by`; the mutation that proves the attribution is `Settings().examiner_id`
  (fresh) in `_run`, not `settings.examiner_id`.
- **`reproduction_report` imports `routes.chunk_start_for` inside the function.**
  `routes` imports `audit` at module scope, so the borrow can only happen at call
  time. Two implementations of the snap would put a reviewer on a different chunk
  boundary than the analyst was on, which is why it is borrowed rather than copied.
- **`find_transcode` returns the LAST match, not the first**, and that is
  load-bearing: a window can be encoded more than once (evicted and re-encoded,
  or re-encoded after an ffmpeg upgrade), and the bytes actually in the cache are
  the last ones written. A `return rec` on the first hit answers with a rendering
  nobody can still play. `test_the_latest_rendering_of_a_window_is_the_one_answered`
  is the only thing standing between that and a plausible-looking edit.
- **`reproduction_report` imports `_source_digest` inside the function too**, for
  the same module-cycle reason as `chunk_start_for` — `digest.py` is safe to
  import at module scope today, but both borrows sit together so a later reader
  moves neither up without checking `routes`.
- **The GPU reproduction is unverified.** There is no NVENC device on this host,
  so every §8 fallback in the suite is a patched `_run` raising on an
  `h264_nvenc` argv. What that cannot test: whether an `h264_nvenc` invocation
  printed by `sfn-video render` actually reproduces on real hardware. §3.1's
  GPU rows are measured; this is not.
- The §6.3 override now writes **both** a WARNING and an audit record from one
  call site. If a later reader deletes one as a duplicate, they have different
  readers: the WARNING is the operational alarm, the record is the durable file.

## 5. Ownership

Held: `src/scalar_forensic/cli.py` (claimed via `cx o --own`), released on this
handoff along with `src/scalar_forensic/video_playback/`,
`tests/test_video_playback.py`, `docs/specs/video-playback-transcode.md`,
`CLAUDE.md`.

## 6. Review

- **Verified:** the two bars and the 21 mutation checks named in §2 (14 for #193,
  7 for #194), each reverted with the tree confirmed clean afterwards; `ruff
  check` and `ruff format --check` on `src tests scripts` on both branches; #193's
  two required checks green before m8 merged it.
- **Not verified:** #194's checks had not reported when I stopped. No GPU here, so
  every §8 fallback path is exercised by making `h264_nvenc` invocations raise,
  never by a real driver fault. No live browser check was done or claimed — this
  work touches no `web/static/`, and nothing yet renders the `rendering` payload.
- **Risk if wrong:** `reproduction_report`'s two branches are one edit away from
  printing a reconstruction under the record's heading — a recipe presented as
  evidence. A mutation covers it now; the comment saying why is what has to
  survive the next edit.
