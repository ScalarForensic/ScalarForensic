# Handoff — scalarforensic-com-c20 (the §6.3 examiner override, server side)

2026-08-14, 08:10 UTC. Manager at retire: `scalarforensic-com-m7` (retiring at
~198k; the next manager inherits this).

## 1. What I was asked to do

Implement the operator's ruling of 2026-08-14 that a spec §6.3 full-copy refusal
is examiner-overridable — server side, audit line, tests, and the §6.3 spec
paragraph.

## 2. Done

- **#184** — the override. `POST /api/video-full?override=true`; 403
  `override-unattributed` (`states.py`) when `SFN_EXAMINER_ID` cannot attribute
  it; a WARNING audit line in `routes.video_full_start` carrying examiner,
  source, digest, verdict, `estimate_bytes`, `limit_bytes`; `JobRequest.override`
  → the job view's `override` field with `jobs.OVERRIDE_NOTICE`;
  `CeilingVerdict.overridable`; `full_copy.overridable` on `playback-info`. 12
  tests in `tests/test_video_playback.py::TestFullCopyOverride`, **9 mutations
  applied, 9 red** — the table and the raw output are in the PR body and its
  first comment, because the scratchpad they were produced in is a tmpfs.
- **#185** — spec §6.3: the "escalated 2026-08-14 and open" paragraph
  **replaced** by the ruling, plus §9's parameter and §10.1's
  `override-unattributed` row.
- **#187** — spec §14 and the `CLAUDE.md` `npm test` bullet, applied **verbatim**
  from `com-c18`'s handoff at `857ecb8` §7a/§7b by script, not retyped.
- **#189** — `CLAUDE.md`: bar re-measured (below) and the override surface added
  to code structure.

**Bar: 991 passed / 5 skipped, coverage 74.58%**, worktree off **`fd31488`**,
`PYTHONPATH=$PWD/src`, `models/` copied in, `git status --porcelain` empty.
`npm test` 58/58 at the same sha. Identical to what I measured at `bd6554a` on
#184's branch, which is expected: #183/#185/#187/#188 add JS and docs only.

Nothing of mine is unfinished. Nothing is unpushed.

## 3. The next concrete step

Not mine and **not started**: `closeFullJob()` in the browser side has no caller,
so closing the player leaves the full-job poll running. `com-c21` found it, left
it alone deliberately, and it is **pre-existing** — not phase 7's and not the
override's. Whoever picks it up should confirm the poll's lifetime against
`video_playback/player.js` before assuming a leak.

## 4. Written down nowhere else

### The trap: `limit_bytes` must come from the *configuration*, not from the limit

`routes.py` used to pass `limit_bytes=verdict.limit_bytes or None` to the runner,
where `None` means "no ceiling configured, do not watch the `.part`". `limit` is
`int(SFN_VIDEO_CACHE_MAX_BYTES * 0.5)`, so **a ceiling too small to halve floors
to 0**, and `or None` reads that 0 as *unbounded*. It was unreachable while every
such job was refused by the estimate — and reachable the instant an override
exists, which put the only encode with **no `.part` watch at all** behind the
only path the ruling opens. It now reads `settings.video_cache_max_bytes > 0`.

How I found it: not by review. I was writing the test that pins constraint 4 of
the ruling (`limit_bytes` unchanged by an override), picked `SFN_VIDEO_CACHE_MAX_BYTES=1`
because it refuses everything, and the assertion I expected to be trivial came
back `None`. **The generalisable part: when you write a test for "X is
unchanged", assert the actual value, not `is not None` and not a comparison to
the same expression the code uses.** `M5` in #184 pins it; `test_a_ceiling_too_small_to_halve_is_still_a_watched_limit`
is the named test.

### `unknown` is overridable as *this implementation's* reading

The ruling was taken about an estimate measured erring high. `unknown` is the
**absence** of an estimate — a longer step, and the operator did not rule on it
by name. I allowed it and said so in §6.3 in those words, on this reasoning: the
enforced number is `limit_bytes`, which comes from the configured ceiling and
does not depend on the estimate existing, so an overridden `unknown` job is
watched exactly as an overridden `refused` one is. What `unknown` withholds is
the forecast, and the forecast is the only thing an override sets aside. `m7`
reviewed and accepted this explicitly, with the instruction that the spec must
not read as though the operator had ruled on `unknown`. **If a later reader
wants to narrow it to `refused` only, that is a defensible change — but it is a
change to a documented reading, not a bug fix.**

### Things in `jobs.py`/`routes.py` that look like bugs and are not

- **The override is refused rather than logged with a null examiner.** That looks
  over-strict for a local tool. It is the second constraint of the ruling: an
  entry naming nobody reads as authority in a log an examiner may have to defend.
  Do not "improve" it into `examiner_id: null`.
- **`full_copy.overridable` is false when `SFN_EXAMINER_ID` is unset even though
  the verdict refuses.** Not a missing feature — it stops the UI offering a
  control the server would 403.
- **The override travels on *every* `view()`, not just the response to the click.**
  Disclosure is for the life of the job and the copy it produced; an analyst
  opening the page later must still see that a gate was set aside and by whom.
- The four traps `com-c17` recorded still hold and I touched none of them:
  `PRIO_PGRP` with `start_new_session=True`, `out_time` (never `out_time_ms`),
  and the `.part` watch against `limit_bytes` rather than the estimate.

### Process

- **Mutation evidence written to a scratchpad is one `rm` from gone.** Mine was
  in `/tmp/claude-1000/…`, a tmpfs. `m7` had me paste the raw tail into a PR
  comment; the table in the body was already self-contained. Put the evidence in
  the PR, not only a path to it.
- The spec file was held by another worker for most of my window. Splitting §6.3
  from §14 into two PRs (m7's call) is why the §6.3 correction did not wait on a
  retiring worker's context. Two unrelated edits to one file are two PRs.
