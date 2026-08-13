# Handoff — `scalarforensic-com-m5` (manager, ScalarForensic)

**There is no CTO on this project**; the manager owns the work and escalates
directly to the operator. Do not spawn a replacement CTO.

## 1. What I was asked to do

Take over from `com-m4`, rescue the gitignored §14 benchmark measurements into
git, and run spec phase 4 with one fresh coder.

## 2. Done

Phases 4, 5 and the measurement fold; phase 6 in flight. `main` at `4c6e522`.

| PR | sha | What |
|---|---|---|
| `#155` | `3ac5442` | Bench report into git; spec §3.5/§3.4/§12/§14/§16, §17 Q1–Q5 closed |
| `#157` | `952b325` | Phase 4a — `capability.py`, probe + `Pipeline` fingerprint |
| `#158` | `57246c4` | Phase 4b — `encode.py`, ffmpeg declared in CI, §3.1's defects pinned |
| `#159` | `6f5879e` | `CLAUDE.md` phase-4 conventions |
| `#162` | `3b191b5` | Phase 5a — `cache_key()`, §6.2 eviction, `KeyedLocks`, atomic publish |
| `#163` | `955334c` | Phase 5b — §6.3 ceiling refusal, `sfn-video purge` |
| `#165` | `2dbfea6` | `CLAUDE.md` bar + `cache.py` surface (by `c16`) |
| `#156`/`#161`/`#166` | | my ledger + runbook folds |

**Bar: 882 passed / 5 skipped, coverage 72.20%**, re-measured by me at `6e09097`
in a worktree with `PYTHONPATH=$PWD/src` and `models/` present, no tracked file
modified. Was 771/5 at 70.22% when I opened.

**Not done: phases 6, 7, 8.** Phase 6 is with `com-c16`.

## 3. The next concrete step

**Review `c16`'s phase-6 PRs, then spawn ONE fresh coder on phase 7.** Phase 6
is the first browser-weight phase; the review that matters is whether §5's player
states are three-state where the question is three-state, and whether every
§10.1 failure has a test that was mutation-proved.

**Phase 7 must not start without reading the §4.3 contention correction** (in the
spec since `#157`) and the `estimate_full_output_bytes()` finding in §6.3: the
job runner is required to check the growing `.part` against the estimate and
abort on overshoot, because the estimate has no codec factor and errs *low* on
HEVC sources.

## 4. What I learned that is written down nowhere else

**The failure mode I opened with, because it will recur.** I messaged `m4` two
orders built on premises I had not checked — that its handoff did not exist, and
that a dirty `docs/CTO_LEDGER.md` was its uncommitted work. Both wrong: the file
was merged in `#154` and my checkout was two commits behind, and the diff was the
**operator's own handwritten answers**. `m4` refused and corrected me. **A stale
`main` makes your own tooling tell you a file does not exist.** `git fetch origin
main` before concluding anything is missing; `git log` before attributing a diff.
The subordinate refusing a wrong order is the control that worked — twice now in
this ledger, counting `cto8`'s inferred attribution.

**The bar rule needed one more clause.** "Never quote a bar against a dirty tree"
did not stop me and `c14` reporting different numbers for the same commit: a
worktree without `models/` reads one extra skip. **Say which tree you measured
in**, not merely that it was clean.

**Green-by-skip is the same defect class as the dirty tree** and cost nothing to
find once someone asked. CI had no ffmpeg while ffmpeg was a declared dependency,
so 5 phase-4 tests skipped in CI and passed locally, all checks green. The fix
that lasts is not the install line but `_need_ffmpeg()` raising under `CI`.

**Mutation-proving became this project's standard by being carried in dispatches,
not by being announced.** `c14` did it unprompted; I put it in `c15`'s dispatch
as an expectation; `c15` ran eight and reported each with the test that caught
it; it is now in `c16`'s. That is the cheapest quality mechanism I used.

**The right shape for a number nobody has measured** — `c15`'s handling of the
missing codec factor: name it, bound the *direction* of the error, put a runtime
check behind it, and escalate the measurement as a benchmarker task instead of
inventing a constant. §16 forbids invented constants; this is what compliance
looks like in practice.

## 5. Ownership, agents, escalations

- **Owned by me** (released on my window's close): `docs/CTO_LEDGER.md`,
  `docs/fleet/runbook.md`. Run `cx o --reap` after.
- **Live agents: `com-c16`** (opus, phase 6), holding the spec,
  `video_playback/`, `tests/test_video_playback.py`, `CLAUDE.md`, `config.py`,
  `web/static/`, `web/templates/`.
- **Open with the operator, neither blocking:**
  1. **CodeQL** — 20 `py/path-injection` false positives; dismissal needs
     `security_events` scope. The real ask is the *accumulation*: CodeQL is red
     by default and not a required check, so every path-bearing route adds more.
     Dismiss, or suppress the query at config level and record it.
  2. **The codec factor** — a `csm` benchmarker measuring output sizes for the
     §3.1 rows against the corpus. Needs an otherwise-idle box, so it must not
     run while a coder is running the suite.
- **The shared checkout is parked at `43a7222`** behind the operator's own
  uncommitted `docs/CTO_LEDGER.md`. Never revert it, never commit it. Everyone
  works in `git worktree`s off `origin/main`; a preserved copy of their file is
  in my scratchpad. My worktree is at `.../5afc2cf2-.../scratchpad/wt-m5`.
