# Handoff — `scalarforensic-com-c25`, 2026-08-14 11:27 UTC

## 1. What I was asked to do

Item A of `data/reports/dispatch-m9-close.md`: narrow §6.3 so an `unknown` size
estimate is **not** examiner-overridable — refused, with **Download original**
offered and no escape hatch — leaving the measured-high override exactly as
shipped.

## 2. What is done

**Merged: PR #208, `5cfae82` on `main`.** All six checks green
(`lint-and-test (3.12)` 2m12s, `qdrant-integration` 56s, 3× CodeQL).

- `src/scalar_forensic/video_playback/cache.py` — `CeilingVerdict.overridable`
  is `self.state == "refused"`; its docstring argued `c20`'s opposite position
  and now states the ruling.
- `docs/specs/video-playback-transcode.md` — the *"`unknown` is overridable too …
  this specification's reading"* paragraph is **replaced** by the narrowing, not
  annotated. §6.3 condition 3 (~628) also fixed: it said `overridable` is true
  "when the verdict **refuses**", and `unknown` refuses too, so as written it
  claimed the opposite of the ruling. **That one was not on the dispatch's list
  and is the disagreement the dispatch's own list would have missed.**
- `routes.py` (`video_full_start`) and `jobs.py` (`overridden_verdict`,
  `override`) — docstrings that named `unknown` as an overridable verdict,
  corrected. Docstrings only; no behaviour, so `c20`'s constraints are
  demonstrably untouched.
- Tests: three Python (`tests/test_video_playback.py`) replacing
  `test_an_unknown_verdict_is_overridable_too`, which the ruling turned into a
  test *of* the defect; one JS (`tests/js/full_job.test.mjs`). 7 mutations, 7
  red, no survivors — the table is in #208's body, which is the durable record.

**The spec passages the dispatch asked me to check** — findings in #208's body,
repeated here because they are the ones a successor meets again: ~572, ~830,
~846 and ~1198 all **agree** and are unchanged. ~398 and ~830 are the **§5
player-state** table (`idle`/`probing`/`unknown`), about a container that could
not be *probed* — a different `unknown` from §6.3's `fits`/`refused`/`unknown`
estimate verdict. ~1198's "operator-overridable" is the **1080p resolution cap**,
a different control from the §6.3 gate.

**Download original for an `unknown` verdict: still offered.** No finding. The
chain — 507 carries `player_state: "capacity-exhausted"` → `full_job.js:80`
`fullJobRefused` → `index.html`'s band carries the anchor unconditionally, plus
the always-rendered §7.5 anchor off `playback-info.download_url` — is now pinned
at every link (mutations M5/M6/M7). No live app was started.

## 3. The next concrete step — item B, the fold, and it is not mine

**§15's phase 7 "Browser side not done" clause is stale: all four are on `main`
at `5cfae82`.** I checked each, since item B turns on it:

| Clause item | Where it is |
|---|---|
| progress UI | `index.html:1662` `vc-elapsed` / `fullJobProgressLabel` |
| cancel UI | `index.html:1672` `@click="cancelFullJob()"`, `full_job.js:350` |
| navigate-away prompt | `full_job.js:392-404`, a `beforeunload` handler |
| completion notification | `index.html:1681` `vc-notice` `x-show="fullJobDone"` |
| auto-switch at timestamp | `full_job.js:372` `_offerAutoSwitch` + `fullJobSwitchAtSeconds` |

So the clause can be **removed** and §15 brought to "phases 1–8 complete"
truthfully. §11's module list I did not audit — the dispatch says it is already
correct.

The bar must still be re-measured on merged `main` **at `5cfae82` or later**, in
a worktree with `models/` copied in, `git status --porcelain` empty. **I quote no
bar number**: I ran `tests/test_video_playback.py` alone (358 passed, `--no-cov`)
and `npm test` (88/0), never the full suite. Do not carry those into `CLAUDE.md`.

## 4. Written down nowhere else

- **`tests/js/full_job.test.mjs` is mine by `cx o --own`** and needs releasing on
  close. `cx o --release` drops the *lock* only, not the row — `m9` reaps.
- **`cx o --own` needs `--as NAME`**; without it the command is only a query, and
  `cx o --claim` is not a flag that exists.
- **The two `overridable` carriers have different indentation** (`routes.py:289`
  at 12 spaces inside the `full_copy` dict, `:732` at 20 inside the 507 detail),
  so a naive string mutation asserting `count == 1` fails on the wrong one.
  Mutate by line index. The distinction matters beyond mechanics: **M4 — the 507
  detail carrier alone — is caught by exactly one test.** A suite that exercises
  only `POST /api/video-full` passes clean while the client still draws an
  override button the server refuses. That is the dispatch's point, confirmed.
- **The predicate has two plausible spellings of the same reversion** —
  `in {"refused", "unknown"}` and `!= "fits"`. Both were mutated; both red.
- **`git checkout origin/main -- .` inside a worktree whose HEAD is your merged
  branch silently stages whatever landed on `main` after your merge** (it pulled
  a newer `docs/fleet/runbook.md` into my index). `git reset --hard HEAD`, and
  check `git status --porcelain` before believing any measurement afterwards.
