# Handoff — `scalarforensic-com-m8` (manager, ScalarForensic)

UTC 2026-08-14 10:24. **There is no CTO on this project**; the manager owns the
work and escalates to the operator via `~/.claude/cx/cto.md` and the gateway
`scalarforensic-cfm-g4`. Do not spawn a replacement CTO.

## 1. What I was asked to do

Take over from `com-m7` and get spec phase 8 — provenance and audit — merged
green, which finishes `docs/specs/video-playback-transcode.md`.

## 2. Done

`main` `5ca812a` → `ea9ddab` (and `#202` behind it). **Phase 8 is shipped, server
and browser, and §7.2 has been demonstrated rather than asserted.**

| PR | sha | What |
|---|---|---|
| `#192` | `0402271` | My runbook opening entry |
| `#193` | `03847d6` | **`video_playback/audit.py`, the §7.2 label and §7.3 records** (`c22`) |
| `#194` | `6f1c7c1` | **`sfn-video render`**, and `sfn-video purge` finally files (`c22`) |
| `#195` | `7064d5d` | `c22`'s handoff |
| `#196` | `5397ddb` | My fold 1 |
| `#197` | `c22e15d` | **The §7.2 label renders the whole pipeline** (`c23`) |
| `#198` | `c706d8a` | `closeFullJob()` gets its caller (`c23`) |
| `#199` | `a37378a` | `c23`'s handoff |
| `#200` | `d1637d6` | My fold 2 |
| `#201` | `ea9ddab` | One download label, both placements (`c24`) |
| `#202` | — | **F1 + F2 from the real-encode check** (`c24`) |

Retired with their work merged: `com-m7` (its window, holding ownership it had
announced as released), `com-c22`, `com-c23`. `com-c24` is live.

## 3. The next concrete step

**Task 3 of `data/reports/dispatch-phase8-close.md`, and it is the last one.**
`c24` has it and starts once `#202` is merged:

- **spec §15** still says "Done: phases 1–3 and the carve" — false since phase 4;
- **`CLAUDE.md`** needs the bar **re-measured on merged `main`** (the file says
  771/5 at `f6cef02`; 1,032/5 and 1,025/5 are *branch* figures — quote neither),
  the `video_playback/` module list (it names none of `capability.py`,
  `encode.py`, `jobs.py`, `states.py`, `audit.py`), the audit log at
  `data/video_audit.log` beside the cache, and `sfn-video render` / `purge`.

When that merges, the spec is finished.

## 4. Ownership map

- **Mine, and I have released them** — verify with `cx o PATH`, do not trust this
  sentence, which is the whole lesson of §7 below: `docs/CTO_LEDGER.md`,
  `docs/fleet/runbook.md`, `docs/handoffs`, `data/reports/dispatch-*.md`.
- **`com-c24`** (live): `src/scalar_forensic/web/static/`, `tests/js/`,
  `src/scalar_forensic/video_playback/`, `tests/test_video_playback.py`,
  `docs/specs/video-playback-transcode.md`, `CLAUDE.md`.
- Run `cx o --reap` after `c24` closes.

## 5. Live agents

| Name | Role | State at my close |
|---|---|---|
| `scalarforensic-com-c24` | coder | live — task 3, the fold. Retire it after that merges and its handoff is in. |
| `scalarforensic-cfm-g4` | gateway | live — the operator's channel. Not yours to kill. |

## 6. Open escalations and pending decisions

**None open that I raised.** One inherited from `m7` is still with the operator:
**§6.3's `unknown` verdict was made overridable as the implementation's reading**,
marked as such in the spec. The operator ruled on the measured-high case, not on
`unknown` by name. It is one spec paragraph and one predicate if they narrow it.

Two standing non-decisions: **CodeQL**'s ~20 `py/path-injection` false positives
on `_resolve_video_path` (closed ruling, not a required check, never suppress);
and **the app on `:8080` is a new process** — `c23`'s `pkill -f sfn-web` took the
operator's down with its own and it restarted it. Reported to `cfm-g4`.

## 7. What I learned that is written down nowhere else

The runbook folds at `#196` and `#200` carry the long versions. These are the ones
a successor meets again.

**A live check is worth exactly what its fixture is, and only the worker knows.**
`c23`'s §7.2 screenshots were a hand-injected payload, not a probe. I found it
only because I opened the PNG myself and noticed the record saying *"the source
carries no audio track"* nine rows above `STREAMS hevc / aac` — and I put it as a
question with three ways it could resolve, explicitly not as a finding. Good
thing: my finding would have been *"`has_audio` is computed wrong at
`routes.py:490`"*, and it was false. `c23` answered with the expensive truth, and
the cheap move — letting me keep believing it was a probe — would have held,
because I had already accepted the PR on substance.

**So the check nobody had done was the one that found the defects.** `c24` ran it
against a real HDR file and returned two: the one recorded argv printed two ways
(`argv.join(' ')` on screen, `shlex.join` in the CLI — only the CLI's survives
being pasted), and `_pipeline_lines` hard-coding ten fields while the record
carries seventeen, so `sfn-video render --at 5` answered about the chunk at 0 and
never said so. **Every mutation test on both sides passed throughout**, because
each side was correct in isolation. Composition is not covered by covering the
parts. `data/reports/c24-task2-label-vs-render.md` is the artifact.

**`_pipeline_lines` was the same defect `Pipeline.describe()` exists to prevent,
reintroduced one layer later.** That docstring says a hand-listed label would
quietly fall behind the fingerprint — and then a hand-listed CLI printer did
exactly that, in the same package, written by the same careful author. **A rule
enforced at one layer is not enforced.** The fix on both surfaces is the same:
walk the payload, lead with the known fields, append what you have never heard of.

**Ask, do not correct.** Three times a worker was right and my draft finding was
wrong or incomplete — this is `m7`'s lesson and `m6`'s before it, and it kept
holding. The formulation that works: state the standard, name the evidence, list
the ways it could resolve, and ask which. `c24` did the same thing upward when it
found F1/F2 and asked before folding §15 rather than after — which is the only
reason they were fixed instead of documented as shipped.

**A handoff saying you released a file is not a release.** `m7` announced its
three ownership rows free in prose, filed `cx q`, and kept its window open — so
`cx o --reap` correctly refused them (a live owner is a live owner) and I could
not fold the ledger until I closed its window. Every dispatch I wrote afterwards
says "release it with `cx o --release`, not in your handoff", and `c22` and `c23`
both did. **Verify ownership with `cx o PATH`, including against §4 above.**

**The PR body is the durable record and must agree with itself.** `#197` shipped
claiming 15 mutations over a 14-row table with a JS bar that disagreed with the
report I had been sent. The code was fine. I asked for the body, not the code, and
got it — the same family as `#177`, where a document's own text claimed committed
files that `.gitignore` had silently declined.

**Check the worktree, then delete it.** 20 GB reclaimed (`/tmp` 59% → 17%) with
`git status --porcelain` clean on each and every local commit matched to a
squash-merged PR first. The heuristic that skips that check once reported two
*live* worktrees as stale on this project.

**`gh pr merge --squash --delete-branch` fails its post-merge local step from a
worktree** (`'main' is already used by worktree`). The merge lands; the error is
cosmetic. Do not re-run it — check `gh pr view N` instead.

**`pkill -f sfn-web` is banned on this box.** Distinct port, kill by PID.
`bitcoin_psql` is mid-transition here with a ~60 GB budget and dead swap — check
`MemAvailable` before anything heavy, and reclaim worktrees as you retire workers.
