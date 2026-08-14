# Handoff — `scalarforensic-com-m9` (closing manager, ScalarForensic)

UTC 2026-08-14 12:05. **There is no CTO on this project**; the manager owns the
work and escalates to the operator via `~/.claude/cx/cto.md` and the gateway
`scalarforensic-cfm-g4`. Do not spawn a replacement CTO.

## 1. What I was asked to do

Succeed `com-m8`, land `#206`, and execute the two remaining items — the §15 /
`CLAUDE.md` fold and the §6.3 narrowing — each via a fresh coder, with the bar
re-measured on merged `main`. **All of it is done.**

## 2. Done

`main` `c65a416` → **`8a747ea`**.

| PR | sha | What |
|---|---|---|
| `#206` | `3a227f1` | `m8`'s ledger record of the §6.3 ruling (reviewed, body written, merged) |
| `#207` | `d345450` | My opening runbook entry |
| `#208` | `5cfae82` | **`unknown` is not examiner-overridable** (`c25`) |
| `#209` | `317b035` | `c25`'s handoff |
| `#210` | `83047ac` | **The fold — §15 at phases 1–8, `CLAUDE.md` true** (`c26`) |
| `#211` | `8a747ea` | `c26`'s handoff |
| — | — | This handoff and the closing runbook fold |

**`docs/specs/video-playback-transcode.md` is finished.** Phases 1–8 and the
carve, with both post-hoc rulings (`#202`'s §7.2 record, `#206`/`#208`'s §6.3
narrowing) folded into the body rather than appended. §17's remainder is
measurement residue — VFR, long-GOP and damaged-index seek behaviour — not
unbuilt behaviour.

**Bar: 1036 passed / 5 skipped at `5cfae82`**, coverage 75.10%, `npm test` 88/0.
Raw output `data/reports/c26-bar-5cfae82.txt`.

Retired with their work merged: `com-m8`, `com-c25`, `com-c26`.

## 3. The next concrete step

**There isn't one on this spec, and that is the finding.** Nothing on
`docs/specs/video-playback-transcode.md` is outstanding. A successor manager
takes new work from the operator, not from this document.

Two standing non-items, both closed rulings — do not reopen either:

- **CodeQL's ~20 `py/path-injection` false positives** on `_resolve_video_path`.
  Not a required check. Never suppress.
- **`requires-python = "==3.12.*"`** is load-bearing; Dependabot reopens it as a
  docker base-image bump. Close it. (`CLAUDE.md` has the seven places a bump
  would touch.)

One genuinely loose thread, small: §11 promises `_stale_evidence_report` would
move to `audit.py` in phase 8. `audit.py` shipped at `#195`; the function is
still `routes.py:66`. `c26` recorded what happened and argued `routes.py` is the
right home — a stale-evidence *report* is a `playback-info` response field, not a
log entry. **It is documented, not hidden.** If anyone disagrees, that is a
design conversation and not a bug.

## 4. Ownership map

**Verify with `cx o PATH`. Do not trust this sentence** — see §7.

Mine at close, and the successor should reap them once my window is gone:
`docs/CTO_LEDGER.md`, `docs/fleet/runbook.md`, `docs/handoffs`,
`data/reports/dispatch-phase8-close.md`, `data/reports/dispatch-m9-close.md`.

`tests/js/full_job.test.mjs` was `c25`'s and fell out when I killed its window.
Run `cx o --reap` after closing mine.

## 5. Live agents

| Name | Role | State at my close |
|---|---|---|
| `scalarforensic-cfm-g4` | gateway | live — the operator's channel. **Not yours to kill.** |

No coder is live. No worktree of mine or my workers' remains: `/tmp` went 75% →
17%, 27 GB reclaimed, each checked clean and each local commit matched to its
squash-merged PR first.

## 6. Open escalations and pending decisions

**None.** The one I inherited — `m7`'s §6.3 `unknown` question — was ruled on by
the operator, recorded in `#206`, and implemented in `#208`. `~/.claude/cx/cto.md`
carries my inbox line; it asks for nothing.

## 7. What I learned that is written down nowhere else

The runbook fold has the long versions. These are the ones a successor meets
again.

**`cx o --release` drops the lock, not ownership.** `--own --as NAME` by a live
claimant moves a row; killing the window drops every row it holds; `cx o --reap`
correctly refuses a live owner. `m8`'s handoff said "Mine, and I have released
them" and they were still his — the same error `m7`'s handoff had made to *him*.
What made it cost minutes instead of an hour is the sentence he wrote directly
after: *"verify with `cx o PATH`, do not trust this sentence"*. **A caveat that
survives its own author's mistake is worth more than a claim that happens to be
true**, which is why §4 above repeats it rather than asserting a clean release.

**A local bar becomes checkable when you reconcile it with CI's.** `c26` measured
1036/5; CI reads 1035/6. Both describe the same run — **1041 collected on each
side** — and the single test that moves is the YuNet real-model test, which
passes with `models/` copied in and skips in CI, which has no ONNX. That
arithmetic is the strongest corroboration available for a number measured on a
machine no reviewer can see, and it is one `grep` of the CI log. This project has
published a false bar twice; "clean tree, named sha" is the rule that prevents
it, and **this reconciliation is the check that catches it**. Recipe in `c26`'s
handoff §4.

**Both workers found defects in places the dispatch said were fine, and the
dispatch was written from two managers' assessments.** I sent `c25` after four
spec passages; two were the §5 **player-state** table, a different `unknown` from
§6.3's — I had conflated two values that share a name. The passage that actually
disagreed with the ruling was not on my list. I told `c26` that §11 needed
nothing, on `c24`'s assessment carried forward unopened; §11 had three defects.
**A section two people have called fine has been read by nobody.** What found all
six was the instruction "audit it and report what you find either way", never
"fix the following" — a list of suspects tells a worker where to stop looking.

**Ask, do not correct — it is now five managers deep on this project and it has
never once failed.** Every one of the corrections above came back as a worker
telling me my premise was wrong, in a dispatch that had invited exactly that.

**Guarantees with two carriers need a test per carrier.** `c25`'s M4: drop
`verdict.overridable` from the 507 detail and the gate still refuses correctly
while the client still draws an override button. A suite exercising only
`POST /api/video-full` passes clean through it. This is the same shape as `m8`'s
"composition is not covered by covering the parts", one layer down — the gate is
not the disclosure.

**When one item's output is another item's measurement, the measurement runs
last.** I reversed the inherited task order for this reason; had I not, the bar
in `CLAUDE.md` would have been stale on the day the spec was declared finished.

**`gh pr merge --squash --delete-branch` fails its post-merge local step from a
worktree** — the merge lands, the error is cosmetic, do not re-run it. Also seen
once: a `gh` GraphQL read timing out mid-`--watch`. That is the network, not a
check; re-read `gh pr checks N` before believing a failure.

**`pkill -f sfn-web` is banned on this box.** The app on `:8080` is the
operator's. Distinct port, kill by PID.
