# Handoff — `scalarforensic-com-m7` (manager, ScalarForensic)

UTC 2026-08-14 08:10. **There is no CTO on this project**; the manager owns the
work and escalates to the operator via `~/.claude/cx/cto.md` (gateway
`scalarforensic-cfm-g3`). Do not spawn a replacement CTO.

## 1. What I was asked to do

Take over from `com-m6` mid-flight, get spec phase 7 merged green — server **and**
browser — then phase 8.

## 2. Done

**Phase 7 is complete and merged, both halves.** `main` `2fdbc3a` → `fd31488`.

| PR | sha | What |
|---|---|---|
| `#176` | `db16634` | m6's handoff |
| `#177` | `f560319` | Benchmark reproduction scripts (by `csm-b2`) |
| `#178` | `8e09490` | Runbook pointer + m6's procedural note |
| `#179` | `1bb24b0` | **Phase 7 server side** (by `c19`) |
| `#180` | `32d9da8` | Bar sha correction |
| `#181` | `c4b6e58` | My fold 1 — runbook + ledger |
| `#182` | `ade6264` | `c19`'s handoff |
| `#183` | `18275e8` | **Phase 7 browser side** (by `c18`) |
| `#184` | `45fe545` | **§6.3 override** (by `c20`) |
| `#185` | `6e19bdb` | §6.3 spec — the ruling replaces the open escalation |
| `#186` | `a809331` | `c18`'s handoff |
| `#187` | `afa18a1` | §14 — the "no JS harness" paragraph, false since `#173` |
| `#188` | `fd31488` | **§6.3 override render** (by `c21`) |

**Bar: 991 passed / 5 skipped, coverage 74.58%** at `a809331`, measured by `c20` in
a worktree with `models/` copied in (without it, 978/6 — the skip count is how you
know the copy happened). `npm test` 58/0 at `#188`. **`CLAUDE.md` still quotes
979/5 at `1bb24b0`** — `c20` is landing the re-quote at `fd31488`; if its PR is not
in when you arrive, that is the first thing to merge.

**§6.3 was ruled and shipped in the same window.** The operator (delegated to
`cfm-g3`) ruled a full-copy refusal examiner-overridable; `c20` implemented it,
`c21` rendered it. Four conditions, all enforced in code and all tested: explicit
per request, attributed or refused (403 `override-unattributed`), disclosed for the
life of the job and on the copy it produced, and **it bypasses the forecast, never
`limit_bytes`**. Ledger `docs/CTO_LEDGER.md` last section carries the full ruling.

## 3. The next concrete step

**Phase 8 — provenance and audit — is the only phase left, and its dispatch is
already written**: `<my scratchpad>/dispatch-phase8.md`. It is thin on purpose and
points at spec §7 entire, §13's last paragraph, §11, §15 item 8, and at `c19`'s
handoff §5 for what phase 8 inherits. Copy it somewhere durable before my scratch
dir goes — **it is on the tmpfs**. If it is gone, the four traps it exists to carry
are: `result.pipeline` never `request.pipeline` (a record built from the request
names the encoder *selected*, not the one that ran — §7.2's exact failure mode);
`sfn-video render` must record `-threads` or a full copy will not reproduce
byte-identically on a different core count; the digest is the **verified** one from
`HashCache`, never the index-time `video_hash`; and refcounts mean one encode can
have N claimants, so decide whether that is one record or N and test it.

Two smaller things, neither blocking:

1. **`closeFullJob()` has no caller** — closing the player leaves the full-job poll
   running. Pre-existing, found by `c21`, deliberately left alone mid-task so as not
   to disturb `c18`'s `beforeunload` contract. Give it to whoever next touches
   `web/static/`.
2. **Two download affordances in one panel** — "Download the original instead" in
   the refusal band and "Download original (clip.mov)" below it. Cosmetic; I saw it
   in `data/reports/live-override-offered-on-refusal.png` and did not file it.

## 4. Ownership map

- **Mine, now released; claim them**: `docs/CTO_LEDGER.md`, `docs/fleet/runbook.md`,
  `docs/handoffs`.
- **`com-c20`** (live): `docs/specs/video-playback-transcode.md`,
  `src/scalar_forensic/video_playback/`, `tests/test_video_playback.py`,
  `src/scalar_forensic/config.py`, `CLAUDE.md`.
- **Free** (released when `c21` and `c18` were closed): `web/static/`, `tests/js/`,
  `ci.yml`, `package.json`.
- Run `cx o --reap` after my window and `c20`'s close.

## 5. Live agents

| Name | Role | State at my close |
|---|---|---|
| `scalarforensic-com-c20` | coder | live ~153k — landing the `CLAUDE.md` bar/override PR, then its own handoff. Retire it after both merge. |
| `scalarforensic-cfm-g3` | gateway | live — the operator's channel. Not yours to kill. |

Retired by me, each with its work merged: `csm-b2` (`#177`), `com-m6` (`#176`),
`com-c19` (`#182`), `com-c18` (`#186`), `com-c21` (`#188`, no artifact — its
evidence file is `data/reports/c21-override-ui-verification.md`, on disk).

## 6. Open escalations and pending decisions

**None open with the operator.** §6.3 was the only one and it is closed. Two
standing items that are *not* decisions:

- **CodeQL** — ~20 `py/path-injection` false positives on `_resolve_video_path`, a
  closed ruling. Dismissal needs `security_events` scope and is the operator's own
  action. CodeQL is not a required check. **Never add a suppression to buy a green
  icon.**
- **One thing I flagged to `cfm-g3` rather than decided silently**: §6.3's `unknown`
  verdict was made overridable as the *implementation's* reading, and the spec says
  so in those words. The operator ruled on the measured-high case, not on `unknown`
  by name. The enforced guarantee is identical — `limit_bytes` does not depend on
  the estimate existing — so I accepted it. If the operator dislikes it, it is one
  spec paragraph and one predicate.

**The operator has offered direct machine access and their own eyes on screenshots**
when a task would otherwise burn tokens. Live browser checks are the case that
matters. Acted on: live-check evidence now goes to `data/reports/` (on disk,
gitignored) instead of a worker's tmpfs scratchpad, so it outlives the worker.

## 7. What I learned that is written down nowhere else

The runbook fold at `#181` carries the long versions of the first two. These are
the ones a successor will meet again.

**A worker self-merged 1750 lines and the review gate simply did not happen.**
`c19` took `#179` from open to merged two minutes after CI went green. The work was
good — 11 mutations reported individually, 3 survivors turned into named tests, each
one a trap `c17` had documented as written down nowhere else. That is exactly why it
is worth recording: **a self-merge that happens to be sound is indistinguishable,
afterwards, from one that is not.** Green CI is not a review; it is the precondition
for one. Every dispatch I wrote after that says "report the PR number and wait" in
those words, and no worker self-merged again.

**Verify the claim, do not assume the cheap explanation.** I flagged `CLAUDE.md`'s
bar as quoted against the branch tip instead of the merge commit — the defect `#151`
existed to fix. `c19` pushed back with evidence: it had re-checked-out `origin/main`
at the merge commit and re-run the whole suite there, so the sentence was already
true and my "fix" would have made it false. Its worktree reflog confirmed it. **A
worker that re-measures is externally indistinguishable from one that retyped a
sha** — the difference is only visible if you ask. Ask.

**A gitignore rule can empty a document's central claim in silence.** `#177` existed
solely to put `b2`'s reproduction scripts in git; its own edited text then claimed
two `taskb_*.log` files were committed, and `.gitignore:59`'s `*.log` had skipped
them without a word. The fix for "a report naming paths not in git" itself named
paths not in git. **`git add` on a directory reports what it added, never what it
declined to add — verify with `git ls-tree` on the pushed branch, not `git status`
on the author's box.** Generalised: if a committed file names a path, that path must
be in git too, or the claim is empty.

**Quiescing the box is a manager operation, and both edges must be announced.** I
inherited this from `m6` and it held: two CPU windows granted (`c18`, `c21`), both
opened and closed by announcement to every holder, `c20` released by name each time.
`c21` asked me to wait for "the operator to free the box" — granting the window is
the manager's, not theirs, and waiting on a person who was never going to be asked
is the same stall that cost `c17` its verification. Correct it when a worker assumes
otherwise.

**The workers out-caught me twice and the pattern is the same one `m6` recorded.**
`c19` corrected my false finding with a reflog. `c21` reported its own **surviving**
mutation (M14) as a finding rather than banking 16 green ones, and the extended test
now covers the refused-start path the line actually guards. Dispatches that state
the standard and then trust the worker keep producing this; supervision would not
have found either.

**Live-check screenshots belong on disk, not in a scratchpad.** `c18`'s are already
gone with its tmpfs; that cost nothing only because its measurements were in its
handoff, which was luck. `c21`'s are in `data/reports/` and a human can open them.
Three pieces of work nearly died on that ramdisk in one day — the rule is now: push
a branch when it becomes expensive to recreate, and write evidence where it outlives
its author.
