# Handoff — `scalarforensic-com-c26`, 2026-08-14 11:48 UTC

## 1. What I was asked to do

Item B of `data/reports/dispatch-m9-close.md` — the fold: bring
`docs/specs/video-playback-transcode.md` §15 from "Done: phases 1–3 and the
carve" to phases 1–8 complete, audit §11's module list, and make `CLAUDE.md`
true again with a bar re-measured on merged `main`.

## 2. What is done

**Merged: PR #210, `83047ac` on `main`.** All six checks green. Docs only; no
source or test file touched, so there is no mutation table — there was no
predicate to mutate.

`docs/specs/video-playback-transcode.md`:

- §15's last paragraph now reads **phases 1–8 complete, and the carve**, verified
  at `5cfae82`, and names the two rulings taken *after* the phase list was
  written — §7.2's recorded-argv record (#202) and §6.3's `unknown` narrowing
  (#206/#208) — folded into the body rather than appended.
- §15's phase 7 **"Browser side not done" clause removed**, replaced by what is
  actually there with symbol names. See §3 below for why removing it was safe.
- §11: three findings, fixed. See §3.

`CLAUDE.md`: the bar (§4), the eleven-module `video_playback/` list, the three
`js/video_playback/` files, `sfn-video render`/`purge`, and the audit log beside
the cache directory — plus one item beyond the dispatch's list: the §6.3 override
bullet still described the gate as it stood before item A merged four commits
earlier. It was not false, only silent on what the operator ruled. A bullet now
pins the narrowing and both `overridable` carriers.

**Nothing is left undone.** The spec is finished: §17's remainder is measurement
residue (VFR, long-GOP, damaged-index seek, multi-sample confirmation of §3.5),
not unbuilt behaviour.

## 3. The next concrete step

**There is none on this spec.** `docs/specs/video-playback-transcode.md` is
closed. What follows is for whoever next touches the file or the bar.

**§11 was audited because I was told to audit it, not because anyone suspected
it, and that is where all three findings came from.** The dispatch said §11 "is
already correct and needs nothing" — `c24`'s assessment, carried forward by `m8`
and by `m9` without either opening the section. What was actually wrong:

| Finding | Evidence |
|---|---|
| the JS line named one file | `full_job.js` (#183) and `rendering.js` (#197) are also on `main`; §11 listed only `player.js`. Now a tree like the Python block. |
| `player.js` was about to be mis-tagged `[carved]` | `git log --oneline --diff-filter=A -1 -- <path>` puts it in **#169, phase 6**. Use that command per file; a phase tag guessed from the neighbours is a guess. |
| a promise that was never kept | §11 said `_stale_evidence_report` "stays in `routes.py` until `audit.py` arrives in phase 8". `audit.py` arrived at #195; the function is still `routes.py:66`. I recorded what happened *and* why `routes.py` is in fact its right home, rather than deleting the sentence quietly — a deleted promise leaves no trace that it was examined. |

The **Python** block was complete and correctly named — the dispatch's own list of
missing modules (`capability.py`, `encode.py`, `jobs.py`, `states.py`,
`audit.py`) was describing `CLAUDE.md`, not §11. Two of them merely lacked a
phase tag their neighbours had.

## 4. Things written down nowhere else

**A local bar that cannot be reconciled with CI's, by naming the exact tests that
differ, is not yet a measurement.** `m9` checked mine rather than taking it: CI
reads **1035 passed / 6 skipped**, I read **1036 / 5**. Same **1041 collected**.
The single test that moves is the YuNet real-model test — it passes in my tree
because I copied `models/` in and skips in CI, which has no ONNX. That
arithmetic is the strongest corroboration a locally-measured bar can get, and it
only works because the collected total is quoted alongside the split. **Recipe
for the next coder measuring a bar:**

1. Measure in a **second, pristine** `git worktree add --detach <sha>`, used for
   nothing but the run — never in the tree that holds your diff. `pytest`
   collects from disk, and a docs-only diff is still a dirty tree by the rule.
2. Capture `git status --porcelain` to a **file, before and after** the run, not
   to the terminal. The after-capture is what makes the measurement checkable by
   someone who was not there; it is also what catches the `git checkout
   origin/main -- .` trap `c25` documented.
3. Copy `models/` in, then confirm the skip count is **5 and not 6**. A 6 means
   the copy did not take and you are one test short of the bar you are about to
   publish.
4. Reconcile against the last CI run on that sha by **naming the tests that
   differ**, not by noting that the numbers are close.

Other things a successor meets again:

- **`gh pr create` does not tell you your PR number before you write the commit
  message.** I put `(#209)` in the subject — #209 was `c25`'s handoff PR, in
  flight behind me — and had to amend and force-push before opening. Leave the
  number out; the squash-merge appends the real one.
- **`m9`'s call on #209 was right and worth reusing:** a docs-only PR in flight
  behind you cannot move the count, so do not rebase and re-measure for it. Say
  in the PR body that you did not, and why.
- **`uv run` in a fresh tmpfs worktree builds a whole `.venv`** (~85 packages,
  a few seconds, plus a hardlink-fallback warning because the uv cache and the
  worktree are on different filesystems). Harmless; budget for it in the first
  command, and do not read the warning as a fault.
- **The shared checkout was never touched.** Both worktrees were on tmpfs under
  the scratchpad; the only thing I wrote to
  `/home/user01/Schreibtisch/gitea/ScalarForensic` is
  `data/reports/c26-bar-5cfae82.txt` (10,097 B, the full run output including
  the coverage table and `EXIT=0`), which is confirmed present. No app was
  started, so nothing came near :8080.
- **Verify an affordance is *armed*, not merely defined.** For the phase 7
  browser check the weak version is grepping for `beforeunload` and calling it
  present. `full_job.js:391` defines `_armLeavePrompt`; what makes it real is
  that `:288` and `:312` call it and `:325` disarms it. A handler defined and
  never registered greps identically to one that works.
- Ownership: I hold `CLAUDE.md` and `docs/specs/video-playback-transcode.md`,
  released on close. `c25` flagged `tests/js/full_job.test.mjs` as still needing
  its row reaped — not mine, but it is the one loose end I know of.
