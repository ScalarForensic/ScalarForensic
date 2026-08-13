# Handoff — scalarforensic-cfm-m2, 2026-08-13 12:12 UTC (retiring at ~204k, operator-caught)

Successor: `scalarforensic-cfm-m3` (live, has read the runbook through my entry tail).
The runbook (`docs/fleet/runbook.md`, cfm-m2 sections) carries the full narrative; this
file adds only what m3 needs to act.

## 1. What I was asked

Successor manager for the iPhone campaign: dispatch-only (never implement, never switch
the shared checkout's branch), verify workers' DONEs by re-measurement, relay operator
rulings, report to `scalarforensic-cfm-g1`.

## 2. Done, with shas (all verified by me: PR MERGED + CI green + bar re-measured)

| PR | sha | what |
|---|---|---|
| #115 | `ca86cd7` | per-run manifest + `SFN_REPORT_DIR` (c1) |
| #116/#119 | `4a05946`/`9e22ed1` | runbook docs (mine) |
| #117 | `0df8035` | benchmark subset, 814 files, seed 20260813 (c3) |
| #118 | `bb4cf41` | face detect folded into batch loop, −8.3% wall / −42% RSS (c2) |
| #120 | `b4582a9` | residual face-pass threading, 3.2× (c5) |
| #121 | `478c131` | bench report `docs/fleet/bench-face-integration-20260813.md` (b1) |
| #122 | `860bee5` | UI items 1+2: FACE header pill, sectioned controls (c6) |
| #123 | `e66fa78` | HEIC hit serving: whitelist from scanner + HEIF→JPEG transcode (c7) |
| #124 | `4bd9418` | POST /api/faces/compare + point_id probes, review-only rejected (c6) |

**Bar at `e66fa78` (current main, shared checkout, clean, models/ present):
618 passed / 5 skipped, coverage 67.63%, ruff clean via CI.** Worktree rule: without
`models/` the same suite reads N−1 passed / 6 skipped (YuNet real-model test skips).

Campaign run COMPLETE (operator-triggered): 8,137 indexed, CSV analysis in the runbook
("3 embed-failed" = in-run duplicate videos, zero real failures; 76 unsupported = 75
`.aae` + operator's `mapping.csv`).

## 3. Next concrete steps (m3)

1. **c6: handoff LANDED** — `docs/handoffs/scalarforensic-cfm-c6-20260813-121211.md`;
   items 3+4 WIP pushed to branch `ui/face-basket` (`1264006`, based on `4bd9418`): js
   parts + 10 wiring tests done (RED BY DESIGN — they are the successor's TODO list),
   `index.html`/`style.css` not started. c6 runs `cx q` next — then
   retire it (`cx k`), spawn its successor **as `com-c`** (operator model policy — memory
   `fleet-model-policy`; my `cfm-c` spawns were inherited drift, self-reported to g1)
   with the handoff + the spec at c4's scratchpad `ui-changeset-2026-08-13-faces.md`
   (READ ITS "## UPDATE" SECTION — the operator refined items 3/4; mixed-origin basket,
   auto pairwise compare, aggregation by max-score-per-hit).
2. **Operator-timed sfn-web restart** — batches #123 + #124 (both flagged in PR bodies).
   Until restart: HEIC hits still 400 in the live UI and the new endpoints 404. After
   restart c4 verifies both. Statics are already live (I pulled after every merge — that
   IS the deploy for static files; pull is safe, it is not a branch switch).
3. **c4 (operator's frontend tester, 89.0k, busy — NOT yours to retire)**: loop contract
   is: coder merges → manager pulls → c4 verifies in the operator's live Chrome
   (force-refetch part files + style.css; cachebust busts only HTML) → defects come to
   the manager as NEED with an evidence file in ITS scratchpad
   (`/tmp/claude-1000/-home-user01-Schreibtisch-gitea-ScalarForensic/cf8c108a-.../scratchpad/`).
4. **Queued, unassigned**: stale-observation prompt should name files; Kalman ETA line
   removal (decoration rule); relabel all-frames-in-run-duplicate videos
   (`cli.py:1762-1767`, misreported as embed-failure); Qdrant client 1.19.0 vs server
   1.17.1 skew; optional full-corpus projection re-bench at `b4582a9` (c5's parting rec).
5. **With the operator, not workers**: `sfn_tags` drop one-liner
   (`curl -X DELETE http://localhost:6333/collections/sfn_tags`) — ruled yes, but MY
   permission classifier denied it and per fleet precedent it is not routed around;
   likely yours will too. CodeQL py/path-injection on #123's transcode — c7 judged it a
   false positive matching the 13 open siblings (path passes `_check_allowed_path`
   first); its dismissal was permission-blocked and correctly left alone.

## 4. Learned, written nowhere else

- **The retirement watcher is BROKEN** (filed high via `cx f`): `cx w` misreads every
  LIVE session as `dead, CTX 0`, so no context nudge ever fires — that is why the
  operator had to catch me at 203.6k by hand. Size workers off `cx s` yourself.
- Campaign Qdrant `localhost:6333` is **read-only for all workers** (operator live on
  the UI). Throwaway instances on other ports for anything write-shaped.
- The permission classifier here blocks: Qdrant collection DELETE, whole-`.env` dumps
  (grep specific keys instead), CodeQL alert dismissal.
- Never `tail` a state-changing git command — a blocked `git pull --ff-only` hid its
  abort behind `tail -1` and left main stale while I reported it moved (runbook, fixed).
- `gh pr merge --delete-branch` fails silently on remote deletion sometimes — verify
  `git branch -r` after; two merged branches survived on origin today.

## Ownership map / live agents / escalations (spec C 3)

- Ownership: `docs/fleet/runbook.md` is mine; **ownership transfers only when my window
  is closed, not on `cx q`** (m1→m2 precedent). c6 holds the 5 UI/routes files until
  killed; c7/c5/b1/c1/c3 released on kill. No locks held (cto.md lock released).
- Live: g1 (CTO), m3 (you), c6 (winding down), c4 (operator's tester). c7 killed by me
  post-verification.
- Open escalations: none at g1 beyond the restart timing; all my DONEs reported.
