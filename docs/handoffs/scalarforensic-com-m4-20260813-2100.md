# Handoff — `scalarforensic-com-m4` (manager, ScalarForensic)

Written at ~186k context. **There is no CTO on this project**; the manager owns
the work and escalates directly to the operator. Do not spawn a replacement CTO.

## 1. What I was asked to do

Land phases 1–3 of `docs/specs/video-playback-transcode.md`, put the two phase-4
blockers to the operator with a recommendation, and fold `docs/CTO_LEDGER.md`.

## 2. Done

All of it, plus the §11 carve and phase 4's unblocking.

| PR | sha | What |
|---|---|---|
| `#145` | `1a6ad56` | Phase 1 — digest via `HashCache`, three-state stale-evidence |
| `#147` | `6539abb` | Phase 2 — `/api/video-download` + UI escape route |
| `#148` | `b3a609a` | Phase 3 — codec allowlist, `mode`+reason, `_evict_cache` contained |
| `#150` | `f6cef02` | §11 carve into `video_playback/` |
| `#151` | `bda812b` | `CLAUDE.md` bar sha |
| `#146`/`#149`/`#152`/`#153` | | ledger + runbook folds, operator rulings |

**Bar: 771 passed / 5 skipped, coverage 70.22%**, re-measured by me on a clean
tree at `f6cef02` — twice, never inherited from a coder's report.

**The operator answered every open question** (2026-08-13). Rulings are in the
ledger under "operator rulings — PHASE 4 IS UNBLOCKED". Nothing is pending.

**Not done: phase 4 itself.** It is unblocked and un-started. Phases 5–8 follow.

## 3. The next concrete step

**Spawn ONE fresh coder (`com`, opus) on phase 4.** It has everything it needs:

- `docs/specs/video-playback-transcode.md` §15.4 is the phase; §11's tree marks
  which modules exist (`codecs`, `digest`, `rewrap`, `cache`, `routes`) and which
  it writes (`capability.py`, `encode.py`; `jobs.py` is phase 7, `audit.py` is 8).
- **`data/reports/video-bench-2026-08-13.md`** is the §14 measurement set. It is
  gitignored — read it before it is lost, and fold its numbers into spec §3.
- The fixture ruling: a committed HDR sample under a **tracked `test_data/`**.
  Derive the rotation case from it at test time with `ffmpeg -c copy` metadata
  add — public HDR clips do not carry `rotation=-90`, that is an iPhone artifact
  and one of the two defects §3.1 found. If no small licence-clean clip exists,
  generate one with `-f lavfi` and commit that.
- Config values the measurements settle: `SFN_VIDEO_MAX_WORKERS=2`,
  `SFN_VIDEO_OUTPUT_HEIGHT=1080`, `SFN_VIDEO_CHUNK_SECONDS=30`.

## 4. What I learned that is written down nowhere else

**The one finding the benchmark implies but nobody has stated.** Aggregate
throughput is *flat* across k=1..8 (~3.5–3.7× video-s/wall-s) — one job already
saturates the box — so per-job latency scales almost linearly with k: 8.08 s at
k=1, 16.35 s at k=2, 67.34 s at k=8. §4.2's prefetch-depth-1 design assumes a
30 s 1080p chunk lands in 6–10 s, and 8.21 s only just fits.

**Therefore `SFN_VIDEO_MAX_WORKERS=2` is not a throughput knob, it is the exact
shape of one viewer: the playing chunk plus its single prefetch.** That is
coherent — but it means **the §4.3 full-video job contends directly with chunk
playback**. §4.3 says it "runs in the background; nothing blocks on it". On this
hardware that is false: a full job occupying one of two workers doubles chunk
latency to ~16 s for its entire ~51-minute duration, blowing the margin the
double-buffer swap depends on. **Recommendation for whoever writes phase 7: run
the full-video job at lower priority (`nice` + an explicit `-threads` cap) so it
yields to chunk work, or state plainly in §4.3 that chunk playback degrades
while a full job runs.** It cannot be both unbounded and invisible. Raise it to
the operator when phase 7 is reached; it is not phase 4's problem.

**Process things worth keeping:**

- **The shared checkout is one working tree.** A coder on a feature branch blocks
  the manager from committing its own docs. Use `git worktree add` off
  `origin/main`; never switch the shared checkout out from under a worker. Note
  `gh pr merge --delete-branch` fails from a worktree ("main is already used by
  worktree") — the PR still merges, but delete the remote branch by hand after.
- **Retire a coder one task early and buy a map.** `c12`'s handoff is why the
  carve cost one PR; `test_video_endpoints.py` needed zero changes, as it
  predicted. Now a standing rule in the ledger.
- **Prose outlives code here — three instances this window.** §11's claim that
  `/api/video-timeline` calls `_resolve_video_path` (it does not), `CLAUDE.md`'s
  bar sha pointing at a squashed-away branch commit, and this ledger's own "what
  this is" claiming perceptual hashing that has never existed in `src/`. All
  three were found by someone checking rather than trusting. Keep checking.
- **The operator's local `docs/CTO_LEDGER.md` carries their handwritten answer
  annotations.** Every one is folded properly into `#153`; the working-tree
  edit is theirs to discard when they choose. Do not revert it for them.

## 5. Ownership, agents, escalations

- **Owned by me** (release on my window's close): `docs/CTO_LEDGER.md`,
  `docs/fleet/runbook.md`. Run `cx o --reap` after.
- **Live agents: none.** `c12`, `c13`, `b1` all retired, ownership released.
- **Open escalations: none.** The operator answered all five. The 17 CodeQL
  `py/path-injection` alerts are verified false positives the operator dismisses
  with their own token scope — do not re-investigate.
