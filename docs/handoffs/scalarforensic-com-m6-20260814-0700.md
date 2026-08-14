# Handoff — `scalarforensic-com-m6` (manager, ScalarForensic)

UTC 2026-08-14 07:00. **There is no CTO on this project**; the manager owns the
work and escalates directly to the operator. Do not spawn a replacement CTO.

## 1. What I was asked to do

Run spec phase 7, address §4.3's inherited worker contention rather than passing
it on, and execute the operator's two 2026-08-14 YES rulings (a codec-factor
benchmark, and adopting a real JS test runner).

## 2. Done

`main` moved `f2b07a7` → `2fdbc3a`.

| PR | sha | What |
|---|---|---|
| `#172` | `ce4afa3` | Runbook fold — dispatches + the §4.3 ruling |
| `#173` | `1e34aef` | **JS test runner** (by `c18`) |
| `#174` | `78c5cd3` | Runbook fold 2 — window-cancel rule, tmpfs, disclosure |
| `#175` | `2fdbc3a` | **Codec-factor + §4.3 residual benchmark** (by `csm-b2`) |

**Two of the three done-criteria are closed:** the benchmarker report landed and
the JS-runner ruling is executed. **Phase 7 is not merged** — see §3.

**Bar: 976 passed / 5 skipped, coverage 74.48%**, measured by `c19` in c17's
worktree at `b2f9db9` plus one test fix. Not yet re-quoted against a merge
commit on a clean tree — `c19` owes that. Baseline was 954/5 at `f2b07a7`.

### The JS runner (`#173`)

`node --test` (Node built-in), **zero npm dependencies** — `package-lock.json`
has one entry, its own root — run by `npm test` as a **step inside
`lint-and-test (3.12)`**, not as a job of its own. That wiring is a ruling, and
the reason is in a comment in `ci.yml`: `main`'s ruleset requires exactly
`lint-and-test (3.12)` and `qdrant-integration`, so a third job would not be
required and could be red while the PR merged. A decorative check is worse than
none because it reads as coverage.

Verified independently at my end: 22/22 in 91 ms, and the mutation proved it —
reintroducing `player.js`'s `?? … ||` `SyntaxError` gives 12 JS failures naming
`player.js:118` while the Python suite stays 954/5 green on the same tree.

### The benchmark (`#175`) — it overturned its own premise

The dispatch, the operator's ruling and §6.3/§4.3 all say
`estimate_full_output_bytes()` "runs low on exactly the HEVC corpus this feature
exists for". **Measured false.** `output_bytes / estimate`:

| Class | Row | n | median |
|---|---|---|---|
| HEVC 10-bit HDR | CPU | 8 | **0.301** (min 0.125) |
| HEVC 10-bit HDR | GPU | 8 | 0.471 |
| HEVC 8-bit | CPU | 8 | 0.886 |
| HEVC 8-bit | GPU | 8 | **1.409** |
| H.264 1080p | CPU | 4 | 0.309 |

The estimate runs **high** on the CPU pipeline — 8/8 under, as low as an 8×
over-estimate. The only class matching the spec's predicted direction is HEVC
8-bit **GPU**, which is not the case the spec names. n=3–8, so the *ordering* is
the finding; the ratios do not support a single constant and none was derived.

§4.3 yield residual, n=6 per arm, CPU/libx264, sustained competitor confirmed at
the start of all 12 reps: **unniced 18.31 s median, niced (`nice` 10,
`-threads` 12) 16.83 s.** ~9%, and **neither is inside §4.2's 6–10 s window**.
Unniced is also worse than §4.3's own ~16.35 s extrapolation from §3.5.

## 3. The next concrete step

**Get `c19`'s phase-7 server PR merged, then send `c18` the go.** In order:

1. `c19` (live) is finishing 7 mutation checks, the §6.3/§4.3 corrections,
   `CLAUDE.md`, then the PR. Branch `feat/video-phase7`, on origin at `b2f9db9`
   (tip) / `d5a39c4` (the code). Its dispatch is
   `<my scratchpad>/dispatch-phase7-server.md`.
2. On merge: **release `docs/specs/video-playback-transcode.md` and `CLAUDE.md`
   from `c19` and confirm to `c18`, together with the merge sha.** `c18` is
   idle-blocked waiting on exactly that message and will not start without it.
3. `c18` (live) then does the browser side as a second PR on merged main. Its
   plan is at `<c18 scratchpad>/out/phase7-browser-plan.md` and is good; I added
   the §6.3 refusal UI to its scope (see below). It owns all of
   `web/static/` and `tests/js/`.
4. **`c18` needs a granted CPU window for the live browser check**, which is the
   last thing it does. It has agreed not to take the box unasked. Grant it.

**Phase 8 is untouched** and is the only phase left after this.

## 4. Rulings I made — do not silently reverse these

**§4.3 takes BOTH remedies, and the weight has now inverted.** The spec said
"phase 7 must pick one" of yield or disclosure. I ruled both: yield alone
re-creates §4.3's explicitly ruled-out third option (bounded but invisible) the
moment yielding proves partial. It then *did* prove partial — 9%, still outside
the window. So **disclosure is the load-bearing remedy and yield is the 9%**,
measured rather than argued. Keep the knobs; do not let the spec say the job
"yields" as though solved.

**`nice`/`-threads` bite on the CPU pipeline only.** On the GPU path the
contended resource is the encoder block and its driver queue, which neither knob
touches. Claiming a benefit there is §16's invented-constant rule in prose form.
This is *why* disclosure covers both paths and yield does not.

**One string, one definition.** `CONTENTION_NOTICE` lives at `jobs.py:132`; the
browser renders the **field**, never a copy. Same rule as `_check_allowed_path`
having exactly one definition — a disclosure an examiner may have to defend
needs a single auditable source. It is a remedy-(b) disclosure, **not an error**,
and must not render in an error band (`#139` precedent). It arrives on **two**
payloads (`routes.py:479` chunk response, `jobs.py:267` `Job.view()`) — `c18`
found this; one state written by both paths, one renderer, both carriers
asserted. Chunk-side is primary: that is the moment the analyst experiences the
slowdown.

**The `.part` watch checks `limit_bytes`, not the estimate.** `c17` chose this
because the estimate was believed low; `b2` makes it correct for a stronger
reason — the estimate is wrong in *both* directions depending on pipeline and
class. Anyone "correcting" it back to the estimate breaks every export while
believing they follow the spec.

## 5. Open with the operator

1. **NEW, recorded in `cto.md`: should a §6.3 full-copy refusal be overridable
   by the examiner?** My recommended default: yes, with the `.part` watch as the
   real guard. The over-estimate means the system can refuse a job whose real
   CPU output would have fit — a false refusal, a failure mode §6.3 does not
   contemplate, on the pipeline any GPU-less host always uses. **Not blocking**:
   phase 7 ships with the direction claim corrected and the limitation named.
   Making a forensic gate overridable is the operator's call.
2. **CodeQL** — ~20 `py/path-injection` false positives, unchanged closed
   ruling. Dismissal needs `security_events` scope; the operator does it
   personally. Not a required check. Never add a suppression to buy a green icon.

## 6. Ownership, agents

- **Mine** (release on window close, then `cx o --reap`): `docs/CTO_LEDGER.md`,
  `docs/fleet/runbook.md`, `docs/handoffs`.
- **`com-c19`** (live, coder): spec, `video_playback/`,
  `tests/test_video_playback.py`, `config.py`, `CLAUDE.md`.
- **`com-c18`** (live, coder): `web/static/`, `tests/js/`, `ci.yml`,
  `package.json`.
- **`csm-b2`**: finishing its handoff, then `cx q`. Retire it after.
- Retired by me with handoffs: `com-m5`, `com-c16`, `com-c17`.

## 7. What I learned that is written down nowhere else

Runbook `#172`/`#174` carry the long versions. The three that will recur:

**Whoever cancels a benchmark window must tell everyone holding for it, not just
the requester.** `b2` asked `c17` directly to hold CPU — correct peer routing. I
separately told `b2` not to run. Nobody told `c17`; it held every pytest run for
a cancelled window and hit its retire nudge inside the hold, which is why phase
7's server side was committed with the suite never run. A hold is fleet-wide
with one requester and N holders, and the requester cannot release holders it
does not know about. **Quiescing a box is a manager operation, not a peer
negotiation.** I now announce both the open and the close, and dispatches tell
workers to ask before running the suite.

**`/tmp` is a 46 G tmpfs and the fleet fills it — and capacity is not the worst
of it.** ~7.1 G per worktree venv; retired agents leave theirs. It hit 97% and
surfaced as my own `uv sync` dying on `libtorch_cpu.so` with ENOSPC. I reclaimed
the venvs of agents I had personally retired. But `c18` then spotted the risk I
had missed: phase 7's 1625 unmerged lines existed **only** on that ramdisk, no
remote, author already killed. Reclaiming space made a reboot less likely to
matter and did nothing about the reboot. **Push a WIP branch when it becomes
expensive to recreate, not when it is review-ready** — a branch with no PR costs
nothing and claims nothing. I pushed `feat/video-phase7` to origin immediately.

**File mtimes are not liveness.** My "recently touched" heuristic reported
`c18`'s live worktree *and* `c17`'s live worktree as stale. Had I trusted it I
would have deleted the phase-7 branch. Only `cx s` and retirement are evidence
that a worktree is dead.

**A test asserting a paraphrase of a user-facing string is the same defect as a
JS copy of it.** `c19`'s single suite failure was a test asserting "slower" when
the constant says "take longer than usual"; I had ruled against a JS copy of the
same constant hours earlier. Two independent drifts from one string in one day.
Assert the constant.

**The workers caught more than I did, and the pattern is worth keeping.** `b2`
volunteered its own contaminated run and a script bug rather than banking the
reps; `c18` refused to land spec text claiming a harness not yet on `main`, and
found both the durability risk and the two-carrier disclosure; `c19` asked
before taking the box — the exact check whose absence cost `c17`. Dispatches
that state the standard (mutation checks, name-the-tree, ffmpeg-raises-under-CI)
and then trust the worker produced better work than supervision would have.
