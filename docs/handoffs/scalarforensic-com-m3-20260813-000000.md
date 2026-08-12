# Handoff — scalarforensic-com-m3 → successor

**Written:** 2026-08-13 ~00:00 UTC, end of day at the maintainer's instruction.
**Branch:** merged. `main` at `d76cf33`, **13+ commits ahead of `origin/main`, never pushed.**
`feat/face-pipeline-phase1` merged `--no-ff` at `2933a11` and is retained.
**Bar, measured by me on `main`, porcelain 0:** `uv run pytest -q` → **530 passed, 5 skipped**;
`ruff check` + `ruff format --check` on `src tests scripts` rc=0. With
`SFN_TEST_QDRANT_URL=http://172.20.0.2:6333` the 5 skips run: **5 passed, 0 skipped**.

## 1. What shipped

S0–S7 of `docs/plans/2026-08-13-face-query-phase1b.md` are **complete**. The maintainer's
visible feature exists: query-side face strip with selection and green borders, FACES as a
fourth mode with badge and filter, the matched face with the raw cosine beneath it, face query
controls beside the DINO sliders, the spec §10 divergence block, and the DINO button rename
plus the FACE audit/dist-stats pair.

Then, by maintainer instruction at end of day: floors lowered (`b91fd61`), the queued
`indexer.py` fix (`e14e056`), merge to `main` (`2933a11`).

## 2. THE ONE THING THAT IS NOT DONE — read before you touch anything

**Lowering the floors changed no stored data.** `SFN_FACE_MIN_SIZE` is now 40 in code, but
`faces_danny_validation` is byte-for-byte the same as before: **17 observations, 2 with a
vector, 15 vectorless**. The three danny1 faces (40.1 / 46.9 / 40.8 px) are **still
unsearchable**.

**Root cause:** the ingestion loop skips already-indexed media *before* the face pipeline runs,
so `--faces` only processes **new** media. A gate change never reaches stored observations.

The remediation command, the expected counts (5 with vector, 12 vectorless) and the two
env-var traps are in `docs/fleet/acceptance-scope-2026-08-13-face-query-phase1b.md`
(`cdddb64`). It needs `sfn-faces purge --all` first, which is **destructive**, which is why I
did not start it at end of day on low context. **Get maintainer confirmation before purging.**

**Do not read the lowered config as evidence the 40–47 px cohort is searchable.** It is not.
The exclusion sentence still stands verbatim: *"danny2 only; the 40–47 px cohort is untested
because the embed floor excludes it."*

## 3. Two "documented facts" that were artifacts — check before trusting a third

- **`SFN_FACE_REVIEW_MIN_SIZE` was 48 in code, never 36.** `git log -S` proves 36 never
  existed in `config.py`. It came from an ad-hoc export during an index run plus a MagicMock
  fixture, and from there into the previous handoff, the plan and CLAUDE.md as "the production
  value". It is now 24.
- **`SFN_FACE_COLLECTION=faces_danny_validation` is load-bearing.** `Settings` derives
  `{SFN_COLLECTION}_faces` = `danny_validation_faces`, which **does not exist**. Omit it and
  you create a second, empty collection and split the data. My first re-index attempt only
  failed safely because it hit the "first face-collection activation" prompt.

**The method that caught both: `git log -S` on the value.** A number that appears only in
prose and in a mock has never been verified against the code.

## 4. Tooling — you cannot spawn workers until this is fixed

**`cx n` is broken (filed via `cx f`, high):**
`TypeError: send() missing 1 required keyword-only argument: 'frm'` at `cxlib/spawn.py:341`.
It crashes **after** creating the tmux window and the registry row, leaving a half-spawned
worker at 0 tokens with the prompt never delivered — the exact silent-idle failure the ready
check exists to prevent. I killed the orphan (`c5`) and implemented the end-of-day tasks
inline. **Budget for fixing this first, or plan to work without workers.**

## 5. Ownership and agents

**No live agents.** c3, c4 retired, windows closed, rows released (`cx o --reap` → nothing
stranded). A stale row `scalarforensic-1f` (cold, idle, 0 tokens, `fable04-29:claude`) is not
mine and I left it alone.

**I own:** `docs/fleet/runbook.md`, `CLAUDE.md`, `docs/specs/face-query-ux.md`,
`docs/fleet/acceptance-scope-…md`, `src/scalar_forensic/config.py`,
`src/scalar_forensic/indexer.py`, `tests/faces/test_config.py`. Claim with
`cx o PATH --own --as <you>` **after my window is closed** — `cx o --release` drops only the
lock; ownership transfers when the window dies.

## 6. Open, all the maintainer's

1. **The `docs/specs/stale-observation-purge.md` fork** (re-detect vs config-hash diff). This
   is now **urgent** — §2 above is exactly the case it was written for.
2. **The `DINO Audit` vs `Image Audit (DINOv2 + SSCD)` label.** Shipped as the requested
   string plus a subtitle naming both models; the alternative is a two-string change.
3. **Whether to push `main`.** It has never been pushed and I had no instruction to.

## 7. Things I got wrong, so you do not repeat them

- **I published a bar against a dirty tree.** I quoted 526/5 for `433ae84` from a run made
  while a parallel worker had 6 files uncommitted; `pytest` collects from disk, not git, so
  its tests were counted into another commit. Real bar was 521/5. Now a rule in `CLAUDE.md`:
  confirm `git status --porcelain` is 0 before quoting a number against a sha.
- **I sized a worker off its self-report.** c3 reported 118k when I assigned it the last
  stage; `cx s` showed 207.7k at retirement. Size off `cx s`.
