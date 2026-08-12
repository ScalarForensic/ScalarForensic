# Handoff — scalarforensic-com-m2 → successor manager

**Written:** 2026-08-12 ~23:30 UTC. **Branch:** `feat/face-pipeline-phase1`, local only,
never pushed. **HEAD at handoff:** see `git log -1`; my last commit is the runbook entry
"S0 DONE". **Bar, re-measured by me at `438213a`:** `uv run pytest -q` → **506 passed,
5 skipped**; `ruff check` + `ruff format --check` on `src tests scripts` rc=0;
`git status --porcelain` → 0. With `SFN_TEST_QDRANT_URL=http://172.20.0.2:6333` the 5 skips
run and pass.

## 1. The one thing to read first

`docs/plans/2026-08-13-face-query-phase1b.md` (`306475b`, Stage 0 corrected in `438213a`).
7 stages. **S0, S1 and S2 are done and ticked.** Your job is **S3–S7**, which is the whole
visible feature: query-side face strip and selection, FACES as a fourth mode with filter,
matched face with the cosine beneath it, face query controls, the §10 divergence block, and
the DINO button rename plus the face model's own audit/dist-stats pair.

## 2. The ruling that settles the argument the last two windows had

The maintainer ruled, verbatim: *"We show it without calibration. Just give us the data and
then we can work with it later. no need for hiding this, the number is just showing what the
tool generated."* **Do not re-escalate this.** It overrode both my recommendation and the
CTO's objection, and §11 makes it the operator's call. Recorded in
`docs/specs/face-query-ux.md` (`fe6fa9c`), the runbook (`98a26fa`), and `CLAUDE.md`
(`5814706`). What stays: the score is labelled a raw model output with no CI and no
calibrated threshold; **0.363 is the SFace authors' reference figure only** — never a
default, a filter, or this deployment's threshold; the banner and the opt-in gate remain;
review-only observations stay vectorless.

## 3. Open decisions — all the maintainer's, none of them mine to take

1. **Production `SFN_FACE_MIN_SIZE`** (the *embed* floor, 64). This now has a second face:
   **danny1.jpeg yields zero searchable probes** at 64 (its faces are ~47 px), so the S3+
   acceptance pass must use **danny2.jpeg** or the feature will look broken when it is not.
   Lowering it puts ~40 px faces into search — the gate with evidential consequence.
2. **Production `SFN_FACE_REVIEW_MIN_SIZE`** — currently 36. Default: keep it. Retention
   only; a review-only point is vectorless and cannot produce a machine match.
3. **Merge `feat/face-pipeline-phase1` into `main`** — local merge, never a push.
4. **The fork in `docs/specs/stale-observation-purge.md`** (`6a22471`): re-detect vs
   config-hash diff, and whether recording the declined prompt on the medium's marker kills
   the need for the command by fixing it at source.
5. **Label**: they asked for "DINO: Dist Stats | Audit"; that modal actually reports DINOv2
   *and* SSCD. Shipping their string plus a subtitle naming both; the alternative
   "Image Audit (DINOv2 + SSCD)" is a two-string change, offered and not yet answered.

## 4. Live agents and ownership

**None.** c1, csm-c1 (tester), p1 and c2 are all retired and their rows released. I own
`CLAUDE.md`, `docs/fleet/runbook.md`, `docs/specs/face-query-ux.md` — release or claim as
you see fit; `cx o PATH --own --as <you>`.

## 5. Written down nowhere else, and each one cost something

- **The app sends no cache headers on `/`, `ignoreCache` does not defeat the browser cache,
  and `/?cachebust=N` does NOT bust `style.css`** — re-set the stylesheet href yourself or
  you will measure the old CSS and report a false pass. Two sessions have now paid for this.
- **`indexer.py:96-104` states the opposite of what Qdrant does.** A named vector cannot be
  added to an existing collection (`VectorParamsDiff` has no `size`), so `--dino` now and
  `--sscd` later is impossible; adding a modality means drop-and-re-index. **Queued and
  unassigned:** make it fail with that instruction instead of a Pydantic error, and delete
  the false comment.
- **A default-pinning test that reads the operator's `.env` pins nothing** —
  `load_dotenv` falls back to `find_dotenv()`. My sweep found 0 remaining flips, but the
  method has a blind spot: **8 of 13 `.env` keys equal their code defaults**, and a test
  pinning one of those passes either way and still pins nothing.
- **My first discriminator for that sweep was broken and its green was worthless** — it
  blocked only the `load_dotenv(None)` fallback while pytest runs from the repo root, where
  the direct `Path(".env")` branch hits first. A positive control caught it; the suite could
  not have. **A discriminator not shown to move something is not evidence.**
- **A CTO order is not evidence.** The CTO ordered a worker pulled off finished work on a
  misattributed commit — it verified the code state and *inferred* the author. When a fact
  goes stale, check **who** changed it before acting.
- `data/faces/face_audit.log` carries one real `query` event with `probe_hash` = 64 zeros,
  from c2's live verification. Append-only evidence; deliberately not deleted. Not a case.
- **The `f` (frontend-tester) role cannot be spawned**: `cx n` refuses the letter though
  `subagent_roles/roles/frontend-tester.md` declares it, adopted 2026-08-11 with operator
  sign-off. Filed via `cx f` (med). Spawn `csm-c` with the role file named as its contract.
- Qdrant is at `http://172.20.0.2:6333` — no host port published. `--faces` alone writes no
  image vectors; `SFN_INPUT_DIR` must be set or `/api/metadata` 403s.

## 6. Full trail

`docs/fleet/runbook.md`, all 2026-08-12 entries. Every number I published there I re-derived
myself; where I relayed a worker's, I said so.
