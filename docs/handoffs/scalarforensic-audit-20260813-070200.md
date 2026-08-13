# Handoff — audit remediation session → successor

**Written:** 2026-08-13 ~07:00 UTC, end of context at the maintainer's instruction.
**Branch:** merged. `main` at `6d63756`; all work merged `--no-ff` from short-lived local
branches (deleted). **Never pushed** — GitHub interaction is allowed in this repo, but do not
push without being asked.
**Bar, measured by me on `main` at `6d63756`, porcelain 0:** `uv run pytest -q` →
**559 passed, 5 skipped, 8.3 s**, coverage **67%** (floor 65 now enforced);
`ruff check` + `ruff format --check` on `src tests scripts` rc=0. The 5 skips ran green
against a throwaway local `qdrant/qdrant` docker container during this session.

## 1. Context: the audit

`docs/audit-2026-08-13.md` is the full repository audit (grade B), recovered from a claude.ai
artifact and committed. **Standing rule from this session: never publish claude.ai artifacts
or post content to any online service — deliverables go into local markdown files.** (The
artifact that briefly held this audit was scrubbed; memory updated.)

Task IDs below (T1…T14) refer to the audit's Task Plan section.

## 2. What shipped this session (all merged to `main`)

| Commit | What |
|---|---|
| `56e7d2f` | **T1** — `httpx2` (unvetted PyPI fork, imported nowhere) → `httpx>=0.27`; `httpx2`/`httpcore2`/`truststore` out of `uv.lock` |
| `7cf0cc3` | **T2** — CI `qdrant-integration` job: Qdrant service container + `SFN_TEST_QDRANT_URL`, runs the 5 exclusion-guarantee tests on every push/PR |
| `efefb1c` | **T5+T6** — `_stored_frame_metadata` helper dedupes the two frame-metadata blocks in `routes/files.py`; the two `except Exception: pass` now `_log.debug` |
| `e7f58ca` | **T10** — audit doc committed with owner rulings; CLAUDE.md suite bar refreshed (was 478/5) |
| `5b56611` | **T3** — `tests/test_ingest_characterization.py`: 29 characterization tests locking the ingest write path. `indexer.py` 17% → **100%**, `cli.py` 40% → 59% |
| `ab2551f` | **T4+T9** — `query_session` call, both frame-metadata scrolls, and hit-metadata read+hash wrapped in `asyncio.to_thread`; `--cov-fail-under=65` in pyproject |

## 3. Owner rulings made this session (recorded in the audit doc — do not re-ask)

- **Q1 / Tag Triage (`routes/tags.py`):** exploratory only — **no test investment**; schedule
  a deprecation review with the co-maintainer per the 3-D-viz precedent. Not yet scheduled.
- **Q2 / concurrency:** a handful of examiners, isolated offline env — T4 uses threads,
  **T8 stays a single shared *sync* client** (no `AsyncQdrantClient`).
- GitHub interaction is allowed; publishing artifacts/online is not (see §1).

## 4. What's next (in order)

1. **T7 — decompose `cli.py:index()`** (~1,490 lines, `cli.py:359–~1847`) into an ingest
   package: scan/dedup → batch embed → upsert → report as importable stages, Typer command
   as orchestration only. **XL — extract one stage per branch/merge, running the T3 tests
   after each.** The T3 characterization tests are the contract: if one fails, behavior
   changed. Acceptance: `index()` < ~150 lines, stages ≥ 75% coverage. This is the
   highest-leverage change in the repo and the reason T3 exists.
2. **T8** — single lifespan-managed sync `QdrantClient` replacing the 13 per-request
   constructions (gotcha: many test patch targets move; patch where the name is *used*).
3. **T11** — fold the 9 payload-index blocks in `indexer.py:118-177` into a loop (T3 test
   `test_new_collection_created_with_full_config_and_indexes` pins the field set).
4. Backlog: T12 (mypy/pyright permissive in CI), T13 (CPU-torch CI index — measure first),
   T14 (docs/ journal split — owner decision pending, audit Q3), Tag Triage deprecation
   review (owner + co-maintainer), raise the coverage floor after T7.

## 5. Gotchas discovered this session

- **The comparability safeguard fires in tests too:** Settings' default
  `SFN_NORMALIZE_SIZE` is 512; the tests' `FakeEmbedder` claims 336, so the second run
  against the same fake collection aborts unless the env pins 336. The `ingest_env` fixture
  does this — keep it if you touch the fixture.
- **Modality order in a dual run is SSCD first** (`models_to_run` builds sscd before dino,
  `cli.py:506-510`), so SSCD inserts and DINO updates. A T3 test asserts this order.
- **The T3 fakes share one `FakeQdrantStore` across client instances** — that is what lets
  two Indexers see each other's points, like a real server. Don't "simplify" to per-client
  state.
- **The CI `qdrant-integration` job has not yet run on GitHub** (nothing pushed). The tests
  it runs are verified locally; the YAML itself is not.
- The audit doc's line references for `routes/files.py` predate the T5/T6/T4 edits — the
  findings are fixed, the line numbers there are historical.

## 6. Still outstanding from the previous handoff (unrelated to the audit)

The face re-index for `faces_danny_validation` (previous handoff
`scalarforensic-com-m3-20260813-000000.md` §2) is **still not done**: the 40–47 px danny1
cohort remains unsearchable; remediation needs the destructive `sfn-faces purge --all` and
**explicit maintainer confirmation first**. Nothing in this session touched it.
