# Fleet runbook — ScalarForensic

## 2026-08-12 — survey window (manager `scalarforensic-com-m1`)

Survey only; no behaviour changed, no worker spawned. Report:
`docs/survey-2026-08-12-open-work.md`, committed `21f65f1`.

Measured at HEAD `b270d47` (2026-08-12T15:22:54+02:00): tree clean — 0 tracked-dirty
files, 0 untracked files (`git status --porcelain` counted both ways). `uv run pytest -q`
→ 480 collected, 475 passed, 5 skipped (all `tests/faces/test_store_integration.py`,
`SFN_TEST_QDRANT_URL not set`), exit 0. `ruff check` and `ruff format --check` on
`src tests scripts` → exit 0, 85 files formatted. No Makefile/justfile/nox/tox; the only
CI file is `.github/workflows/ci.yml`. Qdrant not running (`curl` exit 7).

Branch `feat/face-pipeline-phase1` is 42 commits ahead of `main`, 0 behind, no upstream.

Doc/code disagreements found: CLAUDE.md says 300 tests (measured 480); CLAUDE.md says the
YuNet real-model test skips (it runs — the ONNX is in `models/`); the 2026-08-11 phase-1
plan has 61 unticked boxes for work that all landed as commits.

**5 decisions escalated to the user** (ranked in §4 of the report): embedder choice for
the `danny*` validation run; merge to `main` after validation; Phase 1b vs Phase 2 next;
`purge_all()` missing video rollup points; standalone stale-observation purge command.
