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

### Decision 1 — RESOLVED by the maintainer, 2026-08-12

**Real recognition weights, not the throwaway random-weight embedder.** Overrides the
survey's recommendation (report §4.1). Consequence: the `danny*` run becomes a real
calibration exercise, so the 48 px review-floor judgement and the `danny1.jpeg`
review-only outcome are now evidential findings rather than plumbing checks — and the
manifest's `channel_order`/`mean`/`scale` must match the model exactly, since a wrong
value produces plausible but silently wrong vectors (`faces/embed.py` validates only
shape and `input_size`, not semantics).

Blocked on the maintainer supplying `models/<model>.onnx` + `<model>.onnx.manifest.json`.
RAN `sed -n '20,45p' src/scalar_forensic/faces/embed.py`: nine required manifest fields
(`input_name, layout, channel_order, dtype, input_size, mean, scale, output_name,
embedding_dim`); `input_size` **must be 112** (`embed.py:67` raises otherwise —
arcface-112-v1 is the pinned alignment template) and `embedding_dim` is checked against
the model's real output shape (`embed.py:101`). Decisions 2–5 remain open.
