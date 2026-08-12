# ScalarForensic

Forensic image-similarity tool (FastAPI + Qdrant + DINOv2/SSCD embeddings) built by two
maintainers. Features must have forensic value and be legible to reviewers/courts;
decorative features get removed (precedent: the 3-D background viz, removed 2026-08).

## Commands

- `uv run pytest -q` — full suite (300 tests), hermetic, needs no Qdrant
- `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts` — CI runs exactly these
- `./run.sh sfn-web` — start web UI (wrapper exports venv CUDA libs); boots fine without
  Qdrant and degrades to exact-hash-only mode
- `./run.sh sfn <dir> --dino --sscd` — CLI indexing
- `./run.sh sfn <dir> --faces` — optional face modality (needs `SFN_FACES_ENABLED=true`,
  `SFN_EXAMINER_ID`, a YuNet ONNX and an operator-supplied embedder; `uv sync --group faces`).
  Purge via `uv run sfn-faces purge --media <sha256> | --all`

## Architecture decisions (do not re-litigate)

- Target deployment is distributed across an isolated LAN (see `docs/deployment.md`):
  `SFN_WEB_HOST=0.0.0.0` default and the remote-embeddings endpoint
  (`SFN_EMBEDDING_ENDPOINT`) are intentional, not security oversights.
- Embedding comparability is enforced by safeguards: `SFN_NORMALIZE_SIZE`,
  `SFN_SSCD_N_CROPS`, and model hashes must match what the collection recorded.

## Code structure conventions

- Web endpoints live in `src/scalar_forensic/web/routes/` (APIRouter per topic,
  shared helpers in `_shared.py`); `app.py` keeps only app setup, lifespan, `/`.
- Pipeline logic is the package `src/scalar_forensic/web/pipeline/`; public API is
  re-exported in its `__init__.py` — import sites use `from scalar_forensic.web.pipeline import X`.
- Frontend: the Alpine `sfn()` component is split into part files under
  `web/static/js/` (state/computed/helpers/lifecycle/analysis/evidence/triage/reset),
  merged by `static/app.js` via property descriptors. New component code goes into the
  matching part file; computed getters belong in `computed.js`. Never merge parts with
  `Object.assign` — it evaluates getters instead of copying them. Part `<script>` tags
  in `index.html` must load before `app.js`.

## Gotchas

- `unittest.mock.patch` targets are per-module: patch where the name is *used*
  (e.g. `scalar_forensic.web.routes.files.Settings`, `...pipeline.query.QdrantClient`),
  not the package that re-exports it.
- `scripts/` ignores E402 (imports follow an intentional `sys.path` bootstrap).
- `data/` is gitignored except `data/sample_images/`; huge local test sets
  (`data/images/`, zips) live there untracked. Ingestion CSVs go to `data/reports/`.
- Web UI drop zone fades after 5 s idle (screensaver) and swallows pointer events;
  it wakes on `mousemove` — relevant when driving the UI with browser automation.
- The 14 pytest warnings are a third-party torch `jit.script_method` deprecation — not ours.
- The face real-model test skips unless a YuNet ONNX is present (`models/` is gitignored; fetch
  with `scripts/download_models.py --yunet`). It is the only check that can catch a wrong YuNet
  landmark-column map — the other detector tests build rows from the same assumption as the code.
- Faces have two gates: review (keep the native-resolution crop) and embedding (align + vector).
  Review-only observations are vectorless points — that is what keeps them out of search, not a
  payload filter. Never give them a vector.
- `tests/faces/test_store_integration.py` skips unless `SFN_TEST_QDRANT_URL` is set; it is the
  only test that can observe the exclusion guarantee, since a hermetic test can inspect only the
  `PointStruct` constructor. Run it against a throwaway Qdrant before touching demotion or purge.
