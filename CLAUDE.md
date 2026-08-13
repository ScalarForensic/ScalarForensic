# ScalarForensic

Forensic image-similarity tool (FastAPI + Qdrant + DINOv2/SSCD embeddings) built by two
maintainers. Features must have forensic value and be legible to reviewers/courts;
decorative features get removed (precedent: the 3-D background viz, removed 2026-08).

## Commands

- `uv run pytest -q` — full suite (530 passed / 5 skipped as of 2026-08-13, clean tree), hermetic,
  needs no Qdrant; the 5 skips need `SFN_TEST_QDRANT_URL` and are the only tests that can
  observe the face exclusion guarantee against a real store (CI runs them in a separate
  Qdrant service-container job)
- `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts` — CI runs exactly these
- `./run.sh sfn-web` — start web UI (wrapper exports venv CUDA libs); boots fine without
  Qdrant and degrades to exact-hash-only mode
- `./run.sh sfn <dir> --dino --sscd` — CLI indexing
- `./run.sh sfn <dir> --faces` — optional face modality (needs `SFN_FACES_ENABLED=true`,
  `SFN_EXAMINER_ID`, a YuNet ONNX and an operator-supplied embedder; `uv sync --group faces`).
  Purge via `uv run sfn-faces purge --media <sha256> | --all`

## Contributing workflow

Work goes to `main` through a pull request, never a direct push:

```
git switch -c fix/short-slug
gh pr create --fill
gh pr checks --watch          # lint-and-test (3.12) + qdrant-integration must pass
gh pr merge --squash --delete-branch
```

Ruleset "standard" on GitHub enforces this (PR required, 0 approvals, no force-push, both
CI jobs required). Note it is a **repository ruleset, not classic branch protection** — so
`gh api repos/.../branches/main/protection` answers "Branch not protected"; look at
`/rulesets` instead. Org owners retain a break-glass bypass, so a direct push to `main`
will still succeed for them — treat that as a fire alarm, not a shortcut.

Dependabot PRs that touch `.github/workflows/` cannot be rebased with `gh pr update-branch`
(the CLI token lacks `workflow` scope) — comment `@dependabot rebase` instead.

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
- The app sends **no cache headers on `/`**, and `ignoreCache` does not defeat the browser
  cache — use `/?cachebust=N` when testing UI changes, or you will debug a stale page.
  **`/?cachebust=N` busts only the HTML.** It busts neither `style.css` **nor any
  `/static/js/` part file**, so a live check can measure old CSS *and* old JS and report a
  false pass — or a false RED: on 2026-08-12 a stale `computed.js` made `mergedHits`
  undefined and the feature look broken when it was not. Force-refetch every `<script src>`
  and the stylesheet with `fetch(url, {cache: 'reload'})`, then reload.
- **Never measure the suite against a dirty tree and attribute the number to a commit.**
  `pytest` collects from disk, not from git, so a parallel worker's uncommitted tests are
  counted in *your* run. On 2026-08-12 this published 526/5 for a commit whose real bar was
  521/5. Check `git status --porcelain` is 0 before quoting a bar against a sha.
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
- Face search is uncalibrated by deliberate ruling (2026-08-12): the raw cosine is displayed and
  labelled as such. SFace's 0.363 is the model authors' reference figure only — never a default, a
  filter or a deployment threshold. See `docs/specs/face-pipeline.md` §10.
- **`requires-python = "==3.12.*"` is load-bearing — do not "modernise" it.** `37a8b4b` moved
  the project *down* from 3.13 to 3.12 because the ROCm path pins `pytorch-triton-rocm` by
  literal cp312 wheel URL with a `python_version == '3.12'` marker (`pyproject.toml:89`;
  PyTorch tags that wheel `linux_x86_64`, not manylinux, so uv's index resolver rejects it).
  cp314 wheels do now exist for torch/triton-rocm/onnxruntime, so a bump is *possible* — but
  it means touching 7 places (`requires-python`, the triton URL+marker, `run.sh:8`'s hardcoded
  `.venv/lib/python3.12/...` path, the CI matrix, `uv.lock`, the Dockerfile, and the
  regenerated `vendor/` airgap bundle). 3.12 is supported to Oct 2028 and the hot paths are
  torch/Qdrant, not the interpreter. Dependabot reopens this as a docker base-image bump;
  close it. (Precedent: PR #100, closed 2026-08-13.)
- Pytest's `addopts` in `pyproject.toml` carries `--cov-fail-under=65`, which applies to
  **every** invocation. Any CI job or local run of a test *subset* must pass `--no-cov` or it
  fails on coverage while its tests pass — this silently red-lit `qdrant-integration` for a
  day (fixed in `ci.yml`, see the comment there).
- `indexer.py:96-104` claims Qdrant can add a named vector to an existing collection. It cannot
  (`VectorParamsDiff` has no `size`), so `--dino` now and `--sscd` later is impossible — adding a
  modality means dropping the collection and re-indexing.
