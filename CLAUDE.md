# ScalarForensic

Forensic image-similarity tool (FastAPI + Qdrant + DINOv2/SSCD embeddings) built by two
maintainers. Features must have forensic value and be legible to reviewers/courts;
decorative features get removed (precedent: the 3-D background viz, removed 2026-08).

## Commands

- `uv run pytest -q` — full suite (954 passed / 5 skipped at `8af5266`, 2026-08-14, verified
  clean tree; coverage 73.09% against the 65% floor — measured *after* video-playback
  phase 6, so it describes the layout below), needs no Qdrant; the 5 skips need
  `SFN_TEST_QDRANT_URL` and are the only tests that can observe the face exclusion
  guarantee against a real store (CI runs them in a separate Qdrant service-container job).
  **No longer fully hermetic:** the video encode tests need `ffmpeg` on `PATH`. Without it
  they skip locally and **fail** in CI (`CI` env var set) — a skip is how they went quiet
  for a day, so absence in CI is an error, not a shrug.
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
  **Exception: two subsystems own their routers.** `faces/` and
  `video_playback/` are self-contained packages that export a `router` from
  their `__init__.py`, which `app.py` includes like any other. Video playback
  is split `codecs.py` (container/codec classification and the mode decision),
  `digest.py` (the source SHA-256 and the process-wide `HashCache` handle),
  `rewrap.py` (the PyAV stream copy — *not* an encode), `capability.py` (the
  ffmpeg probe and the pipeline fingerprint), `encode.py` (the ffmpeg re-encode
  — *not* a stream copy), `cache.py` (the bounded viewing-copy store),
  `jobs.py` (the §4.3 full-video job: `Admission`, `JobRequest`, `FullJob`,
  `JobRunner`, and the module singletons `admission` and `runner`) and
  `routes.py`. `cache.py`'s public surface is `cache_key`/`artifact_dir`/
  `chunk_name`/`rewrap_path` (§6.1 layout), `renew_lease`/`release_lease`/
  `lease_state`/`pin`/`protected_videos` (§6.2 protection), `scan`/`evict`
  (§6.2 whole-video LRU), `check_ceiling`/`estimate_full_output_bytes` (§6.3),
  `publish`/`part_path`/`sweep_orphaned_parts`/`ensure_swept` (§10.2),
  `KeyedLocks` (§10.4, single-worker dedup) and `purge` (§13). Never write into
  the store by any other route. `routes/video.py` keeps only the indexing side:
  `/api/video-frame` and `/api/video-timeline`. Spec:
  `docs/specs/video-playback-transcode.md` §11.
- **`rewrap.py` and `encode.py` must never merge.** A rewrap is a PyAV stream
  copy whose output bitstream is bit-identical to its input; an encode is lossy,
  tone-mapped and carries the §7.4 disclosure. Keeping them apart is what stops
  a later reader treating a re-encode as a lossless copy in an evidence viewer.
- **Everything that changes a rendered pixel is a field of
  `capability.Pipeline`, and every field is hashed into the cache key.** No third
  answer: a pixel-affecting setting outside the fingerprint gives one cache entry
  two pictures under one label. A test pins the field set, and `describe()` (the
  §7.2 label) derives from the same fields, so a new field cannot change the key
  while the label keeps describing the old pipeline. Spec §6.1.
- Path-validating security controls get exactly one definition, in
  `routes/_shared.py`: `_check_allowed_path` and `_resolve_video_path`. Never
  re-implement either at a call site.
- Pipeline logic is the package `src/scalar_forensic/web/pipeline/`; public API is
  re-exported in its `__init__.py` — import sites use `from scalar_forensic.web.pipeline import X`.
- Frontend: the Alpine `sfn()` component is split into part files under
  `web/static/js/` (state/computed/helpers/lifecycle/analysis/evidence/triage/reset),
  merged by `static/app.js` via property descriptors. New component code goes into the
  matching part file; computed getters belong in `computed.js`. Never merge parts with
  `Object.assign` — it evaluates getters instead of copying them. Part `<script>` tags
  in `index.html` must load before `app.js`. Two subsystems own a part file of their
  own — `js/faces.js` and `js/video_playback/player.js` — for the same cohesion reason
  their Python packages own a router; they still register on `window.__sfnParts` and
  still load before `app.js`.
- **A wiring test cannot tell you the browser can run the file.** `player.js` once
  shipped a `?? … ||` precedence `SyntaxError` — it did not parse at all — and all
  fourteen of its text-level wiring tests passed against it. There is no JS test
  harness here, so browser-side work is finished by starting a server, force-refetching
  every `<script src>` **and** the stylesheet with `fetch(url, {cache: 'reload'})`,
  reloading, and driving the component. Spec §14 records the gap.

## Gotchas

- **`ffmpeg` is a declared external dependency** (spec §8), installed by CI and the
  Dockerfile, documented in `INSTALL.md`. PyAV covers indexing and the rewrap; only the
  re-encode shells out. The build needs `--enable-libzimg` for `zscale`/`tonemap` — without
  it an HDR source is *refused*, not encoded, because 8-bit output still tagged `bt2020`/HLG
  is the defect §3.1 measured. The capability probe runs a real six-frame
  decode→tone-map→encode→mux; never "fix" it into reading `ffmpeg -encoders`, which is the
  false positive §8 exists to reject.
- **The GPU is used for the *encoder* only.** `-hwaccel_output_format cuda` bypasses
  ffmpeg's autorotate and every portrait clip comes out on its side (§3.1, measured).
  Decode and filtering stay in software; `tests/test_video_playback.py` pins it with a
  fixture carrying a real display matrix, and adding `-noautorotate` fails that test.
- ffmpeg 6 **cannot write rotation side data on an output stream** (`-metadata:s:v rotate=`
  is gone), so the generated HDR fixture gets a display matrix patched into its `tkhd` box.
  Do not replace that with a weakened assertion — the rotation test is one of the two §14
  tests the whole encode path exists to keep honest.
- **`video_playback/` holds process-wide state that a test fixture must reset on both
  sides**, or one test inherits another's answer and it reads as a flake:
  `capability.reset_cache()` (the probe), `digest._reset_hash_cache()` (the `HashCache`
  handle), and `cache.reset_leases()` + `cache.artifact_locks.reset()` +
  `cache._reset_sweep()` (leases, pins, the lock table, the once-per-process `.part`
  sweep), and `jobs.admission.reset()` + `jobs.runner.reset()` +
  `routes.reset_substitutions()` (the shared admission counter, the full-video job
  table — `runner.reset()` kills what is still running — and the GPU→CPU
  cache-lookup hint). `tests/test_video_playback.py`
  has a module-scope **autouse** fixture doing all of them; keep it autouse.
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
- The 15 pytest warnings are both third-party: 14 are a torch `jit.script_method` deprecation
  (all from `test_embedder_preprocessing.py`), and since the 2026-08-13 fastapi bump one is
  `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install
  httpx2 instead`. Neither is ours; the starlette one is a live TODO, not noise.
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
