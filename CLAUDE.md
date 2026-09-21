# ScalarForensic

Forensic image-similarity tool: FastAPI, Qdrant, DINOv2/SSCD embeddings and an
optional face modality. Features must have forensic value; distinguish original
evidence, derived renderings and model outputs.

## Development and verification

- Python is pinned to 3.12. Use the repository's locked dependencies.
- Full suite: `uv run pytest -q`. Keep coverage enabled (65% floor).
- Subsets: `uv run pytest --no-cov -q tests/<file>.py`.
- Lint: `uv run ruff check src tests scripts`.
- Format: `uv run ruff format --check src tests scripts`.
- JavaScript: `npm test`; the built-in Node runner executes component scripts
  with `tests/js/harness.mjs`. Text-only wiring tests do not establish parsing,
  rendering or browser behavior.
- Browser: `npm run test:browser`, using the installed Playwright/Chromium
  development dependencies. The smoke tests use synthetic API responses;
  they do not replace live-store, real-model or release-hardware validation.
- Live Qdrant tests require `SFN_TEST_QDRANT_URL` pointing at a disposable
  server. Never run destructive integration fixtures against case storage.
- Video encoding tests require `ffmpeg` on PATH. Missing ffmpeg causes local
  skips and CI failures. YuNet model tests need the model under `models/`.
- Record exact commands, tree state, skips and warnings with verification.
  Do not attribute a dirty working-tree result to an unchanged commit.

An editable install may point at a different checkout. In a worktree reusing
another checkout's virtualenv, explicitly set `PYTHONPATH` to the worktree's
`src` and invoke that virtualenv's Python with `-m pytest -p no:cacheprovider`.
Do not run `uv sync` or `uv run` there unless a separate environment is intended.

## Running the application

- `./run.sh sfn-web` starts the web UI; the wrapper exports virtualenv CUDA libs.
- `./run.sh sfn <directory> --dino --sscd` indexes images/videos.
- `./run.sh sfn <directory> --faces` needs face enablement, examiner attribution,
  YuNet and an operator-provisioned embedder/manifest.
- `sfn-faces purge --media <sha256>` or `--all` removes face observations.
- `sfn-video purge --media <sha256>` or `--all` removes managed viewing copies.
- `sfn-video render --path <file> [--at <seconds>] [--full]` reads recorded argv,
  not an invocation reconstructed from today's settings.

Use the same configured state directories for web and CLI. Relative defaults
are CWD-relative. The face/video audit logs default to `data/face_audit.log`
and `data/video_audit.log`. Video audit lives beside the cache, not inside it,
so cache eviction/purge cannot erase the audit trail.

## Repository hygiene

Preserve unrelated working-tree changes. Stage explicit reviewed paths, especially
when models or dependency directories are shared symlinks. Respect operator
restrictions on publishing branches containing private snapshots. Normal integration
uses a reviewed PR with the existing Python/JS and Qdrant CI checks.

Keep durable specifications in `docs/specs/`, reproducible measurements in
`docs/benchmarks/`, and substantive audit/release evidence in `docs/audit/`.
Temporary plans, task ownership, session handoffs and cleanup reports belong in
the ignored `.cleanup-notes/` directory or external local scratch space, not
committed coordination trees. Do not create fleet/CTO ledgers or agent dispatch
records in project documentation. Preserve unique engineering decisions before
removing historical narratives; git history retains the originals.

## Architecture and contracts

- Deployment across an isolated LAN is intentional: the `0.0.0.0` web bind and
  remote embeddings endpoint are not accidental internet-exposure assumptions.
  See `docs/deployment.md`; do not silently redesign access control.
- Safeguards enforce normalization size, SSCD crop count and model identity at
  ingestion and similarity-query boundaries. A configuration-mismatch inspection
  override permits metadata/original viewing, not incompatible vector comparison.
- Topic routers live in `web/routes/`; shared path validators live in
  `web/routes/_shared.py`. Video playback owns its router in
  `video_playback/routes.py`; face HTTP endpoints live in `web/routes/faces.py`.
- Ingestion decisions live in `ingestion/`; web computation lives in
  `web/pipeline/`. Import helpers from their owning modules unless intentionally
  exported by the package facade.
- The indexer cannot add a new named vector to an existing collection through
  `VectorParamsDiff`: choose intended modalities at collection creation or
  explicitly rebuild. Never silently delete a collection.
- Frontend `sfn()` is assembled from `web/static/js/` parts by property
  descriptors. Never replace this with `Object.assign`, which evaluates getters.
  Part scripts load before `app.js`. Use the matching part for new behavior.
- Superseded responses must not publish after a session/selection reset.
  Check identity after asynchronous JSON parsing too, not only after fetch.
- When checking a live browser, refresh CSS and every script as well as HTML;
  a query-string cache bust on the HTML alone does not refresh its assets.

## Video playback invariants

See `docs/specs/video-playback-transcode.md` for decisions, limits and rationale.

- Rewrap is a bitstream-preserving PyAV copy; encoding is lossy and separately
  disclosed. Keep `rewrap.py` and `encode.py` distinct.
- `ffmpeg` is an external dependency. HDR probing must exercise actual
  decode/filter/encode with libzimg; an encoder-name listing is insufficient.
- GPU acceleration applies to encoding, not hardware decoding/filtering:
  the latter bypasses autorotation on measured portrait inputs.
- Every pixel-affecting pipeline setting is fingerprinted into the cache key
  and described by the same rendering record.
- Use cache ownership/layout/lease APIs; do not write or delete arbitrary
  files in the store. Keep audit outside automatic retention.
- Full-copy forecast override applies only to a refused estimate and one
  attributed request. Unknown estimates are not overridable. Actual output-size
  limits remain enforced even after forecast override.
- `_override_offered()` supplies both playback-info and start-refusal fields.
  Keep the two response carriers consistent.
- Full-export cancellation requires the private claim returned on start.
  Observing or closing a panel does not authorize cancelling another claim.
- Rendering rows and recorded command come from server audit data, not frontend
  inference. `rendering.js` is the shared renderer for chunks and full copies.
- Tests must reset process-wide probe/hash/cache/lease/admission/job state on
  both sides; retain the autouse lifecycle fixtures.
- Generated rotation fixtures patch the MP4 display matrix because ffmpeg 6
  cannot set output rotation through the old metadata option. Preserve the
  orientation assertions rather than weakening the fixture.

## Face invariants

- Review and embedding gates serve different purposes. Native review crops
  never upscale; aligned embeddings may resample and must be labeled.
- Review-only observations have no named vector. Do not substitute a display
  filter for structural exclusion from similarity search.
- Face observations are scoped to source media and video timecode; content-addressed
  chips can be shared. Purging one observation must not break another's chips.
- Uploaded face indices are bound to a detection token; consumers must carry it.
- Raw face cosine is deliberately shown as uncalibrated, without a confidence
  interval or deployment-calibrated threshold. SFace's reference figure is not
  a deployment threshold. See face spec §10.
- Face enablement gates biometric viewing; stored evidence must not require
  today's inference weights to be available. New inference requires startup
  validation.
- Keep counts, quality exclusions, model provenance and comparability caveats
  visible; unknown compatibility is not a demonstrated mismatch.

## Other pitfalls

- Patch dependencies where used, not merely where re-exported.
- `Settings` reads CWD configuration; hermetic tests isolate operator settings.
- `data/` is ignored except sample images. Never commit case material or models.
- Changing Python requires coordinating the ROCm cp312 wheel/marker, wrapper
  library paths, CI, Docker, lockfile and airgap exports—not just requires-python.
