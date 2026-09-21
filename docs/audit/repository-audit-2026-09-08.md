# ScalarForensic repository audit and improvement plan

## Executive Summary

**Overall health: C — a capable forensic research tool with substantial safeguards and tests, but material gaps in evidence lifecycle and deployment reliability.** The project explicitly remains a PoC (`pyproject.toml:3–4`, `SECURITY.md:5`), so this grade evaluates an isolated-LAN tool, not an internet-facing enterprise service. The top three risks are loss of application-host artifacts in the supplied Compose workflow (A01), incorrect face-observation deletion/retention (A02–A03), and similarity results escaping compatibility validation (A04). Additional urgent defects include an ineffective timeout for silent full-video encoders and unsafe sharing of the mutable face detector (A05–A06). The top three opportunities are durable case storage, regression tests around destructive/recovery paths, and a small extraction of ingestion orchestration into explicit stages (T1–T3, T15). Preserve the existing provenance records, vectorless review-only faces, isolated-LAN architecture, and simple frontend. No Critical finding was verified; six findings are High, with the remainder Medium. Existing lint and JavaScript checks pass locally, but the full Python suite did not complete in this environment, so this report does not certify its coverage or overall pass status. Resolve the High findings before relying on the affected workflows for case work; a wholesale rewrite is unnecessary.

## Repo Map

### Scope and method

Audit date: **2026-09-08**. Tracked revision: **`d9505e08e5daa12893b2ecf72678ae8354d2280c`**; 379 tracked files. The initial working tree already contained untracked `docker-compose.override.yml` and `docs/audit/`; these were left intact. Findings about deployment refer to the **tracked base and named GPU overrides**, not an assumed live configuration. No source, tests, manifests, configuration, or lockfiles were modified. This document is the audit's only repository addition.

Discovery preceded evaluation: mapped tracked files; read manifests, lockfile metadata/package versions, build/CI/configuration, README, installation/contribution/security guidance, deployment documentation, and local architecture instructions; then traced ingestion, query, face identity/purge, and playback execution. Reviewed existing local audit notes as leads and independently inspected the cited implementation. The pre-existing `docs/audit/code-review-2026-08-22.md` is untracked here and is not treated as proof of a finding.

Depth prioritized the ingestion coordinator, persistence, query flow, face storage, security boundaries, and playback resource lifecycle. Tag/discovery mathematics, calibration, CSS/layout, benchmark scripts, historical handoffs/specifications, and individual codec variants received lighter review. Large datasets and model weights were not inspected as evidence content. `.env` variable names were inspected without copying operator values into this report. No production database, deletion, deployment, model download, or browser session was exercised. Git history was not exhaustively scanned for secrets; legal compliance, model accuracy, full dependency vulnerability status, live branch protections, and production performance remain unverified.

### Purpose, users, and stack

ScalarForensic helps forensic examiners find byte-identical, altered, and semantically similar images and sampled video frames; optional faces provide investigative leads, and tags support reference-driven triage (`README.md:3,22–48`; `docs/tag-triage.md:22–65`; `docs/deployment.md:135–162`). The intended topology is one workstation or several roles on an isolated LAN, with operator-managed access control (`docs/deployment.md:3–16`; `INSTALL.md:126`). Two maintainers and deliberately small contributions favor incremental improvements (`CLAUDE.md:3–5`; `CONTRIBUTING.md:1`).

| Layer | Implementation and evidence |
|---|---|
| Runtime/package | Python **3.12 only**, Hatchling, uv; four console entry points (`pyproject.toml:6,24–28,94–99`). Linux is the operational center; CUDA/ROCm deployment guidance and optional MPS device selection exist, but platform parity was not exercised. |
| Embeddings/media | PyTorch, torchvision, Transformers/DINOv2, TorchScript SSCD, Pillow, PyAV; optional OpenCV/ONNX Runtime and HEIF (`pyproject.toml:8–50`; `embedder.py:457,580,678`, under `src/scalar_forensic/`). |
| Storage | Qdrant named vectors and payloads; filesystem derivatives/reports; SQLite hash cache (`indexer.py:53–177`; `embedder.py:169`; `config.py:54–93,211–231`, under `src/scalar_forensic/`). |
| Web/UI | FastAPI/Uvicorn; local Alpine.js, plain JavaScript parts, HTML/CSS, no frontend build pipeline (`web/app.py:72–80,283–288`; `web/static/app.js:1–15`, under `src/scalar_forensic/`; `package.json:4–11`). |
| Validation/shipping | pytest + coverage, Ruff, Node built-in tests; GitHub Actions; wheel/model bundling and Compose (`pyproject.toml:101–117`; `.github/workflows/ci.yml:15–122`; `Dockerfile:43–70`). |

### Architecture sketch

```text
CLI sfn -> Settings / compatibility -> scan + hash + dedup
        -> image decode / video frame extraction -> model embeddings
        -> Indexer -> Qdrant vectors + provenance
        -> local frames, thumbnails, face chips, manifest, CSV, audit log

Browser -> FastAPI /api/analyze -> temporary files + in-memory Session
        -> background analysis -> SSE progress + cached query embeddings
        -> /api/query -> Qdrant -> merged/grouped hits + provenance
        -> allowed-root file routes -> evidence / derived images
        -> face routes -> detector / embedder / FaceStore
        -> playback routes -> capability -> rewrap OR encode -> cache + audit

Tag routes -> TagStore / discovery -> reference/case Qdrant collections
```

Trace anchors: `src/scalar_forensic/cli.py:381,539–616,1230–1255,1508–1562`; `scanner.py:42–56`; `web/routes/analyze.py:58–143,152–231`; `web/pipeline/analysis.py:40–116`; `web/pipeline/query.py:253–303`; `web/routes/_shared.py:13–63`; `video_playback/jobs.py:415–435`; `discovery.py:196,329,395` (all abbreviated paths in this sentence are under `src/scalar_forensic/`).

| Directory/area | Role |
|---|---|
| `src/scalar_forensic/` | CLI, configuration, ingestion, embeddings, collection safeguards, tags and discovery. |
| `src/scalar_forensic/web/routes/` | HTTP adapters by topic; centralized path checks. |
| `src/scalar_forensic/web/pipeline/` | Analysis, querying, face query logic, statistics and provenance. |
| `src/scalar_forensic/faces/` | Optional detector, alignment, quality gates, vectors, chips, identity and purge. |
| `src/scalar_forensic/video_playback/` | Viewing-copy capability, encoding/rewrapping, admission, cache, states and audit. |
| `src/scalar_forensic/web/static/` | Vendored UI assets and topical Alpine component fragments. |
| `tests/` | Python unit/route/characterization tests, gated Qdrant integration, fixtures and Node tests. |
| `scripts/` | Model/dependency download, airgap packaging, export/path tools and benchmark preparation. |
| `docs/` | Deployment/method documentation plus extensive specifications, benchmarks and historical handoffs. |
| `data/`, `models/`, `research/` | Mostly local runtime/input/model/research material; generally ignored (`.gitignore:210–220`). |

**Conventions worth retaining:** snake_case and topical modules; dataclasses for session/results; injected store clients; Qdrant pagination helper; `HTTPException` at HTTP boundaries; explicit failure messages in query results; serialized shared-point upserts; mocks for fast tests plus selective real-store checks (`web/session.py:14–74`; `faces/store.py:61–73`; `indexer.py:22–50`; `web/pipeline/query.py:783–785`; `cli.py:942–952`; all under `src/scalar_forensic/`). Broad exceptions often deliberately preserve per-file progress, but this policy is not uniform. Type annotations exist, with runtime `object`/untyped dictionaries at important boundaries (`web/session.py:66`; `faces/indexing.py:95–118`).

**Surprises:** the operational/provenance design is considerably richer than the PoC label; the CLI's main function still spans 1,622 lines; and the supplied container workflow persists Qdrant but omits the application-host state its own deployment guide requires. Those are architectural facts, not evidence that every installation has suffered data loss.

## Audit Report

Severity reflects consequence and likelihood in the documented topology. **Fact** identifies inspected behavior; **judgment** states the inferred consequence and priority. Findings are grouped by dimension and ordered by severity within each group. Cross-cutting issues are recorded once.

### DevEx and operations

**A01 — High: application artifacts disappear with container removal.**

- **Fact:** Compose mounts only `.env` and `/images` into the app; only Qdrant has a persistent data volume (`docker-compose.yml:44–45,59–77`). App state defaults to relative `data/` locations (`src/scalar_forensic/config.py:58–86,217–231`); video audit lives beside the cache (`src/scalar_forensic/video_playback/audit.py:163–176`). The documented index command uses `docker compose run --rm` (`docker-compose.yml:26–30`).
- **Consequence/judgment:** indexing's frames, thumbnails, reports and optional face chips/logs are written in a disposable container and are not shared with the UI container; playback audit records disappear on UI container recreation. Original evidence remains on the read-only mount, but derived exhibits may become unavailable and historical audit events cannot be recreated from media. This is the first deployment fix. **Action:** T1/T3.

**A17 — Medium: the NVIDIA launch recipe is invalid Compose.**

- **Fact:** `docker-compose.nvidia.yml:11–14` uses `device_requests`. `docker compose -f docker-compose.yml -f docker-compose.nvidia.yml config --quiet` exits 1 under local Compose v5.1.0: `additional properties 'device_requests' not allowed`. The base and AMD configurations validate.
- **Consequence/judgment:** the advertised NVIDIA command fails before application launch (`docker-compose.yml:20–24`). CI does not validate these files (`.github/workflows/ci.yml:15–122`). **Action:** T2; validate a supported GPU reservation stanza and add all three configuration checks.

### Architecture and design

**A04 — High: compatibility is a startup convenience rather than a query invariant.**

- **Fact:** the image-model compatibility check runs in console `start()` (`src/scalar_forensic/web/app.py:280–281`), not ASGI lifespan (`:31–69`). Database failures log and return (`:141–169,188–194`); successful checks inspect only `settings.collection` (`:181–186`). Query execution does not repeat the check and can query the reference collection (`src/scalar_forensic/web/pipeline/query.py:275–303,425–437`).
- **Consequence/judgment:** starting through the ASGI import path, recovering Qdrant after an unavailable startup, or including an incompatible reference collection can expose cosine scores without the promised compatibility refusal. This is a correctness/provenance risk, not a claim that current case vectors are mismatched. **Action:** T6; gate each relevant collection before similarity search, including recovery and explicit inspection mode.

**A08 — Medium: ingestion orchestration is a concentrated change risk.**

- **Fact:** AST measurement finds `index()` at `src/scalar_forensic/cli.py:381–2002`: **1,622 lines**, containing 208 `if`/loop/exception-handler/conditional-expression nodes including nested helpers. Its `_finish_batch` alone is 230 lines (`:765–994`). These are structural counts, not a formal cyclomatic-complexity score. Discovery, hashing, threading, model writes, reporting, face lifecycle and interactive deletion share closure state (`:712,765,1230,1508,1897`).
- **Consequence/judgment:** a change to deduplication or failure handling can alter completion markers or destructive decisions far away. `query_session()` is the next major hotspot at 269 lines/54 such nodes (`src/scalar_forensic/web/pipeline/query.py:253–521`), including duplicated case/reference mode dispatch (`:350–351`). **Action:** T15, after characterization and correctness fixes; extract stages, not new services.

### Security and sensitive-data lifecycle

**A03 — High: purging by source-video digest leaves its biometric observations.**

- **Fact:** CLI promises one-media SHA-256 purge (`src/scalar_forensic/cli.py:2027–2029,2067`), but `FaceStore.purge_media()` filters only `image_hash` (`src/scalar_forensic/faces/store.py:496–499`). Video observations keep source SHA-256 separately in `video_hash` (`src/scalar_forensic/faces/indexing.py:314–320`); video rollups also use it (`faces/store.py:248–253`). An isolated in-memory Qdrant-client probe removed zero points for the video digest and left both seeded observations present.
- **Consequence/judgment:** an operator can believe a video's face data was purged while it remains searchable. **Action:** T5; media-kind-aware purge, verified absence, marker/rollup handling, and reference-safe chip cleanup.

**A07 — Medium: upload limits apply after multipart parsing.**

- **Fact:** FastAPI receives parsed `UploadFile` objects before the handler creates a session or counts bytes (`src/scalar_forensic/web/routes/analyze.py:58–88`). There is no receive-layer limit in application setup (`src/scalar_forensic/web/app.py:72–80`). Installed FastAPI calls `await request.form()` first (`.venv/lib/python3.12/site-packages/fastapi/routing.py:430`); installed Starlette spools file parts (`.venv/lib/python3.12/site-packages/starlette/formparsers.py:146–159,227–235,273–294`). These installed versions match the lockfile (`uv.lock:171–172,1048–1049`).
- **Consequence/judgment:** a large/concurrent upload can consume multipart temporary storage before either the 2 GiB app cap or active-session cap refuses it. This is an application ingress-limit gap, not an assertion that a historical Starlette CVE applies. **Action:** T9; count actual received bytes before parsing and budget concurrent uploads; preserve the handler's cleanup as a second layer.

Security context: `0.0.0.0` and remote embedding are documented design choices (`docs/deployment.md:36–70`). App authentication/TLS is explicitly delegated to operators (`INSTALL.md:126`); no installed proxy or actual LAN isolation was verified. Resolved allowed roots, random upload filenames and constructed subprocess argument arrays are positive controls (`web/routes/_shared.py:13–63`; `web/routes/analyze.py:76–81`; `video_playback/encode.py:181–189`, under `src/scalar_forensic/`). The reviewed application paths did not establish an injection vulnerability or hardcoded production credential. TorchScript loading exists (`src/scalar_forensic/embedder.py:598`), so operator-supplied models remain trusted executable artifacts; no evidence showed a query-upload route loading them as models. Default container execution has no `USER` restriction (`Dockerfile:18–70`); reducing privilege is worthwhile after durable-volume ownership is sorted, without claiming a verified escape.

### Code quality and correctness

**A02 — High: face content identity is confused with observation identity.**

- **Fact:** marker IDs use only frame/image hash (`src/scalar_forensic/faces/store.py:203–214`); stale selection filters only that hash (`:420–450`), while observation IDs include timecode (`:77–88`). Each frame's newly produced IDs are independently compared with stored points (`src/scalar_forensic/cli.py:1545–1552`). Accepted stale deletion removes those IDs (`:1897–1907`). An isolated two-timecode probe selects the first valid observation as stale when processing the second; marker construction also returns the same ID for identical frames from different videos.
- **Consequence/judgment:** repeated/static frames can overwrite marker attribution and propose legitimate observations for deletion. Distinct videos with identical frame bytes and matching timecodes/bboxes also share observation IDs because source-video identity is absent. The confirmation default mitigates deletion but does not repair the candidate list or identity model. **Action:** T4; separate content hashes from source-scoped observation IDs and migrate conservatively.

**A09 — Medium: resumed video ingestion cannot reliably reach complete status.**

- **Fact:** per-spec indexed counters increment only for newly embedded frames (`src/scalar_forensic/cli.py:888–895`). Already-indexed frames are skipped (`:841–844`), but the completion gate counts only new frames and in-run duplicates (`:1747–1759`).
- **Consequence/judgment:** interrupt at 400/500 unique frames, then index the remaining 100: completion still sees only 100. Subsequent runs continue extracting the same video. **Action:** T10; count durable per-modality completion, including verified previously stored frames, without treating failed upserts as success.

**A10 — Medium: stored face probes require an unrelated session configuration.**

- **Fact:** stored point IDs are a supported probe source (`src/scalar_forensic/web/routes/faces.py:727–749`), but search and compare unconditionally pass `entry.query_faces_cfg` into compatibility (`:751–752,827–828`). That field defaults to `None` (`src/scalar_forensic/web/session.py:66`); `check_compat()` dereferences it (`src/scalar_forensic/faces/store.py:168–171`). A direct probe raises `AttributeError`; the wrapper catches only `ValueError` (`web/routes/faces.py:702–705`).
- **Consequence/judgment:** stored-probe requests without prior query-face detection, and comparison before detection, can return 500 on populated metadata. **Action:** T11; validate query-probe state separately and obtain stored-probe compatibility from its actual provenance.

**A11 — Medium: DINO normalization provenance can describe requested rather than actual processing.**

- **Fact:** `DINOv2Embedder` records requested `normalize_size` and only warns if the processor rejects it (`src/scalar_forensic/embedder.py:465–485`). That value is persisted (`src/scalar_forensic/cli.py:904–909`; `src/scalar_forensic/indexer.py:352–356`) and later compared by safeguards (`src/scalar_forensic/safeguards.py:262–267`).
- **Consequence/judgment:** a nonstandard processor can produce different pixels while metadata claims the intended resolution, weakening comparability. The warning branch is verified; occurrence with deployed model weights was not. **Action:** T12; refuse mismatches or record and validate effective preprocessing, including crop geometry.

**A12 — Medium: face distribution statistics silently censor negative scores.**

- **Fact:** the function claims no score floor but calls `threshold=0.0` (`src/scalar_forensic/web/pipeline/faces_stats.py:52–60`); the store forwards it as `score_threshold` (`src/scalar_forensic/faces/store.py:336–342`). The histogram nevertheless represents the full [-1,1] cosine range (`faces_stats.py:95–102`). The query also returns only the top `sample_size` neighbors.
- **Consequence/judgment:** negative scores cannot appear, and large-collection statistics describe selected top neighbors rather than the population claimed in the docstring. **Action:** T13; remove the implicit floor and explicitly label a top-N sample, or implement a genuinely representative population computation if required.

### Performance and resource lifecycle

**A05 — High: silent full-video encoders bypass timeout and size monitoring.**

- **Fact:** `_run_watched()` blocks reading stdout; it checks deadline and file size only upon a `progress` line (`src/scalar_forensic/video_playback/encode.py:286–320`). An isolated silent subprocess sleeping two seconds returned **success after 2.01 seconds with `timeout=1`**.
- **Consequence/judgment:** a silent hung encoder can occupy an admission slot indefinitely; file-size enforcement also stops while progress output stops. This is stronger than a speculative hang: the timeout failure was reproduced without ffmpeg or evidence media. **Action:** T7; independent watchdog and bounded subprocess cleanup.

**A06 — High: concurrent requests share mutable detector state without synchronization.**

- **Fact:** web face detection uses a process-wide cached detector (`src/scalar_forensic/web/pipeline/faces_query.py:50–70`) in a sync route explicitly dispatched to a threadpool (`web/routes/faces.py:495–516`). Detection mutates the network size immediately before inference (`faces/detect.py:69–77`). The CLI pipeline already documents that this detector must not be shared between workers (`faces/indexing.py:111–118`). Image-model lazy initialization is also an unlocked check/load/store (`web/pipeline/_embedders.py:15–31`).
- **Consequence/judgment:** concurrent differently sized images can interleave mutable detector operations; simultaneous cold requests can load duplicate large models. Incorrect detections or allocation spikes are plausible consequences, not measured frequency. **Action:** T8; lock the complete detector operation and lazy construction, with bounded inference admission.

**A13 — Medium: two audit/statistics endpoints block the event loop.**

- **Fact:** async `semantic_stats` calls synchronous database statistics directly, and async `hit_provenance` does likewise (`src/scalar_forensic/web/routes/analyze.py:239–253,267–273`; database calls in `web/pipeline/stats.py:62–70`, `web/pipeline/provenance.py:58` onward). The main query endpoint already uses `asyncio.to_thread` (`web/routes/analyze.py:170–180`).
- **Consequence/judgment:** a slow Qdrant response on these paths delays unrelated requests and SSE flushing in the same worker. **Action:** T14a; use the existing thread-offload convention and a heartbeat concurrency test.

**A14 — Medium: each video slider query repeats serial per-frame network calls.**

- **Fact:** `query_session()` creates a fresh client (`src/scalar_forensic/web/pipeline/query.py:275`), queries each timecode/mode serially (`:352–385`), and repeats the dispatch for references (`:425–437`). The default frame cap is 500 (`src/scalar_forensic/config.py:206–207`).
- **Consequence/judgment:** 500 frames × two modes means up to 1,000 sequential vector requests per video, or 2,000 with reference overlay, before exact-hash work. This is a call-count derivation, not a measured latency estimate. **Action:** T14b; bounded batch queries and latest-request cancellation/debouncing, preserving grouping and score provenance. Full in-memory dedup sets (`src/scalar_forensic/indexer.py:179–215`) are a separate scale limit, but reasonable at the stated tens-of-thousands size; measure before replacing them.

**A19 — Medium: temporary and completed-job state lacks complete lifecycle cleanup.**

- **Fact:** session directories use `mkdtemp` (`src/scalar_forensic/web/routes/analyze.py:70`); only the in-memory store tracks them (`web/session.py:77,103–146`). Lifespan shutdown stops only the reaper (`web/app.py:63–69`). A disconnected analysis cancels its asyncio task but explicitly leaves the underlying thread running (`web/routes/analyze.py:133–138`). Completed full-video jobs remain in `_jobs` until same-digest replacement or test reset (`video_playback/jobs.py:380–413`).
- **Consequence/judgment:** crashes can leave sensitive query uploads outside the TTL registry; repeated distinct exports grow the job table; disconnected clients can continue consuming compute. **Action:** T16; owned temporary-root recovery, cooperative cancellation, and bounded retention. Do not delete arbitrary `/tmp/sfn_*` directories belonging to another live process.

### Testing

**A15 — Medium: test breadth exceeds boundary coverage.**

- **Fact:** core ingest characterization uses a fake Qdrant backend (`tests/test_ingest_characterization.py:1–11`); CI's real-store job runs only face-store integration (`.github/workflows/ci.yml:119–122`). The stored-probe fixture always injects a non-null config (`tests/faces/test_face_probes_compare.py:29–33`), hiding A10. The JS harness explicitly has no DOM (`tests/js/harness.mjs:19–23`).
- **Consequence/judgment:** current tests cannot establish browser rendering, real-store ingestion recovery, or the destructive identity invariants exposed above. This is not a recommendation to replace the many useful unit tests. **Action:** T2 and T17; targeted failure/recovery and real-browser smoke tests. No new overall coverage percentage is claimed.

### Dependencies and reproducibility

**A16 — Medium: deployed dependency identity is not enforced end to end.**

- **Fact:** CI uses `uv sync --dev` without `--locked` (`.github/workflows/ci.yml:42–43,98–99`), and tests against Qdrant `latest` (`:83`) while Compose/bundle pin `v1.17.1` (`docker-compose.yml:42`; `scripts/build_airgap_image.sh:27`). The build backend is unversioned and downloaded independently of the app lock (`pyproject.toml:94–96`; `scripts/download_deps.sh:73–76`). The airgap script's export does not opt out of the default dev group (`scripts/download_deps.sh:39`), which includes web/faces/HEIF/test tools (`pyproject.toml:30–45`), so runtime bundle contents need an explicit group policy.
- **Consequence/judgment:** CI can resolve a stale lock silently, test a different server contract, and produce differently composed bundles. **Action:** T2/T18; lock consistency, deployment-version integration and an artifact inventory. No Python downgrade/upgrade is justified by age alone.

Dependency snapshot: locked and locally installed versions agree for torch 2.11.0+cu128, Transformers 5.15.0, Pillow 12.3.0, FastAPI 0.141.1, Starlette 1.3.1, multipart 0.0.32, Qdrant client 1.19.0, ONNX Runtime 1.28.0, OpenCV 5.0.0.93 and PyAV 18.0.0 (`uv.lock:37,171,599,617,645,810,847,1048,1099,1152`). This does **not** prove they are current or vulnerability-free. A limited official-advisory lookup found that the old Starlette spool-blocking advisory's affected range is below the locked version; it must not be reported as an active CVE here ([Starlette GHSA-2c2j-9gv5-cj73](https://github.com/Kludex/starlette/security/advisories/GHSA-2c2j-9gv5-cj73)). Official PyTorch and Transformers security pages were consulted, but a complete machine-readable lockfile/OS/model scan was not completed ([PyTorch advisories](https://github.com/pytorch/pytorch/security/advisories), [Transformers security policy](https://github.com/huggingface/transformers/security)). Firecrawl account/network access failed; built-in browsing supplied this limited lookup. No applicable known CVE is asserted.

License review is likewise limited: third-party notices exist, and recognition-model rights are explicitly operator-owned (`THIRD_PARTY_LICENSES.md:7–84`). However, that file says no weights are bundled (`:80`) while the airgap Docker image copies `models/` (`Dockerfile:51–54`). T18 should inventory actual distributed weights, fonts, wheels, native libraries and their notices; a legal violation is not established by this audit. The zero-dependency Node test manifest is healthy (`package.json:1–12`; `package-lock.json:1` onward).

### Documentation

**A18 — Medium: operator instructions contradict effective behavior.**

- **Fact:** deployment calls all playback lossless stream-copy (`docs/deployment.md:93–99`), despite the actual full encode path (`src/scalar_forensic/video_playback/jobs.py:427–435`). The remote example sets an endpoint ending `/v1/embeddings` (`docs/deployment.md:44`), while the implementation appends that suffix (`src/scalar_forensic/embedder.py:699,747–748`), yielding a doubled route. `.env.example:53–62` calls one SSCD crop the default; code uses five (`src/scalar_forensic/config.py:242`). `INSTALL.md:101` says normalization defaults to 512; code defaults to 224 (`config.py:44`). README describes query-time re-embedding and a read-only override (`README.md:70`), whereas query uses stored embeddings (`web/pipeline/query.py:263–265`) and CLI mismatch override continues ingestion (`cli.py:616–625`).
- **Consequence/judgment:** operators can misconfigure remote compute, misstate a viewed artifact's transformation, or believe an override prevents writes. **Action:** T19; synchronize the authoritative deployment/configuration docs with actual behavior, and identify historical documents as snapshots. `SECURITY.md:9` also offers only public issue/PR reporting; add a private channel if maintainers can monitor one.

### Strengths to preserve

- **Forensic provenance is designed in:** model hashes, settings snapshots with credential redaction, per-run input manifests and per-vector provenance (`src/scalar_forensic/manifest.py:25–48,84–101`; `indexer.py:350–364`).
- **Review-only faces are structurally excluded:** vector clearing is explicit and covered against a real Qdrant service (`src/scalar_forensic/cli.py:1553–1562`; `tests/faces/test_store_integration.py:1–14,50–65`).
- **Playback has useful safety primitives:** separate rewrap/encode modules, atomic publish, cache ceiling and leases; audit records sit outside cache eviction (`src/scalar_forensic/video_playback/cache.py:57,379,642`; `audit.py:163–176`). Fix the watchdog without discarding these controls.
- **Engineering feedback exists:** Ruff lint/format, coverage floor, Node execution tests, read-only CI token, weekly Dependabot updates (`pyproject.toml:101–117`; `.github/workflows/ci.yml:9–13,45–73`; `.github/dependabot.yml:6–25`).
- **Tests respect operator environments:** the autouse fixture neutralizes `.env` and application environment variables (`tests/conftest.py:1–26`).
- **The frontend and deployment model fit the team:** local static assets, no runtime frontend dependency fetch, modest module boundaries and explicit isolated-LAN assumptions (`src/scalar_forensic/web/static/app.js:1–15`; `docs/deployment.md:3–16,64–70`).

### Verification record

All checks used the existing environment; no dependency synchronization or live case database was used. Logs were written under `/tmp/sfn-audit-20260908-*` and may be transient.

| Check | Observed result / limit |
|---|---|
| `.venv/bin/ruff check --no-cache src tests scripts` | Passed. |
| `.venv/bin/ruff format --check --no-cache src tests scripts` | Passed; 114 files already formatted. |
| `npm test` | Exit 0; local reporter reports six file-level passes, zero failures. Do not substitute the historical 88-test count. |
| Python collection, `--collect-only --no-cov` | 1,041 tests collected. |
| Full Python suite with coverage redirected to `/tmp`, cache disabled | Interrupted after no further visible progress at approximately 90 tests; no completed coverage result. |
| Isolated reproduction of the stalled route test | `test_face_audit_reports_index_time_thresholds_not_current_env` also stalled; faulthandler shows TestClient/AnyIO portal waiting. Bounded diagnostic used a 25-second process timeout. Environment vs. library vs. repository cause was not established. |
| Focused pure Python tests | 101 passed in 2.28 seconds: safeguards, manifests, query evaluation and face store. Coverage disabled for this subset. A separate route-inclusive subset timed out after 90 seconds without completing. |
| Compose base / AMD / NVIDIA `config --quiet` | Base and AMD passed; NVIDIA failed as A17. No containers started. |
| Direct face-store probes | Missing config raises `AttributeError`; identical frame bytes collide as markers; video-digest purge removes zero points; stale selection includes a valid other-timecode observation. Local in-memory client only, not a server integration test. |
| Silent subprocess probe | Two-second no-output process succeeds with a one-second watched timeout, confirming A05. |

## Improvement Strategy

| Theme | Target state and principle | Measurable definition of done |
|---|---|---|
| Durable, source-scoped evidence lifecycle (A01–A03, A09) | Separate source occurrence, content digest and derived artifact identities. Persist all case state and verify destructive operations. | Throwaway-container removal/recreation preserves reports/logs and resolves all seeded references; purge removes all target-video observations and no unrelated ones; interrupted ingest converges on the second run. |
| Compatibility as an enforced boundary (A04, A10–A12) | Scores carry the provenance of the vectors actually used. Unknown/incompatible state has explicit behavior. | Case/reference drift and database recovery cannot yield unvalidated similarity results; stored probes work without fabricated session config; negative cosine fixtures appear in accurately labeled statistics. |
| Bounded background work (A05–A07, A13–A14, A19) | Resource limits must hold independently of client behavior and progress output. | Silent workers terminate within deadline + agreed cleanup grace; concurrent detectors match serial results; oversized bodies stop before full parsing; session/job retention has executable bounds. |
| Small, testable orchestration (A08, A15) | Extract pure decisions from I/O only after tests pin behavior. | Ingestion stages have explicit inputs/results and no shared nested mutation for completion/purge decisions; new boundary cases run in CI; existing coverage floor remains at least 65%. |
| Reproducible operations and truthful docs (A16–A18) | Test the configuration and artifact that operators receive. | CI rejects stale locks and invalid Compose files; integration runs against the shipped Qdrant version; bundle inventory and documented restore procedure match a rehearsed release. |

**Trade-offs:** keep FastAPI, Qdrant, Python 3.12, Alpine, and the single-worker deployment; multi-process sessions/distributed queues are not warranted by the stated audience (`docs/deployment.md:112–118`). Keep operator-managed LAN access control, while documenting that process-level examiner identity is not per-browser authentication. Do not retrofit tamper-proof distributed audit infrastructure without a custody requirement; first retain/back up the existing logs and state their append-mode, non-immutable nature (`src/scalar_forensic/faces/audit.py:21–39`). Do not replace all mocks with integration tests or chase 100% coverage; add tests where current fakes fail to exercise database semantics. Avoid speculative caching, index redesign and dedup database migration before measuring actual corpus size and query latency. Do not mechanically remove broad exceptions that intentionally preserve file-level progress; define failure contracts around the specific unsafe cases. Avoid automatic deletion/migration of existing observations until an operator reviews the dry-run mapping and backup.

**Release exit:** all six High findings resolved with regressions; no newly verified Critical findings; complete Python/JS/lint results on the release environment; pinned Qdrant integration and Compose validation green; restore rehearsal successful. Performance targets should be set against a named corpus/device, not invented in this audit. This plan provides bounded correctness criteria immediately while leaving throughput budgets to the owner.

## Task Plan

Effort: **S <2h; M half-day; L 1–2 days; XL requires breakdown**. Estimates assume one maintainer familiar with this code and available fixtures; operator migration/reindex time is separate. Risk means risk introduced by the change. Each description is scoped for an independently reviewable change. Task references inherit the exact file/line evidence from their linked findings above.

### Milestone 0 — Safety net

Exit: recoverable fixture/case backup, reproducible checks, and failing regressions that expose the highest-risk defects. Tests may be introduced together with their fixes so main remains green.

| ID / title | Description and affected files/areas | Acceptance criteria | Effort / risk | Dependencies |
|---|---|---|---|---|
| **T1 — Capture and rehearse case restoration** | Document and exercise backup of Qdrant plus application-host state before migration. Inventory existing disposable containers before removing them; recover their artifacts where available. Areas: Compose, `docs/deployment.md`, reports/chips/frames/audit paths (A01). | Restore a synthetic case into separate locations; compare report/log checksums and resolve sample hits/chips. Record what is unrecoverable without altering original evidence. | M / Low; actual case backup duration extra | None |
| **T2 — Enforce release configuration checks** | Add `uv sync --locked --dev`, test the shipped Qdrant version, validate base/AMD/NVIDIA Compose, and fix the invalid NVIDIA stanza. Keep required checks in the existing CI jobs. Areas: `.github/workflows/ci.yml`, `docker-compose.nvidia.yml` (A15–A17). | A stale lock or invalid override fails CI; all supported Compose files validate; real-store tests pass on v1.17.1. GPU runtime smoke is separately recorded when hardware is available. | M / Low | None |
| **T0R — Add destructive/recovery regression fixtures** | Seed two videos with repeated frame bytes, partially indexed frames, shared chips and audit records; add silent-worker and compatibility-recovery cases. Extend existing store/ingest/playback tests rather than inventing a test framework (A02–A06, A09). | Tests expose current defects, use throwaway data, and prove unrelated observations survive. At least the purge/identity cases also exercise real Qdrant in CI. | L / Low | T2 for server-backed checks |

### Milestone 1 — Critical fixes

These address the six High findings first, then related correctness defects. The milestone name denotes priority; it does not reclassify findings as Critical.

| ID / title | Description and affected files/areas | Acceptance criteria | Effort / risk | Dependencies |
| **T3 — Persist and share application state** | Add an explicit writable case-data mount and canonical state paths shared by indexing and web containers; document ownership and migration. Keep evidence read-only. Areas: Compose, config examples, deployment/installation (A01). | Index using `run --rm`; restart/recreate web container; frames/chips/reports/logs remain readable with matching checksums. The mount does not shadow model or source files. | M / Medium: paths/permissions | T1, T2 |
| **T4 — Correct face observation identity and stale scope** | Introduce a source-scoped identity for images/video timecodes, retaining content-addressed chips separately. Update markers, stale selection and observation-key migration; produce a reviewable old/new mapping before mutation. Areas: `faces/store.py`, `faces/indexing.py`, CLI face commit, adjudication consumers (A02). | Repeated frames within/across videos coexist; only the intended source/timecode is stale; old adjudication keys remain resolvable or explicitly mapped; reruns are idempotent. | L implementation + L migration validation / High | T1, T0R |
| **T5 — Make video-face purge complete and verifiable** | Distinguish image/source-video scope and delete matching observations, frame markers and rollup, then unlink only unreferenced chips. Validate digest input and log intended scope and observed result. Areas: `faces/store.py`, CLI purge, integration tests (A03). | Target video has zero residual points/search hits; unrelated/shared observations and meta survive; second purge is an explicit no-op; injected failures cannot be reported as completed purge. | L / High: deletion | T1, T0R; coordinate identity schema with T4 |
| **T6 — Gate similarity at the collection boundary** | Move baseline validation into app lifecycle and enforce per-collection compatibility on first use/recovery, with explicit invalidation policy. Preserve an inspection escape hatch with clearly disclosed behavior. Areas: `web/app.py`, safeguards, query/reference paths (A04). | ASGI and console startup agree; unavailable→recovered database is validated; incompatible reference/case collections are blocked; compatible exact-only behavior remains defined. | L / Medium: availability | T0R |
| **T7 — Enforce independent encoder watchdogs** | Monitor time and output size independently of progress parsing and kill/reap the process group on breach. Preserve cancellation and current failure-state/audit semantics. Areas: `video_playback/encode.py`, jobs, playback tests (A05). | Silent subprocess exceeds neither timeout + two-second cleanup grace nor agreed byte-monitor interval; slots/pins release; no child process remains; success/cancel/timeout tests distinguish outcomes. | M / Medium: subprocess lifecycle | T0R |
| **T8 — Serialize unsafe model operations** | Protect detector construction and complete inference, and guard image-model cold initialization. Use bounded local admission; per-thread detectors are optional only if measured contention justifies memory cost. Areas: `_embedders.py`, `faces_query.py`, `faces/detect.py` (A06). | Barrier-controlled concurrent requests at different sizes equal serial outputs; one model load per cache key; errors release locks; overload has a bounded response. | L / Medium: latency/deadlocks | T0R |
| **T9 — Bound ingress before parsing** | Add actual receive-byte accounting plus concurrent-upload admission before multipart spooling; keep route-level accounting and cleanup. Document matching proxy limits for deployments that use one. Areas: app setup/analyze, config, route tests (A07). | Oversized declared and chunked bodies return 413 without full spooling; rejected uploads create no session artifacts; normal multi-file upload and disconnect cleanup work. | L / Medium: streaming semantics | T2 |
| **T10 — Repair interrupted-ingest completion** | Make per-modality completion derive from confirmed stored or successfully committed frames, including resumed frames. Preserve duplicate-source distinctions. Areas: `cli.py`, `indexer.py`, ingest/video tests (A09). | Inject an interruption after 400/500 frames; next run finishes; third run skips extraction. Failed write/failed modality never receives a completion marker. | L / Medium | T0R |
| **T11 — Validate face probe origins** | Separate uploaded-probe configuration from stored-observation provenance and give missing query detection a defined 4xx response. Add concrete typed state at this boundary. Areas: face routes/session and probe tests (A10). | Stored-only probes succeed without prior detection; mixed incompatible probes are refused; compare before detection returns a documented response, never AttributeError. | M / Low–Medium | T6 |
| **T12 — Enforce effective DINO preprocessing** | Reject processor configurations that do not honor the requested geometry, before accepting vectors. If variants must be supported, record actual effective settings through the same safeguard path. Areas: embedder, provenance, preprocessing tests (A11). | Ignored resize or crop arguments fail before upsert; standard processor behavior and metadata remain unchanged. | M / Medium: variant compatibility | T0R |
| **T13 — Correct face-statistics sampling semantics** | Permit a missing score threshold and name the returned top-N sample accurately; do not silently imply a full population. Areas: face stats/store, face UI labels/tests (A12). | Seed negative/zero/positive cosines; counts/histogram include all requested scores; when corpus exceeds N the UI explicitly states the selection/truncation. | M / Low | None |

### Milestone 2 — High-leverage improvements

| ID / title | Description and affected files/areas | Acceptance criteria | Effort / risk | Dependencies |
| **T14a — Offload remaining synchronous audit calls** | Apply existing `to_thread` convention to semantic statistics and hit provenance, with database timeout behavior preserved. Areas: analyze routes/stats/provenance (A13). | During a deliberately delayed database call a parallel heartbeat/SSE request progresses; response payloads/errors remain equivalent. | S / Low | None |
| **T14b — Batch video search dispatch** | Replace serial frame-by-mode calls with bounded batches and retain query-frame associations; reuse scoped clients and avoid obsolete slider requests. Areas: query pipeline, frontend analysis, query tests (A14). | Identical hits/order/provenance on a golden fixture; measured request count is reduced by chosen batch size; memory and maximum in-flight work are bounded; benchmark records before/after. | L / Medium: ranking/merging | T6; T2 |
| **T15 — Extract ingestion stages** | Split discovery/dedup planning, batch commit, video completion, face reconciliation and reporting into small modules with explicit result objects. Keep CLI options/output stable and use the corrected behavior as the baseline (A08). | Each extraction is separately reviewed; characterization tests pass; no closure state controls destructive decisions across stages; no new service/dependency. | XL: five M–L slices / Medium | T4, T5, T10, T0R |
| **T16 — Close temporary/job lifecycle gaps** | Use an owned temporary root with live-process protection, startup recovery and cooperative cancellation; expire finished jobs with a documented policy. Areas: sessions/lifespan/analyze/jobs/config (A19). | Crash-recovery fixture removes only orphaned owned files; active sessions survive; disconnected work stops at safe checkpoints; N completed jobs cannot grow memory without bound. | L / Medium: accidental cleanup | T3, T7, T8 |

### Milestone 3 — Quality and polish

| ID / title | Description and affected files/areas | Acceptance criteria | Effort / risk | Dependencies |
| **T17 — Add a small browser and real-store smoke suite** | Cover upload→query→hit rendering, face selection/compare, and playback failure/download in a real browser; extend integration to resume/purge/identity. Keep heavyweight tools development-only. Areas: tests, CI (A15). | Script execution and visible DOM behavior both verified; no text-only assertion substitutes for rendering; tests use a tiny synthetic corpus and clean their state. | L / Low | T2, Milestone 1 relevant fixes |
| **T18 — Inventory and reproduce airgap artifacts** | Make bundle groups explicit, pin build tooling, inventory wheels/models/native libraries/notices, and attach an advisory-scan result to each release. Rehearse offline install; use a non-root app user once volume permissions are proven. Areas: download/build scripts, Dockerfile, third-party notices (A16). | Inventory matches actual image; intended runtime groups only; lock hashes retained; offline start succeeds; vulnerability/license unknowns are recorded with owners rather than declared clean. | L / Medium: packaging | T2, T3 |
| **T19 — Repair authoritative operator documentation** | Correct remote URL, effective defaults, lossy playback disclosure, override behavior and distribution statements; link one current operations guide and mark historical snapshots. Add private security reporting only if maintainers will monitor it. Areas: README, INSTALL, `.env.example`, deployment, SECURITY, notices (A18). | Commands validated on a synthetic setup; documented defaults compared to Settings; playback docs distinguish source/rewrap/encode; no read-only promise for a write-capable override. | M / Low | T3, T6, T18 for final release wording |

### Quick wins — high impact, S effort

- **NVIDIA schema fix and Compose validation:** the configuration portion of T2 can ship immediately; hardware runtime verification remains separate.
- **Lock enforcement and matching Qdrant CI version:** the lock/version portion of T2 is S; account for any real incompatibility it exposes instead of suppressing it.
- **Offload the two blocking endpoints:** T14a is independently useful and follows an existing pattern.
- **Fix the remote endpoint example and false defaults/disclosures:** the factual corrections in T19 are S; validate examples locally and do not wait for a large documentation rewrite.

### Implementation sketches for the top three substantive tasks

**1. T3 — durable shared state.** Inventory effective paths and recover any existing container-local records before introducing a mount, because mounting over `/app/data` hides its prior contents. Give index and web processes the same explicit case-state mount, with evidence still read-only and correct UID/GID ownership. Test with tiny synthetic media, `run --rm`, and forced web recreation; verify both Qdrant references and historical audit/report bytes. Do not classify logs/manifests as regenerable cache.

**2. T4 — source-scoped face identity.** Define a typed identity such as source kind + source digest + optional frame timecode; keep frame/chip content hashes for deduplication. Use the same identity in markers, stale selection, and observation IDs, and include bbox/alignment version only where appropriate. Inventory legacy IDs and adjudication references, generate a dry-run mapping, then migrate/reindex in a controlled case copy. Gotchas: identical frames at different times, identical frames across different videos, truly duplicate whole source files, and labels tied to old `observation_key` values. Never turn a schema migration into an implicit stale purge.

**3. T5 — verified purge.** Resolve the requested scope explicitly: a source-video digest must select `video_hash`, an image digest must select its image scope. Enumerate observations and markers, delete in bounded batches with acknowledged completion, verify the target scope is empty, then garbage-collect only chips no surviving observation references. Record intent and outcome separately so partial failures remain auditable. Coordinate with T4's identity schema; test shared chips, repeat purge, database failure and cross-case isolation. Existing single-writer/per-case chip-directory constraints still apply (`src/scalar_forensic/cli.py:2069–2079`).

## Open Questions

1. **Case usage:** is this already used for evidential conclusions or still evaluation? Which workflows must be supported before the next release? The PoC label and forensic-facing features imply different release obligations (`SECURITY.md:5`; `README.md:3`).
2. **Current storage:** do deployed overrides already persist app state? Are reports/audit logs trapped in existing containers, and what backup/retention policy exists? A01 evaluates the shipped configuration, not live deployment.
3. **Identity semantics:** should the same bytes at two source paths be one content item with two occurrences? How must old face adjudications survive migration? Source-video/timecode distinctions are necessary, but file-occurrence policy needs an owner.
4. **Access and attribution:** is a single configured examiner per process acceptable for the documented multi-user UI, or must actions be attributable to authenticated browser users? No authentication redesign should be chosen without that requirement.
5. **Inspection override:** should incompatible collections permit only metadata/original viewing, or also explicitly marked experimental similarity queries? CLI currently permits writes under override; documentation must stop promising otherwise.
6. **Operational scale:** maximum corpus, concurrent examiners, longest videos, disk budget, and target query/chunk latency? These determine whether batched search alone suffices and when full-corpus in-memory dedup becomes a problem.
7. **Face statistics:** is the desired chart the best N matches or a representative population distribution? These are distinct measurements and should not share an ambiguous label.
8. **Release environment:** can maintainers reproduce the TestClient/AnyIO stall on the locked CI environment, supply a supported GPU smoke run, and rehearse a truly offline bundle? Local partial checks cannot certify those targets.
9. **Distribution rights:** which actual model artifacts are shipped to recipients and under which permissions? Repository notices alone cannot settle model/native-library obligations.

### Final verification addendum

Completed command: `.venv/bin/python -m pytest tests/test_safeguards.py tests/test_manifest.py tests/test_query_eval.py tests/faces/test_store.py --no-cov -q -p no:cacheprovider` — **101 passed in 2.28 seconds**, exit 0. The full suite was interrupted (exit 130); the route-inclusive subset and isolated stalled-route diagnostic ended at their explicit limits (exit 124). The stack trace localizes the wait to TestClient/AnyIO orchestration but does not establish a repository defect; reproducing it outside this environment remains open. No fresh total-coverage figure is available.

Final document checks found all six requested sections, 19 findings (six High, thirteen Medium), and no missing or out-of-range explicit repository path/line citations. `git diff --stat` remained empty: tracked code/configuration/tests/locks were unchanged. The initial untracked override and audit files remain, with this new report added alongside them. No remediation in the Task Plan was executed.
