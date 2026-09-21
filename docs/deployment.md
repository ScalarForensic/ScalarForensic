# Deployment topologies

ScalarForensic is designed so that its four roles can run on one machine or be
distributed across several machines on an isolated local network, reached over
their local IPs:

| Role | Process | Listens on | Key settings |
|---|---|---|---|
| Vector database | Qdrant | `6333` | — |
| Embedding compute (GPU) | any OpenAI-compatible embeddings server | your choice | — |
| Application host | `sfn` (CLI indexing) / `sfn-web` (web UI) | `SFN_WEB_PORT` (default `8080`) | everything below |
| Clients | browser | — | just the app host URL |

All configuration is environment variables (or `.env` next to the process CWD);
precedence is process env → `.env` → defaults. See `.env.example` for the full
annotated list.

## Optional media parser isolation

The [managed image worker](deployment-media-worker.md) and
[managed video workers](deployment-video-worker.md) isolate encoded-media parsing
from the application's database access and durable writes. Both are opt-in and
configured separately. The image and CPU video workers have no GPU access;
the NVENC renderer has the explicitly documented GPU access. Configure the CLI
and web process consistently for the same case.

Read the [boundary specification](specs/media-parser-isolation.md) and worker
upgrade instructions before enabling them. Local integration does not enable
workers or resolve outstanding decoder, host, or release-security findings.

## Topology 1 — single machine (default)

Everything on one workstation. The defaults work as-is: Qdrant on
`http://localhost:6333`, models loaded locally (GPU/CPU per `SFN_DEVICE`),
web UI on `http://localhost:8080`.

## Topology 2 — separate Qdrant host

Move storage/database load to another machine. On the application host:

```bash
SFN_QDRANT_URL=http://192.168.10.20:6333
SFN_QDRANT_API_KEY=...        # if the Qdrant instance requires it
```

The application host still computes embeddings locally, so it needs the model
files and (ideally) a GPU.

## Topology 3 — full split (Qdrant + GPU embedding server + app host)

Embedding compute moves to a dedicated GPU machine exposing an
OpenAI-compatible embeddings API. On the application host:

```bash
SFN_QDRANT_URL=http://192.168.10.20:6333

SFN_EMBEDDING_ENDPOINT=http://192.168.10.30:9000
SFN_EMBEDDING_API_KEY=...     # if the endpoint requires it
SFN_EMBEDDING_MODEL=<model name the endpoint expects>
SFN_EMBEDDING_DIM=<embedding dimension the endpoint returns>
```

Note the integrity trade-off: a remote endpoint cannot produce a weights hash,
only a config hash derived from endpoint + model name + dimension. These are
stored and displayed with a `config:` prefix and a "cfg-only" chip in the UI so
they are never mistaken for weights hashes. For court-facing work, prefer
locally verified models (Topology 1/2) or pin the remote server's model
artefact out-of-band.

## Reachability matrix

- Clients → application host: `SFN_WEB_PORT` (default `8080`).
- Application host → Qdrant: `SFN_QDRANT_URL`.
- Application host → embedding server: `SFN_EMBEDDING_ENDPOINT` (Topology 3 only).
- Nothing needs to reach the clients; Qdrant and the embedding server never
  talk to each other.
- No internet access is required or expected at runtime: the network policy is
  offline-by-default (`SFN_ALLOW_ONLINE=false`). Set it to `true` (or pass
  `--allow-online`) only for first-time model downloads, then revert.

`sfn-web` binds `SFN_WEB_HOST=0.0.0.0` by default so LAN clients can reach it.
On a machine that is not inside the isolated network, narrow it to
`SFN_WEB_HOST=127.0.0.1`.

## State that lives on the application host

These paths are read/written by both `sfn` (indexing) and `sfn-web` (serving),
so **indexing and the web UI must run on the same machine — or share these
directories** (e.g. one mount) if you split them:

| Setting | Default | Used for |
|---|---|---|
| `SFN_INPUT_DIR` | unset | evidence images root; also the allowed root for file-serving endpoints |
| `SFN_THUMBNAIL_DIR` | `data/thumbnails` | thumbnails written during indexing, served at `/api/thumbnail/…` |
| `SFN_FRAME_STORE_DIR` | `data/frames` | extracted video frames, source for thumbnail regeneration |
| `SFN_HASH_CACHE_PATH` | `data/hash_cache.db` | SHA-256 cache keyed by (path, mtime, size) — path-specific, do not share between machines |
| `SFN_VIDEO_CACHE_DIR` | `data/video_cache` | viewing copies of source videos, written by `sfn-web` on demand — see below |
| `SFN_VIDEO_CACHE_MAX_BYTES` | `8589934592` (8 GiB) | size ceiling for that directory; `0` = no ceiling |
| ingestion reports | `data/reports/` | per-run CSV reports from `sfn <directory>` |

Face chips (`SFN_FACE_STORE_DIR`, default `data/faces`), ingestion manifests
and both audit logs also live on the application host and must be backed up.
Collections, tags and adjudication records live in Qdrant; their source evidence
and referenced files remain application-host responsibilities.

### Playback viewing copies

Playback may serve the original, losslessly rewrap supported streams, or encode a
lossy H.264 viewing copy. Encodes can change pixels (including HDR tone mapping)
and carry a disclosure plus recorded pipeline/command provenance. Only a rewrap
copies the encoded streams bit for bit. Original evidence and indexed frames stay
unchanged. Playback failure leaves original download available.

`SFN_VIDEO_CACHE_DIR` defaults to `data/video_cache`. Its default ceiling is 8 GiB;
`0` disables the ceiling, and an empty directory setting disables derived viewing
copies. Leases and pins protect active work from eviction. Full-copy forecast
refusal can be overridden for one attributed request, but an unknown estimate
cannot; the byte watchdog remains active. The video audit log sits beside the
cache and survives cache purges. Preserve audit logs even when derived files are
regenerated. See [playback specification](specs/video-playback-transcode.md).

## Multi-user limits (web UI)

The web UI supports modest concurrent use out of the box:

- `SFN_MAX_ACTIVE_SESSIONS` (default 32) — further sessions get HTTP 503.
- `SFN_SESSION_TTL_SECONDS` (default 3600) — idle sessions are reaped every 60 s.
- `SFN_MAX_UPLOAD_BYTES` (default 2 GiB) — per-request upload cap, HTTP 413 above.
- `SFN_MAX_CONCURRENT_UPLOADS` (default 2) — concurrent request-body admission.
- `SFN_SESSION_TEMP_ROOT` (default system temp directory plus
  `scalar-forensic-sessions`) — process-owned session uploads; keep space available.
- `SFN_VIDEO_COMPLETED_JOB_MAX` (default 256) — maximum retained finished jobs.
- `SFN_VIDEO_COMPLETED_JOB_TTL_SECONDS` (default 3600) — retention of finished job status.

Use one web worker. `SFN_EXAMINER_ID` attributes the process, not individual browser
users; it is not authentication. Operator-managed LAN access controls and any
reverse-proxy authentication remain necessary for the chosen deployment. Configure
proxy request limits to match the application's upload cap, including multipart
overhead, and allow streaming responses without proxy buffering.

## Consistency constraints across machines

Embeddings are only comparable when produced with identical parameters. The
startup/ingest safeguards will fail fast on mismatches, but when distributing,
keep these identical everywhere embeddings are produced:

- model artefacts (`SFN_MODEL_DINO`, `SFN_MODEL_SSCD`) — verify with the stored
  model hash (`scripts/download_models.py --hash <stored_hash>`)
- `SFN_NORMALIZE_SIZE`
- `SFN_SSCD_N_CROPS`

Collection names (`SFN_COLLECTION`, `SFN_TAGS_COLLECTION`,
`SFN_REFERENCE_COLLECTION`) must match on every machine that talks to the same
Qdrant instance.

## Legal assumptions for the optional face modality

This section states **assumptions, not legal conclusions.**

The face modality is a post-hoc biometric search aid operated by the deploying organisation
(the deployer role). Depending on jurisdiction and use, EU AI Act obligations for post-remote
biometric identification, GDPR/LED duties (including a DPIA and Art. 9/10 special-category
processing), and national law — including authorisation regimes for law-enforcement use — may
apply, with obligations phasing in on the Act's timeline.

**The technical controls in ScalarForensic support such compliance; they do not establish it.**
Jurisdiction-specific legal review before enablement is an operator duty, as is the assessment
of your own role under the applicable framework.

The controls the legal story rests on:

- **Append-mode audit log** (`data/face_audit.log`) recording enablement, index runs and
  purges — what the system did and how it was used, separately from artefact provenance.
- **Enablement record**: first activation writes examiner id, timestamp and a free-text
  authorization reference into the collection metadata and the audit log. Enabling the feature
  is a recorded act, not config drift.
- **Provenance on every artefact**: models, hashes, alignment version, and every gate threshold
  in force (see `docs/face-matching-math.md`).
- **Purge path** for face observations and their chip files.
- **Disabled by default**, with no recognition weights fetched automatically.

Machine output is an investigative lead. Identification is a human conclusion by a named
examiner, and the tool is built to keep that line visible.

## Durable Compose state and recovery

This is the current operations guide. Dated audit reports and historical design
snapshots describe their recorded state, not current operator instructions.
Use the shipped base file explicitly when
validating, so an operator's local override does not change the result:

```bash
docker compose --env-file .env.example -f docker-compose.yml config --quiet
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.amd.yml config --quiet
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.nvidia.yml config --quiet
```

Compose mounts `${SFN_CASE_DATA_DIR:-./case-data}` at `/app/data`, writable by both
`sfn-web` and `docker compose run --rm sfn-web sfn …`. It pins thumbnails, frames,
face chips, hash cache, reports/manifests, and playback cache beneath that mount.
Face and video audit logs live beside their respective stores, at
`/app/data/face_audit.log` and `/app/data/video_audit.log`. Evidence stays mounted
read-only at `/images`; keep the host state directory outside the evidence root.
The mount does not cover `/app/models` or `/app/src`. For separate cases, use
separate Compose project names, state directories, and Qdrant volumes. Do not run
concurrent indexers against one case.

The image runs as UID/GID `10001:10001`. Before first use:

```bash
sudo install -d -m 0750 -o 10001 -g 10001 /srv/scalarforensic/case-data
# Set SFN_CASE_DATA_DIR=/srv/scalarforensic/case-data in the Compose .env.
```

For existing state, review the directory before assigning ownership to UID/GID
10001. Ensure that this user can traverse and read the evidence and mounted `.env`
(e.g. a dedicated group/ACL), while the evidence mount remains read-only. AMD
GPU device access additionally needs the host's video/render group permissions.
Do not assume matching group names imply matching host/container numeric GIDs.
Create the state directory with ownership permitting the image's runtime user to write; do not solve permission errors by making evidence writable.
Check ownership after restoring archives, particularly when changing container UID.
All clients referencing stored paths must retain the same `/images` and `/app/data`
layout. Host-mode indexing must use matching paths or be reindexed for this layout.

### Recover previous container-local artifacts before changing mounts

Do not recreate or remove old containers first: a mount over `/app/data` hides
container-local state, and an already removed `run --rm` container cannot be
recovered through Docker. Inventory existing containers with `docker ps -a` and
record their IDs, image IDs, Compose project, mounts (`docker inspect`), and the
configured custom state paths. Stop writers, then copy each surviving container's
`/app/data` to its own recovery directory using `docker cp CONTAINER:/app/data
RECOVERY_DIRECTORY`. Also recover any state paths configured outside `/app/data`.
Retain each recovered directory separately; do not overwrite conflicting audit
logs or reports. Inventory them with the helper below, reconcile into the new
case directory, then introduce the mount. Record absent reports/logs as
unrecoverable; regenerating an index does not reconstruct historical audit acts.
Original evidence must remain untouched.

### Consistent backup and isolated restore rehearsal

For the shipped single-node Qdrant stack, a stopped storage copy preserves all
collections, including tags, face metadata and adjudications. The following is a
manual maintenance procedure: stop indexing and stop the stack before copying.
Use the same explicit Compose project/file arguments for every command (including
GPU/local overrides used by your deployment). Do not use `down -v`.

```bash
# Choose a fresh restricted backup directory outside evidence and case state.
backup=/secure/backup/case-2026-09-08
mkdir -m 700 "$backup"
docker compose stop
# These commands require the existing containers to remain present.
docker cp "$(docker compose ps -aq sfn-web)":/app/data "$backup/app-data"
docker cp "$(docker compose ps -aq qdrant)":/qdrant/storage "$backup/qdrant-storage"
python3 scripts/verify_case_state.py capture "$backup/app-data" "$backup/app.sha256.json"
python3 scripts/verify_case_state.py capture "$backup/qdrant-storage" "$backup/qdrant.sha256.json"
```

Also preserve the effective configuration (including custom paths), exact app and
Qdrant image IDs/tags, `uv.lock`, model hashes/artifacts, evidence location/hash
inventory and the Compose project/volume mapping. Configuration may contain
secrets; keep it in restricted backup storage. Never copy live SQLite/WAL or Qdrant
storage and call it consistent. Restart the source stack only once the stopped
copies and manifests are complete. Checksums detect changed bytes; they do not
authenticate the backup. Audit logs are ordinary append-mode files, not immutable
or tamper-proof storage.

Restore first into a **separate case directory and fresh Qdrant volume/project**,
using the exact Qdrant image version recorded with that backup. Copy `app-data` into the new host state
location. Create a stopped Qdrant container with the fresh volume, copy the saved
`qdrant-storage` contents into `/qdrant/storage`, and verify both restored trees
with `verify_case_state.py verify RESTORED_ROOT MANIFEST` before starting it.
Retain ownership and permissions. Set the restore project's `SFN_CASE_DATA_DIR`
and evidence mount, use a different web port, and check the resolved Compose
configuration before starting. Do not point the restore at the source database.

After starting the isolated copy, check collection counts/metadata, open a sample
image hit, video frame and face chip, compare preserved report/log checksums, and
confirm original downloads resolve to the same evidence bytes. Exercise disposable
CLI indexing (`run --rm`), recreate only the restored web service, and repeat the
checks. Keep the result with the case backup. Rebuild incompatible indexes only
in the approved case copy; historical reports/logs must survive the rebuild.

The automated local rehearsal is:

```bash
.venv/bin/python -m pytest tests/test_operations_restore.py --no-cov -q
```

It copies synthetic application state into a separate location, resolves saved
frame/chip references, verifies all checksums, and detects a changed log. It does
not certify Qdrant recovery, real browser behavior, GPU availability, or a live
container recreation; those require the isolated deployment rehearsal above.

## Airgap release evidence

On a connected build host matching the target OS/architecture and Python 3.12,
install `uv`, Python's pip download tool, Docker, and optionally Trivy. Run
`bash scripts/download_deps.sh --web` (add `--faces`/`--heif` only when intended).
The exporter uses `--locked --no-default-groups`; build tooling is exported
separately from its pinned build group, and both exports retain lock hashes. The
Docker build installs that tooling into a temporary target removed after building
the application; it is not installed as an application runtime group.
Use a fresh vendor directory for each target/group set: the downloader refuses a
non-empty destination to prevent stale wheels entering the bundle. Runtime groups are chosen explicitly, not inherited from dev.
The image builder's default is core + web; for extra groups prepare vendor first
and use `--no-deps`. Keep Python 3.12 and the selected GPU lock together.

`bash scripts/build_airgap_image.sh --tag RELEASE --scan` creates an artifact
inventory, builds the image as UID/GID 10001, captures actual image IDs, and scans
both app and Qdrant with the installed Trivy. Scanner errors abort the build;
findings are recorded for release-maintainer disposition. Keep scanner version
output, both advisory JSON files and image metadata beside the saved archive.
Without `--scan`, the advisory file explicitly records `not-run`. Neither a
successful command nor unknown license metadata means a release is clean.

`bundle-inventory.json` hashes every wheel/model and both dependency exports plus
`uv.lock`; wheel metadata supplies declared licenses and native member hashes.
`/app/native-inventory.json` records native files and OS packages from the actual
built image. Model rights remain UNKNOWN until the release maintainer associates
the exact hash with its upstream license. Review any unexpected file in `models/`
before building: the entire directory is copied. See `THIRD_PARTY_LICENSES.md`.

A release is not offline-verified until its saved images are loaded on an isolated
host, started with external network access blocked, used to index a tiny synthetic
corpus, and restored using the procedure above. Record the GPU driver/image pair
and actual smoke results. The repository's synthetic wheel-inventory and UID mount
checks do not substitute for this full image rehearsal.

## Browser smoke checks

Development-only Playwright tests load the shipped HTML, CSS and Alpine scripts
in Chromium, with synthetic API responses intercepted by the test. They exercise
upload, hit rendering, face selection/compare, and playback failure with original
download. They verify real DOM behavior, but do not claim backend/model integration.
Real Qdrant tests cover store identity, purge and resumed ingestion separately.

```bash
npm ci
npx playwright install chromium
npm run test:browser
```

CI runs this as a step in the existing required lint/test job. On release hardware,
also start the actual application and exercise the same flows against the tiny
indexed case, force-refetching every script and stylesheet before reloading.

## Collection compatibility and face identity rebuild

`--ignore-config-mismatch` on the web server permits metadata and original-evidence
inspection only. It does not enable incompatible similarity searches or ingestion.
Case and reference collections must match the effective model/preprocessing
configuration before use, including recovery after Qdrant becomes available. Strict validation
uses a bounded diagnostic sample followed by an exact filtered count across all
vector-bearing points; missing provenance also requires rebuilding. DINO validates
both processor settings and the actual tensor dimensions before inference. Changing
model/preprocessing settings requires restarting the web process so cached session
vectors cannot be compared under a different configuration.

Face searches preserve distinct source-video/timecode observations even when frame
bytes are identical. Repeated detection requests publish only the newest result;
query chips are pinned to detection content. Keep ingestion/rebuilding quiescent
during examination: compatibility checks and searches are separate database reads,
not a transaction that freezes a concurrently modified collection.

Face observation identity now includes source-video identity and timecode, while
chip bytes remain content-addressed. Legacy or unknown face identity schemas are
refused with an explicit rebuild requirement. Back up all metadata/adjudications,
reports, chips and logs before rebuilding into a new collection. Old observation
keys are not silently migrated or promised to resolve to new identities; retain
old records for inspection and explicitly review any label reassignment. Existing
indexes are disposable under this remediation's approved migration policy; original
evidence and historical audit records are not.


## Remediation verification boundary (2026-09-08)

See [the implementation and verification report](audit/remediation-results-2026-09-08.md)
for the completed audit tasks and checks. Synthetic integration passes against the
pinned Qdrant v1.19.0 service with the locked qdrant-client 1.19.0.
The release pins apply to fresh deployments. For existing 1.17.x storage, rehearse
upgrades through each intermediate minor version in isolation, following
[Qdrant upgrade guidance](https://qdrant.tech/documentation/upgrades/), or rebuild
disposable indexes into a fresh volume. Never start 1.19.0 directly on the only
copy of an older data volume. These tests do not certify a full
production deployment or model redistribution rights. The subsequent
[image and restore report](audit/release-checks-2026-09-08/image-and-restore-results.md)
records passing offline NVIDIA model execution, synthetic ingestion and whole-state
restore, plus image scans. The scanned validation candidate still requires
security remediation/applicability decisions and redistribution acceptance.
