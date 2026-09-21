# Managed video workers (opt-in)

Video probing, sampled extraction, seeking and bitstream-copy rewrap run in a
CPU-only worker. FFmpeg parsing/filtering and NVENC encoding stay together in
separate chunk and full-export workers. NVIDIA inference stays in the application.
The renderer has GPU-driver access: the operator explicitly accepted that residual
risk. Read the [boundary specification](specs/media-parser-isolation.md) before
enabling this profile. These instructions do not alter a running deployment.

## Build and provision

Build offline from an explicitly selected local application image containing the
locked PyAV/FFmpeg dependencies and NVENC support:

```sh
docker build --pull=false --network=none -f Dockerfile.video-worker \
  --build-arg BASE_IMAGE=YOUR_LOCAL_APPLICATION_IMAGE \
  -t scalarforensic:video-worker .
```

The missing default for BASE_IMAGE is intentional. This is a source overlay, not
a minimal-dependency image. Verify that the base contains no credentials or case
data; baked-in files remain visible. The supervisor resolves the image to its
immutable local identity and refuses image-declared writable volumes.

Use one nonroot host operator with Docker access. The application must have that
same UID/GID for private control files. Parser processes use UID 65534 and the
operator's GID; the operator cannot itself be UID 65534. Provision group-readable
case files and group-traversable directories as appropriate for your evidence
storage policy. The supervisor does not change original evidence permissions.
Unsupported access fails visibly, without an in-process parser fallback.

Choose short, private, distinct state directories beneath an existing non-symlink
parent. Unix socket paths have an OS length limit. The supervisor creates private
state, history, failure-notice and control directories. Do not reuse a generation
across cases. Keep receipts in application-owned durable case storage, outside all
parser mounts; their parent must already exist.

For web uploads, precreate a **dedicated** query-upload directory owned by the
application operator. Do not use the application data directory, a general `/tmp`
mount or the filesystem root. Supply it to all three supervisors as `--uploads`.
The application requires its configured session root to match the CPU worker's
upload mapping. In managed-video mode it makes this dedicated root and new session
directories group-readable/traversable (0750), and completed uploads group-readable
(0640). Existing local mode retains private session directories. Workers receive
this separate directory read-only; upload ownership and deletion remain with the
application.

## Start the three slots

Run these as three independently supervised host processes, using the project
Python environment. When sharing another checkout's venv, set `PYTHONPATH` to
this checkout's `src`. Replace the example case/input/upload paths explicitly.

```sh
python -m scalar_forensic.media_worker.video_supervisor \
  --image scalarforensic:video-worker --role cpu \
  --state-dir /var/tmp/sfn-vcpu --case CASE1 \
  --evidence /YOUR_CASE_INPUTS --uploads /YOUR_QUERY_UPLOADS

python -m scalar_forensic.media_worker.video_supervisor \
  --image scalarforensic:video-worker --role renderer \
  --state-dir /var/tmp/sfn-vchunk --case CASE1 \
  --evidence /YOUR_CASE_INPUTS --uploads /YOUR_QUERY_UPLOADS --gpu 0

python -m scalar_forensic.media_worker.video_supervisor \
  --image scalarforensic:video-worker --role renderer \
  --state-dir /var/tmp/sfn-vfull --case CASE1 \
  --evidence /YOUR_CASE_INPUTS --uploads /YOUR_QUERY_UPLOADS --gpu 0
```

Each process stays in the foreground. SIGTERM removes its exactly owned workers
and publication; history and failure notices remain. A service manager may restart
the supervisor with the same state directory to reclaim owned orphan containers.
It never retries failed evidence automatically. Start the workers before the app:

```sh
export SFN_VIDEO_CPU_STATE=/var/tmp/sfn-vcpu/current.json
export SFN_VIDEO_CHUNK_STATE=/var/tmp/sfn-vchunk/current.json
export SFN_VIDEO_FULL_STATE=/var/tmp/sfn-vfull/current.json
export SFN_VIDEO_WORKER_CASE=CASE1
export SFN_VIDEO_WORKER_RECEIPTS=/YOUR_PRIVATE_CASE_DATA/video-worker.jsonl
export SFN_SESSION_TEMP_ROOT=/YOUR_QUERY_UPLOADS
```

All five `SFN_VIDEO_*` settings are required together. Missing or invalid selected
state fails closed. A full-slot replacement does not block the independent chunk
or CPU slot. With all five settings absent, legacy development mode is unisolated.
For image isolation, configure the [image worker](deployment-media-worker.md) too;
neither profile implicitly enables the other. CLI-only deployments may omit the
upload mapping if every submitted video is beneath the evidence root.

For a containerized application, use the same UID/GID and provide each state
directory as a read-only bind. Overlay only its `control/` and `failures/`
subdirectories as writable binds. The application needs those for admission,
active-request records and failure notices; it does not need writable IPC or the
Docker socket. Use `--client-root` for the corresponding app-visible state path,
`--client-evidence` for the app-visible evidence path and `--client-uploads` for
the app-visible upload path. Keep the app's existing authorized evidence mount
read-only, and mount only its dedicated upload directory writable for staging.
Start supervisors before creating these binds; never let Docker silently create
missing host paths. No Compose override is enabled automatically.

## Limits and lifetime

Each slot admits one operation. Chunk admission waits at most 30 seconds so a
source switch can outlast an existing prefetch; CPU/full admission waits at most
five seconds. Renderer capability is cached only for the validated generation
and requested hardware mode, so metadata does not queue a fresh capability probe
behind every chunk. A source switch may still visibly wait for the old encode;
there is no unbounded queue or automatic cancellation of another request.
Defaults per slot are four CPU quota units, 12 GiB memory (no extra swap), 512
PIDs, 8 GiB scratch tmpfs, 256 MiB temporary tmpfs, a 4 GiB output ceiling and a
one-hour request deadline. These are ceilings, not preallocated memory. Three
slots can still compete for CPU/GPU/memory; budget the application and image
workers separately. Full-export capacity is not unlimited.

Explicit launcher flags `--cpus`, `--memory-bytes`, `--scratch-bytes`,
`--max-output-bytes`, `--job-timeout` and `--rotate-seconds` tune these bounds.
The output ceiling must be smaller than scratch, which must be smaller than the
memory limit. Larger unsupported exports fail instead of silently weakening the
policy. The application deadline/size limits may be tighter. Repeated sources do
not require a new sandbox or a full-file copy per chunk. Initial source hashing
can still read an uncached file once; metadata-only probes record a nullable hash
when it is not already cached.

Workers have no network, application configuration, models, Docker socket or
durable output mount. Root and recursively bound inputs are read-only; scratch is
quota-limited. Capabilities are dropped, no-new-privileges and the default Docker
seccomp policy stay enabled. Only renderer slots receive NVIDIA devices. The host
seals each IPC directory read-only before publishing the socket; the different
parser UID cannot create files there or replace the socket. This relies on trusted
worker bootstrap before any evidence is submitted.

Ordinary hourly rotation waits for an idle slot. Cancellation, protocol failure,
abandoned application requests and deadline expiry remove the exact generation
even when its request lock is held. The client requires a different published
generation before treating failure cleanup as confirmed. If Docker cannot remove
the old worker, the slot remains unavailable; inspect the supervisor error and
ownership history instead of deleting control files to bypass the check.

## Provenance and recovery

The application validates raw RGB frame bytes and metadata, or bounded opaque MP4
bytes; it never reopens worker-produced MP4 to validate it. It assigns the cache
destination, verifies the transfer digest and publishes only a completed result.
Existing rendering audit rows additionally identify the trusted worker runtime;
pipeline fingerprints keep their existing pixel/timing meaning.

Private JSONL receipts include case, generation, immutable image, policy, request,
operation, source mapping/stat identity, nullable source hash and outcome. A
`started` row is durable before submission; `validated` is durable before final
artifact publication. Validation means accepted bytes/metadata, **not** successful
indexing, publication or proof of authenticity. A later publication failure can
therefore have both validated and failed rows. Preserve these alongside each
slot's `generations.jsonl` and the existing video audit log.

When compromise is suspected, review **all** receipts for that case/generation,
including unfinished requests. A JSONL viewer/filter can select their `case_id`
and `generation` fields; keep source paths/stat identities for rows whose hash is
null. The image receipt CLI does not accept this video schema. No automatic
database quarantine/deletion or dedicated video recovery CLI is supplied.
Remediate the decoder or exclude the trigger before rebuilding derived results;
blindly replaying an exploit can compromise another worker. Originals and audit
history must remain intact.

Read-only inputs protect against worker writes, not concurrent operator changes.
Before/after file identity detects ordinary changes; it is not an immutable
snapshot. Reused compromised parsers can silently falsify later outputs in their
generation. Kernel/GPU-driver vulnerabilities and browser-side media parsing
remain outside this containment guarantee.
