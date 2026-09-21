# Managed image parser worker (opt-in)

This isolates image decoding, not video parsing. NVIDIA inference stays in the
application. Read the [boundary and recovery limits](specs/media-parser-isolation.md)
before enabling it. No running deployment is changed by these instructions.

## Build and start

Build from an explicitly selected, already-local application image containing the
locked Pillow/HEIF/NumPy dependencies. The overlay performs no downloads and is
not a minimal-dependency image. Verify the base contains no credentials or case
data: everything already baked into that image remains visible to the parser.

```sh
docker build --pull=false --network=none -f Dockerfile.media-worker \
  --build-arg BASE_IMAGE=YOUR_LOCAL_APPLICATION_IMAGE \
  -t scalarforensic:media-worker .
```

Docker warns that BASE_IMAGE has no default; supplying it is intentional. Use a
short, private state path under an existing non-symlink parent. The supervisor
creates its directory with mode 0700; Unix socket paths must fit the OS limit.
Run as the nonroot operator with Docker access, using the matching project Python
environment (set PYTHONPATH to this checkout's src if reusing another venv):

```sh
python -m scalar_forensic.media_worker.supervisor \
  --image scalarforensic:media-worker --state-dir /var/tmp/sfn-case1 \
  --case CASE1
```

In another process with the same UID/GID, configure the application or ingestion
CLI before starting it:

```sh
export SFN_MEDIA_WORKER_STATE=/var/tmp/sfn-case1/current.json
export SFN_MEDIA_WORKER_CASE=CASE1
export SFN_MEDIA_WORKER_RECEIPTS=/YOUR_PRIVATE_CASE_DATA/decode.jsonl
```

Leave SFN_MEDIA_WORKER_SOCKET and SFN_MEDIA_WORKER_GENERATION unset. The receipts
parent must already exist and be writable by that operator. SIGTERM stops the
supervisor and removes its owned containers and current publication; it retains
history, failure notices and old IPC directories. It never deletes case data.
Do not reuse a running generation across cases.

For a containerized application, add `docker-compose.media-managed.yml` to your
existing Compose files. Set SFN_MEDIA_WORKER_STATE_DIR, CASE, UID and GID to match
the host supervisor, and start the supervisor with `--client-root /sfn-worker`.
Provision application data permissions for that UID explicitly; the override does
not chown existing data. Use an application image containing this managed-client
code. The app receives state read-only and only the failure-notice subdirectory
writable. Neither app nor parser receives the Docker socket. Start the supervisor
before the application. Do not also apply the static-worker template.

The worker has four CPU quota units, four concurrent requests, four queued
requests, 4 GiB memory, 256 PIDs and a 256 MiB scratch tmpfs. It uses runc with no
network or GPU, read-only root, dropped capabilities and no-new-privileges. It
mounts only its own IPC directory, never original evidence. The application CPU
allocation is separate; the accepted benchmark used a four/four allocation on
eight CPUs, not the rejected shared scheduling variant.

### Image limits and upgrades

Export the same `SFN_MAX_IMAGE_PIXELS` setting before starting the host supervisor
and the application. The supervisor forwards the canonical limit to every worker
generation and records it in `generations.jsonl`. Unset or empty uses Pillow's
default; a positive integer sets the guard and `none` explicitly disables it.
The independent source and transport limits remain enforced.

Image protocol v2 carries the application's effective pixel limit on each request.
The worker checks that it matches its fixed startup guard before decoding; it
never changes a process-wide guard for concurrent requests. A mismatch fails
closed and requires correcting configuration and replacing the generation.
Receipts record the guard and protocol version. The static Compose template also
forwards `SFN_MAX_IMAGE_PIXELS`; it must match the application setting.

When updating from protocol v1, rebuild the worker image from the same source as
the application, then restart its supervisor with the new image. Old image
workers are intentionally incompatible and cannot trigger local decoding.
Worker-selection settings must be entirely absent to select local mode;
explicitly empty or incomplete worker settings are configuration errors.

## Recovery

Failures are not proof of malicious content, and successful decoding is not proof
of authenticity. The host replaces failed generations but never retries a source
automatically. Preserve receipts and generations.jsonl, identify the failed
generation and use:

```sh
python -m scalar_forensic.media_worker.receipts /YOUR_PRIVATE_CASE_DATA/decode.jsonl \
  --case CASE1 --generation GENERATION_FROM_HISTORY
```

Review all listed source hashes, including unfinished requests. Remediate the
decoder or exclude the triggering source before rebuilding affected derived
results. Keep originals and audit history. Reprocessing an exploit unchanged may
compromise another generation. There is no automatic database quarantine or
deletion. If Docker is unavailable, restore it and restart with the same private
state directory to reclaim exactly owned orphan containers; inspect errors rather
than assuming removal succeeded.
