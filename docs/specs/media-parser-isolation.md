# Media parser isolation

Status: opt-in managed image and video boundaries implemented locally.
Protect originals and deny worker network access; accept possible undetected
cross-file contamination of derived results within a worker generation.
The operator also approved GPU-device access for the isolated FFmpeg/NVENC
renderer, preserving its existing combined parsing/encoding pipeline. That
exception is implemented in the separate video-renderer policy; it is not a
GPU-free parser or a guarantee against GPU-driver vulnerabilities.
The rejected per-source Docker runner's
[startup measurement](../benchmarks/parser-container-startup-2026-09-12.md)
does not describe the recurring cost of reused workers. This document alone does
not establish production acceptance.

## Implemented image path and current limits

The opt-in image path is implemented. It has five small responsibilities:
`media_access` selects local or worker decoding, `protocol` validates raw messages,
`client` sends requests, `server` runs the existing decoders, and `receipts` records
which sources went through each worker generation. There is no general-purpose
job framework or container-management API in the application.

The worker receives image bytes, not source paths. Pillow/HEIF decoding, EXIF,
native face inputs, stored-chip thumbnails and HEIF previews use this interface.
GPU inference, face alignment, database writes and artifact destinations remain
on the application side. The existing local mode is unchanged unless the worker
environment settings below are supplied. Video has its own explicit configuration
and supervisors, described below; image mode does not implicitly enable it.

The current protocol limits each source to 128 MiB, the combined returned pixels
to 256 MiB, dimensions to 32,768, and the JSON header to 64 KiB. Returns are raw
RGB8; thumbnail requests also allow grayscale L8 to preserve existing rendering.
Oversized inputs are rejected, not resized to bypass a limit. Byte-oriented
ingestion callers already hold their input in memory; the protocol limit does
not retroactively bound those earlier reads. The new file-reading preview/chip
paths check file size before allocation.

Image protocol v2 also binds each request to the application's effective Pillow
pixel guard. The worker rejects a different guard before decoding, keeping its
startup value fixed across concurrent requests. The supervisor forwards
`SFN_MAX_IMAGE_PIXELS` explicitly and records its canonical value in the generation
journal; request receipts record it with the protocol version. Rebuild and restart
image workers when upgrading from v1. Empty worker-selection settings fail closed,
and managed clients check shared generation failure notices both before submission
and before accepting output.

Worker configuration is explicit and all-or-none:

| Application setting | Meaning |
| --- | --- |
| `SFN_MEDIA_WORKER_SOCKET` | Absolute path to the dedicated Unix socket |
| `SFN_MEDIA_WORKER_GENERATION` | New 32-character lowercase hex identity for each runtime instance |
| `SFN_MEDIA_WORKER_CASE` | Operator-assigned case identity; never reuse a generation across cases |
| `SFN_MEDIA_WORKER_RECEIPTS` | Private JSONL log outside the worker-writable IPC directory |
| `SFN_MEDIA_WORKER_TIMEOUT` | Optional total request deadline, default 40 seconds |

Settings for an already-seen generation cannot change within an application
process. A failed client stays failed; changing a timeout cannot revive it.
Generation identity is supplied by the trusted operator, not proof that a new
container was actually created. The operator must enforce replacement across
application restarts as well.

`docker-compose.media-worker.yml` is a standalone, opt-in runtime template, not
an enabled deployment. It requires an image containing this code, pinned by the
operator, and a precreated dedicated IPC directory owned by UID 10001. The socket
is mode 0600, so the application must use a matching UID when accessing it.
Mount only this IPC directory into the application, not the Docker socket.
The template intentionally does not restart automatically with the same identity.

For unattended use, `media_worker.supervisor` now provides independent host-side
replacement. It resolves a local image to its immutable ID, probes each new
generation, then atomically publishes private state. The application rereads that
state for every request (including queued batch items). Normal hourly rotation
allows the previous generation 45 seconds to drain. A client deadline failure
writes a notice outside parser-accessible storage; the host removes the failed
container and starts a fresh generation without replaying the failed source.
The supervisor polls every 250 ms; Docker operations have 15-second deadlines.
Docker failure can prevent physical removal: the supervisor fails closed and
exits visibly, and its durable ownership token permits cleanup on restart.

Managed mode uses `SFN_MEDIA_WORKER_STATE` instead of the socket/generation
settings; CASE and RECEIPTS remain required. Do not combine both modes. Missing
or invalid state fails closed, without local decoding. Managed receipts include
immutable image and policy identities; `generations.jsonl` records lifecycle
events. See [operator setup](../deployment-media-worker.md).

The receipt command lists affected source hashes; it does **not** quarantine
or remove database results automatically:

```sh
python -m scalar_forensic.media_worker.receipts /private/decode.jsonl \
  --case CASE --generation GENERATION
```

Use that list to review/rebuild derived results after remediation. Originals
and audit history are never deleted by this command. Receipts describe decoding,
not successful indexing. Static-mode receipts do not carry managed runtime
identity. The remaining requirements below remain targets where not explicitly
covered by the recorded validation.

## Implemented video path and current limits

`video_access` routes the existing probe, extraction, seek, rewrap and encode APIs
through a separate bounded protocol/client/server. Three independently supervised
slots serve CPU parsing, chunk rendering and full exports. Each slot has one
private Unix socket and one application/host admission lock outside parser
storage. Normal rotation waits for idle; external failure/deadline handling
removes the exact generation even if a request holds its lock. A separate
privileged forwarding broker is unnecessary: the host supervisor alone has Docker
access, while the application owns admission, receipts and publication.

CPU PyAV operations return validated raw RGB frames or metadata; rewrap returns
opaque MP4 bytes. The renderer keeps FFmpeg's existing software decode/filter,
NVENC, audio and mux pipeline. Only renderer slots receive GPU devices. The
application validates bounded artifacts and transfer hashes without parsing their
encoded contents. Existing pixel pipeline fingerprints remain unchanged; rendering
audit rows additionally carry trusted generation/image/policy attribution.

Videos are addressed relative to explicitly scoped, recursively read-only evidence
and optional upload mounts. There is no per-chunk source snapshot/copy. Trusted
hash-cache attribution and before/after file identity detect ordinary mutation,
not adversarial operator replacement. Metadata-only probes can have a null source
hash; their relative path and stat identity remain in receipts. The distinct
parser UID receives group read access to staged uploads, never ownership or
write access. Host-owned IPC is sealed before publication, after trusted bootstrap.

All five application settings are required together: `SFN_VIDEO_CPU_STATE`,
`SFN_VIDEO_CHUNK_STATE`, `SFN_VIDEO_FULL_STATE`, `SFN_VIDEO_WORKER_CASE` and
`SFN_VIDEO_WORKER_RECEIPTS`. Failure never enables local parsing. Queue admission
is bounded to 30 seconds for chunks and five seconds for CPU/full work; each slot
allows one operation. The chunk allowance lets valid prefetch work finish during
a source switch. Capability caching is generation/hardware-mode scoped and still
checks current private state before reuse. Default per-slot
limits are four CPU quota units, 12 GiB memory, 512 PIDs, 8 GiB scratch, 4 GiB
artifact output and one hour per request. Frames are raw RGB8 with at most 256 MiB
per payload; headers are at most 64 KiB. Limits fail explicitly, not by quietly
resizing evidence or relaxing policy. Full exports exceeding the configured
ceiling are unsupported until the operator explicitly provisions larger bounds.

The client fsyncs `started` and `validated` receipts, with validated meaning
accepted output rather than successful indexing/publication. On failure it retains
the active request until replacement is confirmed; disappearance of state alone
is not confirmation. There is no automatic replay or database quarantine, and
no dedicated video receipt recovery CLI. See
[video operator setup and recovery](../deployment-video-worker.md) and
[measured validation](../benchmarks/video-isolation-2026-09-12.md).

## Purpose and delivery boundary

ScalarForensic processes potentially adversarial evidence on an offline LAN with
trusted operators. Isolate encoded-media parsing from database access and durable
case writes without changing forensic processing semantics. Image and CPU video
workers have no GPU devices; the FFmpeg/NVENC renderer is the explicit exception.
Offline operation does not make hostile file content trusted.

The image implementation covers image, EXIF and face-input decoding for ingestion
and the corresponding server-side query/preview paths. The separate video stage
covers probing, extraction, rewrap and playback. An image-only configuration must
not advertise whole-media isolation. NVIDIA inference stays in the application;
NVENC remains the intended video encoder. Redistribution and AMD validation are
outside this delivery.

The [performance experiment](../benchmarks/media-isolation-2026-09-12.md) supports
investigating this architecture, not accepting it. Its reused parser process is
consistent with the revised lifetime decision, but does not establish the final
access boundary or end-to-end performance budget.

## Trust boundary and lifetime

The application reads authorized evidence as opaque bytes, hashes those same
bytes and submits a fixed decode operation. It does not open encoded evidence
with Pillow, HEIF, PyAV or ffmpeg on an isolated path. Face detection, alignment,
embedding and database writes consume validated raw arrays outside the parser.
Model loading remains trusted, operator-provisioned input, not an evidence parser.

Reuse CPU parser containers and decoder processes across sources within one
worker generation. Creation and library initialization are startup or restart
costs, not per-source costs. Keep private writable scratch apart from originals,
durable results and other workers. Do not reuse workers across cases. Recycle at
configured job-count or elapsed-time limits and after suspected compromise.
Those limits bound the recovery scope, not the ability to exploit a decoder.

An exploited decoder may silently falsify subsequent outputs in its generation.
Malicious-file detection is not guaranteed. Read-only access protects originals
against worker writes; it does not prove output correctness. Rerunning the same
exploit without remediation can compromise a replacement worker too.

An operator-controlled service manager starts and replaces workers from a pinned
image. Neither the application nor any parser receives the Docker socket.
The local permission-restricted interface accepts only fixed operations, bounded
source bytes and validated parameters, never client-selected images, commands,
mounts, device requests, environment variables or host paths. Worker deployment
is an explicit operator step. Confirm old-worker termination before trusting its
replacement; a parser's own claim of resetting is not sufficient.

Each parser container must have:

- No network connectivity, application configuration, model mounts, database
  credentials or runtime-control socket. No GPU/device grants except those
  required by the explicitly approved FFmpeg/NVENC renderer described below.
- A read-only root filesystem, unprivileged identity, dropped capabilities,
  no-new-privileges and an enforced syscall policy. No privileged mode or
  unconfined workaround when a required operation fails.
- Only submitted bytes or explicitly scoped read-only case inputs, plus private
  bounded scratch/transport storage. No durable case-output mounts or
  other-worker/shared writable dependency caches.
- Explicit CPU, memory, process-count, scratch-size and wall-time limits,
  plus bounded queue length and total concurrency in the launcher.
- A worker-specific data channel that cannot reach another worker or runtime
  control endpoint. Parser-created filenames and links are never trusted as
  instructions to open host paths.

Measure sustained end-to-end throughput including recurring decode, transfer and
recycling costs; report one-time startup separately. Containers share a host
kernel: this is not a guarantee against kernel vulnerabilities or malicious host
administrators. The application-facing channel is an attack surface too; network
isolation is not a substitute for strict receiver validation.

## Source identity and transport

For the first image stage, use the immutable bytes already read by the
application. Bind each request to a job identifier, source SHA-256, operation,
processing parameters and protocol version. The client supplies those exact
bytes, not an evidence pathname reopened later.
Enforce a finite source-size limit before allocating or copying the snapshot;
record an explicit resource-limit failure for unsupported sizes, without resizing
or weakening the limit automatically. Deployment limits are versioned settings
and must be recorded with the job.

This proves which submitted bytes were processed, not that the original path
cannot change later. Source hashing and stored source identity must use the same
byte sequence; a cached pathname/stat result is not an immutable snapshot.
Large-video snapshot/streaming rules belong to the video design and must not
introduce an unmeasured full-file copy per playback chunk.

Use a length-framed local data stream with a bounded, versioned JSON header and
explicit raw RGB8 payloads. Do not use pickle, object deserialization, arbitrary
NumPy object arrays or an encoded image as the GPU-side decode interface. Keep
bulk pixels off Docker attach/stdout; use an isolated local socket or equivalent
bounded data channel. Logging is separate and capped.

The trusted receiver validates headers before payload allocation: exact schema,
finite numeric values, integer dimensions, allowed channel count and pixel type,
configured pixel/byte limits, exact payload lengths, requested output slots and
job identity. It also rejects duplicate JSON keys, unexpected messages, truncated
streams and excess trailing data. Metadata strings and collections have explicit
limits. Never allocate from an unbounded worker-supplied length.

Copy accepted output into receiver-owned memory before use; do not leave mutable
worker-shared arrays backing inference. Transfer digests detect corruption, not
maliciously fabricated pixels. A compromised parser can return plausible but
incorrect pixels; neither its digest nor schema validation establishes scientific
correctness. The trusted side assigns provenance and storage destinations.

## Image and face semantics

Move encoded-byte decoding behind a small client interface, preserving existing
public call behavior where safe. Keep CPU decode helpers independent of imports
that initialize Torch, GPU runtimes, model loaders or application settings.
This is a targeted extraction, not an embedder or face-pipeline rewrite.

One request can return embedding RGB, native face RGB and EXIF metadata from the
same source. Preserve these existing distinctions:

- Similarity preprocessing retains JPEG `draft()` before pixel decoding, the
  current EXIF orientation format allowlist, RGB conversion, ignored ICC profiles
  and exact short-side Lanczos cap/rounding.
- Face inputs remain full-resolution, EXIF-oriented RGB8. Do not apply JPEG draft
  or the embedding cap to native review/detection pixels.
- Shared decoding may derive embedding pixels only where it is currently
  identical. JPEG and non-allowlisted-format orientation exceptions keep their
  separate embedding decode, inside the same source container.
- Preserve current EXIF presence flags and display metadata on valid inputs.
  Corrupt optional EXIF can retain the existing best-effort result with an
  explicit diagnostic. A transport or isolation failure must never masquerade
  as “no EXIF.”
- Preserve decompression-bomb handling and operator pixel limits; new transport
  limits must produce reported failures, never silent downsampling.

The call-site inventory must include `embedder.py`, `faces/decode.py`,
`faces/indexing.py`, ingestion orchestration, uploads, preview/metadata routes and
stored-chip readers such as `faces/chips.py`. Reopening a derived encoded artifact
is still parsing and must be isolated or eliminated by retaining validated raw
pixels. Parser-produced encoded artifacts must not be reopened in the trusted
application to validate them. Original downloads remain byte-serving operations;
the browser's own media parsing is a separate workstation exposure.

Detection gates, review-only exclusions, face tokens, model identities, batch
ordering/padding, preprocessing compatibility checks and database ownership stay
unchanged. The new execution-environment record identifies parser image and
protocol without pretending that unchanged preprocessing has a new pixel
meaning. Any actual pixel-semantic change requires an explicit compatibility
decision rather than updating expected test hashes.

## Failures, cancellation and observability

The isolated deployment profile refuses required parsing if its launcher,
policy or protocol is unavailable or mismatched. No automatic in-process
fallback. Startup checks verify actual restrictions with a disposable worker;
configuration labels alone are insufficient. A legacy local development mode
can remain explicitly identified as unisolated, but is not security acceptance.

Malformed output, crash, timeout and resource exhaustion invalidate the worker
connection and require replacement before reuse. Cancellation discards the exact
job's incomplete results and invalidates its connection; the next request cannot
consume late data. Clean only runtime-owned scratch, never original evidence.
The service manager reclaims orphaned workers. Admission queues are bounded;
overload is visible rather than allowing unbounded memory growth.

One failed source need not abort unrelated ingestion, but it is counted and
reported as failed, not successfully processed with missing vectors or metadata.
Do not publish partial source results as complete. Record source identity,
operation, policy/image/protocol identity, outcome and bounded diagnostics in
durable application-owned records, outside disposable worker storage. Do not log
raw evidence, full EXIF contents or secrets as parser diagnostics.

The trusted application records a runtime-assigned generation identifier and
request sequence with each source and derived result. When compromise is
suspected, quarantine that generation's derived results; where the first affected
request is unknown, include the whole generation. Invalidation must preserve
originals and audit history. A corrupt-file error alone is not proof of malware.
Recovery uses a clean worker and a patched decoder or exclusion of the triggering
input as appropriate, not a blind automatic retry of every suspicious source.

## Video stage constraints

### Approved NVENC boundary

The renderer retains the existing FFmpeg invocation's source parsing, software
decode/filtering, NVENC encoding and MP4 muxing in the same networkless sandbox.
Its GPU access is granted for encoding, but this is **not** an enforcement claim
that a compromised parser can only invoke encoding: the process can attack the
exposed GPU-driver interface. The operator accepts that residual risk for this
internal NVIDIA deployment. Do not describe the renderer as GPU-free or add
hardware decoding as part of this change.

Give the renderer explicitly scoped read-only evidence access and private,
bounded scratch. It must not receive database credentials, model mounts, the
Docker socket, writable original evidence or durable case-output directories.
The trusted application retains cache ownership, final publication and audit
records. Granting NVENC does not authorize network access, privileged mode or
disabling syscall restrictions. CPU-only image workers stay unchanged.

Preserve the existing FFmpeg pipeline rather than introducing a raw video/audio
handoff. Keep filters, encoder settings, seeking, timestamps, audio and existing
fallback disclosures intact. Record the separate execution policy and immutable
worker image identity. Capability probes must describe the renderer environment,
not accidentally certify a different host FFmpeg installation. CPU fallback, if
needed, stays inside the isolated boundary; isolation failure is never permission
to parse evidence in the trusted process.

### Transport and behavior requirements

Source access for probe, sampled extraction, seek, bitstream-copy rewrap and
full/chunk transcode uses the explicit read-only mappings described above.
Preserve frame selection/hashes and the CLI's intermediate JPEG quality-85
optimized encode/decode behavior. Preserve orientation, VFR timestamps, time
bases, color/HDR interpretation and audio timing across CPU decode and NVENC.
The fixed-rate raw-video prototype is not an acceptable replacement.

Rewrap remains bitstream copy, not an undisclosed encode. Any changed rendering
pipeline receives the corresponding fingerprint and reproducible rendering
record. Do not change the current chunk default as part of image isolation.
Short initial/seek windows followed by longer prefetched windows require their
own cache/timeline design and browser verification.

## Acceptance and rollout

Acceptance requires all of the following for the implemented stage:

1. Differential tests against the pre-isolation paths for JPEG/HEIC/PNG/WebP,
   EXIF orientation edge cases, corrupt files, large dimensions and faces-on/off.
   Compare raw pixels, metadata, ordering and actual pinned-environment model
   outputs; retain native face-review geometry and gate outcomes. Do not claim
   universal cross-GPU determinism from one NVIDIA test environment.
2. Hostile-protocol tests for oversized lengths, overflow, malformed metadata,
   swapped jobs, truncation, trailing data, crashes, cancellation and late replies.
3. Real container denial tests demonstrating no GPU outside the approved NVENC
   renderer exception, and no host/LAN network,
   credentials, original writes/deletions, durable case writes, runtime-control
   access or other-worker access. Verify deliberate process reuse and that
   replacing a generation discards private writable state. Include cleanup,
   admission limits and generation-level result invalidation.
4. Regression tests for ingestion, query/preview, face processing and frontend
   wiring, plus lint/format checks. A code inventory establishes that the claimed
   stage has no remaining trusted-process encoded-evidence decode path.
5. A one-to-five-minute representative end-to-end comparison using read-only
   selected evidence and disposable database/artifact storage, both baseline and
   isolated arms under equal aggregate CPU/GPU/memory budgets. Include faces and
   analyst activity, snapshot/transfer/startup/cleanup costs, and report model
   warm-up separately. Use reversed run order where the time budget permits.
   The original target is at most 1.20 times baseline elapsed time. The operator
   accepted the measured 1.2012 image result as close enough; this is not a general
   waiver for larger regressions or a video performance result. Compare output
   identities and report failures, memory/load and sample limitations.
   Multi-TB endurance is not required for this rough operational assessment.

For this internal forensic-inspection deployment, the operator accepts several
seconds for cold opening and seeking. The earlier provisional three-second
first-frame/two-second seek targets are not release blockers for this delivery;
no further latency optimization is required here. Record measured delays rather
than promising instant playback. Continued playback and transitions were checked
with the unchanged 30-second chunks; muted-browser tests do not establish
subjective audio sync. The earlier five-second chunk experiment is not adopted.

Build and test locally first. Do not change the operator's running deployment,
live database, local Compose override or main branch during implementation.
Review measured results before enabling the isolated deployment profile. No
remote push or release/security clearance follows from this spec or its tests.
