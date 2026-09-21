# Isolated video boundary validation — 2026-09-12

Scope: local internal-NVIDIA implementation in `chore/dead-weight`, working tree
based on `d137392`. Results below were collected with implementation changes
present, **not** from the unchanged baseline commit. No running deployment,
original evidence, main branch or remote was modified. This is a short operational
sample, not multi-TB endurance, release clearance or universal GPU determinism.

Host: AMD Ryzen 9 9900X3D (12 cores/24 logical CPUs), NVIDIA GeForce RTX 4060 Ti
(16,380 MiB reported), driver 595.84. Host client Python 3.12.3; Node v24.14.0.

## Warm rendering differential

Three previously selected, read-only clips covered a 318.5-second HEVC 10-bit HLG
source, an 88.5-second portrait SDR source and a 60.5-second tall SDR source. Each
was rendered at the beginning and midpoint, using unchanged 30-second windows,
1080 output height, audio, software decode/filtering and NVENC. Each pair ran twice
with reversed baseline/worker ordering: **12 pairs / 24 encodes**.

The direct baseline executed the existing `encode()` inside the same worker image
and four-CPU/4-GiB renderer budget, timing its encode operation. The isolated arm
timed the actual application `VideoClient` call, including admission, private
state/receipt I/O, RPC, output transfer, validation and cache publication. Native
imports/capability probing and source hashing were warmed separately. Both arms
used the existing progress watcher and identical pixel/audio settings. Neither arm
used hardware decoding. Output/scratch limits were 1 GiB/2 GiB for this sample.

| Source | Direct mean | Worker mean |
| --- | ---: | ---: |
| HLG landscape, `IMG_1983.MOV` | 6.929 s | 6.938 s |
| Portrait SDR, `IMG_2152.MP4` | 3.142 s | 3.150 s |
| Tall SDR, `IMG_3259.MP4` | 4.140 s | 4.168 s |
| All 12 operations, summed | 56.846 s | 57.023 s |

Measured recurring worker overhead: **0.31%**, approximately **14.7 ms per
operation** on this sample. Every pair produced **identical MP4 SHA-256**, and
repeated positions reproduced their previous hashes. All encodes used NVENC;
none fell back. Original source hashes were unchanged after the run and the
disposable renderer was removed.

This comparison isolates warm request/transport/publication overhead; it is **not**
a new end-to-end ingestion result or a comparison with an unrestricted host
deployment. It excludes one-time worker startup, cold capability/model loading and
uncached source hashing. Cached playback avoids another whole-file read per chunk.
It does not establish performance under simultaneous inference/full-export load.
The previously accepted 20.12% image-ingestion result remains a separate measure.

The baseline itself takes 3–7 seconds to generate these 30-second chunks. Thus the
provisional three-second first-frame/two-second uncached-seek targets are not met
for all clips merely by preserving NVENC. No shorter-window timeline/cache redesign
or changed chunk default was introduced in this isolation change. After reviewing
the results, the operator accepted several-second opens/seeks for forensic file
inspection and requested completion without further cold-latency optimization.

The offline overlay was built with:

```sh
docker build --pull=false --network=none -f Dockerfile.video-worker \
  --build-arg BASE_IMAGE=scalarforensic:internal-nvidia-20260912-patched \
  -t scalarforensic:video-worker-20260912 .
```

Base image: `sha256:8d5b7ed41a82486f93cc3f80b4b9ff0397c064232fb3ed53bc9e057d8cd8877b`.
Tested overlay tag at this run:
`sha256:8c565d5dc1060730f1277d0826b451a1523d6709ff11047a927f6a169096815b`.
The host client used the current worktree source. Later client-only validation
changes do not retroactively change the identity of this measured image.

Local retained measurement inputs and results:

- Selection/hashes/probe metadata: `/tmp/sfn-corpus-review-NNdZ91/selection.json`.
- Driver: ignored `.cleanup-notes/benchmark-video-boundary.py`.
- Results, application receipts and generation history:
  `/tmp/sfn-video-perf-tq2f7bp7/` (`results.json` contains each pair).

The measured command used the shared project venv with `PYTHONPATH=src`:
`python .cleanup-notes/benchmark-video-boundary.py`. Repeating requires the same
read-only local corpus, image, driver environment and explicit disposable paths;
these local artifacts are not committed case material.

## Real container checks

### CPU video extraction differential

The same three clips were sampled at one frame/second, capped at 20 frames per
operation, in a CPU-only four-CPU/4-GiB worker. Six pairs used reversed order and
warm source hashes/imports. The direct arm timed existing PyAV extraction inside
the same image; the managed arm included raw-frame transport, application-side
hash validation and receipts. All **120 frame hashes, indices and timecodes per
arm matched exactly**. Original hashes stayed unchanged and the CPU worker was
removed.

| Source | Direct mean | Worker mean |
| --- | ---: | ---: |
| HLG landscape | 1.438 s | 1.428 s |
| Portrait SDR | 1.207 s | 1.244 s |
| Tall SDR | 2.767 s | 2.892 s |
| All six operations, summed | 10.825 s | 11.128 s |

Measured decode/transport overhead was **2.80%**. This is a short extraction-stage
measurement, not full ingestion with face/model/database work. It does not include
uncached initial source hashing, worker startup or simultaneous analyst/inference
load. Driver: `.cleanup-notes/benchmark-video-frames.py`; retained per-frame hashes,
results and receipts: `/tmp/sfn-video-frames-9ukt23np/`.

### Access restrictions and output parity

A three-second synthetic portrait video with AAC audio exercised CPU extraction,
probe/rewrap and repeated actual NVENC encoding. CPU raw pixel bytes, hashes and
timecodes matched the existing local PyAV path exactly. Artifact byte counts and
SHA-256 matched the received stream. The original stayed unchanged.

Inside both CPU and renderer containers, attempted original overwrite/unlink,
root writes, IPC file creation, IPC unsealing/socket replacement and execution
from scratch were denied. The interface list contained loopback only; no Docker
socket or application `.env` was mounted. Scratch quota was checked. NVIDIA device
nodes were absent from CPU workers and present only in renderers. IPC tests used
the actual sealed directory and distinct worker UID, not just mocked Docker flags.
These tests verify configured restrictions, not resistance to unknown kernel or
GPU-driver exploits; no malicious media exploit was deliberately executed.

Driver: ignored `.cleanup-notes/smoke-video-workers.py`. Retained successful denial
run: `/tmp/sfn-video-806gghj6/`. Its two 1.5-second NVENC renders took 0.258/0.254 s;
the first CPU extraction took 0.244 s including import initialization. All exact
owned containers were removed by supervisor context cleanup.

The independent host watchdog was also exercised against an actually paused CPU
container, with an expired private active-request record and the admission lock
still held. No client failure notice was sent. The supervisor removed the exact
old container (confirmed absent via Docker inspect), started a different
generation and discarded a marker placed in old private scratch. Replacement
took 0.307 s; the replacement was removed on exit. Driver:
`.cleanup-notes/check-video-deadline.py`; result/history:
`/tmp/sfn-video-deadline-ofktsl1z/`.

## Regression checks

From the modified worktree, with the shared venv and explicit source override:

```sh
PYTHONPATH=src /home/user01/Schreibtisch/gitea/ScalarForensic/.venv/bin/python \
  -m pytest -p no:cacheprovider -q
/home/user01/Schreibtisch/gitea/ScalarForensic/.venv/bin/ruff check src tests scripts
/home/user01/Schreibtisch/gitea/ScalarForensic/.venv/bin/ruff format --check src tests scripts
npm test
npm run test:browser
```

Initial whole-feature run: **1,430 passed, 9 skipped**, **82.88% coverage** in
65.74 seconds. The 47 warnings were existing Starlette/httpx, local-Qdrant,
TorchScript and multithreaded-fork warnings. Ruff check and formatting passed
(170 Python files). Node's installed runner reported **12 test files passed**;
Playwright reported **2 passed** using its synthetic API/local-video smoke setup.
Those browser smoke tests alone do not establish real isolated playback behavior.

Independent complete source review found no blocking defects; a separate
non-socket boundary/routing subset independently passed 56 tests. Real cancellation
subsequently exposed an exception-label issue (correct removal but availability
error instead of typed cancellation); final verification below records its fix.

## Final verification and operator acceptance

After fixing typed cancellation, durable receipt appends and source-switch
admission/capability caching, the complete suite passed **1,443 tests, 9 skipped**,
with **83.05% coverage**, in 70.63 seconds. All nine skips require an explicitly
configured disposable `SFN_TEST_QDRANT_URL`; no live case database was used.
The same 47 warning categories remained. Ruff check, formatting and
`git diff --check` passed. No frontend source changed after the successful Node
and two Playwright regression smoke tests above. Independent final code and
documentation re-review found no blockers.

The final real-browser run used current application source/routes/static UI and
three isolated workers, with fixture search results but real video APIs. It
passed all three clips, including immediate source switching while old-source
prefetch was still finishing:

| Source | Open to presented frame | Seek to presented frame |
| --- | ---: | ---: |
| HLG landscape | 8.73 s | 11.35 s |
| Portrait SDR, queued source switch | 11.50 s | 3.73 s |
| Tall SDR | 8.02 s | 4.92 s |

These are single-run usability observations, not controlled throughput estimates.
The first clip played for 35 seconds across the 0→30-second chunk boundary; the
longest sampled interval without time advancing was approximately 0.2 seconds.
There were no JavaScript errors or HTTP 4xx/5xx responses. Eleven generated
chunks had H.264/AAC streams, monotonic per-stream DTS and A/V stream-start offsets
at most 33 ms. Chromium was muted: this does not establish subjective audio sync
or audible transition quality.

Actual cancellation returned `Cancelled` only after full-worker generation
replacement; no artifact was published. The separate read-only upload mapping
successfully probed a staged query video. This is not a full multipart
upload/model-indexing integration test. Original hashes remained unchanged and
all disposable workers/server processes were cleaned up. Final artifacts:
`/tmp/sfn-video-browser-qezxb2ks/`; drivers remain in ignored `.cleanup-notes/`.

The operator accepted the observed several-second inspection delays and requested
completion, not cinema-style playback optimization. This closes implementation
and local verification for the approved boundary. Enabling it in a running
deployment is a separate explicit operator action; no main-branch integration,
remote push, redistribution/licensing clearance or AMD validation is implied.
