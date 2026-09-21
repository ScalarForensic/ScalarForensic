# Reused image worker validation — 2026-09-12

The implemented image worker passed a bounded real-container isolation/reuse
check on this NVIDIA workstation. This is not whole-media security acceptance:
video parsing remains outside the new boundary. Host-managed image-worker
lifecycle validation is recorded below. See the
[implementation and limits](../specs/media-parser-isolation.md).

## Runtime checks

Two disposable generations used the locally available image ID
`sha256:8d5b7ed41a82486f93cc3f80b4b9ff0397c064232fb3ed53bc9e057d8cd8877b`,
with the current worktree source mounted read-only. No image was pulled. This
image contains GPU libraries, but neither worker had GPU device access.

The actual worker CLI ran as UID/GID 1000:1000, with network `none`, read-only
root, all capabilities dropped, no-new-privileges, default seccomp filtering,
256 PIDs, 1 GiB memory without additional swap, two CPUs, and a 64 MiB
noexec/nosuid/nodev scratch tmpfs. Mounts were limited to source code, dedicated
IPC and read-only synthetic evidence. Real evidence and live services were not
used for write-denial tests.

| Check | Result |
| --- | --- |
| Reuse | Two exact-pixel RPCs per generation; socket peer PID matched the inspected worker PID |
| Replacement | Different PID and generation; previous private scratch did not survive |
| Host network | IPv4/IPv6 host loopback canaries refused; host LAN canaries unreachable |
| Raw sockets | IPv4, IPv6 and packet sockets denied with EPERM |
| GPU, Docker socket, test credentials | Unavailable inside the worker |
| Synthetic originals | Write, truncate, unlink and rename each denied with EROFS; hashes unchanged |
| Root filesystem and scratch | Root write denied; execution from scratch denied |

All four canaries were first verified reachable from the host; none received a
worker connection. Denial probes ran with `docker exec` under the same configured
identity and restrictions. They do not simulate a successful parser exploit or
prove resistance to kernel vulnerabilities. Worker-namespace loopback still
exists; network isolation does not prohibit every Internet-socket syscall.

All disposable containers were identified by their unique ownership labels and
removed by inspected container ID. Their synthetic inputs and scratch were also
removed. The running Qdrant service and deployment were untouched.

## Evidence and limits

Implementation checkpoint: `d1f4f0e`. The Python suite passed 1,316 tests with
9 skips (81.77% coverage). All 12 JavaScript test files and both real-browser
smoke tests passed. Ruff checks and
format validation passed across 147 Python source/test files. After simplifying
compatibility imports without changing their meaning, all 42 focused decode and
preprocessing tests passed again.

The private reproduction harness and raw JSON are retained in
`.cleanup-notes/validate-reused-worker-container.*` (run `7e9504ff59fb`). These are
local test artifacts, not committed agent-coordination documents.

The runtime check does not validate cross-worker IPC, external native-code deadlines,
automatic result quarantine, or the performance budget. It establishes the
listed denials and deliberate process reuse on the tested host configuration.
The earlier per-file Docker startup measurement is not a recurring cost of this
reused design.

## Warm-model ingestion comparison

The private benchmark invokes the actual `cli.index` path with DINOv2, SSCD,
faces, EXIF, a fresh on-disk local Qdrant store, and thumbnail/face-chip writes.
Only model construction and the Qdrant connection factories are substituted:
models are real and prewarmed, and the database is disposable rather than the
operator's live server. Three additional preview/metadata pairs are timed after
ingestion; this is not a concurrent analyst-load test.

Both modes use an NVIDIA GeForce RTX 4060 Ti, batch size 8, DINO size 224,
SSCD one-crop mode and the same pinned models. The sample is selected from the
existing read-only corpus, stratified
by format. Run order is baseline, worker, worker, baseline. Models, input
verification/page-cache warming, initial worker startup and final teardown are
outside the timed region. This measures warm reused-worker throughput, not
recycling cost, cold-disk throughput or the operator's full production profile.

The initial allocation capped the application at eight CPUs for baseline, then
four CPUs plus four worker CPUs for isolated mode, all within CPU set 0–7.
Aggregate memory was 20 GiB in either mode. These results are retained, including
the misses against the 20% elapsed-time budget:

| Sample | Baseline seconds | Worker seconds | Combined worker/baseline ratio |
| --- | --- | --- | --- |
| 64 images | 10.539 / 9.147 | 12.055 / 11.876 | 1.2156: **21.56% slower** |
| 128 images | 20.139 / 17.927 | 22.486 / 23.241 | 1.2012: **20.12% slower** |

The 128-image sample contains 42 HEIC, 42 JPEG, 42 PNG and two WebP images.
All 128 were indexed in every arm. Each isolated arm recorded 408 submitted and
408 decoded requests, with no worker failures. Stored vectors and all 380 image
artifacts matched exactly across all four arms. The remaining payload difference
is `faces_processed_at`; the separate face collection metadata differs only in
`enablement.enabled_at`. Equality of the otherwise-complete stored records was
checked after excluding those timestamps (and the harness's documented
`indexed_at` fields). Preview return values were
timed but are not included in the stored-output equality check.

The first 128-image comparison took 83.79 seconds of measured work; model
loading/warm-up took 10.59 seconds separately, and worker readiness was about
0.63 seconds per generation. The baseline varied by roughly 11%, so this is a
rough operational check, not a precise sustained-regression confidence bound.
The unrounded 1.2012 ratio is above the budget and is not counted as a pass.

Raw completed-run evidence is retained privately under
`/tmp/sfn-current-ingest-9bwo1nh3` (64 images) and
`/tmp/sfn-current-ingest-44f6oovz` (128 images). The readable reproduction scripts
live in `.cleanup-notes/benchmark-reused-worker-*.py`; they are test harnesses,
not production deployment management.

### One scheduling alternative — rejected

A final run kept both containers confined to the same eight CPUs (CPU set 0–7),
but let the application use an eight-CPU quota in both modes, with the worker
still capped at four. The summed quotas were 12; the shared CPU set still limited
their combined execution to eight logical CPUs. Torch used eight threads in
both modes. Aggregate memory remained 20 GiB.

This was **worse**, not an improvement: baseline 19.598 / 18.305 seconds versus
worker 24.686 / 24.321 seconds, ratio **1.2930, or 29.30% slower**. The 86.91-second
measured run again indexed all 128 inputs per arm, with exact stored-vector and
380-artifact matches, timestamp-only record differences, and no worker errors.
Model warm-up was 11.07 seconds separately; worker readiness was 0.628 / 0.621
seconds. This alternative is not adopted as a deployment recommendation.

The rigid-allocation scripts are preserved as
`.cleanup-notes/benchmark-rigid-worker-{app,run}.py`; the scheduling variant is
the current `benchmark-reused-worker-{app,run}.py`. Its raw evidence is in
`/tmp/sfn-current-ingest-rb1dqog2/benchmark.json`. Across all three completed
comparisons, measured work totaled about 214 seconds; startup/model warm-up was
separate. All disposable benchmark containers were removed.

### Decision

The original strict 20% gate was missed. The operator subsequently accepted
**20.12%** as sufficiently close to the target for this internal NVIDIA deployment.
This accepts the rough four/four allocation measurement, not a sustained
14-week performance guarantee. The rejected 29.30% scheduling variant remains
rejected. No further image-performance machinery is planned. The image worker
remains opt-in; acceptance does not itself enable or change a running deployment.

## Managed lifecycle verification

The subsequent host supervisor changes were checked in the dirty worktree based
on `2999c9e`, before their own commit. A separate worker image was built offline
from the same local base using `Dockerfile.media-worker`; no deployment image
was replaced. The image's immutable identity is recorded in each generation's
journal and application receipts.

A real supervisor CLI process decoded a synthetic PNG, rotated after 60 seconds,
and returned identical pixels from its replacement. The replacement container
was deliberately paused: a one-second test client deadline produced a failure
notice, the independent supervisor removed that container, and a third generation
decoded identical pixels. SIGTERM removed publication and every test-owned
container. Docker inspection confirmed network none, read-only root, no GPU
requests and only the generation's IPC bind mount. This checks timeout/lifecycle
behavior, not detection of a real exploit or concurrent analyst throughput.
Private logs remain under `/tmp/sfn-managed-pmqvu2lu`; the synthetic reproduction
script is `.cleanup-notes/smoke-managed-worker.py`.

Verification from the worktree, with its src overriding the shared editable install:

```sh
PYTHONPATH=$PWD/src /home/user01/Schreibtisch/gitea/ScalarForensic/.venv/bin/python -m pytest -p no:cacheprovider -q
/home/user01/Schreibtisch/gitea/ScalarForensic/.venv/bin/ruff check src tests scripts
/home/user01/Schreibtisch/gitea/ScalarForensic/.venv/bin/ruff format --check src tests scripts
npm test
npm run test:browser
```

Results: 1,345 Python tests passed, nine skipped, coverage 82.64%; 165 JavaScript
tests and two synthetic browser smoke tests passed. Ruff and whitespace checks
passed. Existing Starlette/httpx and test-runtime warnings were not suppressed;
browser output also warned about conflicting color environment settings. These
checks do not validate a live managed Compose deployment or isolated video.
