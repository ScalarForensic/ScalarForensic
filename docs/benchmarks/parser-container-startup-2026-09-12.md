# Single-use parser container startup cost — 2026-09-12

## Decision

Do not promote the measured `docker run`-per-source runner into the ingestion
pipeline. Its best measured rate was 19.35 jobs/second before source transfer,
decoding, pixel transfer or inference. The previously measured image/inference
stage would need 21.91 images/second to remain within its 20% elapsed-time budget.

This is an early implementation-feasibility warning, not a full-CLI performance
failure or proof that every single-use-container implementation is too slow.
The application and production deployment were not changed.

## Method

The local pinned application image was
`sha256:8d5b7ed41a82486f93cc3f80b4b9ff0397c064232fb3ed53bc9e057d8cd8877b`.
Each job created a new container, started Python with bytecode writes disabled,
imported `PIL.Image` and `pillow_heif`, printed `ready`, exited and was removed.
There were no evidence files, host-directory mounts, GPU grants or database
connections. The image was already local; pulling was disabled. This application
image is a timing fixture, not the proposed minimal production parser image.

Container policy used read-only root, UID/GID 10001, no network, all capabilities
dropped, no-new-privileges, Docker's default syscall policy, a 64-process limit,
256 MiB memory with no additional swap allowance, and no logging driver.
Containers shared CPU cores 0–7; their CPU quotas summed to eight cores at each
concurrency. No privileged or unconfined workaround was used.

For each arm, a host thread pool submitted new jobs for 20 seconds, then drained
the outstanding jobs. Elapsed time includes the drain and Docker CLI calls.
Arms ran in order 1, 4, 8, then 16 and 32. No prestarted reserve was used.

The host Docker daemon, runtime and CLI were not confined to those eight cores;
this is not an equal-total-host-resource comparison with the previous benchmark.
Aggregate allowed container memory also grows with concurrency, reaching 8 GiB
at 32 containers. No representative working-set memory or GPU contention was
measured. The workstation was not exclusive, and there was only one arm per
concurrency, so these are observations rather than stable throughput limits.

## Results

| Concurrent containers | Completed jobs | Elapsed including drain (s) | Jobs/s |
| --- | --- | --- | --- |
| 1 | 85 | 20.047 | 4.24 |
| 4 | 281 | 20.227 | 13.89 |
| 8 | 358 | 20.275 | 17.66 |
| 16 | 373 | 20.398 | 18.29 |
| 32 | 408 | 21.090 | 19.35 |

All 1,505 jobs succeeded. The five timing windows totalled 102.04 seconds.

The [earlier corrected image/inference baseline](media-isolation-2026-09-12.md)
processed 763 images in 29.769 and 28.276 seconds, averaging 29.022 seconds.
Its 20% allowance is 34.827 seconds, or 21.908 images/second. At this startup
test's best rate, 763 jobs would take 39.441 seconds, approximately 36% longer
than that image/inference baseline, without doing its actual work.

That calculation is an extrapolation across different workloads, not measured
end-to-end ingestion overhead. It excludes scanning, hashing, database writes,
faces and any opportunity to overlap runtime work with those stages. A direct
runtime API, minimal image, different resource allocation or clean prestarted
workers could change results. A prestarted reserve must also be replenished at
sustained throughput; measuring only its initial drain would hide startup cost.

## Consequence for implementation

The approved boundary still requires a fresh source lifetime and no writable
state shared across exposed workers. Do not replace it with reused decoders or
ordinary unrestricted subprocesses to meet performance targets.

Before implementing the application integration, investigate a cheaper runtime
mechanism while retaining the denial tests and fail-closed requirements. Options
include removing Docker CLI/lifecycle overhead or a separately reviewed process
sandbox with an unexposed parent and enforced cross-job restrictions. Neither
alternative has been demonstrated here. Any change to the approved boundary
needs an explicit design decision, not a hidden optimization.

## Evidence and cleanup

The throwaway harness and complete per-job timings are retained locally in the
ignored `.cleanup-notes/parser-startup-gate.py`, `parser-startup-first.json` and
`parser-startup-more.json`. Commands were:

```sh
python3 .cleanup-notes/parser-startup-gate.py
python3 .cleanup-notes/parser-startup-gate.py 16 32
```

Container name prefixes were `sfn-parser-startup-e043228a92` and
`sfn-parser-startup-074ee75955`. A final all-container listing filtered by the
benchmark prefix returned no containers. The only running container remained
`scalarforensic-qdrant-1`, up four days. No source evidence or case storage was
read or changed. Main, the local Compose override and the live deployment were
unchanged; nothing was pushed.
