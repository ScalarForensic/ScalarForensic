# Media isolation performance prototype — 2026-09-12

## Outcome

CPU-only decoding with reused workers remains a viable performance direction.
The measured media/inference stages did not slow down and retained exact output
identities. This is **not full ingestion acceptance or a finished security
pipeline**. Production code, configuration and deployment were not changed.

Keep NVIDIA inference and NVENC encoding. CPU-only transcoding was substantially
slower in the representative render tests. The current 30-second chunk setting
missed the provisional browser latency targets. A five-second setting met those
targets in the tested cases, including under media/inference load, but increased
the frequency and cumulative cost of chunk transitions. Do not silently change
the production default on this evidence alone.

## Requirements and scope

- Offline internal NVIDIA deployment, trusted operators, adversarial evidence.
- Forensic processing must be reproducible; isolation must not silently change
  image preprocessing, frame sampling, timestamps or inference semantics.
- Ingestion acceptance means no more than 20% additional elapsed time for the
  same workload, not merely 80% of baseline throughput.
- Subsequent operator decision: a one-to-five-minute representative run is enough
  for a rough load assessment; large-corpus endurance is not a prerequisite for
  this cleanup or local integration. The corrected stage runs total 235 seconds
  and satisfy that exploratory scope. They do not become a full-CLI guarantee.
- Provisional browser targets: first presented frame within three seconds;
  seek response within two seconds; usable continued playback during ingestion.
- Redistribution, AMD validation and unresolved parser patch coverage remain
  separate. This experiment grants no security exception for vulnerable parsers.

## Environment and corpus

RTX 4060 Ti, driver 595.84. Candidate application image:
`sha256:8d5b7ed41a82486f93cc3f80b4b9ff0397c064232fb3ed53bc9e057d8cd8877b`.
It contains application source from `46b4300`; current worktree HEAD was
`f2694a4`. No source overlay or model/dependency update was used.

The existing deterministic 10% subset contains 763 images and 51 videos,
2,037,602,649 bytes. Image coverage: 310 HEIC, 218 JPEG, 233 PNG and two WebP.
Video coverage includes ten HEVC/10-bit/HLG clips. A separate 24-video selection
covers all nine observed codec/pixel-format/transfer groups, taking shortest,
median and longest examples within each group. Its longest clip is 318.51 s.

The exact selection and SHA-256 identities are retained locally in
`/tmp/sfn-corpus-review-NNdZ91/selection.json`; its SHA-256 is
`26a4a8e9ac3e2a45b7f0c8b18ffc53bac12e126c6c77dc0f54cf41f48e17f114`.
All selected input hashes were rechecked after the experiments and matched.
Sources were mounted read-only. No live database, case outputs or main-checkout
files were modified. Source caches were warm; no host cache dropping was done.

## Media and inference comparison

Both arms used the same eight CPU cores, image preprocessing functions, compiled
DINOv2 and SSCD models on NVIDIA, batch size eight and preprocessing cap 331.
DINO normalization was 224. Model warm-up was excluded; partial batches were
padded identically and padding outputs discarded. One reusable CPU container
served bounded raw RGB plus JSON metadata over a Unix socket. It had no NVIDIA
devices or network access. The receiving process checked dimensions, lengths
and transfer hashes before reconstructing RGB images without encoded-file parsing.

For videos, the harness retained the CLI's intermediate JPEG encoding at quality
85 with optimization before preprocessing. It used the existing extraction
functions at 1 fps, maximum 500 frames per video. The 51 videos yielded 457
sampled frames. Source, JPEG and frame identities/timecodes were compared along
with processed pixels and float32 vector bytes from both models.

An initial sequential baseline unfairly gave the worker an overlap advantage.
It was retained as diagnostic evidence but excluded from the comparison below.
The corrected baseline prefetches in a thread, so both arms overlap preprocessing
and inference. Run order was baseline/worker, then worker/baseline.

| Stage | Baseline runs (s) | Worker runs (s) | Median worker/baseline |
| --- | --- | --- | --- |
| 763 images, both models | 29.769, 28.276 | 29.062, 27.655 | 0.977 |
| 457 video frames, both models | 31.295, 30.139 | 29.379, 29.550 | 0.959 |

All processed-pixel/metadata and vector digests matched across both arms, both
repetitions and the earlier sequential experiment. These small differences are
not a claim that isolation makes production ingestion faster; two repetitions
on a non-exclusive workstation do not establish a precise overhead distribution.

This harness excludes CLI scanning/deduplication orchestration, database writes,
frame-store writes, thumbnails, EXIF reporting and face processing. The production
CLI also stages video extraction differently. The test establishes feasibility
of the transfer boundary, not the full CLI's 20% budget. The prototype still
mounts the corpus in the GPU container for the baseline and gives its worker the
whole approved subset read-only: final per-job access control, worker lifecycle,
hostile IPC testing and fail-closed deployment have not been implemented.

## Existing render command

The application's `build_command` was used on the 24 selected videos at their
starts and, for clips over 30 seconds, their midpoints. Each window was at most
30 seconds. Audio and HDR tone mapping were retained. Both encoders used the
same eight-core container allowance and four-thread option. All 58 renders
succeeded. These are render-command timings, not browser latency; some sources
would normally use direct playback/remux instead of forced rendering.

| Encoder | Windows | Median (s) | Nearest-rank p95 (s) | Maximum (s) |
| --- | --- | --- | --- | --- |
| NVENC | 29 | 0.629 | 3.910 | 4.227 |
| libx264 medium | 29 | 1.048 | 7.115 | 9.259 |

Temporary rendered files were removed after recording output hashes and probe
metadata. They are reproducible viewing copies, not deleted source evidence.

## Browser checks

Sandbox-enabled headless Chromium used the real static UI and real playback
routes in a disposable playback-only service. Query hits were fixtures; video
responses were not mocked. No Qdrant or model inference was required by that
service. The browser reached it through its private internal Docker network,
not through a live service or general-purpose external network.

Opening was timed from the UI open action through the first presented video
frame, including the immediate second action to start transcoding. Seeking was
timed through presentation at the requested position. Playback was muted for
automation; audio sync/quality was not assessed. Each service run started with
an empty viewing-copy/hash cache, but source filesystem caches remained warm.
Measurements are one observation per case/condition, not tail-latency guarantees.

| HEVC sample | 30 s chunks: open / seek | 5 s chunks: open / seek | 5 s chunks with load: open / seek |
| --- | --- | --- | --- |
| 318.51 s, 10-bit HLG, 1920×1080 | 5.08 / 4.50 s | 1.90 / 1.23 s | 2.23 / 1.41 s |
| 88.55 s, SDR, 1080×1920 | 2.70 / 2.38 s | 1.03 / 0.88 s | 1.11 / 1.03 s |
| 60.53 s, SDR, 1320×2868 | 4.26 / 2.35 s | 1.04 / 0.95 s | 1.16 / 1.18 s |

The loaded run shared the same eight CPU cores and GPU with an active loop of
representative preprocessing plus DINOv2/SSCD inference. It did not run a full
ingestion job or measure a paired full-ingestion slowdown under playback load.

The HLG clip played for 35 wall-clock seconds in each condition:

- 30-second windows: source position reached 34.77 s; one transition.
- Five-second windows, idle: source position reached 33.54 s; six transitions.
- Five-second windows, loaded: source position reached 33.54 s; six transitions.

There were no page errors or prefetch failures. Sampling at 100 ms observed
individual non-advancing intervals up to approximately 0.2 s. The shorter windows
accumulate about 1.5 seconds of delay across this run, compared with about 0.23 s
for the longer windows. This is an explicit usability tradeoff, not seamless
playback. Shorter windows also change the pipeline fingerprint and can change
lossy encoded bytes; they are not byte-identical replacements for old renderings.

## Remaining acceptance work

1. Implement and test the actual fail-closed parser boundary, covering images,
   EXIF, faces and all video paths. Passing raw pixels is not proof that an
   exploited decoder produced scientifically correct pixels.
2. Preserve VFR timestamps, orientation, color interpretation, audio and
   provenance in the CPU-decode/GPU-encode interface. The earlier synthetic FIFO
   prototype used fixed-rate raw video; it does not settle this contract.
3. When validating the implemented boundary, use a bounded one-to-five-minute
   end-to-end sample with private database/artifact storage, representative face
   workloads and analyst activity. Check the 20% limit on that sample; a multi-TB
   run is not required for the requested rough operational assessment.
4. Evaluate a short initial/seek window followed by longer prefetched windows,
   or another measured way to meet latency without a transition every five
   seconds. This requires protocol/cache/timeline design; it is not implemented.
5. Extend browser coverage, assess transition and audio quality with an analyst,
   and test long recordings and multiple users as needed. Multi-TB endurance is
   deferred, not a blocker for housekeeping or the rough load assessment.

## Evidence and cleanup

Raw measurements, verification scripts and throwaway prototype code are in
`/tmp/sfn-real-perf-cKPobK`: `verified-summary.json`, `stages-prefetch.jsonl`,
`playback.jsonl`, `browser.json`, `browser-5.json`, `browser-5-loaded.json`, and
the associated Python/JavaScript harnesses and logs. Earlier synthetic transfer
results remain in `/tmp/sfn-media-perf-CyeoD1`. These are temporary local artifacts,
not an archival guarantee; operator filenames were not copied into this report.

For local housekeeping, the scripts, manifests, measurements and logs were also
preserved in the ignored, owner-readable archive
`.cleanup-notes/evidence/media-performance-20260912.tar.gz` (SHA-256
`0d29c9789e97b12e8d719ba6641e3526e0e8ea0bba53aea281d1f279cebe505f`).
It excludes rendered MP4s and transient sockets/FIFOs. This private archive is
not tracked or intended for remote publication.

Harness setup failures were resolved without changing production code: caller
UID cache-name lookup, a private executable Torch compilation cache in the GPU
process only, and accessing the internal service by its inspected bridge address
instead of an unpublished localhost port. Failed-run logs were retained.

All benchmark containers and their internal network were removed. The only
remaining running container at verification was the pre-existing
`scalarforensic-qdrant-1`, still up four days. No live database was contacted,
nothing was deployed or pushed, and no driver/model/dependency changes were made.
