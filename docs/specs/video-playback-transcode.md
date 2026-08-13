# Spec: On-demand video transcoding for in-browser playback

Status: **draft v2, post-review** · 2026-08-13 · author `scalarforensic-cfm-g1` (CTO)
Supersedes the "HEVC remedy" pending decision carried in `docs/CTO_LEDGER.md`.
v1 proposed a lazy-HLS segment engine. Two independent adversarial design reviews
(Codex, Opus) rejected it as unproven and materially underspecified; the operator
then chose a simpler shape that removes most of the objections by construction
rather than answering them. §3 records what was measured and, explicitly, what
was not.

---

## 1. What this feature is

The analyst clicks a frame hit — "this face appears at 02:31 in `IMG_3274.MOV`" —
and wants to see what happened around that moment. Today that fails for most of
the corpus: the source is HEVC 10-bit HDR, which browsers cannot decode, so
playback soft-fails.

This feature makes any indexed video watchable in the browser by transcoding
**only what the analyst asks for**, on demand, into a bounded cache.

### The deployment it must survive

Not the development corpus. The real one:

- **Machine 1** holds the evidence and runs the server. **Machine 2** is an
  analyst's browser on an isolated LAN.
- Corpora of **several terabytes**; individual videos of **several gigabytes**,
  possibly hours long.
- Server hardware is **not fixed** — a GPU may or may not be present.
- **Fully offline.** No CDN, no external player, no network calls of any kind.

Any approach whose *interactive* cost scales with source length is rejected: it
makes opening one hit in a 4-hour recording a multi-minute wait.

### Non-goals

- Re-encoding the corpus ahead of time. Unjustifiable at TB scale, and most of it
  would never be watched.
- Modifying originals. Sources are opened read-only, always.
- Treating a viewing copy as evidence. It is a **rendering for human review**
  (§7). The original file and its verified SHA-256 remain the forensic object.
- Serving HEVC to browsers that could decode it natively. Out of scope on
  purpose: the operator's own Chrome *advertises* HEVC and then fails
  (`VideoDecoderPipeline`, no VAAPI), so advertised capability is not trustworthy
  input.
- **Seamless multi-segment streaming (HLS/DASH/MSE).** Rejected in v2; see §4.

---

## 2. What already exists

`#136` built machinery this spec reuses:

| Existing | Location | Status |
|---|---|---|
| Path containment (403 outside allowed roots) | `routes/_shared.py:12` | reused unchanged |
| Absolute-path + extension + regular-file resolution | `routes/video.py:36` | **moves to `_shared.py`, §12** |
| Source digest | `routes/video.py:203` | **replaced, §7.1** |
| Per-source async lock (N clicks → 1 job) | `routes/video.py:398` | extended per artifact |
| Bounded LRU cache + eviction | `routes/video.py:327` | **rewritten, §6** |
| Atomic `.part` write + rename | `routes/video.py:278` | **made mandatory, §10.2** |
| Lossless rewrap (wrong container, right codec) | `routes/video.py:269` | reused unchanged |
| Playback label + stream report | `routes/video.py:442` | extended, §7 |
| Honest rate/ETA display (`_RateTracker`) | `cli.py`, `#139` | reused for job progress, §5 |

The gap is stated by the code itself at `routes/video.py:191`:

> *"Container-level judgement only — no codec is inspected, because nothing here
> can fix a codec the browser lacks: a rewrap moves the same bitstream."*

Correct for a rewrap. This spec adds the mode beyond it.

**Playback modes after this feature:**

```
original    container and codec are browser-safe   -> serve source bytes untouched
rewrap      container wrong, codec fine            -> cached lossless MP4  (exists)
transcode   codec unplayable                       -> cached re-encode     (NEW)
```

---

## 3. Measured evidence

**Method.** Sample `IMG_3274.MOV` from the development corpus: 147 MB, 133.87 s,
1920×1080, `hevc` / `yuv420p10le`, `bt2020` primaries, `arib-std-b67` (HLG) — 10-bit
HDR. Host: RTX 4060 Ti, 24 cores, idle, ffmpeg 6.1.1. Timings are `/usr/bin/time`
wall clock, single job, no concurrency. The tone-map chain used throughout is:

```
zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,
tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p
```

Codec distribution: 57 of a 60-file sample (`find`-ordered, not randomised) are
HEVC; 509 video files total, 5.7 GB, median duration 2.0 s, p90 28 s, max 318 s.

### 3.1 Pipeline comparison (full-file encode)

| Pipeline | Speed | Orientation | Colour tags |
|---|---|---|---|
| `scale_cuda` + `h264_nvenc` | 12.9× | **BROKEN — rotation lost** | `bt2020`/HLG (wrong) |
| CPU tonemap + `h264_nvenc` | 6.1× | correct | `bt709`/`bt709` |
| CPU tonemap + `libx264` | 4.7× | correct | `bt709`/`bt709` |
| CPU naive + `libx264` | 10.7× | correct | `bt2020`/HLG (wrong) |

Two findings that decide the pipeline:

1. **The GPU-filtered path silently loses rotation.** Source carries
   `rotation=-90` side data; `-hwaccel_output_format cuda` bypasses ffmpeg's
   autorotate, so output is 1920×1080 with no rotation data and every portrait
   clip plays on its side. Verified by `ffprobe` on both outputs and visually on
   extracted stills. **A test must pin this** (§14).
2. **Naive 8-bit conversion mis-tags colour.** Output is 8-bit but still tagged
   `arib-std-b67`/`bt2020`, which browsers render washed out with lifted blacks.
   The tone-mapped path emits `bt709`/`bt709`. Tone-mapping costs ~2.3× and is
   required.

### 3.2 Seek cost

Input seeking (`-ss` *before* `-i`) uses the container index:

| Encode | Wall |
|---|---|
| 20 s window at offset 0 | 4.46 s |
| 20 s window at offset 100 s | 4.53 s |

**What this does and does not establish.** It shows seek cost is not proportional
to offset *within a 134 s, well-indexed, short-GOP file*. It does **not** establish
constant cost for multi-hour sources, long-GOP or VFR content, or damaged indexes,
where `-ss` must decode forward from a distant keyframe. v1 generalised this to
"cost independent of source size"; that claim is withdrawn. §14 requires the
measurement on real hardware against a long source before §15 phase 4 is accepted.

### 3.3 Chunk independence

Three adjacent 6 s chunks, each an independent ffmpeg invocation, inspected at
**frame level** (v1 cited container duration, which does not prove alignment):

| Chunk | Wall | Frames | First PTS | Last PTS |
|---|---|---|---|---|
| 6 | 1.51 s | 180 | 36.000 | 41.967 |
| 7 | 1.50 s | 180 | 42.000 | 47.967 |
| 8 | 1.51 s | 180 | 48.000 | 53.967 |

180 frames at 30 fps is exactly 6.000 s, and chunk *n*+1 begins on the frame
immediately after chunk *n* ends. **The frames tile exactly — no overlap, no gap,
no duplicated frames.** (The 6.037 s figure a container-level probe reports is
MPEG-TS packetisation padding, not content.)

### 3.4 What has NOT been measured

Stated so no later phase treats these as settled:

- **4K.** Every figure above is 1080p. §3.1 rates will be materially worse at 4K
  and the CPU path may fall below realtime.
- **Long sources.** Nothing longer than 318 s was encoded; the §3.2 caveat stands.
- **Concurrency.** All timings are a single job on an idle 24-core host. *k*
  concurrent jobs divide those cores.
- **Minimum hardware.** No floor is established. §14 requires one.

---

## 4. Architecture

Two artifacts, both **single-encode MP4 files**. There is no segment format, no
playlist, no shared initialisation data, and no media-source stitching.

```
analyst opens a frame hit
        |
        v
GET /api/video-playback-info    mode=transcode, reason, duration, verified sha256
        |                       (probe only — no encoding)
        v
[play]  -> encode chunk at T, 30 s        ~6-10 s   -> plays
        |                                   |
        |                                   +-> chunk T+30 starts encoding
        |                                       immediately, not at the boundary
        |
        |  reaching chunk end  -> next chunk already encoded and preloaded
        |  seeking to a new T  -> encode a chunk there
        v
[request full video] -> background job, whole file, progress + ETA
        |
        v
   complete -> notify; switch to the full file at the current timestamp
```

### 4.1 Why not seamless streaming

v1 specified lazy HLS with CMAF segments. The reviews established that its
central assumption was unproven — independently encoded fragments sharing one
initialisation segment require matching track IDs, timescales, codec
configuration and **decode-time** continuity (`tfdt`), none of which
`-output_ts_offset` establishes, and §3.3's evidence was gathered in MPEG-TS
where each segment is self-contained. Audio is worse: encoder priming recurs at
every independent encode, accumulating drift across thousands of restarts.

The chunk design removes those failure modes rather than solving them. It also
removes the playlist, the job-priority scheduler, prefetch-cancellation
ownership, the vendored player library, and segment-index validation.

**Cost accepted:** chunk boundaries are visible as a brief glitch. The operator
has explicitly accepted this ("if there is a minimal glitch that is ok for easy
buffering"). Boundaries stop entirely once the full-video job lands.

### 4.2 Chunk playback

`SFN_VIDEO_CHUNK_SECONDS` (default 30). Playback uses **two `<video>` elements**:
while one plays, the next chunk loads hidden in the other; at the boundary they
swap. Each chunk remains an ordinary independent MP4 served by the existing
`FileResponse` range handling. No MSE.

**Prefetch depth is exactly one.** The next chunk is queued the moment the current
one finishes encoding — not when playback reaches the boundary — so by the time
the analyst crosses it, the next chunk is already encoded and preloaded. With a
30 s chunk taking ~6–10 s to produce (§3.1, 1080p), that leaves ample margin, and
depth 1 keeps a bounded amount of speculative work: at most one wasted chunk if
the analyst stops or seeks.

A prefetch made useless by a seek is **left to finish** rather than cancelled. It
is one small job, it is cached if the analyst returns to that position, and
cancellation of speculative work is precisely the ownership problem that made the
v1 scheduler complex (§4.1). Explicit cancellation exists only for the
full-video job (§4.3), where the cost actually justifies it.

Seeking outside the loaded chunk requests a chunk at the new position. This is
what gives random access before the full file exists.

### 4.3 Full-video job

Explicit, operator-initiated (§5), and long: at §3.1's 1080p CPU rate a 4-hour
source is ~51 minutes, ~39 on GPU, **and 4K is unmeasured (§3.4)**. It runs in
the background; nothing blocks on it.

- Progress and ETA come from ffmpeg's own frame-level progress output, rendered
  with `_RateTracker` and `#139`'s honest labelling — *"~4 min remaining at
  current rate"*, never a calibrated-uncertainty claim.
- Cancellable explicitly; navigating away prompts.
- On completion: notification, and if the analyst is still watching chunks, the
  player switches to the full file **at the current timestamp**.

---

## 5. Trigger and player states

**Opening a hit costs no encoding.** `/api/video-playback-info` probes the
container and reports `mode: "transcode"` with a human reason — *"HEVC 10-bit:
this browser cannot decode it"*. (It is not free: it opens the container and
reads stream metadata. The defensible claim is "no transcoding on open", not
"zero compute" — v1 overstated this.)

Every state below must be represented in the UI. `#139` removed a fake error band
from the ingestion display; nothing here may invent a state it cannot observe.

| State | Analyst sees |
|---|---|
| `playable` | plays directly, no controls added |
| `needs-transcode` | reason + **Play** + **Request full video** + **Download original** |
| `chunk-encoding` | spinner with elapsed time; no fabricated percentage |
| `chunk-ready` | plays |
| `chunk-failed` | the failure reason, and Download original as the escape route |
| `full-job-running` | progress bar, rate, ETA, **Cancel** |
| `full-job-done` | sound + animation; button becomes **Open**; auto-switch if watching |
| `full-job-failed` | reason; chunk playback continues to work |
| `cache-disabled` | `SFN_VIDEO_CACHE_DIR` unset — explain, offer Download original |
| `capacity-exhausted` | queue or cache full; explain, offer Download original |

---

## 6. Cache

Two artifact kinds share one bounded store: **chunks** (`{key}/c{start}.mp4`) and
**full copies** (`{key}/full.mp4`), plus existing rewraps.

### 6.1 The cache key must include the pipeline

Not just the source. Key = `sha256(source identity ‖ pipeline fingerprint)`, where
the fingerprint covers hwaccel selection, encoder and rate control, the tone-map
chain, output resolution policy and chunk length. Otherwise flipping
`SFN_VIDEO_HWACCEL`, changing chunk length, or a driver upgrade that flips the
probe yields a cache holding artifacts from two pipelines **inside one video** — a
visible quality shift under a label naming one pipeline. This mirrors the existing
rule that model hashes must match what a collection recorded (`CLAUDE.md`).

### 6.2 Eviction, corrected

`_evict_cache` (`routes/video.py:327`) globs `*.mp4`. Extending it as-is is unsafe:
it would not see files under per-video directories in its size accounting, and it
**would** match and delete a video's own artifacts while it is being watched. The
rewrite must:

- account for the whole cache tree, not one glob;
- evict **whole videos**, least-recently-played first, never single chunks from a
  video currently being watched;
- hold a **playback lease** — HTTP is stateless, so "the video being played" must
  be an explicit, heartbeat-refreshed registration with an expiry, not the
  single-call `keep` argument. Without a lease, eviction can drop a video between
  two of its own chunk requests.
- never evict an artifact with an open reader or an in-flight job.

### 6.3 When one video exceeds the ceiling

`SFN_VIDEO_CACHE_MAX_BYTES` defaults to 8 GiB (`config.py:91`). A long 4K full copy
can exceed that alone, at which point "never exceed the ceiling", "evict whole
videos" and "never evict the video being played" cannot all hold. Resolution:

- **The ceiling is the invariant.** A full-video job whose *estimated* output does
  not fit is **refused before it starts**, with the estimate shown and Download
  original offered. Estimating from measured bitrate is required, not optional.
- Chunk playback is unaffected: chunks are small and bounded.
- In-flight `.part` files count against the ceiling.

### 6.4 Retention

The LRU ceiling is the only automatic retention mechanism. v1 also proposed a TTL
sweep; it is **dropped** — its only distinct job was hygiene, which is better
served by an explicit `sfn-video purge` mirroring `sfn-faces purge` (§13). "Derived
renderings were deleted at time X by examiner Y" is a statement that survives a
courtroom; a background clock is not.

**Cache location is a mount point.** `SFN_VIDEO_CACHE_DIR` on `/dev/shm` makes the
store RAM-backed with no code change. Caution: tmpfs defaults to half of RAM and
shares the box with torch models, so ENOSPC can precede the LRU — the ceiling must
be set below the tmpfs size.

---

## 7. Forensic discipline

### 7.1 Provenance must be verified, not remembered

v1 proposed replacing byte-hashing with a lookup of the `video_hash` recorded at
index time. **Rejected**: it would print, as provenance, a hash of the file *as
indexed* beside a rendering of the file *as it is now*. If the source changed,
the label is a false statement in an evidence viewer.

Instead, back the digest with the existing **`HashCache`** (`embedder.py:169`),
which is keyed on `(normalised real path, mtime_ns, size)`, persists across
restarts, needs no Qdrant — important, since the app boots without it — and
**revalidates against the current file**. Full re-hash only on a cache miss.

If the computed digest differs from the indexed `video_hash`, that is a
**stale-evidence condition**: surface it prominently, do not silently serve.

### 7.2 The rendering must be reproducible

v1 said the label "never learns" whether GPU or CPU ran, while also promising
tone-mapping is "declared, not silent". Contradictory, and §3.1 shows the paths
differ materially. The label records the **actual pipeline**: hwaccel used,
decoder, full filter chain with parameters, encoder and rate control, output
resolution, ffmpeg version, and any audio transformation or omission.

`sfn-video render --path X --at T` must print the exact invocation that produced a
given rendering, so a reviewer can reproduce it.

### 7.3 Attribution

v1 claimed the trigger was "attributable" while recording nothing. Every
transcode — chunk or full — writes an audit record with `SFN_EXAMINER_ID`,
timestamp, source path and verified digest, requested timecode, pipeline
fingerprint, and outcome. `faces/audit.py` and `faces/provenance.py` are the
pattern; this must not invent a second one.

### 7.4 Fidelity disclosure

The label states plainly: this is a **lossy re-encode**, tone-mapped from HDR to
BT.709 (an interpretation, not a measurement), possibly rescaled, with audio
re-encoded or dropped; **fine detail and exact colour must be judged from the
original, not from this rendering**. `skipped_streams` reporting from `#136`
carries over so a silent viewing copy is never mistaken for a silent original.

### 7.5 Download original

`/api/video-download` serves untouched source bytes with the verified digest
displayed. Note the tension with §1: this places a copy of evidence on machine 2.
That is the analyst's deliberate act, it is audited like any other, and the
deployment's handling policy governs it — but the spec must not pretend the
transfer is invisible.

---

## 8. Pipeline and hardware

**ffmpeg is a declared external dependency** (operator ruling, 2026-08-13). The
project currently depends on **PyAV only** (`pyproject.toml:16`), and no ffmpeg
binary is invoked anywhere in `src/` or installed by the Dockerfile — v1 assumed
one silently. Adding it requires: the Dockerfile, the airgap `vendor/` bundle,
`INSTALL.md`, and a documented minimum build (`--enable-libzimg` for `zscale`,
plus NVENC/NVDEC where the GPU path is wanted).

Selection is by **startup capability probe** with `SFN_VIDEO_HWACCEL=auto|cuda|none`
override. The probe must exercise a real decode→tone-map→encode→mux of a few
frames, not merely check that an encoder is listed: decoder, transfer, filter,
pixel format and encoder can each fail independently across driver and build
combinations. A GPU failure at job time falls back to CPU, records the fallback in
the label (§7.2), and never fails the request outright.

Output resolution policy must be stated explicitly — it is the largest single
lever on both cost and disclosure, and v1 never mentioned it.

---

## 9. Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/video-playback-info` | extended: mode, reason, duration, verified digest, job state |
| `POST /api/video-chunk` | encode/serve a chunk at a timecode |
| `POST /api/video-full` | start the background full-video job |
| `DELETE /api/video-full` | cancel it |
| `GET /api/video-job-status` | progress, rate, ETA, terminal state |
| `GET /api/video-download` | original bytes, `Content-Disposition`, verified digest |
| `GET /api/video-playback` | unchanged — `original` and `rewrap` modes |

**Every path-bearing route reuses the existing resolution flow** — absolute path,
`resolve()`, extension check, `_check_allowed_path`, regular-file check. No route
accepts a cache key as its only source identity; a key is never trusted to name a
file. Timecodes are validated against the probed duration, and chunk requests are
bounded, before any queueing or encoding.

Evidence mounts should be **read-only**, closing the window between path
validation and the encoder opening the file.

---

## 10. Failure and lifecycle

### 10.1 Failure matrix

Each of these needs a defined HTTP status, log record, player state (§5) and
retryability: corrupt input; missing video or audio track; unsupported decoder;
ffmpeg non-zero exit; job timeout; GPU failure or saturation; OOM; disk or tmpfs
full; cache directory unwritable or unset; source disappeared or changed
mid-session; malformed duration metadata; queue full. Nothing may retry-storm.

### 10.2 Atomic publication

Mandatory, matching the existing rewrap (`routes/video.py:278`): encode to a
PID-scoped `.part`, `fsync`, atomic rename on success, remove on any failure.
**A truncated artifact must never become a cache hit.** Startup sweeps orphaned
`.part` files left by SIGKILL or host crash.

### 10.3 Subprocess lifecycle

Every ffmpeg invocation carries a timeout; is terminated and reaped on cancel,
client disconnect or shutdown; runs in its own process group so children die with
it; and has bounded captured stderr. Argument construction is never string
concatenation of user input.

### 10.4 Concurrency

- A bounded worker pool, with a **bounded queue** and per-client and per-video
  caps. A LAN host must not be able to grow the queue without limit.
- Deduplication by artifact key, with reference counting: a cancel by one analyst
  must never kill a job another is waiting on.
- Locks must not accumulate unboundedly, and the design must state whether it
  assumes a single ASGI worker process — if not, cross-process deduplication is
  required, since the current in-process dict does not provide it.
- Cache accounting must be serialised, or a bounded overshoot documented.

---

## 11. Code layout

A self-contained subsystem (operator ruling), following the **existing precedent
of `src/scalar_forensic/faces/`**, which `docs/specs/face-pipeline.md` §4 argues
as "plugin shape: optional modality, not a framework".

```
src/scalar_forensic/video_playback/
├── __init__.py      public API — `router`
├── capability.py    hwaccel probe, pipeline selection + fingerprint (§6.1, §8)
├── codecs.py        browser-safe allowlist, mode decision
├── encode.py        chunk and full encode; one path, different -ss/-t
├── jobs.py          worker pool, queue, refcounts, cancellation, lifecycle (§10)
├── cache.py         keys, leases, eviction, purge (§6)
├── audit.py         provenance + examiner record (§7.3), wrapping faces/ helpers
└── routes.py        the APIRouter

src/scalar_forensic/web/static/js/video_playback/player.js   (double-buffered)
```

Honest caveat on the precedent: `faces/` is genuinely optional and env-gated;
video playback is core UI on the default path, so "removable without archaeology"
is a weaker claim here. Cohesion is the justification, not optionality.

**Seams:** `app.py` includes the router; `config.py` gains `SFN_VIDEO_*`;
`index.html` loads `player.js` before `app.js`. `_resolve_video_path`
(`routes/video.py:36`) is **shared with `/api/video-frame` and
`/api/video-timeline`** — v1's move list omitted it. It moves to
`routes/_shared.py` beside `_check_allowed_path`, since by the same
one-implementation rule it is a security control.

`/api/video-frame` and `/api/video-timeline` stay in `routes/video.py`: they serve
the indexing side, not playback.

**`CLAUDE.md` must be updated in the same PR as the carve**, or the checked-in
convention is false for the life of that PR.

---

## 12. Configuration

Every setting needs a name, default and validation rule (`config.py` validates
aggressively; `face-pipeline.md` §13 is the template):
`SFN_VIDEO_CACHE_DIR`, `SFN_VIDEO_CACHE_MAX_BYTES`, `SFN_VIDEO_CHUNK_SECONDS`,
`SFN_VIDEO_HWACCEL`, `SFN_VIDEO_MAX_WORKERS`, `SFN_VIDEO_QUEUE_MAX`,
`SFN_VIDEO_JOB_TIMEOUT`, `SFN_VIDEO_OUTPUT_HEIGHT`, `SFN_FFMPEG_PATH`.

---

## 13. Retention tooling

`sfn-video purge --media <sha256> | --all`, mirroring `sfn-faces purge`. Explicit,
auditable, and the answer to "what derived renderings exist and when were they
destroyed".

---

## 14. Test plan

`addopts` carries `--cov-fail-under=65` on every invocation, so a new subsystem
without tests reds CI; a subset run needs `--no-cov` (`CLAUDE.md`).

**Fixture problem, unresolved and blocking phase 4:** `data/` is gitignored, so a
real 10-bit HDR sample cannot be committed. Either add a tiny generated HDR
fixture (preferred — synthesisable by ffmpeg at build time), or gate on an
env-supplied path and skip, as the YuNet test does. Decide before phase 4.

Required cases: **rotation preserved on both pipelines** (§3.1's defect);
colour tags are `bt709` on output; chunk frame-tiling (§3.3) at frame level;
CPU/GPU output equivalence within tolerance; VFR and long-GOP sources; sources
with no audio, and multi-audio; final partial chunk; odd dimensions; corrupt
index; job timeout, cancel and disconnect leave no `.part` and no orphan process;
eviction during playback; cache-full refusal (§6.3); stale-source detection
(§7.1); every §10.1 failure mapped to its player state; path-containment on every
new route including rejection of a cache key as source identity.

**Measurements required on real hardware before phase 4 is accepted:** 4K rates,
a long-source seek measurement (§3.2), concurrent-job scaling, and a stated
minimum hardware floor with documented degradation below it.

---

## 15. Implementation phases

1. **Digest correctness** — back `_source_digest` with `HashCache`; stale-evidence
   detection; no whole-file hashing on the request path.
2. **Download original** — small, self-contained, immediately useful; the escape
   hatch every later failure state points at.
3. **Codec detection** — allowlist (H.264 8-bit, VP8, VP9, AV1);
   `playback-info` reports `mode: "transcode"` with a reason. No encoding yet.
4. **Encode core** — ffmpeg dependency declared and bundled; capability probe;
   pipeline fingerprint; chunk encode; rotation and colour tests; the §14
   measurements.
5. **Cache** — keyed by pipeline, leases, corrected eviction, ceiling refusal,
   atomic publication, purge command.
6. **Chunk playback** — double-buffered player, seek-to-new-chunk, player states.
7. **Full-video job** — background worker, progress/ETA via `_RateTracker`,
   cancel, completion notification, auto-switch at timestamp.
8. **Provenance and audit** — label with full pipeline record, examiner audit,
   `sfn-video render` reproduction command.

The subsystem carve (§11) happens **between phases 3 and 4**, when there is
something to carve. v1 put it first, creating mostly-empty modules for code that
did not exist; the reviews were right that this is premature. Note it is not a
pure move: `tests/test_video_endpoints.py` and `tests/test_video_playback.py`
patch `scalar_forensic.web.routes.video.*` by name, and `CLAUDE.md`'s first gotcha
is that patch targets are per-module.

---

## 16. Decisions taken in the 2026-08-13 design session

Recorded with reasons. These are one session's choices, not precedent with paid
cost, and may be revisited on evidence.

- **Mode is decided by a server-side codec allowlist**, not browser-advertised
  capability — the operator's Chrome demonstrably advertises HEVC and fails.
- **Two single-encode artifacts** (chunk, full file) instead of a segmented
  stream. Trades background compute for the removal of an entire failure class.
- **30 s chunks, double-buffered**, with a brief glitch at boundaries explicitly
  accepted by the operator.
- **Prefetch depth one**, started when the current chunk finishes encoding rather
  than when playback reaches the boundary; speculative chunks are never cancelled.
- **The full-video job is explicit**, cancellable, with progress and ETA, and
  auto-switches at the current timestamp on completion.
- **ffmpeg CLI is a declared dependency**; bundling cost accepted.
- **Must work with and without a GPU**, selected by runtime probe.
- **Download original** is offered alongside playback.
- **HLS/MSE is rejected for now** (§4.1) and may be reconsidered only if chunked
  playback proves insufficient in real use.

---

## 17. Open questions

1. Output resolution policy (§8) — cap at 1080p, match source, or operator choice.
2. Chunk length: 30 s is chosen, but first-play latency at 4K may argue for less.
3. Minimum hardware floor and behaviour below it (§14).
4. HDR fixture strategy for tests (§14) — blocking for phase 4.
5. Whether the full-video job should be admitted at all on sources whose estimated
   output exceeds a large fraction of the cache ceiling (§6.3).
