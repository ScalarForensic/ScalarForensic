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

Stated so no later phase treats these as settled. 4K rates, long-source seek,
concurrency and the hardware floor have since been measured and moved to §3.5.
What remains open:

- **VFR and long-GOP sources.** Every source measured, including §3.5's
  multi-hour one, is constant-frame-rate short-GOP. `-ss` must decode forward
  from a distant keyframe on long-GOP content; nothing here bounds that cost.
- **Damaged or sparse indexes.** §3.5's long source is a stream-copy
  concatenation of one short file, so it carries a single dense undamaged index
  by construction — the bench report says so explicitly. A real multi-hour phone
  or security-camera capture is likely to differ.
- **Multi-sample confirmation.** §3.5's 4K-rate and seek rows are one sample per
  cell. The relative ordering is the load-bearing finding; the exact multipliers
  are not tight.

### 3.5 Real-hardware measurements (2026-08-13)

Method, hardware table and full per-run figures:
`docs/benchmarks/video-bench-2026-08-13.md`. Same host and ffmpeg build as §3.
The box carried ordinary desktop background load (headless Chrome, `bitcoind`,
low single-digit percent), disclosed there rather than claimed idle.

- **4K encode rates.** CPU tone-map + `libx264` — the correct-colour path that
  requires no GPU — runs at **0.879×, below realtime** on a 4K source
  (152.30 s wall for 133.869 s of video). CPU tone-map + `h264_nvenc` holds
  **1.26×**, a thin margin. `scale_cuda` + `h264_nvenc` stays fastest at 4.42×
  and keeps §3.1's rotation defect. **Single sample per pipeline.**
- **Long-source seek.** Input seeking against a **3 h 43 m** source is flat at
  **5.42–5.97 s** across seven offsets out to 3 h 41 m, with no trend. This
  confirms §3.2 out to multi-hour *on a dense undamaged index only* — see §3.4.
  **One sample per offset.**
- **Concurrency.** Aggregate throughput is **flat at ~3.5–3.7× video-seconds per
  wall-second from k=1 to k=8** (3.71× at k=1, 3.52× at k=8). One job already
  saturates the box via libx264's own threading, so concurrency divides the same
  capacity instead of adding any: per-job wall goes 8.08 s → 67.34 s, ~8.3× at
  k=8. This sizes `SFN_VIDEO_MAX_WORKERS` downward, not upward (§12).
- **First-play latency, 30 s chunk.** **8.21 s at 1080p** (3 reps, 8.17–8.23)
  versus **33.73 s at 4K** (3 reps, 32.99–35.19). 1080p sits inside §4.2's
  assumed 6–10 s margin, tightly. 4K exceeds it by ~4×.
- **Hardware floor.** The operator has ruled this host **is** the deployment
  target, not a stand-in (§16). The figures above therefore *are* the floor;
  no extrapolated floor exists and none should be invented. The host spec is
  recorded in the bench report's hardware table.

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
30 s chunk taking ~6–10 s to produce (§3.1, 1080p; measured at 8.21 s in §3.5,
inside the window but not amply so — and 33.73 s at 4K, which is why §12 caps
output height), that leaves margin, and
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
source is ~51 minutes, ~39 on GPU. Output is capped at 1080p (§12), so §3.5's
4K row — which is 4K in *and* 4K out, 0.879× — is an upper bound on the cost,
not the rate this job will see; downscaling a 4K input to 1080p has not been
timed separately. It runs in
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
lever on both cost and disclosure, and v1 never mentioned it. It is now settled:
**cap 1080p, never upscale** (§12, §16), with the rescale disclosed per §7.4.

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
├── __init__.py      public API — `router`                            [carved]
├── capability.py    hwaccel probe, pipeline selection + fingerprint (§6.1, §8)
├── codecs.py        browser-safe allowlist, mode decision            [carved]
├── digest.py        source SHA-256 + the process-wide HashCache handle [carved]
├── rewrap.py        the PyAV stream copy — lossless, never an encode  [carved]
├── encode.py        chunk and full encode; one path, different -ss/-t
├── jobs.py          worker pool, queue, refcounts, cancellation, lifecycle (§10)
├── cache.py         keys, leases, eviction, purge (§6)               [carved]
├── audit.py         provenance + examiner record (§7.3), wrapping faces/ helpers
└── routes.py        the APIRouter                                    [carved]

src/scalar_forensic/web/static/js/video_playback/player.js   (double-buffered)
```

`[carved]` marks what exists as of the phase 1–3 carve; the rest arrive with
their phases (§15).

Two modules are additions to v1's list, ruled in during the carve:

- **`digest.py`.** The source digest and its `HashCache` handle are neither
  codec logic nor artifact-cache logic. Folding a *hash* cache into `cache.py`,
  which is the *artifact* cache, is exactly the conflation §6.1 warns about. The
  handle is a process-wide singleton over an open SQLite connection, so it has
  to live in exactly one module and be imported as functions — never as the
  `_hash_cache` global.
- **`rewrap.py`.** §2's own table calls the rewrap "lossless rewrap … reused
  unchanged": it is a PyAV stream copy, deliberately not an encode, and
  `encode.py` is specified as the ffmpeg path. Keeping them apart is what stops
  a later reader from treating the rewrap as an encode.

`_stale_evidence_report` stays in `routes.py` until `audit.py` arrives in phase
8; `audit.py` is not created early for one function.

Honest caveat on the precedent: `faces/` is genuinely optional and env-gated;
video playback is core UI on the default path, so "removable without archaeology"
is a weaker claim here. Cohesion is the justification, not optionality.

**Seams:** `app.py` includes the router; `config.py` gains `SFN_VIDEO_*`;
`index.html` loads `player.js` before `app.js`. `_resolve_video_path` is
**shared with `/api/video-frame`** — v1's move list omitted it. It moves to
`routes/_shared.py` beside `_check_allowed_path`, since by the same
one-implementation rule it is a security control.

Correction to v1: `/api/video-timeline` does **not** call `_resolve_video_path`.
It takes a `video_hash` only, validated with `re.fullmatch(r"[0-9a-f]{64}")`,
and never touches a filesystem path. The four callers are `/api/video-frame` and
the three playback routes. The move is still a two-module split and still
correct; there is simply no call in the timeline handler to go looking for.

`/api/video-frame` and `/api/video-timeline` stay in `routes/video.py`: they serve
the indexing side, not playback.

No compatibility re-exports are left behind in `routes/video.py`. The tests read
private symbols off the module without patching, so a stale alias would turn a
loud `AttributeError` into a silently wrong patch target; a missing name is the
signal wanted. Patch targets are rebound per-module instead (`CLAUDE.md`'s first
gotcha).

**`CLAUDE.md` must be updated in the same PR as the carve**, or the checked-in
convention is false for the life of that PR.

---

## 12. Configuration

Every setting needs a name, default and validation rule (`config.py` validates
aggressively; `face-pipeline.md` §13 is the template):
`SFN_VIDEO_CACHE_DIR`, `SFN_VIDEO_CACHE_MAX_BYTES`, `SFN_VIDEO_CHUNK_SECONDS`,
`SFN_VIDEO_HWACCEL`, `SFN_VIDEO_MAX_WORKERS`, `SFN_VIDEO_QUEUE_MAX`,
`SFN_VIDEO_JOB_TIMEOUT`, `SFN_VIDEO_OUTPUT_HEIGHT`, `SFN_FFMPEG_PATH`.

**Settled defaults**, each tied to a §3.5 number rather than to taste:

| Setting | Default | Why |
|---|---|---|
| `SFN_VIDEO_MAX_WORKERS` | `2` | Aggregate throughput is flat from k=1 to k=8 (§3.5), so extra workers buy no capacity and cost per-job latency directly: 8.08 s at k=1, 16.35 s at k=2, 32.47 s at k=4. 2 is the highest k that keeps a queued chunk near §4.2's 6–10 s assumption. |
| `SFN_VIDEO_OUTPUT_HEIGHT` | `1080` | Operator ruling (§16), and §3.5 measures the cost of the alternative: 4K first-play is 33.73 s and 4K CPU tone-map encoding is 0.879×, below realtime. Never upscale — a source shorter than 1080 is passed through at its own height. |
| `SFN_VIDEO_CHUNK_SECONDS` | `30` | Valid **only under the 1080p cap**: a 30 s chunk lands in 8.21 s at 1080p (§3.5), inside §4.2's margin. At 4K the same chunk takes 33.73 s, so raising `SFN_VIDEO_OUTPUT_HEIGHT` above 1080 requires revisiting this value in the same change. |

---

## 13. Retention tooling

`sfn-video purge --media <sha256> | --all`, mirroring `sfn-faces purge`. Explicit,
auditable, and the answer to "what derived renderings exist and when were they
destroyed".

---

## 14. Test plan

`addopts` carries `--cov-fail-under=65` on every invocation, so a new subsystem
without tests reds CI; a subset run needs `--no-cov` (`CLAUDE.md`).

**Fixture strategy (settled, §17 Q4).** `data/` is gitignored, so no 10-bit HDR
sample can be committed there. The HDR fixture is resolved by a three-source
lookup, **first hit wins**:

1. `SFN_TEST_VIDEO_HDR` pointing at a real file — an env gate in the style of the
   YuNet test;
2. a clip present in the tracked `test_data/` directory. Its `README.md` states
   what a dropped clip must carry (10-bit, HLG or PQ transfer, rotation side
   data, a named licence) so the operator can drop one in and the tests pick it
   up with **no code change**;
3. otherwise the fixture is **generated at test time** with `ffmpeg -f lavfi`
   into a tmp dir. ffmpeg is already a §8 dependency, so this adds none.

Tests skip only when ffmpeg itself is absent. **Footage from the operator's
corpus is never committed** — it is case material. The bench report naming a
corpus path is a methodology record, not permission to copy the file.

A generated clip cannot honestly assert everything a real capture can; where a
required assertion needs a property the generator cannot produce, the test is
gated on sources 1–2 rather than weakened to fit source 3.

Required cases: **rotation preserved on both pipelines** (§3.1's defect);
colour tags are `bt709` on output; chunk frame-tiling (§3.3) at frame level;
CPU/GPU output equivalence within tolerance; VFR and long-GOP sources; sources
with no audio, and multi-audio; final partial chunk; odd dimensions; corrupt
index; job timeout, cancel and disconnect leave no `.part` and no orphan process;
eviction during playback; cache-full refusal (§6.3); stale-source detection
(§7.1); every §10.1 failure mapped to its player state; path-containment on every
new route including rejection of a cache key as source identity.

**Measurements required on real hardware before phase 4 is accepted: satisfied
by §3.5** (2026-08-13) — 4K rates, long-source seek, concurrent-job scaling and
the hardware floor are all measured, the last as a policy statement (this host
*is* the target, §16) rather than an extrapolation, so "degradation below the
floor" has no subject. The residue is named in §3.4 and stays open: VFR,
long-GOP and damaged-index seek behaviour, and multi-sample confirmation of
§3.5's single-sample rows.

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

Done: phases 1–3 and the carve. In the event only `tests/test_video_playback.py`
needed rewiring — all six of `test_video_endpoints.py`'s string targets belong to
`/api/video-frame` and `/api/video-timeline`, which stay put.

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

### Operator rulings closing §17 Q1–Q5 (2026-08-13)

- **Output resolution is capped at 1080p** (`SFN_VIDEO_OUTPUT_HEIGHT`, §12).
  Never upscale; a rescale is disclosed on the label per §7.4; the cap is
  operator-overridable. Full quality is served by **Download original** (§7.5),
  not by the viewing copy — that is what makes the cap acceptable rather than a
  loss of evidence.
- **Chunk length stays 30 s**, and is valid **only under the 1080p cap**. §3.5's
  33.73 s first-play at 4K forecloses raising the cap without revisiting chunk
  length in the same change.
- **The hardware floor is this host** — the operator has ruled it the deployment
  target, not a stand-in. Its spec is recorded in
  `docs/benchmarks/video-bench-2026-08-13.md`, and §3.5's figures are the floor.
  There is no extrapolated floor and none is to be invented.
- **HDR fixture**: the three-source lookup in §14 (env gate → tracked
  `test_data/` → `ffmpeg -f lavfi` generation).
- **A full-video job is refused before it starts** when its estimated output
  exceeds **50% of `SFN_VIDEO_CACHE_MAX_BYTES`**. The estimate is shown and
  Download original is offered (§6.3).

---

## 17. Open questions

Q1–Q5 are **closed**; the rulings are in §16. What remains open is measurement
residue, not design: VFR, long-GOP and damaged-index seek behaviour, and
multi-sample confirmation of §3.5's single-sample rows (§3.4).
