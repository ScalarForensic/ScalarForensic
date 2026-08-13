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
timed separately. It runs in the background — but **"nothing blocks on it" is
false, and the spec must not claim it.**

**It competes with chunk playback for the same two workers.**
`SFN_VIDEO_MAX_WORKERS=2` is not a throughput knob: §3.5 measured aggregate
throughput flat from k=1 to k=8, so 2 is the shape of exactly *one viewer* —
the chunk being played plus its single §4.2 prefetch. A full-video job holds one
of those two for its whole ~51-minute run, which puts chunk encoding at k=2 for
the duration: **8.21 s → ~16.35 s** per chunk (§3.5's k=2 row), outside the
6–10 s window the double-buffered swap depends on. The analyst who starts a full
export therefore makes their own live playback worse, and today nothing tells
them so.

Two candidate remedies, both **phase 7's call, neither implemented here**:

1. **Yield.** Run the full-video job at lower priority (`nice`) with an explicit
   `-threads` cap, so chunk work wins the contention instead of splitting it.
   Costs a longer export; needs measuring, since §3.5 shows the box is already
   saturated by one job.
2. **Accept and disclose.** Leave the contention and say so in the UI — *"chunk
   loading will be slower while the full export runs"* — which is honest but
   makes the degradation the analyst's problem.

What is ruled out is the third option: leaving it both unbounded and invisible.

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

**Three states phase 6 added, because the list above has no answer for "we have
not asked yet".** v1's table answers one question — *can this play?* — with two
verdicts, and a verdict is exactly what the player does not have before
`playback-info` returns, or after a container refuses to open. This subsystem is
already three-state in four other places (`stale_evidence`, `mode`,
`Capability`, §6.3's `unknown`/`fits`/refusal) for the same reason.

| State | Analyst sees |
|---|---|
| `idle` | no video selected; the player claims nothing |
| `probing` | `playback-info` in flight — a spinner, **not** a verdict |
| `unknown` | the container could not be probed: the probe's own words, and Download original |

`unknown` is the one that matters. `#147` shipped "unknown displayed as
mismatch" in an evidence viewer, and reporting an unprobeable file as
`needs-transcode` is that defect one layer up — a claim about a stream nobody
read. `codecs._playback_mode` has returned `unknown` as a fourth mode since
phase 3; `states.MODE_TO_STATE` carries it through to the UI instead of
flattening it on the way. A test asserts the state is neither `playable` nor
`needs-transcode`.

`full-job-running`, `full-job-done` and `full-job-failed` are **phase 7's** and
are deliberately not in phase 6's set: §5 says nothing here may invent a state
it cannot observe, and with no job endpoints nothing can enter them.
`states.PHASE_7_STATES` names them so the omission is visible rather than
accidental.

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

**What the fingerprint is, decided in phase 4** (`video_playback/capability.py`,
`Pipeline`). The fingerprint is `sha256` over the canonical rendering of exactly
these nine fields, and the rule is that *every* field of `Pipeline` is hashed —
so the question "is this in the key?" is the same question as "is this a field?",
with no third answer:

| Field | Why it changes pixels |
|---|---|
| `hwaccel` | GPU and CPU encoders do not produce the same picture |
| `decoder` | fixed at `software`; §3.1's GPU-filtered decode loses rotation |
| `filter_chain` | the tone-map chain and the scale expression, verbatim |
| `encoder` | `libx264` vs `h264_nvenc` |
| `rate_control` | preset and CRF/CQ |
| `output_height` | the §16 cap |
| `chunk_seconds` | changes where every artifact boundary falls |
| `audio` | codec and bitrate of the re-encoded track |
| `ffmpeg_version` | an encoder's output changes between builds |

Deliberately **outside** it, because none of them changes a pixel:
`SFN_VIDEO_MAX_WORKERS`, the cache directory and ceiling, queue and timeout
settings, `SFN_EXAMINER_ID`, the source path, and the timecode a chunk starts at
— that last is source identity and chunk arithmetic, the other half of the key.

Two consequences worth stating rather than discovering:

- **Upgrading ffmpeg invalidates the whole cache.** Accepted, and it is the
  conservative direction: a rendering carries a label naming its pipeline (§7.2),
  and a label that names a build which did not produce the file is a false
  statement in an evidence viewer.
- **§8's job-time GPU fallback lands under a different key**, because the
  fallback *is* a different `Pipeline`. That is the correct behaviour — the CPU
  encode is not the GPU encode — and it means a video half-encoded either side of
  a driver failure holds two key sets, never one key with two pictures in it.

A test pins the field set, so adding a field is a deliberate act rather than a
silent cache-wide invalidation, and the §7.2 label is derived from the same field
set — a label that named fewer fields than the key hashes would describe a
pipeline it does not fully describe.

**Decided in phase 5: `chunk_seconds` stays in the key for both artifact kinds.**
It is pixel-affecting for a *chunk* — it moves every encode boundary — and not
for the §4.3 full-file artifact, which is one encode of the whole source however
long a chunk is, so one fingerprint means changing `SFN_VIDEO_CHUNK_SECONDS`
needlessly re-encodes every cached `full.mp4`. That cost was weighed and
accepted: it is paid rarely (§16 forecloses changing chunk length without
revisiting the resolution cap in the same change) and it is recoverable by
re-encoding. The alternative — a fingerprint covering different fields per
artifact kind — replaces `Pipeline`'s one rule ("every field is hashed", with no
third answer to "is this in the key?") with a per-kind table someone has to keep
correct forever, and a wrong entry in that table is the §6.1 defect itself. A
rare recoverable cost beats a permanent second rule. The reasoning is repeated
next to `cache_key()` in `video_playback/cache.py`.

**The key's other half is a directory.** Artifacts live at
`{cache_dir}/{source_digest}/{key}/…`, with the lossless rewrap at
`{cache_dir}/{source_digest}/rewrap.mp4` — a rewrap runs no pipeline, so its only
identity is the source's. Nesting under the source digest is what makes §6.2's
"evict whole videos" a directory removal rather than an inference over opaque key
names; a video rendered by two pipelines holds two `{key}/` subdirectories and is
still one eviction unit.

### 6.2 Eviction, corrected

The `*.mp4` glob `#148` narrowed as containment could not see files under
per-video directories in its size accounting, and would have matched and deleted
a video's own artifacts while it was being watched. **Rewritten in phase 5**
(`video_playback/cache.py`, `evict()`):

- it accounts for the **whole cache tree**, `.part` files included (§6.3);
- it removes **whole videos**, least-recently-played first — recency is the newest
  mtime anywhere under the video, so serving one chunk makes the whole video
  recent, which is what "least-recently-*played*" means when a play touches one
  file out of forty;
- a **playback lease** protects the video on screen. HTTP is stateless, so the
  player registers the video explicitly (`POST /api/video-lease`) and refreshes it
  on a heartbeat; the lease expires on its own (`SFN_VIDEO_LEASE_SECONDS`, §12) so
  a closed tab cannot pin the cache forever. Lease state is **three-valued** —
  `held`, `expired` (the heartbeat stopped) and `none` (never registered in this
  process, which is what a second worker says about a video the first is serving).
  A boolean would collapse the last two into "nobody is watching";
- an in-flight write holds a **pin** (`pin()`), a refcount covering the `.part`
  the lease cannot: `FileResponse` streams its body after the handler returns, so
  the lease is the readers' mechanism and the pin is the writers'.

**The bounded overshoot, stated rather than discovered.** When every unprotected
video is gone and the store is still over the ceiling, `evict()` reports
`over_ceiling` and logs it; it does not delete a leased video. §6.3 is what keeps
the ceiling an invariant — by refusing jobs that would not fit *before* they run —
and breaking a playback in progress to recover bytes from a residue of a smaller
ceiling is the worse trade.

Legacy top-level `{sha256}.mp4` files from the pre-directory layout are counted
and evictable as their own entry rather than deleted on sight: they are still
valid rewraps, and a layout change should not throw away a warm cache.

### 6.3 When one video exceeds the ceiling

`SFN_VIDEO_CACHE_MAX_BYTES` defaults to 8 GiB (`config.py:91`). A long 4K full copy
can exceed that alone, at which point "never exceed the ceiling", "evict whole
videos" and "never evict the video being played" cannot all hold. Resolution:

- **The ceiling is the invariant.** A full-video job whose *estimated* output does
  not fit is **refused before it starts**, with the estimate shown and Download
  original offered. Estimating from measured bitrate is required, not optional.
- Chunk playback is unaffected: chunks are small and bounded.
- In-flight `.part` files count against the ceiling.

**Implemented in phase 5** as `cache.check_ceiling()`, against the §16 ruling:
the limit is **50% of `SFN_VIDEO_CACHE_MAX_BYTES`** — half, not all, because a
store that one video can fill is not a cache. The verdict is **three-valued**:
`fits`, `refused`, and `unknown` for a source whose container does not report the
duration, bitrate and coded height the estimate needs. `unknown` refuses the job
as well, and is kept distinct anyway: "this video is too big for the cache" and
"this file would not say how big it is" are different sentences, and a boolean
would print the first for the second condition. The estimate is reported on
`playback-info` (`full_copy`) whether or not it refuses, so the analyst reads the
number and not only the verdict.

**The estimate is uncalibrated, and that is a phase 7 obligation, not a gap to
ignore.** It scales the source's measured bitrate by the *area* ratio of the §16
output cap — `(min(ih, H)/ih)²`, since the cap never upscales — and applies **no
codec factor**, because none is measured: §3.5 timed the encodes and recorded no
output sizes, so there is no ratio in this repository to apply and inventing one
is what §16 forbids. The direction of the error is knowable even though its size
is not: a CRF-23 H.264 encode of a 10-bit HEVC source at the same resolution is
usually *larger* than its source, so the estimate runs low on exactly the corpus
this feature exists for. **The phase 7 job runner must therefore check the growing
`.part` against the estimate and abort on overshoot rather than trusting it.**

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
| `POST /api/video-chunk` | encode the chunk at a timecode; returns JSON, not bytes |
| `GET /api/video-chunk` | serve an already-encoded chunk; **never encodes** |
| `POST /api/video-full` | start the background full-video job |
| `DELETE /api/video-full` | cancel it |
| `GET /api/video-job-status` | progress, rate, ETA, terminal state |
| `GET /api/video-download` | original bytes, `Content-Disposition`, verified digest |
| `GET /api/video-playback` | unchanged — `original` and `rewrap` modes |
| `POST /api/video-lease` | register/refresh the §6.2 playback lease; `release=true` drops it |

**Why the chunk endpoint is two verbs (phase 6).** v1's table had one
`POST /api/video-chunk` returning the media. A `<video>` element cannot POST —
it issues a `GET` with `Range` headers and nothing else — and it will not wait
the §3.5 8.21 s an encode takes without its own media stack concluding the
source is broken. So the POST does the work and returns JSON (`chunk_url`,
`next_chunk_start`, the pipeline that ran, the §5 state); the GET serves bytes
out of the cache through `FileResponse` range handling and 404s on a miss
instead of encoding. That also keeps the GET idempotent and cacheable, and stops
a request that starts an encode from being disguised as a fetch.

The GET carries `fp`, the fingerprint of the pipeline whose rendering to serve —
a video encoded on two hosts has two pictures (§6.1). **It is a selector inside
that video's directory and never an identity**: `path` is the identity, through
the same resolution flow, and a non-hex `fp` is refused before any path is
built.

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

**Implemented in phase 6** as one table, `video_playback/states.py`, beside the
§5 state list it maps onto — two tables that have to agree, so they live in one
module. `classify()` turns an exception into a row; every row names an HTTP
status, a §5 state, a `retryable` flag and, when retryable, a `Retry-After`.
Phase 7's job runner maps the same conditions onto `full-job-failed` through the
same function rather than re-deriving them.

| §10.1 condition | kind | status | §5 state | retry |
|---|---|---|---|---|
| corrupt input (container will not open) | `corrupt-input` | 422 | `unknown` | no |
| missing video track | `no-video-track` | 422 | `chunk-failed` | no |
| **missing audio track** | — | 200 | `chunk-ready` | — |
| unsupported decoder / no usable pipeline | `no-encode-pipeline` | 503 | `chunk-failed` | no |
| ffmpeg non-zero exit | `encode-failed` | 422 | `chunk-failed` | no |
| job timeout | `job-timeout` | 504 | `chunk-failed` | 30 s |
| **GPU failure or saturation** | — | 200 | `chunk-ready` | — |
| OOM (encoder killed by SIGKILL) | `encoder-killed` | 507 | `capacity-exhausted` | 60 s |
| disk or tmpfs full | `disk-full` | 507 | `capacity-exhausted` | 60 s |
| cache directory unwritable | `cache-unwritable` | 503 | `cache-disabled` | no |
| cache directory unset | `cache-unset` | 503 | `cache-disabled` | no |
| source disappeared | `source-disappeared` | 404 | `chunk-failed` | no |
| source changed mid-session | `source-changed` | 409 | `chunk-failed` | no |
| malformed duration metadata | `malformed-duration` | 422 | `chunk-failed` | no |
| queue full | `queue-full` | 503 | `capacity-exhausted` | 15 s |

The two rows with **no kind** are the ones that are not failures by the time a
caller sees them, and the table says so rather than omitting them: a GPU that
fails at job time falls back to CPU (§8) and produces a chunk, and a source with
no audio is encoded with `-an`. A matrix is as wrong when it invents a failure
as when it drops one, and a test pins that neither kind exists.

**Retryability is advertised, never acted on server-side.** The response states
whether a retry could help and how long to wait; the player obeys it. That is
what "nothing may retry-storm" means operationally — a client looping on a
`retryable: false` row is looping on a condition that will not change.

**Two conditions are decided from the returncode, not from a message.** A
negative returncode is a signal; `-9` with no stderr is the OOM killer, which is
`capacity-exhausted, retry later` and emphatically not "this file cannot be
encoded". Parsing that out of ffmpeg's prose would break the first time it
reworded itself, so `EncodeError` carries `returncode` and `timed_out`.

**The §8 fallback answers a GPU fault and only a GPU fault.** A timeout is not
retried on CPU — the CPU path is the slower one (§3.1: 6.1× vs 2.7×), so the
retry would spend `SFN_VIDEO_JOB_TIMEOUT` again and fail again — and neither is
a SIGKILL, because a second encoder started under memory pressure turns one
refused request into two dead ones.

### 10.2 Atomic publication

Mandatory. **Implemented in phase 5** as one function, `cache.publish()`, used by
both writers (`encode.py` and `rewrap.py`): write to a PID-scoped `.part`, `fsync`
the file, `os.replace` onto the final name, `fsync` the directory, and remove the
`.part` on any failure. **A truncated artifact must never become a cache hit** —
without the file fsync the rename can reach the disk before the data does, and a
host crash in between leaves a file that *looks* finished under a name callers
treat as verified; without the directory fsync the rename itself is not durable.

`cache.sweep_orphaned_parts()` removes `.part` files left by SIGKILL or a host
crash. The pid in the name is what separates them from a live sibling process's
in-flight encode, which must not be touched; pid reuse delays a delete by one
sweep and is documented rather than engineered away. It runs **once per process on
first cache use** (`ensure_swept`) rather than in `app.py`'s lifespan, because the
subsystem owns its store (`CLAUDE.md`) and the cache directory is a request-time
setting — a deployment that never plays a video has none to sweep.

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
  **Done in phase 5**: `cache.KeyedLocks` holds one `asyncio.Lock` per key only
  while a caller wants it, refcounting waiters and deleting the entry when the
  last one leaves, so the table's size is exactly the number of in-flight callers.
  It replaces `_remux_locks`, a dict that grew one entry per source digest ever
  seen and was never cleared. **It deduplicates within one ASGI worker process
  only.** Two worker processes can both encode the same artifact; both publish
  atomically to the same path (§10.2), so the cost is wasted CPU, never a corrupt
  file. Cross-process deduplication, if it is ever wanted, is phase 7's.
- Cache accounting must be serialised, or a bounded overshoot documented.

---

## 11. Code layout

A self-contained subsystem (operator ruling), following the **existing precedent
of `src/scalar_forensic/faces/`**, which `docs/specs/face-pipeline.md` §4 argues
as "plugin shape: optional modality, not a framework".

```
src/scalar_forensic/video_playback/
├── __init__.py      public API — `router`                            [carved]
├── capability.py    hwaccel probe, pipeline selection + fingerprint (§6.1, §8) [phase 4]
├── codecs.py        browser-safe allowlist, mode decision            [carved]
├── digest.py        source SHA-256 + the process-wide HashCache handle [carved]
├── rewrap.py        the PyAV stream copy — lossless, never an encode  [carved]
├── encode.py        chunk and full encode; one path, different -ss/-t
├── jobs.py          worker pool, queue, refcounts, cancellation, lifecycle (§10)
├── cache.py         keys, leases, eviction, purge (§6)               [carved]
├── states.py        §5 player states + the §10.1 failure matrix       [phase 6]
├── audit.py         provenance + examiner record (§7.3), wrapping faces/ helpers
└── routes.py        the APIRouter                                    [carved]

src/scalar_forensic/web/static/js/video_playback/player.js   (double-buffered)
```

`[carved]` marks what exists as of the phase 1–3 carve; the rest arrive with
their phases (§15).

`states.py` is a third addition, ruled in during phase 6. §5's state list and
§10.1's failure matrix are two tables that have to agree — a failure whose state
the UI has no branch for is a failure the analyst never sees — and keeping them
in one module is what makes disagreement visible. It is not in `routes.py`
because phase 7's job runner is its second caller: a full-video job maps the same
conditions onto `full-job-failed` through the same `classify()`.

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
`SFN_VIDEO_CACHE_DIR`, `SFN_VIDEO_CACHE_MAX_BYTES`, `SFN_VIDEO_LEASE_SECONDS`,
`SFN_VIDEO_CHUNK_SECONDS`, `SFN_VIDEO_HWACCEL`, `SFN_VIDEO_MAX_WORKERS`,
`SFN_VIDEO_QUEUE_MAX`, `SFN_VIDEO_JOB_TIMEOUT`, `SFN_VIDEO_OUTPUT_HEIGHT`,
`SFN_FFMPEG_PATH`.

**Settled defaults**, each tied to a §3.5 number rather than to taste:

| Setting | Default | Why |
|---|---|---|
| `SFN_VIDEO_MAX_WORKERS` | `2` | Aggregate throughput is flat from k=1 to k=8 (§3.5), so extra workers buy no capacity and cost per-job latency directly: 8.08 s at k=1, 16.35 s at k=2, 32.47 s at k=4. 2 is the highest k that keeps a queued chunk near §4.2's 6–10 s assumption. |
| `SFN_VIDEO_OUTPUT_HEIGHT` | `1080` | Operator ruling (§16), and §3.5 measures the cost of the alternative: 4K first-play is 33.73 s and 4K CPU tone-map encoding is 0.879×, below realtime. Never upscale — a source shorter than 1080 is passed through at its own height. |
| `SFN_VIDEO_LEASE_SECONDS` | `120` | Four missed 30 s heartbeats — long enough that a chunk encode at §3.5's 8.21 s never drops the lease under §4.2's margin, short enough that a crashed browser stops protecting a video within a couple of minutes (§6.2). |
| `SFN_VIDEO_QUEUE_MAX` | `8` | Admitted chunk encodes, running **and** waiting — §10.4's "a LAN host must not be able to grow the queue without limit". Four §3.5 chunk times deep at the default 2 workers (8.21 s at k=1, 16.35 s at k=2), so a full queue drains in well under a minute; beyond it the request is refused with a `Retry-After` rather than left hanging. Validated `>= SFN_VIDEO_MAX_WORKERS`. |
| `SFN_VIDEO_CHUNK_SECONDS` | `30` | Valid **only under the 1080p cap**: a 30 s chunk lands in 8.21 s at 1080p (§3.5), inside §4.2's margin. At 4K the same chunk takes 33.73 s, so raising `SFN_VIDEO_OUTPUT_HEIGHT` above 1080 requires revisiting this value in the same change. |

---

## 13. Retention tooling

`sfn-video purge --media <sha256> | --all`, mirroring `sfn-faces purge`. Explicit,
auditable, and the answer to "what derived renderings exist and when were they
destroyed".

**Shipped in phase 5**: `cli.video_app` (entry point `sfn-video`) over
`cache.purge()`. `--all` confirms interactively, exactly one scope is accepted,
and the report names every purged digest so the output *is* the record. A
playback lease bounds automatic eviction and deliberately does **not** stop a
purge — an examiner deleting a rendering on purpose is the act §6.4 keeps
explicit precisely so that it happens. Nothing here touches a source file or an
indexed frame JPEG; a purged rendering is regenerated by playing the video again.

The examiner id is printed with the result but **not yet filed to an audit log**:
`video_playback/audit.py` is phase 8, and stubbing it early would create the
mostly-empty module §15 says the carve exists to avoid. Phase 8 owns wiring this
call into it.

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
   *Server side done*: `POST`/`GET /api/video-chunk`, `states.py`, the admission
   gate, `SFN_VIDEO_QUEUE_MAX`, and every §10.1 row pinned by a test.
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
