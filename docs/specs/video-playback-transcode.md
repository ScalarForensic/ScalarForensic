# Spec: On-demand video transcoding for in-browser playback

Status: **draft v1** · 2026-08-13 · author `scalarforensic-cfm-g1` (CTO)
Supersedes the "HEVC remedy" pending decision carried in `docs/CTO_LEDGER.md`.
All performance figures below are **measured on this machine**, not estimated; the
method is recorded in §3 so they can be re-run on the real server.

---

## 1. What this feature is

The analyst clicks a frame hit — "this face appears at 02:31 in `IMG_3274.MOV`" — and
wants to see what happened around that moment. Today that fails for most of the corpus:
the source is HEVC 10-bit HDR, which the browser cannot decode, so playback soft-fails.

This feature makes any indexed video playable in the browser by transcoding **only the
parts the analyst actually watches**, on demand, into a bounded cache.

### The deployment it must survive

Not the test corpus. The real one:

- **Machine 1** holds the evidence and runs the server. **Machine 2** is an analyst's
  browser on an isolated LAN. Files never live on machine 2.
- Corpora of **several terabytes**; individual videos of **several gigabytes**, possibly
  hours long.
- Server hardware is **not fixed** — a GPU may or may not be present.
- **Fully offline.** No CDN, no external player, no network calls of any kind.

Every design choice below follows from those four facts. In particular: **any approach
whose cost scales with source file length is rejected**, because it makes opening one hit
in a 4-hour recording a multi-minute wait.

### Non-goals

- Re-encoding the corpus ahead of time. Storage cost at TB scale is unjustifiable and
  most of it would never be watched.
- Replacing or modifying originals. Sources are opened read-only, always.
- Making the viewing copy an evidential artifact. It is a **rendering for human review**,
  labelled as such (§7). The original file and its SHA-256 remain the forensic object.
- Serving HEVC to browsers that can decode it natively. Deliberately out of scope: the
  operator's own Chrome *advertises* HEVC support and then fails to decode
  (`VideoDecoderPipeline`, no VAAPI), so advertised capability is not trustworthy input.

---

## 2. What already exists

`#136` built most of the surrounding machinery, and this spec reuses all of it:

| Existing | Location | Reused as-is |
|---|---|---|
| Path containment (403 outside allowed roots) | `routes/_shared.py:12` | yes |
| Source digest keyed to the indexer's `video_hash` | `routes/video.py:203` | **with a fix, §8** |
| Per-source async lock (N clicks → 1 job) | `routes/video.py:398` | extended per segment |
| Bounded LRU cache + eviction | `routes/video.py:327` | extended, §6 |
| Lossless rewrap for wrong-container/right-codec | `routes/video.py:269` | yes, unchanged |
| Honest playback label + stream report | `routes/video.py:442` | extended, §7 |

The gap is stated by the code itself, at `routes/video.py:191`:

> *"Container-level judgement only — no codec is inspected, because nothing here can fix a
> codec the browser lacks: a rewrap moves the same bitstream."*

That boundary is exactly right for a rewrap. This spec adds the third mode beyond it.

**Playback modes after this feature:**

```
original    source container and codec are browser-safe  -> serve source bytes untouched
rewrap      container wrong, codec fine                  -> cached lossless MP4 (exists)
transcode   codec unplayable                             -> cached HLS segments  (NEW)
```

---

## 3. Measured evidence

Sample: `IMG_3274.MOV`, 147 MB, 133.87 s, 1920×1080, `hevc` / `yuv420p10le`,
`bt2020` primaries, `arib-std-b67` (HLG) transfer — i.e. 10-bit HDR, representative of
the corpus (95% of a 60-file sample is HEVC). Host: RTX 4060 Ti, 24 cores, ffmpeg 6.1.1.

**Full-file transcode, four pipelines:**

| Pipeline | Speed | Orientation | Colour |
|---|---|---|---|
| `scale_cuda` + `h264_nvenc` | 12.9× | **BROKEN — rotation lost** | untone-mapped |
| CPU tonemap + `h264_nvenc` | 6.1× | correct | correct |
| CPU tonemap + `libx264` | 4.7× | correct | correct |
| CPU naive + `libx264` | 10.7× | correct | **wrong tags** |

Two findings that decide the pipeline:

1. **The GPU-filtered path loses rotation.** Source carries `rotation=-90` side data;
   `-hwaccel_output_format cuda` bypasses ffmpeg's autorotate, so output is 1920×1080 with
   no rotation data and every portrait clip plays on its side. Confirmed by `ffprobe` on
   both outputs and visually on extracted stills. A sideways video in an evidence viewer is
   a worse defect than a colour shift.
2. **Naive 8-bit conversion mis-tags colour.** Output is 8-bit but still tagged
   `arib-std-b67`/`bt2020`; the browser renders it washed out with lifted blacks. The
   tone-mapped path emits correct `bt709`/`bt709`. For a tool where clothing or vehicle
   colour can matter, this is the same class of problem as an uncalibrated number on
   screen. Tone-mapping costs ~2.3× and is worth it.

**Segment-level transcode — the load-bearing measurement:**

Input seeking (`-ss` *before* `-i`) uses the container index, so cost depends on window
length, not file length or offset:

| Encode | Wall |
|---|---|
| 20 s window at offset 0 | 4.46 s |
| 20 s window at offset 100 s | 4.53 s |

Three adjacent 6 s segments, each encoded as an independent process:

| Segment | Wall | Duration | First frame | Timeline start |
|---|---|---|---|---|
| 6 | 1.51 s | 6.037 s | keyframe | 35.98 s |
| 7 | 1.50 s | 6.037 s | keyframe | 41.98 s |
| 8 | 1.51 s | 6.037 s | keyframe | 47.98 s |

Independently produced, yet the timeline positions are exactly 6.000 s apart. That is what
`-force_key_frames` plus `-output_ts_offset` buys, and it is what makes seamless stitching
in the player possible.

**Conclusion: 1.5 s of CPU produces 6 s of playable video — ~4× realtime headroom with no
GPU at all, at a cost independent of source size.** This is the whole basis of the design.

---

## 4. Architecture

Lazy HLS: a playlist describing the *whole* timeline, whose segments are transcoded only
when requested.

```
analyst clicks hit at 02:31
        |
        v
GET /api/video-playback-info      -> mode=transcode, reason, duration, source sha256
        |
        v
[Prepare for playback]  (§5)      -> eager jobs for the segments around 02:31
        |
        v
GET .../index.m3u8                -> virtual playlist, all N segments advertised
        |
        v
hls.js requests segment 25        -> cached? serve : transcode(25) then serve
        |                              1.5 s
        v
playback starts; prefetch keeps [playhead-X, playhead+X] warm
```

Because the playlist is virtual, **seeking anywhere is free**: the analyst drags to
1:40:00 and the player requests only that segment. Segments never watched are never
produced. Hit-anchored viewing is not a separate feature — it is what happens when the
player starts at the hit's segment and no earlier segment is ever requested.

### Prefetch and job control

- A **bounded worker pool** runs segment jobs; a per-`(digest, index)` lock deduplicates,
  reusing the `_remux_locks` pattern already in `video.py`.
- **Priority**: a segment the player is blocking on always preempts prefetch work.
- **Seek cancels stale prefetch.** Without this, an analyst scrubbing across a 4-hour
  recording queues hundreds of dead jobs and starves the segment actually being waited on.
  This is the single most important piece of the job logic.
- Prefetch depth `X` is configurable and small by default.

### Segment format

**fMP4 / CMAF**, HLS v7: one init segment plus media segments per video. Avoids MPEG-TS
90 kHz timestamp quirks, supports the widest codec set, and the same fragments could serve
DASH later without re-cutting.

### Player

`hls.js`, **vendored into `web/static/`** (Apache-2.0). No CDN — the deployment is offline
and the CSP forbids external hosts. It follows the existing frontend convention: player
code goes in the matching Alpine part file under `web/static/js/`, never into `app.js`.

---

## 5. Trigger

Transcoding is **operator-initiated, then automatic within the session**:

1. Opening a hit costs **zero** compute. `/api/video-playback-info` reports
   `mode: "transcode"` with a human reason — *"HEVC 10-bit: this browser cannot decode
   it"* — and the UI shows a **Prepare for playback** control.
2. On click, the first 1–2 segments around the hit timecode are transcoded eagerly, so
   playback begins after ~1.5 s rather than after a full-file encode.
3. From then on the sliding prefetch window keeps playback smooth without further asking.
4. A **Download original** control sits alongside, always available (§9).

Rationale: nothing derived from evidence is written to disk without a deliberate act, and
that act is attributable. It also means a server is never made to transcode videos that
were merely glanced at — which matters when the corpus is terabytes.

---

## 6. Cache: three mechanisms, three distinct jobs

These were chosen separately and must not be conflated:

| Mechanism | Governs | Setting |
|---|---|---|
| **Sliding window** | what is *produced* ahead of the playhead | `SFN_VIDEO_PREFETCH_SEGMENTS` |
| **TTL sweep** | when *idle* segments are reclaimed on a long-running server | `SFN_VIDEO_SEGMENT_TTL` |
| **LRU ceiling** | the hard cap that must never be exceeded | `SFN_VIDEO_CACHE_MAX_BYTES` |

- The **LRU ceiling is the invariant.** When the cache exceeds it, whole least-recently-
  played *videos* are evicted — never individual segments out of the middle of a video,
  which would leave holes that re-transcode mid-playback and stutter.
- The **TTL sweep is hygiene, not capacity.** A case machine left running for days should
  not accumulate derived renderings of evidence indefinitely. The sweep is what makes the
  cache's contents roughly track current work rather than everything ever opened.
- **Eviction never touches the video being played.** The existing `keep` parameter in
  `_evict_cache` already encodes this and is extended to the segment directory.

**Cache location is a mount point, not a feature.** `SFN_VIDEO_CACHE_DIR` pointed at
`/dev/shm/...` makes the whole cache RAM-backed with no code change, since tmpfs is a
filesystem and `FileResponse`, range serving and eviction all work unchanged. Deployments
that want derived copies to never touch persistent storage get that by configuration. An
in-process bytes cache was considered and rejected: it means hand-rolling HTTP 206
partial-content handling that Starlette currently provides correctly for free.

---

## 7. Forensic discipline

- The viewing copy is a **rendering, not evidence**, and the UI says so. The existing
  playback label is extended to state the mode, the reason, that colour has been
  tone-mapped from HDR, and the **source SHA-256** — the same value the indexer stored as
  `video_hash`, so a rendering is always tied back to the point that produced the hit.
- **Originals are opened read-only and never modified.** All output goes to the cache dir.
- `skipped_streams` reporting (from `#136`, e.g. Live-Photo LPCM audio that has no MP4
  mapping) carries over, so a silent viewing copy is never mistaken for a silent original.
- Tone-mapping is **declared, not silent**. An HDR→SDR conversion is an interpretation;
  the label names the operator (`tonemap=hable`) so a reviewer can reproduce it.

---

## 8. Hardware capability

One pipeline; encoder and decoder chosen by a **startup capability probe**, with
`SFN_VIDEO_HWACCEL=auto|cuda|none` as an explicit override.

- **GPU present**: NVDEC decode, GPU tone-map, NVENC encode. **The rotation defect in §3
  must be fixed and pinned by a test** — this is the one place the fast path is known to
  produce wrong output.
- **No GPU**: `zscale`/`tonemap` + `libx264`. Measured at ~4× realtime headroom for 1080p,
  which the segment design absorbs; at 4K the GPU path matters much more, which is why the
  probe exists.

The output contract is identical either way, so the cache, the player and the label never
learn which ran.

### A scale defect to fix first

`_source_digest` (`video.py:203`) SHA-256s the **entire** source file to key the cache. On
a multi-GB video that is a full read before playback can start, and its
`lru_cache(maxsize=256)` will thrash against a TB corpus. The indexer already computed and
stored this value as `video_hash`; the playback path should look it up and only fall back
to hashing when it is absent. This should land before the transcode work, since the
transcode path inherits the same key.

---

## 9. Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/video-playback-info` | extended: `mode`, reason, duration, segment count, sha256 |
| `POST /api/video-playback/prepare` | start eager jobs around a timecode |
| `GET /api/video-hls/{digest}/index.m3u8` | virtual playlist over the whole timeline |
| `GET /api/video-hls/{digest}/init.mp4` | CMAF init segment |
| `GET /api/video-hls/{digest}/{n}.m4s` | segment; transcodes on demand if absent |
| `GET /api/video-download` | **original bytes**, `Content-Disposition`, sha256 shown |
| `GET /api/video-playback` | unchanged — `original` and `rewrap` modes |

`/api/video-download` is deliberately the untouched source: for serious review the analyst
opens it in VLC or mpv locally, where codec support is a non-issue, and can verify the hash
against the one the indexer recorded.

---

## 10. Code layout: a self-contained subsystem

**Operator ruling, 2026-08-13:** this ships as a *plugin* — one folder owning the whole
concern, integrated at a few explicit seams — rather than logic spread across the existing
route and pipeline modules. The goal is readability and maintainability: a reviewer should
be able to read video playback in one place, and the subsystem should be removable or
replaceable without archaeology.

```
src/scalar_forensic/video_playback/
├── __init__.py      public API — `router`, nothing else imported elsewhere
├── capability.py    hwaccel probe, encoder/decoder selection (§8)
├── codecs.py        browser-safe allowlist, mode decision (original|rewrap|transcode)
├── rewrap.py        lossless MP4 rewrap — moved from routes/video.py, unchanged
├── segments.py      single-segment transcode: seek, tonemap, keyframe/ts alignment
├── playlist.py      virtual CMAF playlist over the whole timeline
├── jobs.py          worker pool, per-(digest,index) locks, prefetch, seek-cancellation
├── cache.py         TTL sweep + whole-video LRU (§6)
└── routes.py        the APIRouter — playback, info, prepare, hls, download

src/scalar_forensic/web/static/js/video_playback/
├── player.js        Alpine part file for the player
└── vendor/hls.js    vendored, Apache-2.0, offline
```

### The seam

Integration with the rest of the app is deliberately narrow and enumerable:

1. `app.py` includes `video_playback.router` — the only Python import site.
2. `config.py` gains the `SFN_VIDEO_*` settings, alongside the existing cache settings.
3. `index.html` loads `player.js` before `app.js`, per the existing frontend convention.

### What moves, and what does not

**Moves in:** the playback concern in full — `/api/video-playback`,
`/api/video-playback-info`, and `#136`'s rewrap machinery (`_needs_remux`, `_remux_to_mp4`,
`_evict_cache`, `_source_digest`). Splitting "rewrap" from "transcode" across two modules
would defeat the purpose, since they are two answers to one question: *how do I make this
source playable?*

**Stays in `routes/video.py`:** `/api/video-frame` and `/api/video-timeline`. These serve
the *indexing* side — re-extracting a stored frame, describing hit positions — and are
consumed by the results list, not the player.

**Shared, not duplicated:** `_check_allowed_path` stays in `routes/_shared.py`. Path
containment is a security control for every file-serving endpoint in the app and must have
exactly one implementation.

> **Note for reviewers:** `CLAUDE.md` states that web endpoints live in
> `web/routes/` with an APIRouter per topic. This subsystem is a deliberate,
> operator-approved deviation for cohesion, and it keeps the convention *internally* —
> `video_playback/routes.py` is still one APIRouter per topic. Do not "correct" it back
> without re-reading this section. `CLAUDE.md` should be updated when phase 1 lands.

---

## 11. Implementation phases

Each phase is independently shippable and independently reviewable.

0. **Carve the subsystem** — create `video_playback/`, move the existing playback code in
   unchanged, rewire the single import in `app.py`. Pure refactor: the suite must stay at
   its current bar with no behavioural change, which is what makes the following phases
   readable diffs rather than mixed move-and-modify noise.
1. **Digest fix** (§8) — look up `video_hash`, stop hashing GB files on the request path.
2. **Download original** (§9) — small, self-contained, immediately useful.
3. **Codec detection** — `_needs_transcode()` allowlist (H.264 8-bit, VP8, VP9, AV1);
   `/api/video-playback-info` reports `mode: "transcode"` with a reason. No encoding yet.
4. **Segment engine** — capability probe, encoder selection, single-segment transcode,
   keyframe/timestamp alignment pinned by tests against a real HDR sample. **The rotation
   regression from §3 gets an explicit test.**
5. **Playlist + cache** — virtual m3u8, per-segment locks, TTL sweep, whole-video LRU.
6. **Player + prefetch** — vendored `hls.js`, sliding window, seek-cancels-prefetch.
7. **Label** — extend the playback label with mode, reason, tone-map operator, sha256.

---

## 12. Decisions recorded (do not re-litigate)

Made by the operator during the 2026-08-13 design session:

- Server-side **codec allowlist** decides the mode, not browser-advertised capability —
  because the advertised capability is demonstrably wrong on the operator's own Chrome.
- **fMP4/CMAF** segments, not MPEG-TS.
- **Whole-video LRU** for the ceiling, **plus** a segment TTL; distinct roles per §6.
- **Eager first 1–2 segments** on an explicit prepare action, then automatic sliding
  prefetch — not fully automatic on open, and not manual for every segment.
- **Must work with and without a GPU**, selected by runtime probe.
- **Download original** is offered alongside playback.

## 13. Open questions

1. Segment duration. 6 s is the HLS convention and what §3 measured; shorter improves
   seek responsiveness and worsens per-segment overhead. Wants one measurement at 2 s and
   4 s on the real server before being fixed.
2. Concurrency ceiling for the worker pool — how many analysts a server should serve at
   once is a deployment question, not a code one, but it needs a default.
3. Whether an audio-only fallback is worth having for sources whose video stream cannot be
   decoded at all.
