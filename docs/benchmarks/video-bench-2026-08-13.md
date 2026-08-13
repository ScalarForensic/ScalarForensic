# Video playback §14 benchmarks — 2026-08-13

Benchmarker: `scalarforensic-csm-b1`. Repo `main` at `43a7222`, clean. Raw
`/usr/bin/time -v` logs and the concurrency driver script are in the bench
scratch dir (not under `data/`, not committed):
`/tmp/claude-1000/-home-user01-Schreibtisch-gitea-ScalarForensic/098f80ee-3b25-494d-bd9b-31bc15a8b965/scratchpad/video-bench/`.
Large intermediate `.mov`/`.mp4` files were deleted after measurement to
reclaim ~14 GB; the `.log`/`.txt` files (one per run) are kept as the
methodology trail.

## Hardware — stated minimum floor

The operator has ruled this machine **is** the deployment target, not a
stand-in for one. Its spec is therefore recorded as the floor, not an
extrapolation:

| | |
|---|---|
| CPU | AMD Ryzen 9 9900X3D, 12 cores / 24 threads |
| RAM | 91 GiB total (48 GiB available at measurement time) |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16380 MiB VRAM, driver 595.84, compute 8.9 |
| Kernel | Linux 7.0.0-28-generic x86_64 |
| ffmpeg | 6.1.1-3ubuntu5 (`--enable-libzimg`, `--enable-libx264`, `--enable-libx265`, NVENC path present) |

This is the same host and ffmpeg build §3 used. No other hardware was tested;
there is no separate "floor" figure to derive — this machine's numbers below
**are** the floor.

**Box was not perfectly idle.** Two other Claude sessions (`llm-browser-com-c3/4/5`)
run headless Chrome and a Node test process on this host continuously; a
long-running `bitcoind` process holds a steady ~5% CPU. `uptime` 1-minute load
was checked before each phase (0.3–0.5 before starting; it climbs to the
job's own contribution during a run and decays back after). No other
CPU/GPU-heavy job was observed active during any timed run — the ambient load
is background daemons in the low single-digit percent, not a competing
benchmark. This is disclosed per the dispatch rule rather than omitted; it
does not invalidate the concurrency numbers but means "idle" here is
"ordinary desktop background," not a bare-metal box with nothing else
installed.

## Sources — synthesised, not real 4K/multi-hour footage

No real 4K or multi-hour source exists in the corpus (§3 saw max 318 s). Per
the dispatch:

- **4K source**: `IMG_3274.MOV` (the §3 dev-corpus HDR sample, found under
  `/media/user01/.../Gitea_Backup/input_scalar/IMG_3274.MOV` — read-only,
  not under this repo's `data/`) upscaled 1920×1080 → 3840×2160 with
  `libx265 -crf 18 -preset fast`, HLG/bt2020 tags preserved, 133.869 s,
  matches §3's sample exactly (verified against ffprobe before use). This is
  a real HDR *frame content* source at 4K resolution — not synthetic colour —
  per the dispatch's instruction to prefer this over invented HDR.
- **Long source**: 100× concatenation of the same `IMG_3274.MOV` via
  `ffmpeg -f concat -c copy` (stream copy, no re-encode) → 13387.043 s
  (3 h 43 m). **This is a best-case index**: concat-copy produces a single
  dense, undamaged keyframe/index structure. It does **not** exercise
  long-GOP, VFR, or damaged-index behaviour, and §3.2's caveat about those
  cases is *not* settled by this measurement — only the offset-proportionality
  question is.

## 1. 4K encode rates (§3.1 pipeline table, repeated at 2160p)

**Method**: single job, no concurrency, `/usr/bin/time -v` wall clock, 1 rep
each (single-sample — recorded as such, not averaged from repeats). Input:
the 4K HDR source above (133.869 s). Tone-map chain identical to §3:
`zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p`.
libx264/nvenc preset not specified in §3; used `medium` (libx264) and `p4`
(nvenc) here — record this if the number is ever compared apples-to-apples
against §3's 1080p figures.

| Pipeline | Command tail | Wall | Speed | Colour tags (output) |
|---|---|---|---|---|
| `scale_cuda` + `h264_nvenc` | `-hwaccel cuda -hwaccel_output_format cuda -vf scale_cuda=3840:2160:format=nv12 -c:v h264_nvenc -preset p4` | 30.27 s | **4.42×** | `bt2020`/HLG (wrong, as §3.1) |
| CPU tonemap + `h264_nvenc` | `-vf "<tonemap>" -c:v h264_nvenc -preset p4` | 106.19 s | **1.26×** | `bt709`/`bt709` (correct) |
| CPU tonemap + `libx264` | `-vf "<tonemap>" -c:v libx264 -preset medium` | 152.30 s | **0.879×** | `bt709`/`bt709` (correct) |
| CPU naive + `libx264` | `-vf format=yuv420p -c:v libx264 -preset medium` | 70.17 s | **1.907×** | `bt2020`/HLG (wrong, as §3.1) |

**Verdict, settling §3.4's prediction: the CPU tone-map + libx264 path — the
one decisions require for correct colour without a GPU — falls below
realtime at 4K (0.879×).** CPU tonemap + nvenc stays above realtime (1.26×)
but with a thin margin. Naive CPU (wrong colour, not used) is comfortably
above realtime. GPU-only stays fastest but keeps §3.1's known rotation defect
(not re-tested here — same code path, same defect mechanism, already pinned
by a required test per §14).

**One sample per pipeline.** Re-run before treating any single figure as
tight; the relative ordering (GPU-scale fastest, CPU-tonemap-x264 slowest and
sub-realtime) is the load-bearing finding, not the exact multiplier.

## 2. Long-source seek cost (§3.2, repeated on a multi-hour source)

**Method**: `-ss OFFSET -i <long source>` (input seeking, container-index
based) `-t 20` window, CPU tonemap + libx264 (the pipeline decided for
correct-colour chunk encodes), `/usr/bin/time -v`, 1 rep per offset, 7
offsets spanning the full 13387 s duration.

| Offset | Wall |
|---|---|
| 0 s | 5.65 s |
| 100 s | 5.54 s |
| 1800 s (30 min) | 5.55 s |
| 3600 s (1 h) | 5.91 s |
| 7200 s (2 h) | 5.44 s |
| 10800 s (3 h) | 5.97 s |
| 13300 s (near end, 3 h 41 m) | 5.42 s |

Range 5.42–5.97 s, no trend with offset — **confirms §3.2's finding holds out
to 3h41m of offset, on this index.**

**This is a best-case number and must not be read as a general guarantee.**
The source has a dense, undamaged, constant-GOP index by construction (100×
stream-copy concatenation of one 30 fps short-GOP file). §3.2's named risks —
damaged indexes, long-GOP, VFR — are untested here and remain open per §3.4.
A real multi-hour phone or security-camera capture is very likely to differ.

## 3. Concurrent-job scaling, k = 1/2/4/8

**Method**: k simultaneous chunk encodes launched together (`&` + `wait`),
each an independent ffmpeg process, CPU tonemap + libx264, 30 s chunk, 1080p,
distinct offsets per job (0, 500, 1000, ... s into the long source, no
overlap) to avoid page-cache sharing artefacts. Batch wall = wall clock from
launch of job 0 to completion of the last job. Per-job wall from each
process's own `/usr/bin/time -v`. **No `-threads` cap was set** — each ffmpeg
invocation uses libx264's own auto-threading (defaults to core count), so at
k≥2 jobs oversubscribe the 24 logical threads; this is realistic default
behaviour, not an isolated one-core-per-job model, and the throughput numbers
below already reflect that contention.

| k | Per-job wall (mean) | Per-job spread | Batch wall | Aggregate throughput (video-s encoded / wall-s) |
|---|---|---|---|---|
| 1 | 8.08 s | n/a (1 job) | 8.08 s | 3.71× |
| 2 | 16.35 s | 16.31–16.39 s | 16.40 s | 3.66× |
| 4 | 32.47 s | 32.29–32.67 s | 32.67 s | 3.67× |
| 8 | 67.34 s | 66.03–68.27 s | 68.28 s | **3.52×** |

**Aggregate throughput is flat (~3.5–3.7×) across k = 1 to 8 — it does not
increase with concurrency.** A single job already saturates most of the
box's cores via libx264's own threading; adding more concurrent jobs divides
the same aggregate throughput among more chunks rather than adding capacity.
The practical effect: **per-job latency scales almost linearly with k**
(8.08 s → 67.34 s, ~8.3× at k=8 vs k=1), because each job gets a shrinking
share of the same saturated CPU.

**This is the number that should size `SFN_VIDEO_MAX_WORKERS`, and the
finding argues for a low cap, not a high one.** §4.2's prefetch-depth-1
design assumes a 30 s 1080p chunk lands in ~6–10 s; that margin holds at k=1
(8.08 s, inside the assumed window) and roughly at k=2 (16.35 s, already over
it), but is blown by k=4 (32.47 s) and badly blown by k=8 (67.34 s). Recommend
`SFN_VIDEO_MAX_WORKERS=2` as the ceiling that keeps a queued chunk inside
§4.2's latency assumption on this hardware; higher settings buy no aggregate
throughput and directly cost prefetch margin.

## 4. First-play latency, 30 s chunk, 1080p vs 4K

**Method**: CPU tonemap + libx264 (the correct-colour pipeline), 30 s window,
3 reps each, `/usr/bin/time -v`, single job (no concurrency — isolates the
resolution effect from the k effect measured above).

| Resolution | Rep 1 | Rep 2 | Rep 3 | Mean |
|---|---|---|---|---|
| 1080p | 8.23 s | 8.17 s | 8.23 s | **8.21 s** |
| 4K (2160p) | 32.99 s | 33.02 s | 35.19 s | **33.73 s** |

**1080p sits inside §4.2's assumed 6–10 s margin (8.21 s mean, tight but
inside).** **4K blows it by ~4×** (33.73 s vs. the 6–10 s assumption). At 4K,
prefetch depth 1 cannot keep up with a 30 s chunk on this hardware under the
decided pipeline — the analyst would sit on a spinner well past chunk
boundary if the source is played at full 4K resolution.

Note: §16 already caps output resolution at 1080p by operator decision
(pending ledger item 3, "1080p is ok, full quality can be archived by
download"), so this 4K number is a **stress test of the assumption**, not a
number the shipped path is expected to hit under that ruling — but it
confirms §17 Q2's concern was correctly raised, and forecloses ever raising
the resolution cap without also revisiting chunk length.

## Summary for phase 4 gate

- 4K CPU-tonemap+libx264 (the correct-colour, no-GPU-required path): **0.879×, below realtime.**
- Long-source (3h41m, best-case index) seek cost: **flat, 5.4–6.0 s, confirms §3.2 out to multi-hour.**
- k=8 aggregate throughput: **3.52× video-seconds/wall-second — no gain over k=1's 3.71×.**
- Recommended `SFN_VIDEO_MAX_WORKERS`: **2** (keeps queued-chunk latency inside §4.2's 6–10 s assumption; higher k buys no throughput, only worse latency).
- 4K first-play latency (30 s chunk): **33.73 s mean — ~4× over §4.2's margin; confirms operator's 1080p-cap ruling is load-bearing, not optional.**
