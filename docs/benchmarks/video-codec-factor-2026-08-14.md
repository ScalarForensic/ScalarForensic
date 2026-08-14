# Video codec output-size factor, and the §4.3 yield residual — 2026-08-14

Benchmarker: `scalarforensic-csm-b2`. Repo `main` at `f2b07a7`, clean tree in a
dedicated worktree (`PYTHONPATH=$PWD/src`). Raw per-run JSON, the driver
scripts, the full 509-file corpus probe and the task-B logs are committed
alongside this report — `docs/benchmarks/video-codec-factor-2026-08-14-scripts/`
(`probe_all.py`, `run_bench.py`, `task_b.py`, `samples.py`, `results.jsonl`,
`corpus_probe.tsv`, `taskb_run.log`, `taskb_niced.log`) — not left on the
bench scratch dir's tmpfs, which does not survive a reboot and would have
quietly broken this report's own reproduction instructions. Encoded `.mp4`
outputs were deleted immediately after each run's size was recorded, to
avoid accumulating case-derived bytes on disk; only filenames/metadata from
the corpus are in the committed files, never footage bytes.

## Hardware and build

Same host and ffmpeg build as `docs/benchmarks/video-bench-2026-08-13.md` and
spec §3/§3.5 — the operator's ruling that this machine **is** the deployment
floor, not a stand-in, still applies; no other hardware was tested.

| | |
|---|---|
| CPU | AMD Ryzen 9 9900X3D, 24 threads (`nproc`) |
| RAM | 91 GiB total, 28–35 GiB available during these runs |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16380 MiB, driver 595.84, compute 8.9 |
| Kernel | Linux 7.0.0-28-generic x86_64 |
| ffmpeg | 6.1.1-3ubuntu5 (same build as §3/§3.5) |

Box was not perfectly idle (ordinary desktop background load, consistent with
§3.5's disclosure) for task A — this does not affect byte counts, only wall
time, which is recorded but not the deliverable. Task B's box state is
recorded separately in §4, since latency there is exactly what is measured.

## Corpus

The operator's real dev corpus at
`/media/user01/SAM_870_SATA/Gitea_Backup/input_scalar/` (read-only; not
copied into this repo — corpus footage is case material and is never
committed). 509 video files (`.MOV`/`.MP4`), matching spec §3's corpus size
exactly, so this is the same population §3 sampled from, probed in full:

- 494 HEVC / 15 H.264
- HEVC: 74 ten-bit (`yuv420p10le`, `arib-std-b67` transfer — HDR) / 420
  eight-bit (`yuvj420p` or `yuv420p`, `bt709`)
- Resolutions: 336 files at 1440×1080, 78 at 1920×1080, 61 at 1920×1440
  (portrait-native, reported pre-rotation), 17 at 1080×1920. **No source in
  this corpus exceeds 1920 on its long edge** — there is no real 4K sample
  here, matching §3.4/§3.5's own note that no real 4K or multi-hour source
  exists in the corpus.
- 151 of 509 report a non-integer `r_frame_rate` (e.g. `65/3`, `50/3`,
  `89/3`) from `ffprobe`, consistent with VFR capture, though `r_frame_rate`
  alone does not distinguish real VFR from a container-timing artifact on a
  very short clip — see §3's gap below.

Full per-file probe (codec, pixel format, resolution, duration, bit rate,
transfer/primaries) is `corpus_probe.tsv` in the scripts dir; `probe_all.py`
is the exact script that produced it (one `ffprobe -show_entries` call per
file, `-select_streams v:0`).

---

## 1. Method — task A

For each sampled source, the driver (`run_bench.py`) calls the repository's
own functions directly rather than reimplementing them, so the numbers are
about the real code, not a parallel model of it:

- `scalar_forensic.video_playback.codecs._stream_report()` — the exact probe
  `check_ceiling()` uses for `duration_ms`, `bit_rate`, `video_height`.
- `scalar_forensic.video_playback.cache.estimate_full_output_bytes()` — the
  §6.3 estimator, unmodified, called with `output_height=1080` (the
  `SFN_VIDEO_OUTPUT_HEIGHT` default).
- `scalar_forensic.video_playback.capability.select()` — builds the real
  `Pipeline` for a `Capability` with `encoder="libx264", hwaccel="none"`
  (CPU row) and a second with `encoder="h264_nvenc", hwaccel="cuda"` (GPU
  row), each with `tonemap_ok=True` so an HDR source takes the real §3.1
  tone-map chain rather than being refused. This is *not* the probe path
  (§8) — it is a hand-built `Capability` per row, so both rows always run
  regardless of what this specific host's probe would pick.
- `scalar_forensic.video_playback.encode.build_command()` — the exact argv,
  called with `start=None, duration=None` (full-file encode, matching what
  the §4.3 job runs), `has_audio=True`.

Each command is run once via `subprocess.run`, wall-clock timed but **not
concurrent with anything else task A cares about** — output byte count does
not depend on box load. `output_bytes / estimate_bytes` is recorded, the
output file is deleted immediately (freed, not kept), and the process moves
to the next sample. One rep per (source, row) cell — see the n stated below
before treating any single ratio as tight, same caveat §3.4/§3.5 already
carry for this repository's other single-sample cells.

Example command pair (`IMG_2668.MOV`, 68.4 s HEVC 10-bit HDR source):

```
# CPU row
ffmpeg -nostdin -hide_banner -v error -i IMG_2668.MOV \
  -vf "scale=-2:'min(ih,1080)',zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,\
tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" \
  -c:v libx264 -preset medium -crf 23 \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -c:a aac -b:a 128k -movflags +faststart -fflags +genpts -f mp4 -y out.mp4

# GPU row — same filter chain, different encoder/rate-control
  -c:v h264_nvenc -preset p4 -rc vbr -cq 23 ...
```

## 2. Result — task A, the deliverable ratio

`output_bytes / estimate_full_output_bytes()`, per class, CPU and GPU rows
separately (they do not produce the same bytes — GPU/`nvenc -cq 23` runs
consistently larger than CPU/`libx264 -crf 23` at the same nominal quality
setting, by roughly 1.5–2×, across every class below):

| Class | Row | n | min | median | max |
|---|---|---|---|---|---|
| HEVC 10-bit (HDR) | CPU | 8 | 0.125 | 0.301 | 0.356 |
| HEVC 10-bit (HDR) | GPU | 8 | 0.266 | 0.471 | 0.848 |
| HEVC 8-bit | CPU | 8 | 0.203 | 0.886 | 1.536 |
| HEVC 8-bit | GPU | 8 | 0.365 | 1.409 | 1.938 |
| H.264 1080p | CPU | 4 | 0.190 | 0.309 | 0.340 |
| H.264 1080p | GPU | 4 | 0.388 | 0.563 | 0.682 |
| VFR (odd `r_frame_rate`, best-effort) | CPU | 3 | 0.261 | 0.451 | 0.485 |
| VFR (odd `r_frame_rate`, best-effort) | GPU | 3 | 0.317 | 0.708 | 0.963 |

Full per-file rows (source codec/pix_fmt/resolution/duration/bytes, output
bytes, estimate bytes, ratio) are in `results.jsonl`; the table above is
computed straight from it, no filtering beyond excluding failed runs (there
were none — every sampled encode completed and produced non-empty output).

**This measurement does not confirm §6.3's stated direction of error.** The
spec text (§6.3, §4.3) reasons that a CRF-23 H.264 re-encode of a 10-bit HEVC
source is "usually larger than its source," and since the estimate at
1080p-in/1080p-out is approximately the source's own bitrate×duration (area
ratio 1.0, no downscale in this corpus — every sample here is ≤1920×1080), a
larger-than-source output was expected to push the ratio above 1. It does not,
for the CPU row on either HEVC class, and does so only occasionally on the
GPU row:

- **HEVC 10-bit (HDR), CPU: every one of 8 samples is *under* the estimate**
  (max 0.356). The tone-map-to-BT.709 pass that §3.1 costs at ~2.3× appears
  to also *shed* bits relative to the source's own 10-bit HEVC encode — the
  phone's HDR capture is evidently encoded at much higher visual fidelity
  (closer to visually-lossless) than CRF 23 targets, so despite the codec
  disadvantage (H.264 vs HEVC at the same resolution), CRF 23 comes out
  smaller, not larger, on every sample measured.
- **HEVC 8-bit, CPU: median is close to 1 (0.886) but still under it on 6 of
  8 samples**; the two that exceed it (`IMG_1637.MOV` 1.536, `IMG_1964.MOV`
  0.997) are both short clips (0.9–3.0 s) with correspondingly low source
  bitrate, where CRF 23's roughly-fixed quality target does exceed a source
  that was itself lightly compressed.
- **GPU rows run larger across every class** (`h264_nvenc -cq 23` is
  measurably less bit-efficient than `libx264 -crf 23` at the same nominal
  CQ/CRF number, consistent with general encoder-quality literature, not
  specific to this corpus) — and the GPU row for HEVC 8-bit *does* cross 1 at
  the median (1.409), meaning **the estimate genuinely runs low, but only for
  the GPU row on already-lightly-compressed 8-bit sources**, not for the
  10-bit HDR corpus the spec's own text names as the risk case.
- **The one class where the spec's predicted direction holds cleanly is
  HEVC 8-bit / GPU.** Every other class/row combination in this sample stays
  under 1, several substantially so (HEVC 10-bit CPU: as low as 0.125 — the
  estimate can overshoot the real output by 8×).

**Named implication for §6.3, stated as a finding, not a fix (this is a
measurement dispatch, code is out of scope):** the uncalibrated estimate is
not uniformly a low-side risk. On this sample it is frequently a *high-side*
over-estimate for the CPU pipeline, which is the pipeline `select()` picks
whenever the host has no working GPU encoder — i.e., the estimate can refuse
a full-video job (§6.3's `refused` verdict) for a file whose real CPU output
would in fact have fit. The GPU pipeline is closer to the spec's assumed
direction but the single class that crosses 1 with a wide margin is HEVC
8-bit, not HEVC 10-bit HDR.

**What this does not settle.** n is 3–8 per cell — small enough that a
single outlier moves the median noticeably (see `IMG_1637.MOV` above). The
ordering (GPU rows larger than CPU rows; HEVC 10-bit CPU rows the smallest
relative to estimate) is the load-bearing finding; the exact ratios are not
tight enough to derive a single multiplicative "codec factor" constant from
this alone — see §5.

## 3. Classes not sampled

- **4K.** No real 4K source exists in this 509-file corpus (§3.4/§3.5 already
  established this for the encode-*rate* measurement; it holds here too, for
  the same corpus). §3.5's 4K row was produced by upscaling a 1080p source
  with `libx265 -crf 18` to synthesise 4K *pixel dimensions* for a rate
  measurement. That method is unsuitable for an output-*size* measurement:
  upscaling invents no detail, but downscaling that invented (interpolated,
  not native) detail back to 1080p for the §4.3 pipeline would not exercise
  the bit density a real 4K capture has, and reports a size ratio that is an
  artifact of the synthesis, not evidence about real 4K sources. Rather than
  publish a number that looks like a class in the table above but measures
  something else, this class is named as **unmeasurable from this
  repository's corpus**, not filled in.
- **Long-GOP.** Not distinguishable from the container-level probe used to
  build `corpus_probe.tsv` (no per-file GOP structure was extracted — that
  would need a full frame-type scan, out of scope for an output-size
  measurement). Named as unmeasured, same as §3.4 already states for seek
  cost.
- **VFR** is sampled at best-effort: `corpus_probe.tsv`'s `r_frame_rate`
  column is `ffprobe`'s guess, not a direct variable-frame-rate flag, and it
  cannot separate genuine VFR capture from a short-clip container-timing
  artifact. The 3 samples used all report frame counts consistent with a low
  but plausible frame rate (`65/3` ≈ 21.7 fps, `50/3` ≈ 16.7 fps, `89/3` ≈
  29.7 fps) rather than the clearly-spurious `600/1`/`240/1` values seen
  elsewhere in the corpus (excluded as probable timing artifacts, not real
  content). No long-duration VFR source exists in the corpus to test whether
  VFR output size behaves differently at scale.

---

## 4. Task B — the §4.3 yield residual

**Settings, from `scalarforensic-com-c17` directly (phase 7 implementer):**
`SFN_VIDEO_JOB_NICE=10` (applied with `setpriority(PRIO_PGRP)` on a
`start_new_session` child, so every ffmpeg thread carries it) and
`SFN_VIDEO_JOB_THREADS=0` → ffmpeg `-threads max(1, os.cpu_count()//2)` = 12
on this 24-thread host. c17: "the knobs bite on CPU/libx264 only — chunk
encodes take neither knob. Measure both pipelines: NVENC residual is
unmeasured and I claim nothing about it." This report accordingly measures
the CPU pipeline only.

**Method.** The real §4.3 competitor holds a worker for ~51 min;
this corpus has no source that long, and synthesizing a genuinely
tens-of-minutes source is expensive on this host's tight `/tmp` (tmpfs,
shared with other sessions — 6.3 GiB free at the time this was designed).
`scalarforensic-com-m6` flagged (correctly) that a short-lived competitor
that exits mid-measurement doesn't test the steady state §4.3 describes.
Resolution: `task_b.py`'s `Sustainer` relaunches the full-copy job the
instant one run exits, for the whole span of the chunk-encode reps, so a
competitor is running at every timed rep's start — a **synthesized sustained
competitor**, explicitly not claimed equivalent to one continuous 51-minute
job. Full-job source: an 11.4-minute concat-copy of `IMG_2668.MOV`
(§3.5's own method for producing a long source from stream-copy
concatenation). Chunk source: `IMG_2668.MOV` directly, 30 s window at
alternating starts (`10.0`/`30.0` s, both safely inside its 68.4 s duration).

In practice, at CPU/libx264 with a live chunk-encode competing for the same
core budget, one full-job run took the whole span of both 6-rep measurements
(0 restarts recorded in either run — `Sustainer.restarts` stayed 0
throughout), so both runs in fact tested against the *same continuous* run
of the 11.4-minute source, not a mid-run handoff. The relaunch design is
still the correct one to have built (a shorter or faster-finishing full job
would have needed it), and is kept in the report because it is what made
`competitor_running_at_start=True` a checked invariant rather than an
assumption.

**Box state during the measured runs.** Window coordinated by
`scalarforensic-com-m6`: `c17` at its retire nudge, running nothing;
`c18` idle, its work merged (#173); `m6` held its own pending pytest run
until this reported back. `/tmp` (tmpfs) at 12 GiB free after `m6` reclaimed
~21 GiB from retired agents' worktree venvs — noted because a full tmpfs
mid-run would look exactly like a slow encode, and this run did not hit
that. Non-fleet load present throughout, as it is on this host at all
times (§3.5): `bitcoind` ~4.5% steady, Xorg, an NX session. `uptime` load
average during the runs: 16.56/16.03/8.41 at start, 28.71/23.36/13.30 by the
end of the niced run — both figures reflect this measurement's own two
concurrent ffmpeg processes (the sustained competitor plus the timed chunk
encode) at 24 threads, not third-party load; "quiet" here means no
suite and no other session's encode, not load 0.

**Discarded first attempt.** Before the coordinated window below, a first
unniced rep was run and **discarded**, for reasons recorded rather than
silently dropped:

1. `scalarforensic-com-m6` measured the box directly during that window and
   found `pytest -q` running at 101% CPU (started 08:42:31, gone by
   08:42:45 — short bursts, not continuous) with both `c17` and `c18`
   mid-implementation and expected to keep running the suite repeatedly.
   Load average 1.40/2.78/1.76 — "this box has not been quiet recently."
   Per the dispatch, a coder's session not showing `busy` is not sufficient
   evidence of a clean box, and this confirms why: the contamination here
   was a burst that would not necessarily show as sustained "busy" state.
2. Independent of contention, the first script draft had a bug: chunk start
   offsets (`10, 50, 90, 130, 170` s) ran past the 68.4 s chunk source's own
   duration for 3 of 5 reps, so those reps hit EOF early and returned
   near-instant "wall" times (0.14–0.17 s) that measure nothing. Fixed
   before any counted run: chunk starts now alternate `10.0`/`30.0` s, both
   leaving a full 30 s window inside the source, and `chunk_cmd()` asserts
   the invariant so a future regression fails loudly instead of producing a
   silently-short chunk.
3. On re-inspection, that same first draft's competitor (an 11.4-minute
   source finishing in ~145 s at CPU speed) would have exited partway
   through a 5-rep, ~40 s run even without the box-state issue — the
   `Sustainer` relaunch design above replaces the single-Popen design that
   produced the discarded rep, for this reason as well as m6's.

Both fixes were self-tested under the current (noisy) box before requesting
a window — two short runs (6 reps each) against small synthetic sources
(137 s), confirming the assertion holds and `competitor_running_at_start` is
`True` at every rep including across a mid-run restart. Neither self-test is
counted as data; both used disposable sources, not the corpus files this
report's numbers are drawn from.

Task A (§1–§3, all byte counts) was unaffected by any of this — output size
does not depend on what else the box was doing — and was measured before
this contamination was discovered.

### Result

Chunk-encode wall time (30 s window, **CPU/libx264 pipeline only**,
tone-mapped 10-bit HDR chunk source), 6 reps each, under a **synthesized
sustained competitor** (11.4 min concat-copy source, relaunched across reps
by `Sustainer` — not one continuous 51-minute job; see method above):

| Condition | Pipeline | Competitor | n | min | median | max |
|---|---|---|---|---|---|---|
| Unniced, unthreaded | CPU/libx264 | synthesized sustained (11.4 min concat) | 6 | 18.19 s | 18.31 s | 18.84 s |
| `nice 10`, `-threads 12` | CPU/libx264 | synthesized sustained (11.4 min concat) | 6 | 16.64 s | 16.83 s | 17.15 s |

GPU/`h264_nvenc` residual: **not measured, no claim made** (§ above — the
knobs c17 landed do not touch the encoder block or its driver queue).

Full per-rep data (`competitor_running_at_start` checked true on all 12
reps, `Sustainer.restarts` counted): `taskb_run.log` (unniced) and
`taskb_niced.log` (niced), committed alongside the scripts.

**§4.3's own projection was optimistic — say so plainly.** §4.3 extrapolates
~16.35 s for k=2 from §3.5's throughput-halving model (8.21 s baseline ×2).
This measurement's *unniced* median is **18.31 s, above that projection**,
under a real (if synthesized) competing full-job encode rather than an
extrapolation from an aggregate-throughput ratio. §3.5's baseline itself was
8.21 s with no competitor; this report's 18.31 s is the number that should
replace ~16.35 s wherever §4.3 currently cites it.

**The residual does not close the gap §4.2 depends on.** Both conditions sit
well outside the 6–10 s window the double-buffered chunk swap assumes.
`nice 10` + `-threads 12` reduces the median by **~8%** (18.31 s → 16.83 s) —
real (the two conditions' min/max bands do not overlap) and free, but nowhere
near enough to re-enter the 6–10 s window. **The yield remedy shrinks the
penalty; it does not solve it.** This is a correction to the spec's §4.3
framing, which proposed "yield" and "accept and disclose" as if either
alone might suffice: on this measurement, yield is worth keeping (it is a
free ~8%) but disclosure is not the belt-and-braces half of the remedy — it
is the part carrying the actual weight, since the analyst's chunk playback
is measurably degraded either way. At this depth (`nice 10`, half the CPUs
capped on the competitor) the chunk encode is still sharing a saturated
box — consistent with §3.5's concurrency finding that one job already
saturates this host via libx264's own threading, so a competitor that still
gets half the CPUs remains a heavy one. A deeper cap (lower `-threads`, or a
higher `nice` delta) was not tested — this report measures the settings
`c17` actually landed, not a sweep.

**Scope, per c17 directly:** `nice`/`-threads` apply to the CPU/libx264 full
job only. Whether the GPU pipeline (`h264_nvenc`) suffers comparable
contention when a full-video job runs on the encoder block is **not
measured here and this report makes no claim about it** — the encoder
block and its driver queue are not touched by either knob, so a CPU-side
remedy provides no reason to expect the same improvement on that path.

---

## 5. Review

**Scope discipline.** Task A measured exactly what §6.3's estimator needs —
`output_bytes / estimate_full_output_bytes()` — using the repository's own
estimator and pipeline-selection code rather than a reimplementation, so a
future drift between this report and the code it describes is at least
detectable (re-run `docs/benchmarks/video-codec-factor-2026-08-14-scripts/run_bench.py`,
`PYTHONPATH=$PWD/src`, against the current `cache.py`/`capability.py`/
`encode.py` and the operator's corpus). No code under test was edited.

**What the numbers support.** With n=3–8 per class, this report should not be
read as delivering a single "the codec factor is X" constant — it explicitly
does not, per §2's last paragraph, and the dispatch asked for n and spread
to be stated rather than a false-precision point estimate. What it does
support, with reasonable confidence given the sample: the CPU pipeline
(the one selected on any host without a working GPU encoder, which per §8 is
the fallback path, not a rare one) undershoots the §6.3 estimate on both HEVC
classes sampled, sometimes by close to an order of magnitude. That is the
opposite of the risk direction the spec text currently states, and readers
of §6.3/§4.3 should not treat "the estimate runs low" as settled without
reading this report's caveats.

**Known threats to validity, beyond the stated ns.** (1) All samples are
short-to-medium duration (0.9 s–68.4 s); the corpus has nothing longer, so
whether the ratio drifts with duration past a couple of minutes is
untested — extrapolating these ratios to the ~4-hour job §4.3 describes is
exactly the kind of leap this report's own §3 refuses to make for 4K. (2)
Sample selection within each class was duration-stratified (evenly spaced
across the sorted-by-duration list) rather than random, which avoids
clustering all samples at the corpus's dominant 1–3 s mode but is not a
statistically drawn sample either. (3) `-crf 23`/`-cq 23` are the values
already fixed in `capability.py`'s `_RATE_CONTROL`; this report measures
their actual output size, it does not evaluate whether 23 is the right
choice.

**Task B.** Measured against the settings `c17` stated directly, under a
coordinated clean window with box state recorded (§4). n=6 per condition is
thin for a ~8% delta between medians (18.31 s vs 16.83 s) — real, in the
sense both conditions are consistently far from each other's range (no
overlap between the two `min`/`max` bands), but not a tight number. The
sustained-competitor design is a deliberate departure from a literal
51-minute job, disclosed as such rather than presented as equivalent; a
reviewer who needs the literal-51-minute-job number should treat this as a
lower bound on how much of the contention `nice`+`-threads` removes; a
harsher test (a competitor that never finishes and truly runs the whole
measurement) was not attempted, and might show a larger or smaller
improvement — this report does not know which. The GPU-path residual is
named as unmeasured, per c17, rather than guessed from the CPU numbers.

**One thing this report got wrong before catching it.** The first attempt at
both tasks shipped with bugs that would have produced a plausible-looking
but false number if not caught: task A's estimator call was correct from the
start, but task B's first script both ran through a coder-contaminated
window undetected by "not busy" alone, and had a chunk-start arithmetic bug
producing near-zero wall times that read as suspiciously good in isolation.
Both are recorded in §4 rather than silently fixed, per the dispatch's own
instruction that a named gap is worth more than a number covering it.
