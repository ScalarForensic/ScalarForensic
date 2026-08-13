# Benchmark: face detection folded into the batch loop (PR #118)

2026-08-13 · scalarforensic-cfm-b1 · before/after ingest benchmark of `bb4cf41`
(perf(faces): fold face detection into the batch loop's shared decode).

## Result

| Arm | Code | Run 1 | Run 2 | Mean wall | Max RSS |
|---|---|---|---|---|---|
| before | `0df8035` (pre-#118, git worktree) | 183.66 s | 182.74 s | 183.20 s | 9.93 / 9.82 GiB |
| after | `bb4cf41` (main; tip `9e22ed1` is docs-only) | 164.81 s | 171.34 s | 168.08 s | 5.75 / 5.71 GiB |

**Wall time: −15.1 s mean, −8.3 %** (paired run 1: −10.3 %, run 2: −6.2 %).
**Peak memory: −42 %** (10.4 GB → 6.0 GB max RSS) — the after arm no longer holds a
second decode of the batch for face detection.

Correctness cross-check: all four runs produced identical counts — 763 images +
457 video frames indexed (1,220 items per embedder, 0 failed), faces
1,010 detected / 254 comparable / 319 retained for review (170 confidence, 54 pose,
20 sharpness, 75 size) / 437 rejected (215 confidence, 222 size).

## Subset

814 files at `/media/user01/SAM_870_SATA/Gitea_Backup/input_scalar_bench10`, built by
`scripts/make_benchmark_subset.py` (see `docs/fleet/benchmark-subset.md`).
Manifest: seed `20260813`, fraction `0.1`, strata heic 310/3098, jpg 218/2182,
other_image 235/2351, video 51/509.

## Methodology

- Hardware: RTX 4060 Ti (CUDA, torch 2.11.0+cu128), subset on SATA SSD. GPU idle
  (0 % util) before and between runs; no campaign ingest running.
- Isolation: throwaway Qdrant container on `127.0.0.1:6335` (deleted after);
  campaign instance on 6333 never touched. Each run got a fresh collection pair
  (`bench_<run>` / `bench_<run>_faces`), fresh hash cache, fresh thumbnail/frame/face
  store dirs — so nothing was ever deduplicated away. First-activation prompt answered
  `bench-20260813` per run.
- The before arm ran from a git worktree at `0df8035` with its own `uv sync --frozen`
  venv (no dependency changes in #118) and the shared checkout's `models/` symlinked in.
  The after arm ran from the shared checkout on `main`.
- The campaign `.env` is cwd-resolved, so the worktree would not see it while the
  shared checkout would; to keep arms identical, every relevant `SFN_*` variable was
  pinned explicitly on the command line for both arms (campaign-matching values:
  `SFN_BATCH_SIZE=128`, `SFN_DEVICE=auto`, `SFN_NORMALIZE_SIZE=512`,
  `SFN_SSCD_N_CROPS=5`, `SFN_DUPLICATE_CHECK_MODE=hash`, `SFN_EXTRACT_EXIF=true`,
  `SFN_VIDEO_FPS=1.0`, `SFN_VIDEO_MAX_FRAMES=500`, `SFN_FACE_CROP_DILATION=0.25`,
  `SFN_EXAMINER_ID=bench`, absolute model paths).
- Warm/cold handling: the OS page cache was primed over the whole subset before any
  timed run (`cat * > /dev/null`), and arms were interleaved
  (before_r1 → after_r1 → before_r2 → after_r2) so neither arm alone paid residual
  cold effects. Model load is included in every run's wall time (fresh process each run).
- Timing: `/usr/bin/time -v` around the full command.

Command shape (per run, `$D` = fresh per-run scratch dir):

```
echo bench-20260813 | \
SFN_QDRANT_URL=http://localhost:6335 SFN_COLLECTION=bench_<run> \
SFN_FACE_COLLECTION=bench_<run>_faces <pinned SFN_* as above> \
SFN_THUMBNAIL_DIR=$D/thumbnails SFN_FRAME_STORE_DIR=$D/frames \
SFN_FACE_STORE_DIR=$D/faces SFN_HASH_CACHE_PATH=$D/hash_cache.db \
SFN_FACES_ENABLED=true \
/usr/bin/time -v ./run.sh sfn /media/user01/SAM_870_SATA/Gitea_Backup/input_scalar_bench10 \
  --dino --sscd --faces --report $D/report.csv
```

## Raw wall times

| Run | Order | Elapsed | Max RSS (KB) |
|---|---|---|---|
| before_r1 | 1 | 3:03.66 | 10,415,028 |
| after_r1 | 2 | 2:44.81 | 6,027,924 |
| before_r2 | 3 | 3:02.74 | 10,296,632 |
| after_r2 | 4 | 2:51.34 | 5,988,508 |

Full logs, `time -v` output and per-run ingestion CSVs were kept in the session
scratchpad (`runs/<run>/{log.txt,time.txt,report.csv}`); bench Qdrant collections and
the throwaway container were deleted after the runs.
