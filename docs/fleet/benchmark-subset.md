# Benchmark subset

`scripts/make_benchmark_subset.py` builds a deterministic ~10% stratified sample of a
media corpus so pipeline speed is measurable in ~3 minutes instead of ~35. Strata are
`heic`, `jpg`, `other_image` (png/webp/gif/bmp/tiff) and `video` (mov/mp4/m4v/avi/mkv),
sampled independently so the subset preserves the corpus' format proportions. Files are
**copied**, not symlinked — decode benchmarks need real I/O.

## Building the subset

```
uv run python scripts/make_benchmark_subset.py \
    /media/user01/SAM_870_SATA/Gitea_Backup/input_scalar \
    /media/user01/SAM_870_SATA/Gitea_Backup/input_scalar_bench10
```

The destination **must be a sibling of the corpus, never inside it** — the campaign
scanner walks `input_scalar` recursively and would ingest the subset as new evidence.
The script refuses a destination inside the source and refuses a non-empty destination.
The source directory is only ever read.

Determinism: the default seed is fixed (`20260813`), the candidate list is sorted
before sampling, and each stratum has its own derived RNG — so the same corpus + same
seed always selects the byte-identical set of files, and growth in one stratum never
changes another stratum's selection. Override with `--seed N` / `--fraction F` if you
need a different sample.

Every run writes `benchmark_subset_manifest.json` into the subset directory: seed,
fraction, per-stratum totals and selected counts, and the full list of chosen relative
paths. Quote the manifest when reporting benchmark numbers.

## Running a bench ingest against the subset

**Never edit the campaign `.env`.** Override the collection names on the command line
only, and give the bench run its own collections so it cannot touch campaign data:

```
SFN_COLLECTION=bench_subset SFN_FACE_COLLECTION=bench_subset_faces \
    ./run.sh sfn /media/user01/SAM_870_SATA/Gitea_Backup/input_scalar_bench10 --dino --sscd
```

**Bench runs need a THROWAWAY Qdrant — never `localhost:6333` while the campaign
runs.** The campaign instance is live evidence storage; even a separately named
collection on it competes for its I/O and memory and skews both the benchmark and the
campaign. Start a disposable instance on another port and point the run at it:

```
docker run --rm -d -p 6444:6333 -v /tmp/qdrant-bench:/qdrant/storage qdrant/qdrant
SFN_QDRANT_URL=http://localhost:6444 \
    SFN_COLLECTION=bench_subset SFN_FACE_COLLECTION=bench_subset_faces \
    ./run.sh sfn /media/user01/SAM_870_SATA/Gitea_Backup/input_scalar_bench10 --dino --sscd
```

Tear the container down (and delete `/tmp/qdrant-bench`) when done; bench collections
are disposable by design.

## Tests

`tests/test_benchmark_subset.py` covers stratification, determinism, and the
safety refusals against a tmp-dir fixture — hermetic, no real corpus access.
Remember `--no-cov` when running the file on its own:

```
uv run pytest tests/test_benchmark_subset.py -q --no-cov
```
