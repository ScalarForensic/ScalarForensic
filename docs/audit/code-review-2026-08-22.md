# Code audit — bugs and logic flaws

**Date:** 2026-08-22 · **Tree:** clean at `d9505e0` (origin/main), no code changed by this
audit. · **Method:** line-by-line read of the whole `video_playback/` package plus
`web/routes/_shared.py`, with two traced sweeps over the core indexing modules
(`indexer/embedder/video/discovery/config/safeguards/calibration/tags/manifest/cli`) and the
faces package (`store/indexing/chips/detect/embed/routes/faces_query/faces_stats`). Every
finding below was confirmed against the surrounding code, not inferred from names.

Severity is about forensic consequence first, likelihood second.

---

## HIGH

### H1 — Face search/compare 500s when probing by stored point id without a prior query-faces run

`src/scalar_forensic/web/routes/faces.py:751`, `:827` · root cause `src/scalar_forensic/faces/store.py:169`

`FileEntry.query_faces_cfg` starts life as `None` (`web/session.py:66`). Both endpoints accept
probe origins that do not require it — `/api/faces/search` takes standalone `point_ids`
("probes come from two origins", faces.py:727) and `/api/faces/compare` works from an
`image_hash` alone. When a file was uploaded but `query-faces` never ran for it,
`_compat_block(store, cfg)` calls `FaceStore.check_compat(None)`:

```python
hard_mismatches = [
    f"{f}: collection has {meta[f]!r}, this run has {getattr(cfg, f)!r}"
    for f in _HARD_FIELDS
    if f in meta and meta[f] != getattr(cfg, f)
]
```

`getattr(None, "embedder_model_hash")` raises **AttributeError**. `_compat_block` catches only
`ValueError` (faces.py:702–705), so it escapes as an unhandled 500 — on every populated
collection (an empty meta point returns early at store.py:158–159 and masks the bug). The test
suite misses it because `tests/faces/test_face_probes_compare.py:32` always injects a cfg.

*Failure:* examiner pastes a known observation's id to find twins → 500, no audit entry.
The feature is dead on the documented input combination.

### H2 — `sfn-faces purge --media <video sha256>` is a silent no-op for video-derived face points

`src/scalar_forensic/faces/store.py:496–499`, consumed by `cli.py:2067`

```python
def purge_media(self, image_hash: str) -> PurgeResult:
    flt = Filter(must=[FieldCondition(key="image_hash",
                                       match=MatchValue(value=image_hash))])
    return self._purge_by_filter(flt)
```

Video-frame face points carry `image_hash` = **the extracted frame JPEG's** SHA-256; the video
itself appears only in the separate `video_hash` payload field (`faces/indexing.py:316–320`;
rollups carry *only* `video_hash`, store.py:253). Nothing resolves `video_hash → frame hashes`.
Passing the video's own digest — the identifier an operator naturally has from the ingestion
CSV — matches zero points, prints `Purged 0 face point(s)`, and logs a successful-looking
purge event.

*Failure:* `sfn-faces purge --media <sha-of-surveillance.mp4>` reports success while every
biometric observation derived from that video stays live and searchable. This is the erasure
guarantee §13 models for viewing copies; here it fails silently.

---

## MEDIUM

### M1 — Interrupted video ingest can never be marked complete; every later run re-slices it forever

`src/scalar_forensic/cli.py:1747–1759`, interplay with `cli.py:888–895` and `cli.py:841–844`

The completion gate counts only frames embedded **in this run**
(`vf_indexed_by_spec`) plus in-run duplicates (`_vf_dup_count`). Frames already in Qdrant are
marked `_S_SKIP_IDX` ("already indexed in Qdrant") in `_finish_batch` and counted nowhere.

*Failure:* run 1 dies at 400/500 frames of a video (marker never written). Run 2 re-slices,
finds 400 hashes already present, embeds 100 → gate sees `100 + 0 >= 500` → false. Run 3:
all 500 are `_S_SKIP_IDX` → `0 + 0 >= 500` → false. The recovery mechanism
(`mark_video_complete` / `get_all_video_info()["complete"]`, indexer.py:217–268) is
unreachable in exactly the crash-recovery scenario it was built for, so the video pays full
container decode + slicing on every subsequent run forever.

### M2 — DINOv2 provenance records the *requested* resolution even when the processor ignored it

`src/scalar_forensic/embedder.py:462–486`, consumed at `indexer.py:356`, compared at
`safeguards.py:262–267`

When `AutoImageProcessor.from_pretrained` silently keeps its built-in size despite
`size={"shortest_edge": normalize_size}`, the code detects this — and only emits a stderr
warning. `self.normalize_size` still reports the requested value, which becomes
`dino_normalize_size` in the Qdrant payload and is exactly what `check_collection_compat`
compares against current settings.

*Failure:* deployment A honors `SFN_NORMALIZE_SIZE=224`; deployment B loads a variant whose
processor ignores it. Both collections record identical model hashes *and*
`dino_normalize_size=224`; the safeguard passes; cross-collection cosine similarity is
meaningless — the one failure mode the safeguards exist to refuse, with a stderr line as the
only trace.

### M3 — Byte-identical video frames collide: marker overwritten, sibling observations flagged stale

`src/scalar_forensic/faces/store.py:209` (marker id = `uuid5(…, f"face-marker:{image_hash}")`),
`store.py:420–451` (`stale_face_points` scopes purely by hash), `cli.py:1545–1551`

Observation point ids correctly include `frame_timecode_ms`, so two extracted frames with
identical bytes (static scene, frozen timestamp, black lead-in/out) produce two distinct
observation sets sharing one `image_hash`. But (a) the second commit overwrites the first
frame's marker — same deterministic id — silently replacing its `n_detected/n_kept/rejected`
counts; and (b) `stale_face_points(frame2_sha, produced_ids_of_frame2)` finds frame 1's points
"not produced by this run" and queues them into the end-of-run prompt as *"stored by an earlier
run and not produced again by this one"* (cli.py:1862) — a false statement inviting
confirmation of deletion of legitimate evidence. Only the interactive default (`False`) stands
between the prompt and the deletion. No test covers the collision.

### M4 — Shared YuNet detector raced across FastAPI threadpool threads

`src/scalar_forensic/web/pipeline/faces_query.py:53–70` (`_detector_cache`),
`src/scalar_forensic/faces/detect.py:76–77`

`query_faces` is sync `def` → Starlette threadpool (default 40 workers). Two concurrent
requests get the same cached `YuNetDetector`, whose `detect()` mutates shared net state:

```python
self._net.setInputSize((dw, dh))
_, rows = self._net.detect(bgr)
```

`n_dropped_noncanonical += 1` (detect.py:94) is likewise unsynchronized. The project's own
rule is stated in `faces/indexing.py:111–114`: *"YuNet (cv2.FaceDetectorYN) is not thread-safe,
so a pooled pass must never share self.detector across threads"* — and the indexing pass obeys
it with per-thread detectors plus an embed lock. Only the web query path violates it.

*Failure:* corrupted/mixed detections or cv2 errors under concurrent examiners — plausible on
the stated distributed-LAN deployment.

### M5 — A full-video job refused at admission reports itself as a generic encode failure

`src/scalar_forensic/video_playback/jobs.py:113–114` raises inside `_run`,
`jobs.py:448–481`, `states.py:335–381`

`POST /api/video-full` returns success (job view, state `full-job-running`) before the
admission check runs — it happens inside the spawned task (`async with admission.enter(...)`,
jobs.py:419). When the queue is full, `Admission.enter` raises `QUEUE_FULL.as_http()` — an
`HTTPException` — inside the try. `classify_full_job` has no branch for `HTTPException`, so it
falls through to the catch-all `ENCODE_FAILED` row ("This chunk could not be produced: …").

*Failure:* under load the analyst sees a job start, then fail with kind `encode-failed` and a
chunk-flavoured reason instead of §10.1's `queue-full` row with its Retry-After. The failure
matrix's identity guarantee ("the kind is what a log record and a test key on") is broken on
exactly the contention case §10.4 exists for.

### M6 — `_run_watched` timeout is only enforced when progress lines keep arriving

`src/scalar_forensic/video_playback/encode.py:286–321`

The deadline is checked inside the `for line in proc.stdout` loop, i.e. only when ffmpeg
emits another `-progress` block. An encoder that stalls *without output* (hung demuxer,
deadlocked filter graph, NFS source) blocks forever on `readline`; nothing fires
`SFN_VIDEO_JOB_TIMEOUT`, the worker thread and its admission slot leak for good, and §10.3's
"the encoder was stopped" promise does not hold. The chunk path is not affected —
`communicate(timeout=…)` enforces regardless of output — so this is specific to the watched
(full-job) runner. Cancellation still works (`_kill_group` forces EOF); only the timeout is
toothless on a silent hang.

---

## LOW

### L1 — Re-indexed content never updates `image_path` (or EXIF flags) on an existing point

`src/scalar_forensic/indexer.py:409–448` — existing points get vector + model provenance only;
core payload keeps the first-seen path. With dedup mode `filepath`/`both`, content moved from
`/a/img.jpg` to `/b/img.jpg` embeds, updates the vector, and leaves `image_path=/a/img.jpg` in
Qdrant while the CSV reports the new file indexed. Once `/a/` is wiped between cases, the
collection's pointer for that content is permanently dead and hit-image serving fails.

### L2 — Pixel formats without subsampling digits are judged decodable, contradicting the stated rule

`src/scalar_forensic/video_playback/codecs.py:221–233` vs `:256–257`

`_pixel_profile` returns `(bits, None)` for formats like `gbrp`/`gray` — its comment says these
are "not 4:2:0 [and] no browser decodes them" — but `_decode_verdict` only rejects when
`chroma is not None and chroma != "420"`; `None` passes as unknown. An 8-bit GBR High-4:4:4
H.264 then classifies as "decodes natively", mode `original`, and the browser shows a broken
player instead of a transcode offer.

### L3 — Non-sha256 directories in the cache store are invisible to ceiling, eviction and purge

`src/scalar_forensic/video_playback/cache.py:355–367` — `scan()` skips any directory whose name
isn't a sha256 hex digest. Anything placed there (or written by a future layout change) counts
against no ceiling, is never evicted, and survives `sfn-video purge --all` — for a store whose
§13 contract is "derived renderings deleted".

### L4 — `JobRunner._jobs` retains every finished job forever

`src/scalar_forensic/video_playback/jobs.py:386–395` — finished `FullJob`s stay keyed by
digest until a *new job for the same video* replaces them. Each holds request, settings,
result, rendering. Unbounded growth proportional to distinct videos ever fully exported per
process lifetime.

### L5 — Face score statistics are censored at cosine ≥ 0 while claiming the full distribution

`src/scalar_forensic/web/pipeline/faces_stats.py:55–60, 101` — `search_faces(…, threshold=0.0)`
is an active Qdrant floor, not "no floor": negative-cosine observations never reach the sample,
so the histogram's entire negative half is structurally unreachable and min/percentiles are
computed over a censored population. The module's own docstring claims "exact search with no
score floor". In an uncalibrated-by-ruling display, hiding the left tail is exactly the kind of
silent shaping §10 of the face spec rules out.

### L6 — `face_audit` reports a healthy medium as "not comparable" when it has zero surviving faces

`src/scalar_forensic/web/pipeline/faces_stats.py:196–206` — a medium with a marker but no
observations (detector found nothing: routine) yields `row={}` → `_stored_config` produces an
all-`None` attribute bag → every recorded hard field "mismatches" `None` → ValueError →
`ok=False` with the comparability warning shown for a perfectly ordinary result. `check_compat`'s
None-means-unknown convention holds for fields *absent from meta*, not for a bag of Nones.

### L7 — Face read/browse endpoints bypass the modality gate

`src/scalar_forensic/web/routes/faces.py:143` (`by-image`), `:155` (`explain`), `:359–403`
(chip endpoints) — `_require_faces(settings)` guards query/search/compare/dist-stats/audit but
not the observation browser, explainer or chip downloads. With `SFN_FACES_ENABLED=false`,
biometric crops remain downloadable by anyone with a valid id/hash. If the intent were "disable
new queries, keep exhibits viewable", `audit` being gated while `by-image` is not makes the
asymmetry look accidental rather than ruled.

### L8 — Oversized face upload → 500 instead of the intended 400

`src/scalar_forensic/web/routes/faces.py:515–518` catches `(OSError, ValueError)`, but Pillow's
decompression-bomb guard raises `Image.DecompressionBombError`, a `RuntimeError` subclass, out
of `load_for_detection` (decode.py keeps the guard active deliberately). A >~178 MP upload
crashes with an unhandled exception instead of "not a usable image".

### L9 — Batch-size calibration cache validates neither device nor geometry

`src/scalar_forensic/calibration.py:56–65` vs `save_cached_batch_size` at `:68–101` — the
saver persists `device` and fit parameters; the loader checks only positivity. After a GPU→CPU
reconfiguration or a raised `SFN_NORMALIZE_SIZE`, the run silently adopts the old batch size
(OOM risk / skewed throughput). Performance knob only, but the stored `device` field shows the
intent and the load side never honours it.

### L10 — Small playback races and label inaccuracies (grouped)

- `video_playback/routes.py:653–663` — chunk GET: eviction could remove the file in the window
  between `is_file()` and `renew_lease()`; the serve then 404s/errs. Microseconds wide, but
  the lease is registered after the existence decision.
- `routes.py:506` (`_cache_hit`) labels a cache-hit rendering with
  `duration_seconds=float(video_chunk_seconds)` — the final short chunk is labelled with the
  full nominal length in the §7.2 row.
- `cache.py:551` — with no ceiling configured, `check_ceiling` computes the display estimate at
  hardcoded 1080 rather than `settings.video_output_height`; cosmetic inconsistency with the
  transcode-mode estimate.
- `estimate_full_output_bytes` (`cache.py:505–534`) applies the area ratio to the container's
  *whole* bitrate including audio, and the output re-encodes audio at fixed AAC 128k — both
  push the estimate around. Documented as uncalibrated; noted because the direction is known.

---

## Verified-clean areas

- **Rewrap/publish atomicity**: pid-scoped `.part` + fsync-file/rename/fsync-dir
  (`publish`, cache.py:642–664); orphan sweep with the documented pid-reuse hole; two-worker
  concurrent publish converges to one valid artifact.
- **Leases/pins/KeyedLocks**: refcount arithmetic, three-state lease semantics and lock-table
  lifetime all trace correct on the single-event-loop model they document; the multi-worker
  limitation is stated, not hidden.
- **The §6.3 override narrowing**: `overridable == state == "refused"` is consistently enforced
  at both publish points (routes.py:289 and the 507 detail at routes.py:732); `limit_bytes`
  derives from the configured-ceiling predicate, not falsiness; the `.part` watch backs the
  forecast up with a real invariant.
- **Exclusion-guarantee core (faces)**: vectorless review-only points + mandatory post-upsert
  vector clear, ordered so a failed clear never commits idempotency; vectorless probes refused
  on both origins; purge's delete-then-reference-check ordering.
- **Index/query preprocessing symmetry** (image side): both paths share the cap arithmetic;
  SSCD lands on identical sizes on both paths (the rare `int()` truncation is deterministic
  and symmetric).
- **Point-ID construction** (image/frame namespaces), batch preconditions, serialized upserts.
- **Hash handling**: streaming SHA-256, `HashCache` mtime_ns+size validation, WAL discipline;
  `digest.py`'s singleton keyed by DB path with fail-open-to-direct-hash fallback.
- **files.py path controls**: hex64 validation before joins, `resolve()` +
  allowed-root containment everywhere, thumbnail regen re-validates candidate paths from
  Qdrant payloads before touching disk.
- **config parsing**: malformed values refuse rather than fall back; `.env` precedence
  deliberate.
- **tags/manifest**: deterministic uuid5 ids, `created_at` preservation, atomic writes via
  temp+`os.replace`, secret redaction in manifests.

## Suggested order of attack (for whenever fixes are taken)

1. H1 + H2 (broken guarantees on ordinary input),
2. M5 + M6 (failure-matrix integrity under contention/hang),
3. M1 + M2 (silent divergence between recorded and actual state),
4. M3/M4, L5–L8 (faces correctness/gating),
5. the rest.
