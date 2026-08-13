# Handoff — scalarforensic-com-c9 → successor (2026-08-13)

Retiring on context, mid-task. Branch `fix/query-faces-generation` is pushed at
`origin/fix/query-faces-generation`, one WIP commit on top of `9e71a96`.

## Shipped before this (no action needed)

PR #136, merged as `9e71a96` — in-browser playback of source videos from a
frame hit. Bar at merge 690/5 (shared checkout). Worktree `wt-vid` and branch
`feat/video-playback` removed; nothing outstanding except the two items m3
routed to the operator side (server restart, c4's visual pass) and the CodeQL
`py/path-injection` dismissals, which are a maintainer call.

## Task in flight: query-faces overwrite race

CONFIRMED defect. c4 pinned it as reqid 690's 404 body (`content-length 31` =
`{"detail":"unknown face index"}`).

### What the defect actually is

`entry.query_faces` is per-`FileEntry`, so a POST for file B does **not** clobber
file A's server state. The overwrite is on the client, and the URL shape lets it
become an evidential problem:

1. `index.html` runs `loadQueryFaces()` on `$watch('selectedFileId', …)`. Two
   files means two POSTs in flight; the first can resolve **last**, leaving file
   A's `queryFaces` on screen after selection has moved to file B.
2. `faces.js` rebuilt the chip URL from `this.selectedFileId` at render time
   instead of using the server's stamped `chip_url`. So A's indices got aimed at
   B's entry — 404 when B has fewer faces (what c4 saw), and **B's face shown
   under A's identity label** when B has as many or more. The silent case is the
   dangerous one.
3. Independently: re-detecting the same file (different video frame) reassigns
   indices, so any chip URL still on screen addresses a different face.

### What is already implemented on the branch

- `routes/faces.py`
  - `_detection_token(faces, cfg)` — sha256 over `config_hash` plus each face's
    `index|bbox|det_conf|embedding_status`, truncated to 16 hex. Content-derived,
    not a counter, so re-detecting an identical view reproduces the token and
    does *not* invalidate URLs the examiner is still looking at.
  - `chip_url` is now `/api/faces/query-chip/{session}/{file}/{token}/{index}`;
    the response also carries `detection_token` at top level.
  - `query_chip` takes `token` and returns **409** on mismatch instead of
    answering from the current detection.
- `static/js/faces.js` — `loadQueryFaces` captures `fileId` and an ordinal
  `_queryFacesSeq`; a response is dropped unless it is still the newest request
  *and* the file is still selected. `queryFaceChipUrl(face)` returns
  `face.chip_url`. New `markQueryFaceStale(face)`.
- `static/js/state.js` — `_queryFacesSeq`, `queryFacesStale`.
- `static/index.html` — chips pass `face` (not `face.index`), `@error` marks the
  tile stale, and the status label reads `stale — re-detect` for those tiles.

Hard constraints held so far: chips still memory-only + `no-store`; review-only
faces untouched (still vectorless); single-file sessions behave identically
(one detection, token matches, same URLs modulo the extra path segment).

### TODO — the red tests are the list

`uv run pytest tests/faces/test_query_faces.py -q --no-cov` → **2 failed, 7 passed**

1. `test_query_faces_endpoint_never_returns_a_vector` — asserts
   `chip_url.endswith("/query-chip/s/f/0")`. Update to the token shape; import
   `_detection_token` and build the expected suffix from it rather than
   hardcoding a digest.
2. `test_query_chip_serves_the_in_memory_crop_with_no_store` — GETs
   `/api/faces/query-chip/s/f/0`. Needs the token segment. Note the mock entry's
   `query_faces_cfg` is an auto-created `MagicMock`; `_detection_token` reads
   `getattr(cfg, "config_hash", "")` and `str()`s it, which is stable for one
   mock object, so compute the token from the same `entry` the test patches in.
3. `test_query_chip_404s_on_unknown_index` currently passes **by accident** — the
   3-segment URL no longer matches the route, so it 404s from routing, not from
   the bounds check. Rewrite it with a valid token so it tests what it claims.
4. Add: 409 on a stale token; token changes when the detection changes; token is
   reproduced when the same faces are re-detected (the no-churn property).
5. Add a `tests/test_static_wiring_web.py` check that `queryFaceChipUrl` returns
   the server's `chip_url` and does not build a URL from `selectedFileId` — that
   is the regression that actually bit.
6. Frontend has no live check: the chrome-devtools browser was held by another
   session for my whole window. A frontend tester should drive a two-file session
   and confirm no chip ever renders under the wrong identity.

### Also owed (m3 granted the file, I never claimed it)

Document `SFN_VIDEO_CACHE_DIR` (default `data/video_cache`; empty disables
rewrapped playback) and `SFN_VIDEO_CACHE_MAX_BYTES` (default 8 GiB; `0` = no
ceiling) in `docs/deployment.md`. Claim it first:
`cx o /home/user01/Schreibtisch/gitea/ScalarForensic/docs/deployment.md --own --as <you>`.
Both are already documented in `config.py` next to the frame-store settings.

## Claimed files

`src/scalar_forensic/web/routes`, `src/scalar_forensic/web/static`,
`src/scalar_forensic/config.py`. Release or re-claim as m3 directs.

## Fences that applied

Worktree + PR flow (never switch the shared checkout's branch), Qdrant :6333
read-only, no `.env` edits, `/media/user01/...` read-only. Routes/config changes
need an `sfn-web` restart — operator side, standing grant, do not perform.
