# Handoff — `scalarforensic-com-c14`, video playback phase 4

UTC 2026-08-13 22:09:36. Manager: `scalarforensic-com-m5`. There is no CTO on
this project.

## 1. What I was asked to do

Rescue the §14 bench measurements out of gitignored `data/` and fold them into
the spec (PR 1), then implement §15 phase 4 — ffmpeg declared, capability probe,
pipeline fingerprint, chunk encode, and tests pinning the rotation and `bt709`
colour defects.

## 2. What is done

All merged to `main`. Bar **821 passed / 5 skipped** at `6f5879e`, verified on a
clean tree (`git status --porcelain` empty), coverage **70.99%** against the 65%
floor. Both CI checks green on every PR.

| PR | sha | What |
|---|---|---|
| `#155` | `3ac5442` | `docs/benchmarks/video-bench-2026-08-13.md` (verbatim); spec §3.5 added, §3.4 rewritten, §12 defaults, §14 fixture strategy, §16 rulings, §17 reduced to residue. Bar unchanged 771/5. |
| `#157` | `952b325` | `video_playback/capability.py`; §12 settings in `config.py`; colour transfer/primaries on the stream report; §6.1 fingerprint decision and the §4.3 contention correction in the spec. |
| `#158` | `57246c4` | `video_playback/encode.py`; ffmpeg installed in CI + Dockerfile, documented in `INSTALL.md`; the two §14 tests. |
| `#159` | `6f5879e` | `CLAUDE.md`: bar, conventions, gotchas. |

**Not done, deliberately:** phase 5 (cache). Not started, not begun in any
branch. `jobs.py` and `audit.py` do not exist and were not stubbed.

## 3. The next concrete step

**Phase 5 — cache (§6).** It was blocked behind `capability.py` and is now
unblocked. In order:

1. **`Pipeline.fingerprint()` gives you the second half of the §6.1 key and
   nothing else.** It is `sha256` over the nine pipeline fields. The *first*
   half — source identity — is `digest.py`'s verified SHA-256, and §6.1's key is
   `sha256(source identity ‖ pipeline fingerprint)`. That concatenation does not
   exist yet; there is no `cache_key()` function anywhere. Write it in
   `cache.py`, not in `capability.py`: the fingerprint is a property of the
   pipeline, the key is a property of the artifact store.
2. **Read the note I left in spec §6.1 before you key anything.** `chunk_seconds`
   is a fingerprint field, which is right for a chunk — it moves every encode
   boundary. It is *wrong* for the §4.3 full-file artifact, which is one encode
   of the whole source however long a chunk is. As it stands, changing
   `SFN_VIDEO_CHUNK_SECONDS` needlessly invalidates every cached `full.mp4`.
   That is the conservative direction so I left it, but it is your decision to
   make deliberately: keep it, or split the fingerprint per artifact kind.
3. **`_evict_cache` in `cache.py` is containment, not the §6.2 rewrite.** `#148`
   deliberately narrowed its glob to top-level `{sha256}.mp4` rewraps so it
   could not delete the `full.mp4` and chunk files later phases put in the same
   store. Its own docstring says so. The real §6.2 work — whole-tree accounting,
   whole-video LRU eviction, and the **playback lease** (heartbeat-refreshed
   with an expiry, because HTTP is stateless and the single-call `keep` argument
   cannot express "being watched") — is all still ahead of you.
4. **`encode()` already returns the pipeline that actually ran**
   (`EncodeResult.pipeline`), which on a GPU→CPU fallback is *not* the one you
   asked for. Key the published artifact off `result.pipeline`, never off the
   pipeline you selected before the call, or a fallback lands under the GPU's
   key and you get one key holding two pictures — the exact defect §6.1 names.

## 4. Things written down nowhere else

### The fixture machinery (`tests/test_video_playback.py`)

This is the part that would cost the most to re-derive. §14's three-source
lookup is implemented as two fixtures, both near the top of the file:

- `_operator_hdr_clip()` — sources 1 and 2: `SFN_TEST_VIDEO_HDR` pointing at a
  file, then a `hdr_sample.*` in the tracked `test_data/` directory. `test_data/`
  exists with a `README.md` stating what a dropped clip must carry. It is empty
  of clips and will stay that way: **nothing from the operator's corpus is ever
  committed.**
- `_generate_hdr_clip()` — source 3: `ffmpeg -f lavfi` producing a 64×96,
  0.6 s, 10-bit `yuv420p10le` HEVC clip tagged `bt2020` primaries /
  `arib-std-b67` transfer / `bt2020nc` matrix, muxed to `.mov` with the `hvc1`
  tag.
- `hdr_hlg_mov` and `hdr_rotated_mov` are the two fixtures tests actually
  request. `hdr_rotated_mov` uses an operator-supplied clip **only if that clip
  really carries rotation** — a supplied clip silently missing it would turn the
  rotation test green while testing nothing.

**The `tkhd` display-matrix patch.** ffmpeg 6 cannot write rotation side data on
an *output* stream: `-metadata:s:v:0 rotate=-90` is accepted and silently does
nothing (I verified this — the output had no side data at all), and
`-display_rotation` is an input-side option. PyAV is no help either: it exposes
no stream-level side data at all (`VideoCodecContext` has no `side_data`
attribute), which is why `_display_rotation()` and `_ffprobe_video_stream()`
shell out to **ffprobe** rather than using PyAV like the rest of the file.

So `_patch_display_matrix()` writes a real matrix into the container:

- Find `b"tkhd"` in the file; the box starts 4 bytes earlier (the size field).
- The 3×3 matrix sits **48 bytes into the box** — after version+flags (4),
  creation and modification times (8), track ID (4), reserved (4), duration (4),
  reserved (8), layer (2), alternate group (2), volume (2), reserved (2).
- 36 bytes, nine big-endian int32: `[a b u; c d v; x y w]`, with `a b c d x y`
  in 16.16 fixed point and `u v w` in 2.30. `_rotation_matrix()` writes
  `[0, -65536, 0, 65536, 0, 0, 0, width<<16, 1<<30]`.

The result is **genuine side data**, not a metadata string: `ffprobe` reports
`"side_data_type": "Display Matrix"` with `"rotation": 90`. This is not a
workaround around a weakened test — it is what makes the strong test possible.
Do not replace it with a metadata tag; that path does not work.

### The mutation check — the rotation test is proven, not assumed

**I ran it.** I added `-noautorotate` to `build_command`'s base argv, ran
`pytest -k rotation`, and got:

```
AssertionError: rotation was lost: the source's display matrix did not become geometry
1 failed, 142 deselected
```

then reverted the edit. The test asserts a 64×96 source comes out **96×64** —
the display matrix has to become *geometry*, which is only true because decode
and filtering stay in software so ffmpeg's autorotate runs.

Two things follow. **Trust it**: it is not a test that passes because nothing
touches it. And **re-prove it** if you change the encode path — particularly if
anyone reintroduces `-hwaccel_output_format cuda`, which is the specific thing
§3.1 measured breaking this and which the current design avoids by using the GPU
for the *encoder only*.

The colour test (`test_output_is_tagged_bt709`) is not mutation-checked the same
way; it asserts `bt709` on transfer, primaries *and* matrix plus `yuv420p`
against a source that is demonstrably `arib-std-b67`/`bt2020`, which is a strong
enough shape that I did not spend the window on it. Worth doing if you touch the
tone-map chain.

### `mock.patch` targets and module-level state in `video_playback/`

`CLAUDE.md`'s first gotcha bites here. Current state:

- **`capability.py` holds a process-wide cached probe** — module globals
  `_cached` and `_cache_lock`, with `capability(settings, refresh=False)` and
  `reset_cache()`. Any test that touches the probe **must** call `reset_cache()`
  in a fixture on both sides, or it inherits another test's answer.
  `TestCapabilityProbe` has an autouse fixture doing exactly that; copy it.
- **`digest.py` holds the same shape** for the `HashCache` handle (`_hash_cache`,
  `_hash_cache_lock`, `_reset_hash_cache()`), and the existing `roots` fixture
  already resets it on both sides because the handle points into `tmp_path`.
- **`cache.py` holds `_remux_locks`**, a dict of `asyncio.Lock` per source
  digest. It is never cleared. Not a problem today; it is one of the "locks must
  not accumulate unboundedly" items §10.4 names, and phase 5 owns it.
- Patch `vp_encode._run` (not `subprocess.Popen`) to simulate an encode failure —
  `TestGpuFallback` does this, and it is why that test can exercise the fallback
  without a GPU. Patch `vp_encode.subprocess.Popen` when you want to test the
  *timeout* path, because that is where the timeout lives.
- `tests/test_video_endpoints.py` needed **zero** changes across all of phase 4,
  same as in the carve: its string patch targets all belong to
  `/api/video-frame` and `/api/video-timeline`, which live in `routes/video.py`
  and are untouched by playback work.

### Running the suite from a git worktree

I did all of phase 4 in a worktree because the shared checkout could not
fast-forward (see §5). The venv is installed editable against the **main**
checkout, so a worktree imports the main checkout's `src/` and your edits are
invisible. Two things make it work:

```
ln -s /home/user01/Schreibtisch/gitea/ScalarForensic/.venv .venv
PYTHONPATH=$PWD/src uv run --no-sync pytest -q
```

`PYTHONPATH` wins over the editable install's path entry. **Without it you will
test the other checkout's code and not notice.** Also: a fresh worktree has no
`models/`, so the YuNet test skips and the bar reads 820/**6** rather than
821/**5**. Copy `models/` in before quoting a bar, or your number will not match
`CLAUDE.md`'s.

### The CI skip guard

`_need_ffmpeg()` raises `AssertionError` when `CI` is set and `pytest.skip`s
otherwise. Before `#158`, CI ran **795 passed / 11 skipped**; after, **820/6**,
identical to local. All 5 recovered tests are real assertions now. `ci.yml`'s
`qdrant-integration` job deliberately has **no** ffmpeg install — it runs only
`tests/faces/test_store_integration.py`. There is a comment there saying so, and
saying it needs the install if its selection ever widens.

### Things that contradicted the spec or the source material

- **§4.3's "nothing blocks on it" was false** under `SFN_VIDEO_MAX_WORKERS=2`.
  Found by `com-m4`, corrected in `#157`. A full-video job holds one of the two
  workers for ~51 minutes, putting chunk encoding at k=2 — 8.21 s → ~16.35 s
  (§3.5), outside §4.2's 6–10 s window. Two remedies are named in §4.3 as
  **phase 7's call**; neither is implemented.
- **§8's "GPU path" is not §3.1's fastest row.** §3.1's fastest pipeline
  (`scale_cuda` + `h264_nvenc`, 12.9×) is the one that *breaks rotation*.
  "hwaccel=cuda" in this codebase therefore means software decode, software
  filtering, GPU **encoder only** — the 6.1× row. This is written into
  `select()`'s docstring; do not "optimise" it back.
- **A build without libzimg cannot be handled by falling back.** Encoding an HDR
  source without the tone-map chain produces §3.1's second defect — 8-bit pixels
  still tagged `bt2020`/HLG. `Capability.unavailable_reason(hdr=True)` **refuses**
  and points at Download original. SDR still works on such a build. This is a
  three-state answer, not a boolean, and it is deliberate.
- The §17 Q1–Q5 answers were in `docs/CTO_LEDGER.md`'s "pending user decisions"
  as inline `-->` replies from the operator, not in any spec. They are now in
  §16 and §17 is residue only.

## 5. One thing that is not mine and is still open

The shared checkout `/home/user01/Schreibtisch/gitea/ScalarForensic` **cannot
fast-forward past `43a7222`**: `docs/CTO_LEDGER.md` is dirty with the operator's
handwritten `-->` answers and `#153`/`#154`/`#156` all touched that file.
`com-m5` owns it, has preserved a copy, and escalated to the operator. Neither
of us reverts or commits it. **Work in a git worktree** — that is the documented
arrangement here, not a workaround. My worktree is at
`/tmp/claude-1000/.../8a535664-.../scratchpad/wt-phase4` and can be removed with
`git worktree remove`.

## 6. Ownership

Released on merge of this PR: the spec, `video_playback/`,
`tests/test_video_playback.py`, `config.py`, `Dockerfile`, `INSTALL.md`,
`pyproject.toml`, `.github/workflows/ci.yml`, `docs/deployment.md`, `CLAUDE.md`,
`test_data`.

Note `pyproject.toml` and `docs/deployment.md` were claimed and **never
modified** — I took them expecting to declare ffmpeg there and it belonged in
`INSTALL.md` and the Dockerfile instead.
