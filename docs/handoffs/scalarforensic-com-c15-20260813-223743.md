# Handoff — `scalarforensic-com-c15`, video playback phase 5 (cache)

UTC 2026-08-13 22:37:43. Manager: `scalarforensic-com-m5`. There is no CTO on
this project.

## 1. What I was asked to do

Implement §15 phase 5: the cache keyed by pipeline, playback leases, the §6.2
eviction rewrite, ceiling refusal, atomic publication, and `sfn-video purge`.

## 2. What is done

**Phase 5 is complete and merged.** Bar **882 passed / 5 skipped** at `955334c`,
clean tree (`git status --porcelain` empty), coverage **72.20%** against the 65%
floor, measured in a worktree with `PYTHONPATH=$PWD/src` and `models/` copied in.
Both required checks green on both PRs.

| PR | sha | What |
|---|---|---|
| `#162` | `3b191b5` | `cache_key()`, per-video store layout, `evict()`, leases + pins, `KeyedLocks`, `publish()`, `sweep_orphaned_parts()`, `POST /api/video-lease`, `SFN_VIDEO_LEASE_SECONDS`. Spec §6.1, §6.2, §9, §10.2, §10.4, §12. |
| `#163` | `955334c` | `check_ceiling()` + `estimate_full_output_bytes()`, `full_copy` on `playback-info`, `sfn-video purge`, `_stream_report` gains `video_width`/`video_height`/`bit_rate`. Spec §6.3, §13. |

**Not done, deliberately:** phase 6 (chunk playback) and everything after.
`jobs.py` and `audit.py` still do not exist and were **not** stubbed.
`CLAUDE.md` was **not** updated — I never held it. Its bar line still says
821/5 at `57246c4`; the true bar is now **882/5 at `955334c`**, and its "Code
structure conventions" paragraph does not yet mention `cache.py`'s new public
surface. That is a one-PR job for whoever holds `CLAUDE.md` next.

## 3. The next concrete step

**Phase 6 — chunk playback.** Everything it needs from the cache exists:

1. `cache.artifact_dir(cache_dir, source_digest, fingerprint)` gives the
   directory, `cache.chunk_name(start)` the filename. **Key off
   `EncodeResult.pipeline`, never off the pipeline you selected** — on a §8 GPU
   fallback they differ, and keying on the request files a CPU encode under the
   GPU's key. `encode.py` already returns the one that ran.
2. Hold `cache.pin(source_digest)` around every encode and serve, and renew the
   lease on every chunk request. The player must call `POST /api/video-lease` on
   a heartbeat (default 120 s ttl, so ~30 s beats) and once with `release=true`
   when the analyst closes the video.
3. Call `cache.evict(cache_dir, settings.video_cache_max_bytes)` after
   publishing, with no `protect=` argument — the default reads leases and pins
   for you. `protect=set()` exists for tests and disables that.
4. `cache.check_ceiling()` is written but **nothing calls it on a job path**,
   because phase 5 runs no jobs. Phase 7 must call it before starting a full
   copy, and must also do what §6.3 now says explicitly: **check the growing
   `.part` against the estimate and abort on overshoot.** See §4 below for why.

## 4. Things written down nowhere else

### The size estimate is uncalibrated and biased low

`estimate_full_output_bytes()` scales the source's measured bitrate by the
*area* ratio of the §16 cap and applies **no codec factor**. That is not an
oversight: §3.5 timed the encodes and recorded no output sizes, so there is no
measured re-encode ratio in this repository, and §16 forbids inventing one. But
the direction of the error is knowable — a CRF-23 H.264 encode of a 10-bit HEVC
source at the same resolution is usually *larger* than its source — so the
estimate runs **low on exactly the HEVC corpus this feature exists for**. It is
therefore a screen, not a guarantee, and the phase 7 job runner has to enforce
the ceiling itself by watching the `.part`. This is in the docstring and in
§6.3; it is the one thing in phase 5 that a later phase can quietly get wrong.

If someone wants to close it properly: measure output sizes for the §3.1 rows
on the operator's corpus and turn the factor into data, the way §3.5 turned the
concurrency guesses into data.

### Why the store nests under the source digest

§6 writes the layout as `{key}/c{start}.mp4`, but §6.2 requires eviction of
*whole videos*, and `key = sha256(source ‖ fingerprint)` is opaque — you cannot
get back to "which video is this?" from it. So artifacts live at
`{cache_dir}/{source_digest}/{key}/…` and the rewrap at
`{cache_dir}/{source_digest}/rewrap.mp4` (a rewrap runs no pipeline, so its only
identity is the source's). One video is one directory; eviction is `rmtree`.
A video rendered by two pipelines holds two `{key}/` subdirectories and is still
one eviction unit. Spec §6.1 records this.

**Legacy top-level `{sha256}.mp4` files** — the pre-`#162` layout — are still
counted and evictable as their own entry rather than deleted on sight. Any store
on the operator's machine written before `3b191b5` holds them, and they retire by
LRU. `scan()` returns them with `legacy=True`.

### Leases vs. pins — they are not redundant

`FileResponse` streams its body **after** the handler returns, so a request
cannot bracket a read; that is the whole reason a lease exists and why it has to
be an explicit heartbeat rather than a `keep=` argument. The **pin** is the
other half: an in-flight *write* whose `.part` lives inside a directory LRU would
otherwise remove, and a lease can expire under an encode slower than its ttl.
Use both. `protected_videos()` is the union.

`LeaseState` is three-valued and must stay that way. `none` is not `expired`: a
second ASGI worker process says `none` about a video the first is actively
serving, and a boolean would report that as "nobody is watching".

### The module state a fixture must reset

Adds to `c14`'s list (`capability._cached`, `digest._hash_cache`):

- `cache._leases`, `cache._pins` — reset with `cache.reset_leases()`.
- `cache.artifact_locks` — `.reset()`.
- `cache._swept`, the once-per-process part-file sweep flag — `_reset_sweep()`.

`tests/test_video_playback.py` has one **module-scope autouse** fixture
(`_clean_cache_state`) doing all three on both sides. It is autouse because a
leaked lease silently changes what a later eviction test evicts, which reads as
a flake rather than as pollution.

### The sweep is not in `app.py`'s lifespan

§10.2 asks for a startup sweep. `cache.ensure_swept()` runs it once per process
on first cache use instead, because `app.py` keeps only app setup (`CLAUDE.md`)
and the cache directory is a request-time setting — a deployment that never plays
a video has no store to sweep. If a lifespan hook is ever wanted anyway, the
function is idempotent and safe to call from one.

Orphan detection is by the **pid in the `.part` name**, which is what separates a
crashed writer's scratch file from a live sibling process's in-flight encode. Pid
reuse delays a delete by one sweep; that is documented, not engineered away.

### `KeyedLocks` is single-worker dedup, stated as such

§10.4 asked for the assumption to be stated. It is, in the class docstring and
in §10.4: two ASGI worker processes can both encode the same artifact, and both
publish atomically to the same path, so the cost is wasted CPU and never a
corrupt file. Cross-process deduplication, if it is ever wanted, is phase 7's.

### Mutation checks — eight, all caught

The standing rule now. `#162`: single-file eviction → `test_a_whole_video_goes_
never_one_chunk`; `keep = set()` → the lease and pin tests; lock entry never
deleted → all three `TestArtifactLocks`; `.part` skipped in accounting →
`test_part_files_count_against_the_ceiling`. `#163`: `unknown` collapsed to
`fits` → `test_an_unestimable_source_is_unknown_not_refused`; fraction 1.0
instead of 0.5 → two `TestCeilingRefusal`; area scaling dropped → `test_estimate_
scales_by_the_output_area`; purge ignoring `--media` → two `TestPurge`. Each
reverted.

### CodeQL

`#162` went red on **three new** `py/path-injection` alerts at
`web/routes/_shared.py:57,61` — the same closed-ruling class as the existing 17,
new instances only because `POST /api/video-lease` is a new taint source into
`_resolve_video_path`. CodeQL does not model `Path.relative_to` as a sanitizer.
`com-m5` re-read the code, confirmed the class, and escalated; dismissal needs
`security_events` scope and is the operator's action. **Do not add a suppression
comment to buy a green icon.** CodeQL is not one of the two required checks;
`#163` was green.

## 5. The shared checkout, still

`/home/user01/Schreibtisch/gitea/ScalarForensic` still cannot fast-forward:
`docs/CTO_LEDGER.md` is dirty with the operator's handwritten answers. Unchanged
from `c14`'s §5 — work in a worktree. Mine is at
`/tmp/claude-1000/.../e7bd14ed-.../scratchpad/wt-phase5` and can be removed with
`git worktree remove`. Note a fresh worktree reads **881/6**, not 882/5, until
`models/` is copied in — and copy it, do not symlink it: `.gitignore` has
`models/` with a trailing slash, which does not match a symlink, so a symlinked
`models` shows up as untracked and makes the tree dirty for bar-quoting.

## 6. Ownership

Released on merge of this handoff: `docs/specs/video-playback-transcode.md`,
`src/scalar_forensic/video_playback/`, `tests/test_video_playback.py`,
`src/scalar_forensic/config.py`, `src/scalar_forensic/cli.py`, `pyproject.toml`.

`pyproject.toml` was touched in `[project.scripts]` only (one line,
`sfn-video`), per the constraint `com-m5` attached to the grant.
