# Handoff — `scalarforensic-com-c16`, video playback phase 6 (chunk playback)

UTC 2026-08-14 02:30. Manager: `scalarforensic-com-m5` (at its retire nudge when
this was written; report to whoever the operator spawns). There is no CTO on this
project.

## 1. What I was asked to do

Correct `CLAUDE.md`'s stale bar line in its own PR, then implement §15 phase 6:
double-buffered chunk playback, seek-to-new-chunk, §5's player states, and every
§10.1 failure mapped to the state it produces and pinned by a test.

## 2. What is done

**Phase 6 is complete and merged.** Bar **954 passed / 5 skipped** at `8af5266`,
coverage **73.09%** against the 65% floor, clean tree (`git status --porcelain`
empty), measured in
the worktree `/tmp/claude-1000/.../faea21c3-.../scratchpad/wt-c16` with
`PYTHONPATH=$PWD/src` and `models/` **copied** in (not symlinked — `.gitignore`'s
trailing slash does not match a symlink, and an untracked `models` makes the tree
dirty for bar-quoting).

| PR | sha | What |
|---|---|---|
| `#165` | `2dbfea6` | `CLAUDE.md`: bar 882/5 → measured at `6e09097`, coverage 72.20%; `cache.py`'s public surface in the conventions paragraph; the `video_playback/` reset gotcha. |
| `#168` | `b239c51` | Server side: `POST`/`GET /api/video-chunk`, `states.py`, the admission gate, `SFN_VIDEO_QUEUE_MAX`, spec §5/§9/§10.1/§11/§12/§15. |
| `#169` | `8af5266` | Browser side: `js/video_playback/player.js`, the `index.html` markup, `style.css`, spec §6.1 and §14. |

**Not done, deliberately:** phase 7 (full-video job) and phase 8. `jobs.py` and
`audit.py` still do not exist and were **not** stubbed. The three `full-job-*`
states of §5 are **not** implemented — §5 forbids inventing a state that cannot
be observed, and with no job endpoints nothing can enter them.
`states.PHASE_7_STATES` names them so the omission is visible rather than
accidental.

## 3. The next concrete step

**Phase 7 — the full-video job (§4.3).** In order:

1. **`cache.check_ceiling()` is written and still nothing calls it on a job
   path.** `playback-info` reports its verdict for display; the job runner must
   *enforce* it before starting a full copy, and must also do what §6.3 says
   explicitly: watch the growing `.part` against `estimate_full_output_bytes()`
   and abort on overshoot. `c15`'s handoff §4 explains why — the estimate applies
   no codec factor and therefore runs **low on exactly the HEVC corpus this
   feature exists for**. It is a screen, not a guarantee.
2. **`routes._Admission` is the thing `jobs.py` replaces.** It is a semaphore
   plus an admitted-count, deliberately not a job runner: no cancellation, no
   refcounts, no progress. Replace it wholesale rather than growing it. Its
   docstring says so.
3. **Map full-job failures through `states.classify()`**, not a second table.
   Every §10.1 row already resolves; a full-job failure differs only in the
   *state* it lands in (`full-job-failed` rather than `chunk-failed`), so add the
   mapping at the call site, not a parallel matrix.
4. **§4.3's contention is still unaddressed and still true.** A full-video job
   holds one of the two workers for ~51 minutes, putting chunk encoding at k=2 —
   8.21 s → ~16.35 s (§3.5), outside §4.2's 6–10 s window that the double-buffered
   swap depends on. §4.3 names two remedies (yield with `nice` + `-threads`, or
   accept and disclose in the UI) and rules out the third — leaving it unbounded
   *and* invisible. **Phase 7 must pick one.** The player has nowhere to say it
   today.

## 4. Things written down nowhere else

### The defect that cost the most to find, and cannot be found in CI

**A host whose GPU probes clean and then fails at encode time never hit its own
cache.** `select()` keeps returning the GPU pipeline; a §8 fallback files the
artifact under the CPU key; the lookup misses forever; every chunk re-encodes.
The cache silently stops being a cache, on exactly the hardware §8's fallback
exists for, and no CI runner has a GPU so nothing there can see it. It surfaced
because `test_the_second_request_is_a_cache_hit_and_does_not_re_encode` patches
`vp_encode._run` to raise on a hit and the second request re-encoded.

Two pieces fix it and **the distinction between them is the whole thing**:

- `routes._relocate_on_fallback` keeps the *artifact* invariant — the published
  file is `os.replace`d under `result.pipeline`'s key. A GPU key never holds a
  CPU picture.
- `routes._substitutions` is a **lookup hint and never a key**: selected
  fingerprint → the whole `Pipeline` that last actually ran. Selection is
  unchanged, so a recovered GPU is used again at the next genuine miss, and the
  direct key wins when both artifacts exist (`test_the_direct_key_wins_when_
  both_exist`). It holds `Pipeline` objects and not fingerprints so a substituted
  hit labels itself with the encoder that produced the bytes (§7.2).

It is process-wide state (reset it: `routes.reset_substitutions()`), and it
**does not survive a restart** — deliberately. A fallback host pays one wasted
GPU attempt after a restart; persisting the table would make a repaired GPU
permanently unused. Spec §6.1 records all of this.

### The §8 fallback now answers a GPU fault and only a GPU fault

A timeout is not retried on CPU (the CPU path is slower — §3.1's 6.1× vs 2.7× —
so the retry spends `SFN_VIDEO_JOB_TIMEOUT` again and fails again), and neither
is a SIGKILL (a second encoder under memory pressure turns one refused request
into two dead ones). `EncodeError` carries `returncode` and `timed_out` so this
is read from the exit status rather than parsed out of ffmpeg's prose, which
would break the first time ffmpeg reworded itself.

### Why the chunk endpoint is two verbs

§9 listed one `POST /api/video-chunk` returning media. A `<video>` element cannot
POST — it issues a `GET` with `Range` and nothing else — and it will not wait
§3.5's 8.21 s for an encode without its own media stack concluding the source is
broken. POST does the work and returns JSON; GET serves bytes and 404s on a miss.
`fp` on the GET is a *selector inside that video's directory*, never an identity;
`path` is the identity, and a non-hex `fp` is refused before any path is built.

### The live browser check is mandatory, and it is not the reason you think

`player.js` shipped a `?? … ||` precedence **`SyntaxError`** — the file did not
parse *at all* — and **all fourteen text-level wiring tests passed against it.**
A wiring test reads a file; it cannot tell you the browser can run it. This is
`CLAUDE.md`'s `/?cachebust=N` gotcha generalised, from the other direction.

So for any browser-side work here: start a server, force-refetch **every**
`<script src>` and the stylesheet with `fetch(url, {cache: 'reload'})`, reload,
and drive the component. What I measured this way, against a 7 s HEVC 10-bit .MOV
at `SFN_VIDEO_CHUNK_SECONDS=2`:

- first chunk ready in 454 ms; prefetch depth exactly 1;
- boundary swap moved buffer 0→1, start 0→2, next 4, and prefetched 4;
- seek to 2.5 s issued **no** request; seek to 6.1 s encoded the chunk at 6 with
  `next_chunk_start` null;
- `t=99999` → 422 `timecode-out-of-range`, rendered `chunk-failed`, retry
  withheld; a synthetic `queue-full` gated Retry for 3 s then offered it;
- close fired `POST /api/video-lease?...&release=true`;
- **a served chunk decodes in Chrome** — 2.000 s, 320×240, no media error — on a
  source Chrome cannot open. That is the feature working end to end.

**The gap this leaves, recorded in spec §14 and repeated here because it outlives
phase 6:** there is no JS test harness in this repository. `player.js` is pinned
by text-level wiring tests plus a manual browser run; the markup inside
`.vc-block` is wiring-pinned only, because rendering it needs a full analysis
session with Qdrant and an indexed corpus. Closing it means a real JS test
runner — a new dependency and the **operator's** decision, not a phase of this
spec.

### The three-state discipline, where it landed

`playable` / `needs-transcode` / **`unknown`**. A container that will not open has
said nothing about whether it would play, so reporting it as `needs-transcode`
would be `#147`'s "unknown displayed as mismatch" one layer up.
`codecs._playback_mode` has returned `unknown` since phase 3;
`states.MODE_TO_STATE` carries it through instead of flattening it. `idle` and
`probing` were added for the same reason: before anything is asked, the player is
not entitled to a verdict. A test asserts the state is neither `playable` nor
`needs-transcode`.

### Mutation checks — thirteen, all caught, all reverted

`unknown` collapsed into `needs-transcode` → `test_an_unprobeable_container_is_
unknown_and_not_needs_transcode`. SIGKILL branch dropped → `test_an_oom_killed_
encoder_is_capacity_exhausted_not_a_bad_file`. Timeout branch dropped →
`test_a_job_timeout_is_a_504_and_retryable`. `ENOSPC` reclassified as unwritable
→ `test_a_full_filesystem_is_capacity_exhausted_and_retryable`. Admission gate
removed → `test_the_queue_refuses_rather_than_growing_without_limit`. Chunk-start
snapping removed → four `TestChunkSnapping`/`TestChunkPlayback`. Fallback
relocation removed → six tests. GET's 404 guard removed → `test_a_get_never_
encodes_a_missing_chunk`. Substitution lookup removed → three `TestFallbackCache
Lookup`. Timecode bound removed → two. Stale-source check removed →
`test_a_changed_source_is_409_and_never_encoded`. Fallback guard removed → two
`TestEncodeFallbackLimits`. `fp` hex check removed → `test_a_fingerprint_is_
never_accepted_as_the_identity_of_a_file`.

### Module state a fixture must reset

Adds to `c14`'s and `c15`'s lists: `routes.admission.reset()` and
`routes.reset_substitutions()`. Both are in `tests/test_video_playback.py`'s
module-scope **autouse** `_clean_cache_state`, which also now resets
`capability.reset_cache()`. An admission counter left above zero turns a later
chunk request into a spurious 503, which reads as a flake. `CLAUDE.md` names all
of them.

### CodeQL

`#168` went red on `py/path-injection` at
`video_playback/routes.py:80,175,236` and `codecs.py:53` — the same closed-ruling
class as the existing 17, new instances only because `/api/video-chunk` is a new
taint source into `_resolve_video_path`. CodeQL does not model `Path.relative_to`
as a sanitizer. **Do not add a suppression comment to buy a green icon**;
dismissal needs `security_events` scope and is the operator's action. CodeQL is
not one of the two required checks.

## 5. The shared checkout, still

`/home/user01/Schreibtisch/gitea/ScalarForensic` still cannot fast-forward:
`docs/CTO_LEDGER.md` is dirty with the operator's handwritten answers. Unchanged
since `c14`'s §5 — work in a worktree. Mine is at
`/tmp/claude-1000/.../faea21c3-.../scratchpad/wt-c16` and can be removed with
`git worktree remove`. A fresh worktree reads **953/6** rather than 954/5 until
`models/` is copied in.

## 6. Ownership

Released on merge of this handoff: `docs/specs/video-playback-transcode.md`,
`src/scalar_forensic/video_playback/`, `tests/test_video_playback.py`,
`CLAUDE.md`, `src/scalar_forensic/web/static/`, `src/scalar_forensic/config.py`.

`config.py` was claimed for one setting (`SFN_VIDEO_QUEUE_MAX`) and touched
nowhere else. `src/scalar_forensic/web/templates/` was granted and **does not
exist** — the shipped page is `src/scalar_forensic/web/static/index.html`.
