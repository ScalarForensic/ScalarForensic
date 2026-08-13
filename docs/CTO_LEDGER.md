# CTO ledger — ScalarForensic

Written by the CTO; rewrite in place, keep it tight, commit every fold.
`git log -S` on this file is the project's decision history.

Seeded 2026-08-12 by cto8 (`21f65f1`). Final fold 2026-08-13 by
`scalarforensic-cfm-g1`. Raw record: `docs/fleet/runbook.md` and the
`~/.claude/cx/cto.md` git history.

> ## THE CTO ROLE IS RETIRED (2026-08-13, operator's decision)
>
> `scalarforensic-cfm-g1` folded and exited without a successor. **There is no
> CTO on this project.** The manager owns the work and **escalates directly to
> the operator**, not upward to a CTO session that does not exist. Nobody should
> spawn a replacement CTO; if the operator wants one later they will run
> `cx cto` themselves.
>
> This file remains the project's decision record. **The manager now writes it**
> — same discipline: rewrite in place, keep it tight, commit every fold.

## what this is

Forensic image-similarity tool: FastAPI UI + CLI over a Qdrant vector store,
embedding with DINOv2 and SSCD, matching by cosine similarity plus **exact**
hashing (SHA-256/MD5), with an env-gated **face modality** as a second identity
axis. **There is no perceptual hashing** — this line claimed it for two eras and
no such code has ever existed (`grep -riE 'phash|perceptual|dhash|imagehash'`
over `src/` returns nothing; `faces/chips.py`'s `ahash` is a SHA of the aligned
crop, not a perceptual hash). Corrected 2026-08-13 after the operator asked what
the "perceptual-hash modality" pending item meant — the question exposed the
claim. Third instance in this ledger of prose outliving the code it described.

**Design rule from `CLAUDE.md`: a feature must have forensic value and be
legible to a court; decorative features get removed.** Target deployment is a
distributed isolated LAN, fully offline.

## current state (2026-08-14)

- `main` at **`2dbfea6`**. Bar **882 passed / 5 skipped**, coverage 72.20%
  against the 65% floor — **re-measured by `m5` at `6e09097`**, in a worktree
  with `PYTHONPATH=$PWD/src` and `models/` present, no tracked file modified.
  **A worktree without `models/` reads one extra skip** (the YuNet test), which
  is how 821/5 and 820/6 reconciled at phase 4; *always say which tree you
  measured in*, not merely that it was clean. Was 771/5 at 70.22% before phase
  4: phase 4 added 50 tests, phase 5 added 61.
- Campaign complete: 8,137 files indexed. Cropper delivered as the standalone
  repo `portable_face_cropper` (8138→4537 crops, 0 failures, 35m01s).
- **Fleet: manager `com-m5`, one coder `com-c16` on phase 6.** `com-c12`
  (phases 1–3), `com-c13` (the carve), `csm-b1` (the §14 measurements),
  `com-c14` (phase 4) and `com-c15` (phase 5) all retired with ownership
  released; `com-m4` retired at ~210k. **Three handoffs are required reading
  for anyone touching `video_playback/`**:
  `docs/handoffs/scalarforensic-com-c14-20260813-220936.md` (patch targets,
  module state, the worktree `PYTHONPATH` trap, the encode-path facts),
  `...-com-c15-20260813-223743.md` (phase 5's cache, and phase 6's four
  concrete steps in its §3) and `...-com-m4-20260813-2100.md`.
- **The §14 measurements are safe in git** — `#155` `3ac5442`. They were
  produced into `data/reports/`, which `data/*` gitignores, and lived one `rm`
  from gone; they are now `docs/benchmarks/video-bench-2026-08-13.md`, copied
  verbatim with their caveats. Spec §3.5 carries the load-bearing figures, §3.4
  was rewritten down to the real residue (VFR, long-GOP, damaged indexes,
  multi-sample confirmation), §12 gained a defaults table tying each value to a
  number, §14 settled the fixture strategy, and **§17 Q1–Q5 are closed into
  §16**. *A benchmarker writing its report under `data/` is a loss risk the
  dispatch should preempt: name a tracked output path when you spawn one.*
- **PHASES 4 AND 5 ARE DONE** (`#157`/`#158`/`#159`, `#162`/`#163`). **Phase 6
  is in flight** with `com-c16`; phases 7–8 remain. The operator answered every
  open question on 2026-08-13. **Two things now sit with them**: the CodeQL
  dismissals, and whether to fund a benchmarker for the missing codec factor —
  both below.
- **The shared checkout is PARKED at `43a7222` and everyone works in
  worktrees.** `docs/CTO_LEDGER.md` is dirty there with **the operator's own
  handwritten `-->` answers**, and `#153`/`#154`/`#156` all rewrote that file
  upstream, so it cannot fast-forward. The substance of those answers is already
  folded (§16 of the spec, and below); the working-tree edit is the operator's
  to clear with `git checkout docs/CTO_LEDGER.md && git pull`. **Neither revert
  it nor commit it.** A preserved copy is in `m5`'s scratchpad. This is not a
  blocker for anyone: `git worktree add` off `origin/main` is the documented
  arrangement, and phase 4 was built entirely that way. **From a worktree, the
  suite needs `PYTHONPATH=$PWD/src`** — the venv is installed editable against
  the main checkout, so without it you test the *other* checkout's `src/` and do
  not notice. An untracked `docker-compose.override.yml` is nobody's — leave it,
  never stage it, and remember it makes `--porcelain` non-empty.

### shipped and verified in the m3 window

UI items 1–4; frame display at stored resolution end-to-end (`#128`–`#130`);
HEIC serving (`#123`); compare/point-probes (`#124`); the CLI trio
(`#131`/`#132`/`#134`); error surfaces (`#133`); in-browser playback of source
video from a frame hit (`#136`); query-face chip generation race fixed (`#137`,
proven by a pixel-signature check); runbook (`#138`); run-progress display
restored (`#139`).

### landed this session

- `#140` — recovered m3's runbook entry, written but never committed before its
  window closed.
- `#141` — the 5 **true** CodeQL alerts: `ci.yml` had no `permissions:` block
  (default token scope), and three handlers interpolated exception text into
  client responses. The worst was `video.py`'s `get_video_info` returning
  `{"error": str(exc)}` straight to the browser via `/api/query-metadata`.
- `#142` — **spec: on-demand video transcoding** (`0dbc1cd`),
  `docs/specs/video-playback-transcode.md`. This is the current work queue; §15
  is the phase plan.
- `#143`, `#144` — the CTO's last two folds.

### landed in the m4 window — spec phases 1–3 are COMPLETE

Three PRs by `com-c12`, in order, both required checks green on each. Bar
re-measured by the manager at the end, on a clean tree, not inherited.

- `#145` `1a6ad56` — **phase 1, digest correctness.** `_source_digest` is backed
  by the indexer's persistent `HashCache`, offloaded with `asyncio.to_thread`,
  and degrades to a direct hash when the DB is missing, disabled or corrupt —
  never a 500. Stale-evidence detection is **three-state on purpose**:
  `stale_evidence: None` means *not checked*, and only a real comparison clears
  a file. 725/5.
- `#147` `6539abb` — **phase 2, download original.** `/api/video-download`
  through the same resolution flow as every path-bearing route. Also killed a
  shipped defect the manager found reviewing `#145`: `videoPlaybackDigestMatchesHit`
  returned false when the indexed hash was merely *absent*, and the UI rendered
  that as "does NOT match the video_hash recorded for this frame" — **unknown
  displayed as mismatch, in an evidence viewer.** The UI now renders the
  server's three states and a real mismatch gets a banner, not a tooltip. 740/5.
- `#148` `b3a609a` — **phase 3, codec detection.** Server-side allowlist
  (`h264` 8-bit, `vp8` 8, `vp9` 10, `av1` 10; 4:2:0 only), `mode` gains
  `transcode` and `unknown` with a human reason sentence. No encoding.
  Also took the **`_evict_cache` containment fix**: candidates are restricted to
  top-level `{sha256}.mp4`, so the `full.mp4` and chunk artifacts later phases
  add can no longer be deleted mid-play. That is containment, **not** the §6.2
  lease/whole-video rewrite, which stays in phase 5 — the code says so. 771/5.

Two review findings the manager raised and `c12` folded in, worth keeping as
the shape of defect this codebase produces: a path interpolated into
`playback_url` **unquoted, one line above** a correctly `quote()`d
`download_url` (iPhone filenames contain `#` and `&`); and `video-download`
blocking on a full-file SHA-256 before the first byte moved — worst possible
place for a silent multi-minute stall, since that endpoint is the escape route
a *failing* playback points at. It now emits the digest header only from a
`HashCache` hit (new `HashCache.peek()`) and streams immediately on a miss; an
absent header means "not computed", never "unverified".

### the §11 carve is also DONE — `#150` `f6cef02`, `#151` `bda812b`

`com-c13`, one PR, **zero change in the test count** — the shape a pure move
should have. Layout as built: `video_playback/{__init__,codecs,digest,rewrap,
cache,routes}.py`; `routes/video.py` 779 → 106 lines, keeping only the indexing
side (`/api/video-frame`, `/api/video-timeline`); `_resolve_video_path` moved to
`routes/_shared.py` beside `_check_allowed_path`. **No compatibility
re-exports** — a missing name is the signal you want, and `test_video_playback.py`
reads ten private symbols off the module without patching. `CLAUDE.md` and spec
§11 updated in the same PR; §11's tree now marks which modules exist and which
arrive with their phases.

Two module placements were ruled by the manager, not escalated — §11's module
list was the spec author's drafting, whereas the operator ruled only "a
self-contained subsystem":

- **`digest.py`** for the source digest and the `HashCache` handle. Folding a
  *hash* cache into `cache.py`, the *artifact* cache, is the conflation §6.1
  warns about, and the handle is a process-wide singleton over an open SQLite
  connection — duplicate it across two modules and phase 1 silently regresses
  with no failing test.
- **`rewrap.py`** for `_remux_to_mp4` / `_repair_timestamps`. §2's own table
  calls this "lossless rewrap … reused unchanged": a PyAV stream copy,
  deliberately not an encode, and `encode.py` is the ffmpeg path.
- `_stale_evidence_report` stays in `routes.py` until `audit.py` arrives in
  phase 8; `audit.py` is not created early for one function.

**A false statement in §11 was corrected in the same PR and is worth naming**:
v1 said `_resolve_video_path` is "shared with `/api/video-frame` and
`/api/video-timeline`". The timeline handler takes a `video_hash` only and never
touches a filesystem path. Caught by the outgoing coder writing its handoff,
verified by the manager. Same failure mode as the "review floor 36" — a claim
that lived only in prose.

**The handoff is why the carve cost one PR.** `com-c12` was retired at 150.8k
*before* starting it, and spent its last turn writing every `mock.patch` target
string, the symbol placement, and the module-level state that must not be
duplicated. `test_video_endpoints.py` then needed zero changes, exactly as it
predicted. Retiring a coder one task early and buying a map is cheaper than
letting it start a churn-heavy refactor 40k below the retire nudge.

### landed in the m5 window — PHASE 4 IS COMPLETE

Four PRs by `com-c14`, both required checks green on each. Bar re-measured by the
manager on a clean tree, not inherited.

- `#155` `3ac5442` — the measurement fold (above).
- `#157` `952b325` — **`capability.py`**: a real 6-frame decode → tone-map →
  encode → mux probe, and `Pipeline`, the §6.1 fingerprint over nine fields with
  the field set pinned by a test. Its docstring records what is deliberately
  **not** in the key (worker count, cache dir and ceiling, queue/timeout,
  examiner, source path, chunk timecode) — the half that normally goes unwritten
  and then gets silently violated. `ffmpeg_version` **is** a field: two builds
  are two pipelines, and invalidating the cache on upgrade is the conservative
  direction for a rendering whose label names its pipeline (§7.2).
- `#158` `57246c4` — **`encode.py`**, ffmpeg declared in CI, the Dockerfile and
  `INSTALL.md`, and §3.1's two defects pinned by tests.
- `#159` `6f5879e` — `CLAUDE.md` conventions.

Three findings from this window worth keeping:

- **CI had no ffmpeg, so 5 of the new tests were green-by-skip** (795/11 in CI
  vs 820/6 local). `c14` raised it rather than shipping the silence. Fixed by
  declaring the dependency CI already had on paper, plus a guard — `_need_ffmpeg()`
  **raises** when `CI` is set and skips otherwise, so dropping the install line
  cannot quietly restore the skip. `qdrant-integration` deliberately has no
  ffmpeg (it runs one faces test); the workflow says so, and says what to do if
  its selection widens.
- **The §14 tests are mutation-verified, not assumed.** Adding `-noautorotate`
  makes the rotation test fail with "rotation was lost"; `c14` ran that and
  reverted it. ffmpeg 6 **cannot** write rotation onto an output stream
  (`-metadata:s:v rotate=` is accepted and silently does nothing) and PyAV
  exposes no stream-level side data, so the generated fixture carries a real
  display matrix patched into its `tkhd` box, documented to the byte offset. The
  assertion was not weakened to fit the tooling. **Do this for any test that
  pins a defect** — an unproven pin is a claim, not a test.
- **§8's "GPU path" is not §3.1's fastest row.** The 12.9× `scale_cuda` pipeline
  is the one that *breaks rotation*, so `hwaccel=cuda` in this codebase means
  software decode, software filtering, GPU **encoder only** — the 6.1× row. It
  is in `select()`'s docstring. Do not "optimise" it back.

Also `Capability.unavailable_reason(hdr=True)` **refuses** on a build without
libzimg rather than falling back, because encoding HDR without the tone-map chain
reproduces §3.1's second defect — 8-bit pixels still tagged `bt2020`/HLG. SDR
still works there. Three-state, deliberately, like `stale_evidence` and `mode`.

### phase 5 is COMPLETE — `#162` `3b191b5`, `#163` `955334c`

Two PRs by `com-c15`, both checks green. Bar 882/5 at 72.20%, re-measured by the
manager. `cache_key()` is `sha256(source ‖ fingerprint)`; the §6.2 lease/pin/
whole-video eviction **replaces `#148`'s containment**; `KeyedLocks` replaces the
unbounded `_remux_locks` (§10.4); `publish()` fsyncs file *and* directory;
`sweep_orphaned_parts()`; §6.3 ceiling refusal at the operator's 50% ruling; and
`sfn-video purge --media | --all` per §13.

**`chunk_seconds` stays in the key for both artifact kinds** — `c15`'s call,
recorded in `cache_key()`'s docstring and §6.1, and the manager agreed. Changing
`SFN_VIDEO_CHUNK_SECONDS` needlessly re-encodes cached `full.mp4`, a rare
recoverable cost; the alternative is a per-artifact-kind field table, which is a
*second* place for the §6.1 defect to live. Cheap side of the trade.

**Eight mutations across the phase, each caught by a named test, each reverted.**
Single-file eviction, empty keep-set, undeleted lock entry, `.part` excluded from
accounting, `unknown`→`fits`, `CEILING_FRACTION` 1.0, dropped area scaling, purge
ignoring `--media`. Combined with `c14`'s `-noautorotate` check, **mutation-proving
a defect-pinning test is now this project's standing expectation**, and it is
carried in the dispatch rather than left to be rediscovered.

**THE OPEN FINDING — `estimate_full_output_bytes()` has NO codec factor**, because
none has ever been measured: §3.5 timed the encodes and recorded no output sizes,
and §16 forbids inventing one. The error direction is knowable and bad: a CRF-23
H.264 encode of a 10-bit HEVC source at the same resolution is usually *larger*
than its source, so the estimate runs **low on exactly the HEVC corpus this
feature exists for** (57 of a 60-file sample). §6.3 now says plainly that it is a
screen, not a guarantee, and that **phase 7's job runner must check the growing
`.part` against the estimate and abort on overshoot**. Closing it properly means
measuring output sizes for the §3.1 rows against the operator's corpus — a
benchmarker task, not a coder's, and it needs an otherwise-idle box.

## the video playback work — READ THE SPEC, IT IS THE PLAN

`docs/specs/video-playback-transcode.md` is **draft v2, post-review**. v1 proposed
a lazy-HLS segment engine; two independent adversarial reviews (Codex + Opus,
2026-08-13) both returned "do not merge". The operator then chose a simpler shape
— two single-encode MP4 artifacts, a 30 s chunk on play plus an explicit
background full-file job — which removes most of the blockers by construction
instead of answering them.

**Do not re-propose HLS/MSE/CMAF.** §4.1 records why it was rejected and what
evidence would be needed to revisit it. The reviews are the reason the spec is
trustworthy; their findings are already integrated, so the spec's §3 is careful
about what was measured and §3.4 lists what was not.

**Sequencing that matters:**

- **Phases 1–5's predecessors are all merged**: phases 1–3 (`#145`, `#147`,
  `#148`), the §11 carve (`#150`, `#151`), the measurement fold (`#155`) and
  **phase 4** (`#157`, `#158`, `#159`). **Phase 5 is in flight**; phases 6–8
  follow in §15's order.
- **Phase 5 is no longer blocked**: `Pipeline.fingerprint()` exists, and it is
  the second half of §6.1's cache key `sha256(source identity ‖ pipeline
  fingerprint)`.
- **`#148` CONTAINED the `_evict_cache` defect; it did not fix it.** Candidates
  were narrowed to top-level `{sha256}.mp4` so chunk artifacts could not be
  deleted mid-play. The §6.2 lease/whole-video rewrite is phase 5's, and the
  code says so. Do not read the containment as the fix.
- **`cache.py`'s `_remux_locks` grows without bound** — an `asyncio.Lock` per
  source digest, never cleared. §10.4 names it; phase 5 owns it.
- **Spawn ONE fresh coder per phase, not a continuation.** §11's tree marks
  which modules exist; `jobs.py` is phase 7 and `audit.py` is phase 8, and
  neither is created early as an empty module.

`docs/benchmarks/video-bench-2026-08-13.md` is the §14 measurement set, in git
since `#155`. Read its caveats with its numbers: the concurrency figures are
sound, the multi-hour seek figure is a best-case index by construction, and the
4K rows are one sample each.

## closed rulings — do NOT re-escalate

- Uncalibrated per-face cosine IS displayed, labelled raw; 0.363 is the model
  authors' reference figure and never a threshold, filter or default
  (spec `face-pipeline.md` §10).
- SFace is the embedder (Apache-2.0, OpenCV Zoo). Floors lowered
  `SFN_FACE_MIN_SIZE` 64→40, `SFN_FACE_REVIEW_MIN_SIZE` 48→24 (`b91fd61`).
- `purge_all` must never `delete_collection` — the enablement record is an
  auditable act and survives routine purges.
- `sfn_tags` collection dropped. DBs are test-only: drop + reingest is free
  (~3 min for 10%).
- Restarts are **always** allowed (standing operator grant); canonical recipe
  (`setsid nohup`) is in the runbook.
- **Stored-data-first** is standing: never a UI workaround over missing data.
- **The ingestion run-progress display is an operator exemption from the
  decoration rule.** `#134` removed it; the operator reversed that and `#139`
  restored it. Do not re-remove. The Kalman ±1σ "calibrated" band stays gone —
  its Q/R were hand-picked constants, which is the same defect class as an
  uncalibrated face cosine.
- **20 `py/path-injection` CodeQL alerts are false positives** (17 + 3 added by
  `#162`, when `POST /api/video-lease` became a new source into
  `_resolve_video_path`; the manager read `_shared.py:53-63` and confirmed the
  class rather than inheriting the ruling), verified
  independently this session, not inherited: every flagged path passes through
  `_check_allowed_path` (`routes/_shared.py:12`), which calls `resolve()` —
  normalising traversal and symlinks — *before* `relative_to()` against resolved
  roots, and fails closed when no root is configured. CodeQL does not model
  `Path.relative_to` as a sanitizer. Dismissal needs `security_events` scope, so
  it is an operator action; agents cannot do it.
- Cropper disk-only risk is **CLOSED**: `portable_face_cropper` has a GitHub
  remote and `main` (`8dc490c`) matches it. The `c58c58f` in older notes is
  merely an ancestor.

## standing rules

- **Forensic value is the acceptance test, not feature completeness.**
- **`SFN_FACE_STORE_DIR` is set PER CASE** (operator discipline, not an invariant).
- Commit **explicit paths**, never a broad `git add`.
- **A default-pinning test must supply its OWN empty env file** — `.env` leaks
  into the process env and `find_dotenv()` fallback pins nothing.
- A value that appears only in prose and a mock needs `git log -S` before it is
  trusted (the "review floor 36" that never existed in code).
- **Never quote a test bar against a dirty tree**; `git status --porcelain` must
  be empty first.
- **Retire a coder one task early and buy a map.** A worker approaching the
  retire nudge should spend its last turn writing the handoff, not starting the
  next unit of work. `com-c12`'s handoff is why the §11 carve cost one PR; the
  alternative this ledger already records is an agent dying holding uncommitted
  work.
- **A two-state boolean over a three-state question is a fabricated claim.**
  "Not checked" must never render as "checked and failed". This is the same
  defect class as the uncalibrated face cosine and `#139`'s fake error band,
  and it had shipped in `computed.js` until `#147`. Whenever a UI states a
  verdict, ask what it shows when the input is *absent*.
- **Verify an inherited verdict before acting on it.** Two claims carried in
  this ledger turned out stale within one session (the cropper remote, the test
  bar). Checking cost one command each.
- **The operator annotates their working copy of this file by hand.** Their
  answers arrive as `-->` margin notes in an uncommitted
  `docs/CTO_LEDGER.md`. Fold the *substance* into the ledger properly and
  leave the working-tree edit alone: never commit the margin notes into the
  decision record, and never `git checkout` them away — reverting a user's
  edit is not an agent's call. Write your own folds from a `git worktree` off
  `origin/main` so their working copy is never in your way.

## how this fleet fails — kept instances

- **cto8 verified the code and INFERRED the author, then ordered on the
  inferred half.** Attribution is load-bearing; `git log -1 --format=%an` costs
  one command. Subordinates checking orders caught it.
- **m3 published 526/5 from a dirty-tree run** (a parallel worker's uncommitted
  tests were counted). Now a `CLAUDE.md` gotcha.
- **Ad-hoc env exports became "documented facts"** twice (review floor 36;
  `SFN_FACE_COLLECTION` load-bearing but recorded nowhere).
- **An agent died holding uncommitted work.** m3 wrote a runbook entry and its
  window closed before it committed; only an ownership-row check found it.
  Recovered as `#140`. A dead agent's owned paths are worth `git diff`-ing
  before the row is reaped.
- **`m5` opened by ordering its predecessor on two premises it had not
  checked** — that `m4`'s handoff did not exist, and that a dirty
  `docs/CTO_LEDGER.md` was `m4`'s uncommitted work. Both were wrong: the
  handoff was merged in `#154` and the local checkout was two commits behind
  `origin/main`, and the diff was **the operator's own handwritten answers** in
  their working copy, already folded properly by `#153`. `m4` refused and
  corrected it. **A stale `main` makes "the file does not exist" a lie your own
  tooling tells you — `git fetch origin main` before concluding anything is
  missing**, and never attribute an uncommitted diff without `git log`/`git
  diff` first. This ledger's own standing rule ("verify an inherited verdict")
  applies to verdicts you form yourself in your first five minutes.

## operator rulings, 2026-08-13 — PHASE 4 IS UNBLOCKED

All five open items were answered in one pass. Recorded as closed rulings; do
not re-escalate them.

1. **HDR test fixture — RULED: commit a small public sample under a tracked
   `test_data/`.** This dissolves the blocker rather than answering it: the
   constraint was only ever that `data/` is gitignored, and a *new tracked
   directory* has no such rule. `test_data/` is for fixtures a test needs and a
   reviewer must be able to see; it is not `data/`, which stays gitignored for
   evidence. Manager's implementation note: the committed clip must be genuinely
   10-bit HDR (`yuv420p10le`, `bt2020`, HLG or PQ) and redistributable, and the
   **rotation** case is derived from it at test time with an `ffmpeg -c copy`
   metadata add — public HDR clips do not carry the `rotation=-90` side data that
   is an iPhone artifact, and stream-copying it on costs nothing and stays exact.
   One committed file covers both §3.1 defects. If no suitable public clip is
   small and licence-clean, generate one with `ffmpeg -f lavfi` and commit *that*
   — a generated file in `test_data/` satisfies the ruling's shape and is
   licence-clean by construction.
2. **Hardware — RULED: this PC is the deployment target.** Not a proxy for it.
   That collapses §17 Q3: the **minimum hardware floor is this machine's measured
   spec**, recorded as a measurement of the target rather than an extrapolation.
   The §14 measurements run here, and the 4K and multi-hour sources are
   synthesised because the corpus has none — **which means the seek numbers are a
   best case and understate the damaged-index, long-GOP and VFR risk §3.2 names.**
   That caveat travels with the number, in the §3.4 style.
3. **§17 Q1 output resolution — RULED: 1080p cap.** "Full quality can be achieved
   by download" — which is exactly §7.4's position, that fine detail is judged
   from the original and never from the rendering. The cap and the Download-original
   route are one decision, not two.
4. **§17 Q5 — RULED as recommended:** refuse a full-video job before start above
   an estimated 50% of `SFN_VIDEO_CACHE_MAX_BYTES`, showing the estimate and
   offering Download original.
5. **CodeQL — RULED: the operator dismisses the 17.** Agents still cannot; it
   needs their `security_events` scope.

## pending user decisions

Four items the operator asked *about* rather than ruled on. Answers below; none
blocks any work.

- **`stale-observation-purge.md`** — a design note for a command that does not
  exist (`docs/specs/stale-observation-purge.md`, status "proposal, not
  implemented"). The real gap it names: stale face observations are detected
  only *inside* an index run, and a re-run skips the medium, so the operator is
  never asked again and the keys survive only in `face_audit.log`. The open fork
  is how a standalone command would find them — re-detect (faithful, costs a
  full detection pass) or config-hash diff (cheap, but over-reports, so it is
  usable for *inspect* and never as a delete predicate). **Nothing is broken
  today**; this is deferred work, not a defect.
- **Audit-button label** — not a new feature. It is the shipped `DINO Audit` /
  `FACE Audit` button pair in the analysis toolbar (`index.html:1055`, `:1067`).
  An earlier window asked whether the wording should change and got no answer;
  the string stands. Nothing to review unless the operator dislikes the label.
- **Perceptual-hash modality** — **there is none, and there never was.** See the
  correction at the top of this file: the "what this is" line claimed perceptual
  hashing for two eras against code that has never contained any. Matching is
  DINOv2/SSCD cosine plus exact SHA-256/MD5. The item was asking whether to
  *add* a perceptual-hash axis; it remains unbuilt and unproposed.
- **Cropper `.gif`** — ScalarForensic **does** support GIF (`scanner.py:17`).
  The skipped files were 2 `.gif` inputs in the separate `portable_face_cropper`
  run; nobody ruled on re-running to include them. Low stakes, different repo.

Closed since the last fold: **HEVC remedy** — superseded by the `#142` spec. The
answer is on-demand segment transcoding keyed to what the analyst actually
watches, not pre-built viewing copies; the earlier NVENC-transcode
recommendation was measured and found to silently drop rotation metadata.
**Cropper remote** — was never actually open; the push had already happened.
