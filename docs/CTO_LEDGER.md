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
embedding with DINOv2 and SSCD, matching by cosine similarity plus exact and
perceptual hashing, with an env-gated **face modality** as a second identity
axis. **Design rule from `CLAUDE.md`: a feature must have forensic value and be
legible to a court; decorative features get removed.** Target deployment is a
distributed isolated LAN, fully offline.

## current state (2026-08-13, late)

- `main` at **`bda812b`**. Bar **771 passed / 5 skipped**, coverage 70.22%
  against the 65% floor — **re-measured by `m4` on a clean tree at `f6cef02`**,
  twice, not taken from either coder's report; `#151` is docs-only. Was 710/5 at
  69.35%; phases 1–3 added 61 tests and the §11 carve added zero, which is what
  a pure move should show. **`CLAUDE.md`'s stale 559/5 line is now fixed** and
  cites `f6cef02`.
- Campaign complete: 8,137 files indexed. Cropper delivered as the standalone
  repo `portable_face_cropper` (8138→4537 crops, 0 failures, 35m01s).
- **Fleet: manager `com-m4` only; no live workers.** `com-c12` delivered phases
  1–3, `com-c13` the carve; both retired with their ownership released. Two
  stranded `cfm-g1` rows were reaped at the manager's open. **Everything
  downstream is blocked on the operator** — nothing is left running, on purpose.
- **The shared checkout is a real constraint, not a formality.** One working
  tree serves every session, so a coder on a feature branch means the manager
  cannot commit its own docs there without riding along in the coder's PR.
  `git worktree add` off `origin/main` is the way to commit owned docs while a
  branch is checked out; do not switch the shared checkout out from under a
  worker. An untracked `docker-compose.override.yml` is nobody's — leave it,
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

- Phases 1–3 and the §11 carve are **DONE and merged** (`#145`, `#147`, `#148`,
  `#150`, `#151`). **Phase 4 is next and there is nothing unblocked before it.**
- **Phase 5 is blocked behind phase 4, not merely queued**: the cache key is
  `sha256(source identity ‖ pipeline fingerprint)` (§6.1), and the pipeline
  fingerprint is produced by phase 4's `capability.py`, which does not exist.
  Do not start phase 5 "in parallel".
- Phase 4 is **blocked on two things**: the HDR test-fixture decision (§14 — a
  gitignored `data/` means no committed sample; generate one or gate-and-skip like
  the YuNet test), and the real-hardware measurements §14 requires (4K rates,
  long-source seek, concurrent scaling, minimum hardware floor).
- §17 carries five open questions; 1, 3 and 5 are operator calls — all three
  are in "pending user decisions" above with a recommendation.
- **When phase 4 is unblocked, spawn ONE fresh coder**, not a continuation. It
  inherits `video_playback/` with `encode.py`, `capability.py`, `jobs.py` and
  `audit.py` unwritten, and §11's tree marks exactly which modules exist.

The `_evict_cache` `*.mp4` glob defect the spec found is **contained in `#148`**
and no longer an open item; the full §6.2 rewrite remains phase 5.

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
- **17 `py/path-injection` CodeQL alerts are false positives**, verified
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

## pending user decisions

**Phase 4 of the video work is blocked on 1 and 2.** Both are put to the
operator with a recommendation; the full argument is the `com-m4` runbook entry
of 2026-08-13.

1. **HDR test fixture** (spec §14, §17 Q4) — `data/` is gitignored so no 10-bit
   HDR sample can be committed, and §14's rotation and `bt709` colour-tag tests
   are exactly the two defects §3.1 found. *Recommend both, layered*: a
   generated `ffmpeg -f lavfi` fixture as the CI default (ffmpeg is already a
   §8 dependency, so this adds none) plus a gate-and-skip on an env-supplied
   real-corpus path, YuNet-style, for what a synthetic clip cannot honestly
   assert. Generated-only makes green CI mean less than it looks; gated-only
   means the assertions never run.
2. **Real-hardware measurements** (§14, §17 Q3) — every §3 number is 1080p,
   single-job, on the idle dev host; §3.4 says so. Three questions: is the dev
   host representative or is there target hardware; are there real 4K and
   multi-hour sources to point a benchmarker at (§3 saw max 318 s, median
   2.0 s) or should it synthesise; and what is the minimum hardware floor —
   that one is a policy statement, not a measurement. *Recommend* holding the
   `csm` benchmarker until the first two are answered, and scheduling it after
   phase 3 merges regardless: concurrency scaling is only valid on an
   otherwise-idle box, so it must not run while a coder is running the suite.
3. **Spec §17 Q1, output resolution** — *recommend cap at 1080p by default*,
   never upscale, rescale disclosed per §7.4, operator-overridable. 4K is
   unmeasured, and §7.4 already says fine detail is judged from the original.
4. **Spec §17 Q5, full-video job near the ceiling** — *recommend refusing
   before start* above an estimated 50% of `SFN_VIDEO_CACHE_MAX_BYTES`, showing
   the estimate and offering Download original. §6.2's lease cannot save the
   watched video when the overflowing artifact is the job's own output.
5. **CodeQL dismissals** — 17 verified false positives; one-liner supplied to
   the operator, needs their token scope.
6. **`stale-observation-purge.md`**: re-detect vs config-hash diff (carried).
7. **Audit-button label**: shipped string + subtitle stands unless objected (carried).
8. **Perceptual-hash modality** absent (carried).
9. **Cropper `.gif` inputs** — 2 files were skipped on the delivered run; nobody
   has ruled on whether to re-run including them (carried, low stakes).

Closed since the last fold: **HEVC remedy** — superseded by the `#142` spec. The
answer is on-demand segment transcoding keyed to what the analyst actually
watches, not pre-built viewing copies; the earlier NVENC-transcode
recommendation was measured and found to silently drop rotation metadata.
**Cropper remote** — was never actually open; the push had already happened.
