# Handoff — scalarforensic-cfm-m1, 2026-08-13 (iPhone campaign prep)

Successor: read `docs/fleet/runbook.md` (entry "[ScalarForensic, cfm-m1, 2026-08-13]")
first — it carries the full state. This file adds only what a successor needs to act.

## State in one paragraph

All five prep items are DONE; the ingestion run is deliberately NOT started (operator
triggers it interactively). Main is at `906c4b5` (runbook merge #113); the bar is
**561 passed / 5 skipped** at `64908be`, verified on a clean tree WITH the campaign
`.env` present, ruff check+format rc=0. Merged PRs: #109 (pillow-heif in dev group),
#111 (SFace batch-1 chunking + suite `.env` hermeticity), #112 (efficiency audit
`docs/fleet/pipeline-efficiency-2026-08-13.md`), #113 (runbook state).

## Per-stage numbers (from the audit, 43-image sample + 509-video probe)

HEIC decode 171 ms vs JPG 19 ms (no HEIF draft mode); face pass re-decodes native-res
(HEIC 103 ms) + YuNet 35 ms; DINO 27.8 ms/img warm @512 fp16, SSCD 15.5 ms 5-crop;
GPU util 35–55 % mean, VRAM 4.4 GB @43 imgs. Videos: 4,856 s total footage → ≈4.9 k
frames at 1 fps, cap 500 never binds, slicing ≈27 fps wall. Projection: **~35 min
full-corpus wall (upper 45), GPU busy ~9 min, face pass dominates at ~17 min.**
No perceptual hash exists in the codebase; sha256+md5 only.

## Machine-local, deliberately untracked

- `docker-compose.override.yml` — publishes Qdrant on `127.0.0.1:6333` (the compose
  file's own documented debugging pattern, `docker-compose.yml:46-49`); the container
  was recreated with it, volume intact. Without it the `.env`'s
  `SFN_QDRANT_URL=http://localhost:6333` is unreachable from the host.
- `.env` — campaign config (collection `iphone_campaign_2026`, faces enabled, all
  derived data on `created_by_scalar/`). Gitignored as always.

## Open items (unassigned)

1. Per-run manifest into `created_by_scalar/` (config snapshot + model hashes + file
   list); fold the CSV `--report` path into config while at it (env var missing).
2. Parallelize the face pass (~13 min saving; per-thread YuNet instances needed).
   Post-run acceptable per audit.
3. DECIDE at g1: drop `sfn_tags` (holds tags referencing the deleted
   `danny_validation` points).

## Inline-vs-dispatch justification (asked by g1)

(1) Inline was deliberate: `cx n` was filed BROKEN by my predecessor (spawn.py:341,
half-spawned orphans) and never confirmed fixed, and the items were small,
serial, and verification-heavy (profile → interpret → fix → re-measure). (2) Cost
of dispatching these: each item needed the profiler context to review anyway —
the audit/fix loop would have round-tripped through me at comparable wall time;
the real cost of inline was context burn (retired at ~170 k). (3) Recommendation:
successor dispatches the manifest and face-pass-parallelization items to coder
workers (they are now well-specified), keeps profiling/verification and anything
touching the shared checkout's branch state to itself.

## Review

- Bar re-verified before this handoff: `git status --porcelain` clean except the two
  named untracked files; suite run includes the campaign `.env` (hermeticity fixture
  active). PR merge states checked via `gh pr view` (#109/#111/#112/#113 all MERGED).
- Qdrant end-state verified by REST: collections = `sfn_tags`,
  `faces_danny_validation` (1 point: the enablement record); `localhost:6333` answers.
- Deletion executed exactly per operator list; `data/models`, `data/sample_images`,
  `data/face_audit.log` retained (calibration and audit-trail dependencies).
