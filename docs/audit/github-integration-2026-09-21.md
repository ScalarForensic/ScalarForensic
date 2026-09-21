# GitHub integration preparation — 2026-09-21

This integration packages the completed local source, cleanup, and face-crop
fixes on top of GitHub `main` at
`d9505e08e5daa12893b2ecf72678ae8354d2280c`. The earlier 23 local commits are
consolidated from the tracked state at
`735f496f4b6f9d0b614ca47a98e2de667995e902`; that local commit and its private
snapshot ancestors are not parents of this integration branch.

## Publication boundary

- Only tracked source, tests, deployment files and documentation are included.
- No new or modified photos are included relative to GitHub `main`. The
  existing curated image blobs are identical to those already on GitHub.
- New files in `data/sample_images/` are ignored by default. Existing tracked
  fixtures remain tracked; adding another sample requires an intentional override.
- Local operator configuration, personal photos, scratch notes, model artifacts
  and dependency directories are excluded. Local originals are retained.
- The eight compressed audit additions contain JSON records or text logs;
  they do not contain image payloads.

The historical private-branch statements in older audit documents describe
their original local branches, not this branch's ancestry. Preparing this
branch does not constitute a deployment or approval to redistribute the
historical container candidates described in those records.

## Changes

The integrated work hardens ingestion recovery, source identity, query/session
lifecycles and provenance; introduces supervised opt-in image/video parsing
workers; and removes superseded compatibility paths and coordination material.
The [local integration record](local-integration-2026-09-21.md) describes the
worker implementation and deployment prerequisites.

The subsequent cleanup removes redundant wrappers, source-text tests and stale
comments. Face processing now uses roll-invariant pose/landmark checks, creates
thumbnails directly from native crop pixels, and records the quality-gate
algorithm version while preserving legacy stored-probe compatibility. See the
[face gate contract](../specs/face-review-embedding-split.md).

## Verification

The publication worktree used the existing locked Python 3.12 environment with
`PYTHONPATH` selecting its own `src`. No dependencies were reinstalled. All 461
tracked files matched the approved local source before adding this record.

| Check | Result |
| --- | --- |
| Full Python suite with coverage | 1,497 passed, 9 skipped; 83.19% coverage; 74.91 s |
| JavaScript | 12 test files passed |
| Chromium smoke tests | 2 passed, using synthetic API responses |
| Ruff lint and format | Passed; 170 Python files formatted |
| Base, AMD and NVIDIA Compose configurations | Passed with `.env.example` and explicitly selected tracked files |
| Whitespace check | Passed |
| Face-fix review | No actionable findings |

Python ran with `-m pytest -p no:cacheprovider -q -rs`, coverage enabled, and
`SFN_TEST_QDRANT_URL` unset. All nine skips are disposable-Qdrant integration
checks: six face-store cases, two server/snapshot cases and one interrupted-ingest
case. The Docker daemon socket is unavailable locally; the existing
`qdrant-integration` GitHub job must pass before merge. No live case database
was used. The other required GitHub job is `lint-and-test (3.12)`.

The 50 Python warnings come from Starlette/httpx deprecation, local-Qdrant
limitations, TorchScript deprecation and multiprocessing fork usage. Browser
tests also emitted the existing `NO_COLOR`/`FORCE_COLOR` warning. No test failure
was suppressed. Test logs remain in ignored local scratch storage.

This verifies the source integration, not current container security, GPU
deployment, existing-volume upgrades or release-scale model accuracy. Worker
protocol changes still require the rebuild/restart procedure documented in
[deployment-media-worker.md](../deployment-media-worker.md#image-limits-and-upgrades).
