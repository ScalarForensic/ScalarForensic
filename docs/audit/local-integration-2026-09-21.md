# Local parser integration validation — 2026-09-21

## Source and disposition

Local `main` was fast-forwarded from `5dd6102` to
`6c61f06919c082ed3b64af11cabdbccfd9b054aa`. This includes the eight previously
separate image/video isolation commits through `4af8e0c` and the reviewed fixes
in `6c61f06`. No commits remain unique to `chore/dead-weight` relative to `main`.
The worktree and its private historical evidence are retained.

Source integration and local regression verification are complete. No remote
fetch, pull, push, publication, deployment, database migration or host update was
performed. The pre-existing untracked `README2.md` and
`docker-compose.override.yml` were preserved and their SHA-256 hashes rechecked.

## Review corrections

- Explicitly empty image-worker selection settings now raise a configuration
  error instead of allowing local decoding. Managed/static settings cannot mix.
- Image clients check the shared generation failure notice before submitting
  work and before accepting a response. A response arriving after invalidation
  is discarded and recorded as failed; removing a notice does not revive an
  already invalidated client.
- Image protocol v2 binds every request to the application's effective Pillow
  pixel guard. The worker verifies equality before decoding, keeping its startup
  guard fixed across concurrent requests. The supervisor forwards the canonical
  setting and the static Compose template forwards the configured override.
- Receipts record the guard and protocol version. Generation journals attribute
  the guard only on the launch event, so recovery of an older generation cannot
  incorrectly assign the new supervisor's configuration to it.

The seven configuration/invalidation regressions and four pixel-limit/launcher
regressions failed before their fixes and passed afterwards. A further regression
reproduced and closed the recovered-generation attribution issue. Independent
review found no remaining integration blocker after these corrections.

## Verification

The final source was tested both in the isolated worktree and after integration
from the main checkout, with `PYTHONPATH` explicitly selecting that checkout's
`src`. The worktree used the main checkout's existing Python 3.12 environment;
dependencies were not reinstalled.

| Check | Result |
| --- | --- |
| Full Python suite, corrected worktree | 1,464 passed, 9 skipped; 83.15% coverage; 68.95 s |
| Full Python suite, merged main | 1,464 passed, 9 skipped; 83.15% coverage; 69.78 s |
| Focused image/worker suite | 180 passed |
| Ruff check and format | Passed; 170 Python files formatted |
| JavaScript | 12 test files passed |
| Playwright | 2 synthetic browser smoke tests passed |
| Managed and static image-worker Compose configuration | Both passed |
| Whitespace and modified documentation link targets | Passed |

Full Python command from `main`:

```sh
PYTHONPATH="$PWD/src" timeout 240 .venv/bin/python -m pytest -p no:cacheprovider -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
```

The nine skips require a disposable `SFN_TEST_QDRANT_URL`; no live case database
was used. The final suite reported 50 warnings in the existing Starlette/httpx,
local-Qdrant, TorchScript and multithreaded-fork categories. Node and Playwright
ran before the backend corrections; their frontend source and smoke fixtures
were unchanged by those corrections. Playwright uses synthetic API responses,
not a complete isolated deployment.

The initial sandboxed Python run stalled and was interrupted; Chromium failed
at launch with `Operation not permitted`. Permission-enabled local reruns passed.
No application isolation policy was weakened. Both Compose checks used
`.env.example` and explicitly selected tracked files, excluding the operator's
override. Test logs are retained locally under
`.cleanup-notes/evidence/local-integration-2026-09-21/` and are not release artifacts.

## Remaining release prerequisites

The Docker daemon socket was unavailable even outside the execution sandbox.
Consequently this session did not rebuild images, repeat container denial/GPU
checks, rescan native dependencies, or rehearse a database restore/upgrade.

Image workers must be rebuilt and their supervisors restarted alongside the
updated application because protocol v1 workers are intentionally incompatible.
Follow the [image-worker upgrade instructions](../deployment-media-worker.md#image-limits-and-upgrades)
before enabling the profile. Source integration does not enable it automatically.

The [September 12 security findings](internal-nvidia-2026-09-12/README.md)
remain historical findings requiring current artifact assessment; containment
does not patch native libraries or establish host-kernel safety. No new security
or redistribution acceptance is claimed. Existing-volume migration, AMD and
redistribution validation remain outside this local source verification.

The [standalone stale-face inspect/purge command](../specs/stale-observation-purge.md)
remains a deferred feature proposal. It is separate from the completed isolation
implementation and was not added or exercised against case data here.
