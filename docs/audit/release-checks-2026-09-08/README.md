# Release check follow-up — 2026-09-08

**Further verification:** [Complete image and whole-case restore results](image-and-restore-results.md) now record offline execution of all bundled models, GPU ingestion, real whole-state restoration, image scans and a local two-image candidate bundle. Two image-build defects were fixed. Security/redistribution acceptance remains open.

## Completed

- Aligned Compose, CI, airgap bundling and installation examples to Qdrant server **1.19.0**, matching the locked Python client **1.19.0**. Pulled image digest: `sha256:057ee3a8da769fe7310dd3537b4dc7583bf87a95ce8ac43c0af5a46bc580d1fc`.
- Reproduced the incompatible 1.19.0/1.17.1 pair with the client's explicit compatibility predicate against a disposable server. The same check passes against 1.19.0. The client checks compatibility on a daemon thread, so a short process with no warning is not sufficient evidence.
- Added `tests/test_qdrant_release.py` to the existing CI integration gate. It verifies compatibility and downloads/uploads a real Qdrant collection snapshot into a fresh collection, comparing point IDs, payloads and vectors and confirming review-only points remain excluded from search. This is a same-version, same-server synthetic restore; it does not verify whole-case restoration or migration of existing volumes.
- Nine targeted integration checks passed. The final full suite after dependency changes passed **1,122 tests**, **80.44% coverage**, in **46.34 seconds**. The original `pytest.log` is not present in the repository; these are the recorded historical results. Seventeen existing warnings concern Starlette deprecation, local Qdrant behavior and TorchScript deprecation; none reports client/server incompatibility.
- Ruff check/format (136 files), all three explicit Compose variants, shell syntax, `git diff --check` and offline lock consistency passed.
- Real offline DINOv2 inference passed on an **RTX 4060 Ti**, Torch **2.11.0+cu128**, cuDNN **91900**, with `torch.compile` enabled. Two synthetic RGB inputs produced distinct, finite, normalized 1,024-dimensional vectors. See [gpu-dino.json](gpu-dino.json). No evidence images were loaded.
- Host GPU diagnosis: the inherited `LD_PRELOAD=/usr/NX/lib/libnxegl.so` caused an abort with `Invalid handle. Cannot load symbol cudnnGetVersion`. Removing `LD_LIBRARY_PATH` alone did not resolve it; clearing both overrides for the child process allowed the minimal convolution and full DINOv2 inference to pass. No persistent host environment or application workaround was installed.

## Advisory results

The live PyPI advisory scan covered the locked core, web, faces and HEIF dependency export (77 entries). [Audited requirements](audited-requirements.txt), [Python results](python-advisories.json) and [npm results](npm-advisories.json) are preserved. The export is a record for advisory lookup, not an installation file; release installs continue using hashed exports.

- Updated locked **h2 4.3.0 → 4.4.1**, clearing `PYSEC-2026-3628` / `GHSA-6hr6-w5qg-qmwg`.
- **setuptools 81.0.0 remains flagged** for `PYSEC-2026-3447` / `GHSA-h35f-9h28-mq5c`. The scanner reports two records for the same advisory, not two distinct issues. Its reported fix is 83.0.0, but Torch 2.11.0 requires `setuptools<82`. An explicit resolver check confirmed the conflict. The final lock preserves Torch 2.11.0, Torchvision 0.26.0 and the existing CUDA package versions. Resolve through a supported upstream dependency combination or a documented release risk assessment; no advisory was ignored in the scan.
- The setuptools advisory concerns Unicode exclusions during source-distribution creation, particularly on macOS filesystems. This is not evidence of an exploitable application endpoint; it still needs release triage.
- `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` were **not auditable through PyPI**. These skipped records prevent a claim of complete coverage.
- npm audit reported **zero vulnerabilities** in the locked JavaScript dependency set.

Official references: [Qdrant 1.19.0 release](https://github.com/qdrant/qdrant/releases/tag/v1.19.0), [upgrade guidance](https://qdrant.tech/documentation/upgrades/), [h2 advisory](https://github.com/python-hyper/h2/security/advisories/GHSA-6hr6-w5qg-qmwg), [setuptools advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c).

## Remaining acceptance

The complete image, all-model smoke, GPU ingestion and synthetic whole-case restore now pass; see the [image/restore report](image-and-restore-results.md). Image scans and the [license/advisory review](license-advisory-review.md) are complete as evidence gathering, with findings still requiring remediation or release-owner acceptance. AMD, production-scale targets and upgrades of existing live storage were not validated.

The earlier snapshot checks used disposable container `sfn-release-check-20260908` on loopback port 16333 with no host mounts; it was removed after verification. The existing live Qdrant 1.17.1 container, live data, operator `.env` and Compose override were not changed. Work remains uncommitted.
