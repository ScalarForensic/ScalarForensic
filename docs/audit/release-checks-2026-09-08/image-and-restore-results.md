# Image and whole-case restore verification — 2026-09-08

## Result

Built and exercised a complete **local NVIDIA validation candidate**, including the core, web, faces and HEIF dependencies and all four model artifacts present in `models/`. Functional checks passed. **Security and redistribution acceptance remain open; this is not an approved release.** No production deployment, live-data operation, commit or push occurred.

Candidate tag: `scalarforensic:release-check-20260908`.
Application image ID: `sha256:e447fb5a1f85c5cc08d72710640af1041fbb4469a49f80f57a351c97e8bc5701`.
Qdrant: `qdrant/qdrant:v1.19.0`, digest `sha256:057ee3a8da769fe7310dd3537b4dc7583bf87a95ce8ac43c0af5a46bc580d1fc`.
Exact metadata: [final-images.json](final-images.json).

The compressed two-image bundle was stored locally at `release-artifacts/2026-09-08/scalarforensic-validation-candidate.tar.gz`, with `SHA256SUMS` beside it. Loading the individual exported images and final compressed bundle was recorded as passing. The original `combined-archive-load.log` is not present in the repository. Artifact files, downloaded wheels and requirements exports are ignored by Git.

## Build findings fixed

1. Real setuptools wheels contain nested metadata for vendored dependencies. `bundle_inventory.py` previously rejected them as multiple wheel metadata records. It now selects only root-level distribution metadata while retaining native-member/license-file inventory. The synthetic vendored-metadata regression failed before the fix and passed afterwards.
2. The initial image installed GCC without the standard C development headers. Real offline GPU inference failed in Triton with `fatal error: stdlib.h: No such file or directory`. Added `libc6-dev` to the Dockerfile. The corrected image completed compiled DINOv2 and SSCD inference.

The build used 82 wheels with hash-checked, offline Python installation; OS packages/base image were obtained during the connected build. This validates **offline operation**, not building the Debian base from an airgapped host. The recorded wheel/model/native inventories describe files, not redistribution rights. The original `image-build.log` and `bundle-inventory.json` are not present in the repository, so their full contents cannot be recovered from this report.

## Runtime evidence

| Check | Result and boundary |
| --- | --- |
| All model artifacts | Passed with `--network none`, UID 10001 and read-only synthetic evidence. DINOv2 produced two distinct normalized 1,024-dimensional vectors and SSCD two distinct normalized 512-dimensional vectors, both with GPU compilation enabled. YuNet executed (zero detections on patterns); SFace produced normalized 128-dimensional vectors from synthetic crops. This is model execution, not face-accuracy validation. |
| Host/GPU | NVIDIA RTX 4060 Ti, Torch 2.11.0+cu128. Docker GPU passthrough passed. The image did not inherit the host NoMachine preload that affected the previous host-only test. |
| Real CLI ingestion | Passed for two synthetic PNGs and a three-second H.264 test-pattern video. Both DINOv2 and SSCD enabled; three video frames indexed, no ingestion failures. |
| Web | Real `sfn-web` startup and root HTML passed. Original image/frame/video HTTP downloads matched on-disk SHA-256. Web inspection ran on CPU; GPU inference was exercised through the model and CLI checks. |
| Offline isolation | Model-only container used `--network none`; CLI, database and web communicated on a Docker `--internal` network with no published ports. |
| Non-root/persistence | Model and application containers used UID 10001. An attempted write to the synthetic evidence mount failed as required. CLI-generated state persisted in a separate writable state directory. |
| Whole stopped-state restore | Stopped both synthetic services, copied all application state and Qdrant storage, then verified **11 application files and 816 database files** against SHA-256 inventories before startup in fresh containers on a second internal network. |
| Restored behavior | Exact comparison passed for **6 database points**, all payloads/vectors, exact vector search results, **11 artifact files** and **6 HTTP original downloads**. Includes two images, three indexed frames and one video anchor. This verifies the full small synthetic case, not a production-size case or old-version migration. |

Preserved evidence: [model result](image-model-smoke.json), [application hashes](app-state.sha256.json), [database hashes](qdrant-state.sha256.json). The original `image-model-smoke.log`, `image-ingestion.log`, `runtime-before.log` and `whole-case-restore.log` are not present in the repository; the table above records their historical conclusions, not independently recoverable log output. Reproduction helpers and the synthetic-only environment are preserved alongside this report; they use the documented `/tmp/sfn-release-artifacts` staging directory and refuse a nonempty fixture directory.

## Image scanning and release disposition

Trivy 0.69.3 scanned exported image archives with a freshly downloaded database. Counts below are package/advisory records, **not distinct vulnerabilities or demonstrated exploits**:

| Image | Critical | High | Medium | Low | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Application | 8 | 273 | 686 | 741 | 24 |
| Qdrant | 3 | 87 | 93 | 86 | 5 |

Raw compressed reports: [application](app-scan.json.gz), [Qdrant](qdrant-scan.json.gz). [Machine-readable summary](image-scan-summary.json) separates OS, Python, JavaScript and Rust SBOM scope. The image scan identified Python findings beyond the earlier requirements-only scan, including pip and Torch metadata. No findings were suppressed.

Read the [license and advisory review](license-advisory-review.md) for component-specific evidence and applicability limits. FFmpeg crafted-media findings deserve particular attention because media parsing is an application function. Kernel-header records are not evidence that the container supplies a vulnerable running kernel; the 32-bit Perl finding is distinguished from this verified 64-bit interpreter. Such triage does not clear all other findings.

The release still needs:

- Fixes or accepted applicability/risk decisions for image findings, especially media parsers; reevaluate and rerun affected validation after changes. A clean scanner exit by itself is not a vulnerability acceptance policy.
- Resolution of SSCD weight redistribution scope, GPLv3/CUDA distribution compatibility, required notices and corresponding-source obligations for the actual bundle.
- Acceptance of the setuptools build-path assessment, or a separately tested compatible Torch/backend migration. The old lock constraint was not overridden.
- Any AMD-specific, production-scale, deployment-environment or historical-volume upgrade requirements. No such acceptance target was supplied; this pass used the available NVIDIA workstation and a fresh synthetic case.

## Verification and cleanup

The changed inventory/restore unit subset passed **4 tests**. Ruff check/format (136 Python files), shell syntax and `git diff --check` passed. The prior full Python-suite result remains 1,122 passed / 80.44% coverage; this pass adds real-image evidence and did not rerun that entire suite.

All five synthetic ingestion/web/database containers and both test networks were removed. The existing live `scalarforensic-qdrant-1` remained running on 1.17.1. Operator `.env`, Compose override and case data were preserved. Built images, ignored wheel/model inputs and the local candidate archive remain available.

Automatic approval review rejected a Docker-socket mount into the scanner because of host-control access. Scanning exported archives completed successfully without socket access; no approval remains blocked by that rejection.
