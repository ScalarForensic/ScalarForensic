# Audit remediation results — 2026-09-08

## Outcome

**Image follow-up:** [Full candidate build and restore verification](release-checks-2026-09-08/image-and-restore-results.md) passed offline all-model execution, GPU ingestion and synthetic whole-case restoration. Image vulnerability findings and redistribution acceptance remain open.

**Release follow-up:** [Additional checks and evidence](release-checks-2026-09-08/README.md) align Qdrant to 1.19.0, verify a real synthetic snapshot restore and host DINOv2 GPU inference, and record the live advisory scan. The final suite now has **1,122 passing tests** with **80.44% coverage**.

Implemented the approved repository changes across milestones 0–3, with regression coverage and independent subsystem reviews. **Initial verification: 1,120 Python tests passed with 80.44% coverage.** Eight JavaScript test files and one real Chromium smoke passed. The release follow-up above supersedes these initial counts; this is not certification for forensic use or an airgap release.

The [original audit](repository-audit-2026-09-08.md) remains a historical report against revision `d9505e08e5daa12893b2ecf72678ae8354d2280c`; its original line references, grade and baseline test limitations describe that revision, not the modified checkout. The [remediation design](../specs/audit-remediation.md) records the durable architecture and verification requirements.

## Approved decisions and execution

The user selected all milestones in priority order, disposable/rebuildable indexes, and metadata/original viewing only under a configuration-mismatch override.

At the time of this historical verification, changes were uncommitted in the original checkout. The operator `.env`, pre-existing `docker-compose.override.yml`, historical audit material and live case data were preserved. Only synthetic collections were used; the separately created Qdrant container was removed after verification. Nothing was deployed or pushed.

## Milestone results

| Milestone / tasks | Implemented behavior | Representative evidence |
|---|---|---|
| **0 — Safety net**, T0R/T1/T2/T3 | Shared application state survives disposable CLI containers; Compose GPU syntax is valid; CI uses locked dependencies and a pinned real Qdrant service. Added checksummed application-state restore rehearsal and interrupted-ingest regressions. | `docker-compose.yml:61`; `.github/workflows/ci.yml:132`; `scripts/verify_case_state.py:15`; `tests/test_ingest_characterization.py` |
| **1 — Integrity**, T4/T5/T6/T10 | Face identities and collapse preserve source video/timecode; stale selection and purge respect that scope, including legacy image purge and shared chips. Completion counts only acknowledged writes and includes previously committed frames. Incompatible ingestion cannot activate the face collection. Case/reference similarity checks fail closed, including database recovery and unknown provenance. | `faces/store.py:32,643`; `web/pipeline/faces_search.py:67`; `ingestion/commit.py:17`; `ingestion/video.py:45`; `cli.py:364`; `safeguards.py:291,379` |
| **1 — Runtime safety**, T7/T8/T9 | Encoder deadline/size monitoring works without progress output; cancellation reaps subprocesses. Upload byte/concurrency admission precedes multipart spooling. Shared model loads/inference and mutable face detection are serialized. | `video_playback/encode.py:240`; `web/ingress.py:20`; `web/pipeline/_embedders.py:44`; `web/pipeline/faces_query.py:149` |
| **2 — Correctness and performance**, T11/T12/T13/T14 | Stored face probes use their own provenance. DINO checks declared and actual tensor geometry. Face statistics include negative cosine scores and identify selected top-N neighbors. Video searches use at most 64 vectors per RPC and 32 frames of intermediate results; frame association, ranking and reference separation are tested. Blocking statistics/provenance work is offloaded. Superseded query/face responses cannot overwrite newer selections. | `web/routes/faces.py:782`; `embedder.py:523`; `web/pipeline/faces_stats.py:52`; `web/pipeline/query.py:729`; `web/routes/analyze.py:268`; `web/static/js/analysis.js:48`; `web/static/js/faces.js:53,271` |
| **2 — Maintainability/lifecycle**, T15/T16 | Extracted discovery, planning, acknowledged commit, completion, face reconciliation and reporting into six ingestion modules. Sessions own temporary namespaces and worker leases; startup recovers orphan namespaces safely. Completed playback jobs have age/count limits. | `ingestion/`; `web/session.py:169,208,216`; `video_playback/jobs.py:383,442` |
| **3 — Release tooling/docs**, T17/T18/T19 | Added Chromium smoke in existing CI gate; explicit runtime/build dependency exports; temporary build tooling; wheel/model/native inventories with hashes and license unknowns; optional advisory-check release refusal; non-root UID 10001 image; revised deployment/rebuild/restore guidance. | `.github/workflows/ci.yml`; `scripts/download_deps.sh:45,80`; `scripts/build_airgap_image.sh`; `scripts/bundle_inventory.py:24`; `Dockerfile:75`; `docs/deployment.md` |

Source paths abbreviated above are under `src/scalar_forensic/`, except repository-root configuration, scripts, tests and docs.

## Review findings resolved before completion

Independent reviews found additional defects beyond the first implementation batches:

- Strict validation initially inspected only 1,024 points. It now uses an exact filtered count to reject incompatible or missing provenance anywhere in the vector-bearing population (`safeguards.py:291`; `tests/test_query_compatibility_gate.py:160`).
- Statistics and uploaded-tag queries could consume obsolete session embeddings after settings changed. They now validate cached configuration, and tag reference vectors are freshly retrieved instead of surviving collection rebuilds (`web/pipeline/stats.py:67`; `web/routes/tags.py:56,152`).
- Correct declared processor sizes did not guarantee correctly sized tensors when transformations were disabled. A regression using the real Transformers processor now verifies rejection before model inference (`tests/test_embedder_preprocessing.py`; `embedder.py:523`).
- Publishing a temporary namespace before locking it allowed another process's recovery sweep to remove it. Namespace creation now locks a hidden staging directory before atomic publication (`web/session.py:216`).
- Older face detection could replace newer server state. Detection publication now checks a request generation, readers snapshot faces/config together, and chips fingerprint crop/vector/timecode content. Active detection holds a session lease (`web/routes/faces.py:473,530`).

## Verification

| Check | Final result |
|---|---|
| Initial full `pytest -q`, isolated Qdrant at port 16333 | **1,120 passed**, **80.44%** coverage, 43.90 seconds; no skipped tests reported |
| `npm test` | **8 test files passed**, zero failures |
| `npm run test:browser` | **1 Chromium smoke passed**; real HTML/Alpine/DOM with synthetic API interception |
| Ruff check / format | Passed; **135 Python files** formatted |
| CPU, AMD, NVIDIA Compose validation | All passed using `.env.example` and explicit tracked Compose files |
| Shell syntax / `git diff --check` | Passed |
| `uv lock --check --offline` | Passed; 90 packages resolved |

Python/Chromium tests required execution outside sandbox threading/process restrictions. Fixtures remained synthetic. The operations worker also verified a cached offline application-wheel build, a fresh dependency-free install containing packaged HTML, and isolated UID-10001 persistence/read-only-evidence behavior. These narrower checks do not replace a full built-image startup test.

## Remaining release checks and trade-offs

1. **Client/server support — resolved for fresh deployments:** server pins now match qdrant-client 1.19.0; compatibility and integration checks pass. Existing live 1.17.1 storage was not upgraded. Follow the versioned backup and sequential upgrade guidance before deployment.
2. **Real deployment:** host DINOv2 GPU inference and a synthetic Qdrant snapshot restore now pass. The subsequent complete image and synthetic whole-case restore checks also pass; release hardware/scale acceptance and old-volume upgrades remain outside that rehearsal. Rehearse the documented restore and actual-image startup on release hardware before relying on them.
3. **Advisories/licenses:** live Python/npm scans now have recorded results. The h2 advisory is fixed; setuptools remains constrained by Torch, CUDA wheels were not auditable through PyPI, and image scans and redistribution evidence review have since completed; findings and legal acceptance remain open. See the follow-up for exact scope and findings.
4. **Rebuild boundary:** schema-v2 face identities deliberately require rebuilding old indexes. Preserve historical adjudication/audit records separately; they are not silently migrated to new observation IDs.
5. **Concurrency/scale:** validation and search are separate database operations. Do not rebuild a collection concurrently with examination. Strict exact counts and fresh tag-vector reads trade some latency for correctness. Video RPC/intermediate sizes are bounded; complete returned forensic results can still grow with the number of matching frames. No production-scale latency or memory target was supplied or certified.
6. **Scope calibration:** retained the existing workstation/isolated-LAN architecture and manual access-control model. No enterprise infrastructure, model changes, silent dependency upgrades or wholesale CLI rewrite was introduced. The extracted CLI remains substantial; further decomposition should follow measured maintenance needs.

The only human decisions still needed concern release hardware, existing-volume upgrade acceptance, redistribution rights, and deployment acceptance targets. No additional approval is needed to inspect or test these uncommitted changes.
