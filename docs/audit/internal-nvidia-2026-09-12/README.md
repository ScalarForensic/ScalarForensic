# Internal NVIDIA validation — 2026-09-12

**Source follow-up:** the opt-in [image and video isolation boundary](../../specs/media-parser-isolation.md)
was subsequently implemented and locally validated. See the
[image measurements](../../benchmarks/media-worker-validation-2026-09-12.md) and
[final video validation](../../benchmarks/video-isolation-2026-09-12.md#final-verification-and-operator-acceptance).
Those changes address the isolation implementation work below; they do not patch
the native libraries in these historical candidate images or grant release clearance.

## Disposition

**Functional validation passed; security acceptance remains blocked.** This is
an internal-use validation candidate, not approval to process untrusted evidence
or to redistribute the bundle. No live deployment, database upgrade, driver
change, model change, or security exception was performed.

Application source is commit `46b4300` on the local `chore/dead-weight` branch.
The branch still includes the operator's dirty-snapshot base and must not be
published without explicit approval. These are image-only security experiments;
the repository's standard Dockerfile and dependency lock remain unchanged.

## Exact candidates

| Candidate | Local tag | Image ID |
| --- | --- | --- |
| Application, final | `scalarforensic:internal-nvidia-20260912-patched` | `sha256:8d5b7ed41a82486f93cc3f80b4b9ff0397c064232fb3ed53bc9e057d8cd8877b` |
| Qdrant 1.19.0, Debian-updated | `scalarforensic-qdrant:internal-1.19.0-20260912` | `sha256:6b4852b96c3d83b5ec892d7f0ef98e0bd8e5207704715475dfc3c45ec9e89d97` |

The application was rebuilt from an archive of committed source, copied model
artifacts and the existing hash-checked wheel inputs. Worktree and input-source
`uv.lock` both hashed to
`2dc1e9c321323bf638fc66e9b49ca6a4d1e831484a2f0ffef6a3cb251506a0ec`.
The first build reused an OS layer, so a second build used `--pull --no-cache`.
Base: `python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`.
The refreshed intermediate app image was
`sha256:d0c73b726fcee8d3494550422d0f57e17a2973288a636071ba7982c68ce05596`.
The [build-input inventory](build-inputs.json.gz) records wheel/model hashes;
the separate pinned pip requirement records the final added wheel.

Qdrant's current same-version amd64 upstream digest
`sha256:6c0652f8d6925b22f2f6f0e0a5365a6c9dbc8768bd6e70ccc1cdc14847e452a0`
still had the previous OS findings. A separate image applied Debian's supported
package updates without replacing the Qdrant binary/version. The application
then received only the official pip 26.2.1 wheel, installed offline with a hash
requirement; Torch, CUDA and model dependencies were not changed. Reproduction
recipes are [Qdrant.Dockerfile](Qdrant.Dockerfile),
[Pip.Dockerfile](Pip.Dockerfile) and [pip requirement](requirements-pip.txt).
They are validation recipes, not modifications to the standard build workflow.

## Functional evidence

- NVIDIA RTX 4060 Ti, 16,380 MiB, driver 595.84; host kernel 7.0.0-30-generic.
- The final image ran with UID 10001, network disabled and read-only synthetic
  evidence. The attempted evidence write failed as required. DINOv2 produced
  2x1024 and SSCD 2x512 finite, normalized, distinct vectors, both compiled on
  GPU. SFace produced 2x128 finite normalized vectors; YuNet ran and detected no
  faces on the generated patterns. This is execution, not face-accuracy testing.
  [Complete model log](model-smoke.log.gz) preserves optimization warnings.
- On the refreshed app, actual CLI ingestion with both GPU image backends
  indexed two generated PNGs and three frames from a three-second H.264 video,
  with zero read/preprocessing/embedding failures.
- Before restore, the private web service returned six original downloads whose
  hashes matched disk. Baseline: six points, both named-vector search probes,
  11 artifact files and six HTTP checksums.
- Both synthetic services were stopped. All 11 application and 816 database
  files matched SHA-256 inventories before restored startup. The restored
  service matched the full baseline: IDs, payloads, vectors, rounded exact-search
  scores, artifact hashes and original-download hashes. The final pip-patched
  image repeated this verification against the same restored baseline.
- The final pip-patched image repeated all four model checks. `pip check` passed.
  The real synthetic playback capability probe selected `h264_nvenc`/CUDA with
  `tonemap_ok=true`, no notes, using Debian FFmpeg 7.1.5-0+deb13u1.
- [Application checksums](app-state.sha256.json) and
  [database checksums](qdrant-state.sha256.json) preserve the stopped-state evidence.
  Harnesses are the retained `../release-checks-2026-09-08/` model, fixture and
  runtime helpers plus `scripts/verify_case_state.py` from this source commit.
  The complete synthetic [runtime baseline](runtime-baseline.json.gz) retains
  payloads, vectors, search results and HTTP hashes used for equality checks.

The host initially could not write manifests into a container-owned results
directory; manifests were written beside it instead and capture/verify reran
successfully. The runtime helper also warns about an insecure API key because
it passes the synthetic empty string as a key; no credential was configured.
No product workaround was introduced for either harness issue.

## Fresh security results

Trivy 0.69.3 scanned exported images without a Docker socket mount. Database
updated at `2026-09-12T07:04:21Z`, downloaded at `11:26:10Z`; see
[database metadata](scanner-db.json). Scans used vulnerability-only offline
analysis after the database download. No ignore list or severity filter was used.

Counts are package/advisory records, not distinct exploits or reachable defects:

| Final scope | Critical | High | Medium | Low | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Application Debian | 8 | 280 | 689 | 744 | 20 |
| Application Python/bundled metadata | 0 | 2 | 2 | 1 | 0 |
| Qdrant Debian | 3 | 52 | 61 | 59 | 1 |
| Qdrant dashboard dependencies | 0 | 8 | 3 | 0 | 0 |
| Qdrant Cargo SBOM | 0 | 0 | 0 | 3 | 0 |

Raw evidence: [application](app-scan.json.gz), [Qdrant after](qdrant-scan.json.gz),
[Qdrant before](qdrant-before-scan.json.gz). The updated Qdrant removed all 93
Debian records with available fixes, including 30 high records; no remaining
Debian record in either final image has a fixed version in this scan. The
[util-linux](https://security-tracker.debian.org/tracker/CVE-2026-53614) and
[OpenSSL](https://security-tracker.debian.org/tracker/CVE-2026-14456) updates were
checked against Debian's security records. Dashboard/Cargo dependencies were not
rebuilt or silently removed.

Pip 26.2.1 removed the six findings against pip 25.0.1, but the final scanner also
identified vendored components and warned about third-party SBOM accuracy:

- `setuptools 70.3.0`: pip's `vendor.txt` attributes its `pkg_resources` copy to
  this version. The vendored tree has no setuptools package or `package_index.py`;
  its `sdist.py` is pip's distribution wrapper, not setuptools' affected FileList
  implementation. The flagged download/sdist components are not in this copy.
- `msgpack 1.1.2`: the final image resolves `Unpacker` to
  `pip._vendor.msgpack.fallback`, not a native extension. CacheControl calls
  `msgpack.loads`, whose implementation constructs a fresh unpacker each call.
  That path does not meet the [advisory's reuse-after-error condition](https://github.com/msgpack/msgpack-python/security/advisories/GHSA-6v7p-g79w-8964).
  This is bounded applicability evidence, not a claim about all MessagePack use.
- Installed `setuptools 81.0.0` and Torch 2.11.0+cu128 findings remain. The earlier
  sdist-path assessment is retained, not automatically accepted. Forcing
  setuptools 83 conflicts with the current Torch dependency constraint. A
  Torch/backend migration needs its own supported-version validation.

## Remaining release blockers and boundaries

1. **Untrusted-media parsing:** actual FFmpeg input decoding is used, so its
   HEVC/PNG/other crafted-media findings cannot be dismissed because synthetic
   inputs worked. PyAV 18.0.0 ships separate native libraries, including
   libavcodec 62.28.102, which require separate patch/version coverage. A
   supported patched parser build and isolation design need validation; no
   exploit was run and no arbitrary distro/backport change was made.
2. **Other native/dashboard findings:** retain per-component applicability and
   upstream remediation work. On the updated Qdrant, Perl is 64-bit and
   Archive::Tar is absent, narrowing two critical records; this does not clear
   the unrelated Perl regex finding or the image as a whole.
3. **Host kernel:** linux-libc-dev is headers, not the executing kernel. The host
   enables SMB server as a module and SMB Direct support, with no loaded ksmbd
   entry observed. This is not proof of a patched host or immunity to module
   activation; a host patch/configuration assessment remains separate.
4. **Scale and hardware:** no production-size case, long-duration workload,
   hostile-input corpus, AMD target or old live-volume migration was tested.
5. **Licensing:** redistribution/AMD were explicitly deferred. Internal-use scope
   is not a blanket legal clearance; no model license was relabeled and no
   GPL/CUDA compatibility exception or security risk acceptance was invented.

Do not deploy this candidate to untrusted evidence on the strength of this
functional report. Next priority is the media-parser remediation/isolation
decision, not another broad dependency bump or a scanner-ignore policy.

## Retention and cleanup

All five synthetic service containers and both internal networks were removed.
Stopped-state backup copies and raw temporary evidence remain under
`/tmp/sfn-internal-nvidia-w80gIJ`; temporary files may disappear after reboot.
Candidate images remain local. The live `scalarforensic-qdrant-1` stayed on
1.17.1; no operator evidence, main-checkout files, shared venv or model files
were modified. Nothing was pushed or deployed.
