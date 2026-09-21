# Model, native-license and advisory review — 2026-09-08

Evidence review only; no redistribution approval or security exception is granted here. Dependencies and release artifacts were not modified. Sources were checked against official upstream material; Firecrawl failed DNS resolution, so browsing used the available web tool. Installed wheel metadata and license files were inspected in `.venv`; the final image inventories must be reconciled separately.

## Model redistribution evidence

`Dockerfile` copies **everything in `models/`**. An operator-supplied model therefore becomes a distributed artifact when present in the build context. The directory currently contains DINOv2, SSCD, YuNet and SFace, despite the source repository not tracking weights.

| Artifact | Verified evidence | Remaining release action |
| --- | --- | --- |
| `dinov2-large` | Meta explicitly licenses DINOv2 code **and weights** under Apache 2.0. The local Hugging Face README also declares `apache-2.0`. This concerns the ordinary DINOv2 backbone, not later separately licensed biology/X-ray models. [Meta README](https://github.com/facebookresearch/dinov2#license), [license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE). | Tie the actual content hash to the downloaded revision; include the license and applicable notices in the airgap artifact. The local model README alone is not the complete license. |
| `sscd_disc_mixup.torchscript.pt` | Meta's repository has MIT terms, but its README calls these the **codebase** license. It supplies this exact pretrained model for reproducing results and notes that original training used fonts/emoji not licensed for redistribution. That statement concerns training assets and does not by itself prohibit distributing weights; equally, it is insufficient to establish an explicit weight grant. [README](https://github.com/facebookresearch/sscd-copy-detection#pretrained-models), [MIT license](https://github.com/facebookresearch/sscd-copy-detection/blob/main/LICENSE). | **Weight redistribution remains unresolved.** Obtain an authoritative clarification or maintainer/legal determination covering the checkpoint. Alternatively exclude it from the distributable build context and document separate operator provisioning. Do not relabel the checkpoint as research-only or MIT without resolving the scope. |
| `face_detection_yunet_2023mar.onnx` | At the downloader's pinned OpenCV Zoo commit `f12e12798e8314f7c074a6656816c048dcc95b7a`, the model-directory README licenses all files under MIT. [Pinned README](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/README.md), [license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE). | Retain MIT copyright/permission text and reconcile the image hash with the downloader's expected SHA-256. |
| `face_recognition_sface_2021dec.onnx` | Local SHA-256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` matches the [pinned upstream LFS pointer](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_recognition_sface/face_recognition_sface_2021dec.onnx). That directory explicitly licenses **all files** under Apache 2.0. [Pinned README](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_recognition_sface/README.md), [license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_recognition_sface/LICENSE). | Record this provenance and include Apache terms/applicable notices. The generic recognition-model warning in `THIRD_PARTY_LICENSES.md` does not establish that this particular SFace artifact is research-only. |

These findings establish upstream license declarations, not acceptance of the intended deployment or all third-party rights.

## PyTorch and CUDA native components

PyTorch 2.11.0 has BSD-style redistribution conditions; retain its bundled `LICENSE` and `NOTICE`, including transitive notices. Its license does not relicense NVIDIA libraries. [PyTorch 2.11 license](https://github.com/pytorch/pytorch/blob/v2.11.0/LICENSE).

The CUDA 12.8.1 EULA grants conditional application redistribution, lists redistributable components in Attachment A and includes a Linux-specific provision. Distribution must satisfy the agreement, including application functionality, access and consistent downstream terms. Preserve notices and map the **actual shipped files** to these terms; package names or successful `pip install` are insufficient. [CUDA 12.8.1 EULA, §§1.1–2.6](https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html).

Installed cuDNN 9.19.0.56 carries its own SDK agreement and supplement; the latter names runtime `.so`/`.h` files as distributable subject to the agreement. Its restrictions include not imposing source-disclosure, derivative-licensing or free-redistribution requirements on the SDK. [cuDNN 9.19 EULA](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.19.0/reference/eula.html). ScalarForensic's `LICENSE` is GPLv3: **the intended combined distribution and any applicable exception/permission require an explicit compatibility determination**. This review neither concludes that aggregation is prohibited nor assumes that placing libraries in a container resolves it.

Wheel metadata can mislead: installed `nvidia-nccl-cu12==2.28.9` declares `LicenseRef-NVIDIA-Proprietary`, while its bundled `License.txt` contains BSD redistribution conditions matching [NCCL's tagged upstream license](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/LICENSE.txt). NVTX declares Apache 2.0; NVSHMEM, cuSPARSELt and CUDA bindings carry separate terms. Review their actual license files and bundled third-party notices individually. Native-inventory hashes and retained `/usr/share/doc` do not, on their own, satisfy GPL/LGPL corresponding-source obligations for other native components such as FFmpeg. Native legal acceptance remains open.

## Setuptools advisory disposition

The preserved Python scan flags `setuptools==81.0.0` twice for one issue, `CVE-2026-59890` / `GHSA-h35f-9h28-mq5c`, and reports 83.0.0 as fixed. The [upstream advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c) currently displays affected versions through 82.0.1 and no patched version, but the [upstream 83.0.0 release notes](https://github.com/pypa/setuptools/blob/main/NEWS.rst#v8300) explicitly document the Unicode-normalization fix. Preserve this metadata discrepancy rather than suppressing the advisory.

**Assessment: the documented release path does not invoke the vulnerable sdist operation.** Dependencies are downloaded with `--only-binary=:all:`; Docker installs those wheels offline and builds the application wheel with Hatchling. The advisory concerns Unicode mismatches in setuptools `MANIFEST.in` exclusions while creating source distributions, not ordinary wheel installation or model inference. Linux alone is not proof of immunity: manually mismatched Unicode names remain possible if that code path is used. Torch/Triton kernel compilation is not, by itself, source-distribution creation.

Recommended disposition is a release-scoped non-applicability/risk record for this build and runtime, subject to release-owner acceptance. Keep 81.0.0 flagged in raw scan output, prohibit source-distribution publishing from this runtime, keep secrets outside build contexts, and reassess if build backends or packaging workflows change. This does not claim upstream dependency wheels were built securely.

The installed Torch 2.11.0 wheel explicitly requires `setuptools<82`; forcing 83 or uninstalling setuptools breaks declared dependency compatibility. [Torch 2.12.0 still has the cap](https://github.com/pytorch/pytorch/blob/v2.12.0/setup.py). A supported upstream **candidate** is Torch 2.13.0 / Torchvision 0.28.0: [Torch 2.13 source](https://github.com/pytorch/pytorch/blob/v2.13.0/setup.py) requires `setuptools>=77.0.3`, allowing 83, and [official wheel instructions](https://pytorch.org/get-started/previous-versions/#v2130) list CUDA 12.6, 13.0 and 13.2, plus CPU/ROCm. They do not list the current CUDA 12.8 combination. Migration therefore needs a new backend choice, lock, wheel-metadata verification and GPU/model regression checks; it is not a validated drop-in fix. No dependency changes were made.

## Qdrant image scan triage

Reviewed Trivy 0.69.3 output `/tmp/sfn-release-artifacts/qdrant-scan.json`, created `2026-09-08T16:40:49Z`, for the amd64 Qdrant 1.19.0 image recorded in the audit README. Counts are package/advisory records, not unique vulnerabilities or proven remotely reachable defects:

| Scope | Critical | High | Medium | Low | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Debian 13.6 packages | 3 | 80 | 93 | 83 | 5 |
| JavaScript packages | 0 | 7 | 0 | 0 | 0 |
| `qdrant/qdrant.spdx.json` Rust/Cargo SBOM | 0 | 0 | 0 | 3 | 0 |

All critical records concern `perl-base==5.40.1-6`: `CVE-2026-13221`, `CVE-2026-42496`, `CVE-2026-8376`; none has a fixed version in this scan. The last concerns **32-bit Perl builds**, so amd64 image metadata is a strong triage lead, but confirm the installed interpreter configuration before recording non-applicability. [Debian advisory](https://security-tracker.debian.org/tracker/CVE-2026-8376). A subsequent container probe confirmed `archname=x86_64-linux-gnu-thread-multi` and `ivsize=8`, supporting non-applicability of that specifically 32-bit issue to this image. Other Perl findings need component presence and reachable-use checks; an OS package finding does not prove a Qdrant HTTP endpoint is affected.

Thirty high OS records have fixes: three util-linux advisories (`CVE-2026-53612/53613/53614`) repeated across nine binary packages, fixed at source-package version `2.41.5-0+deb13u1`; and OpenSSL `CVE-2026-14456` across three packages, fixed at `3.5.7-1~deb13u2`. Debian confirms the [util-linux security version](https://security-tracker.debian.org/tracker/CVE-2026-53614) and [OpenSSL security version](https://security-tracker.debian.org/tracker/CVE-2026-14456). The former involves mount operations/privilege conditions; the latter OpenSSL QUIC server use. Review reachability, but prefer a rebuilt image using the distro security updates, then rescan and repeat startup/restore checks.

All seven JavaScript high records have reported fixes: Browserslist 4.28.2 → 4.28.7 (two records); js-yaml 4.3.0 → 4.3.1 (one); nanoid 3.3.16 → 3.3.18 (one), and 5.1.5 → at least 5.1.16 to cover all three reported 5.x findings. These are scan-reported versions; rebuild the relevant upstream dashboard assets and validate its lock before use. Determine whether flagged packages are build-time tooling or shipped executable code. Updating this repository's unrelated npm lock does not fix Qdrant's dashboard.

A fresh upstream image digest or a reviewed same-version rebuild is a mitigation candidate, not an accomplished fix. Updating OS packages alone cannot repair statically bundled Rust dependencies or JavaScript assets. The three low Cargo records are from an embedded SBOM, not proof of exhaustive Rust/native binary coverage. Keep release security acceptance open until fixes or documented applicability decisions cover the findings. Preserve the raw report alongside the final image digest.

## Built application image scan follow-up

Reviewed `/tmp/sfn-release-artifacts/app-scan.json`, created `2026-09-08T16:54:34Z`, targeting `/reports/app-final.tar`. The release run reports successful offline GPU/model and synthetic restore checks for this corrected image; those functional checks do not establish advisory non-applicability.

| Scope | Critical | High | Medium | Low | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Debian 13.6 packages | 8 | 273 | 680 | 739 | 24 |
| Python metadata | 0 | 0 | 6 | 2 | 0 |

There are 913 distinct OS advisory IDs; **no OS finding in this scan has a fixed version**. The current stable image/packages therefore cannot be called clean simply because the build ran the latest available `apt-get` installation. This differs from the older Qdrant OS package set above.

The eight critical records are:

| Package/version | Advisory | Bounded applicability finding |
| --- | --- | --- |
| `libglib2.0-0t64 2.84.4-3~deb13u3` | `CVE-2026-58016` | Malformed D-Bus introspection XML; trace use of `g_dbus_node_info_new_for_xml`, not arbitrary XML processing. [Debian record](https://security-tracker.debian.org/tracker/CVE-2026-58016). |
| `libmbedcrypto16 3.6.5-0.1~deb13u1` | `CVE-2026-34873`, `CVE-2026-34875` | TLS 1.3 session-resumption impersonation and FFDH public-key export overflow. Source-package mapping to a crypto library does not prove the TLS server path is present; confirm symbols, consumers and enabled algorithms. [TLS record](https://security-tracker.debian.org/tracker/CVE-2026-34873), [FFDH record](https://security-tracker.debian.org/tracker/CVE-2026-34875). |
| `libxml2 2.12.7+dfsg+really2.9.14-2.1+deb13u3` | `CVE-2026-6653` | XML internal-subset/entity handling UAF. The package version deliberately contains older 2.9.14 code; do not infer immunity from the 2.12.7 prefix. [Debian record](https://security-tracker.debian.org/tracker/CVE-2026-6653). |
| `linux-libc-dev 6.12.107-1` | `CVE-2026-43185` | The advisory concerns executing kernel `ksmbd` SMB-direct code. `linux-libc-dev` supplies build headers, not the running kernel; this record does not establish vulnerable kernel execution in the image. Review the actual host kernel/configuration separately. [Debian record](https://security-tracker.debian.org/tracker/CVE-2026-43185). |
| `perl-base 5.40.1-6` | `CVE-2026-13221`, `CVE-2026-42496`, `CVE-2026-8376` | The release run verified 64-bit Perl, which excludes the specifically 32-bit prerequisite of `CVE-2026-8376`. The other regex/archive findings still require installed-module and reachable-use checks. |

Debian classifies the GLib, Mbed TLS and libxml2 entries above as minor/no-DSA for trixie, while this scanner labels them critical. Preserve both assessments. Fixes recorded for Debian unstable are **not** approved trixie upgrade instructions.

Of the 273 high OS records, 53 attach to kernel headers, and 153 repeat 17 FFmpeg source advisories across nine binary packages. The remaining 67 span other packages. FFmpeg `7:7.1.5-0+deb13u1` is a priority for unresolved runtime assessment because the application actually transcodes media. Findings include crafted HEVC (`CVE-2026-64831`), PNG (`CVE-2026-66040`), VobSub (`CVE-2026-64830`) and CFHD/AVI (`CVE-2026-70632`) inputs. Establish which demuxers, decoders, filters and input routes execute before accepting risk; synthetic media success does not exercise adversarial cases. PyAV's wheel-bundled FFmpeg libraries are a separate native artifact from Debian FFmpeg and require their own version/coverage mapping.

Python findings are six pip records, one setuptools record and one Torch record. Installed pip 25.0.1 has five medium and one low findings involving package installation/archive/index handling; the highest reported fix covering these is **26.2.0**. Offline hashed wheel inputs reduce relevant attack opportunities but do not retroactively prove the bootstrap installer or wheels safe. Setuptools remains assessed above. Trivy additionally identifies `torch==2.11.0+cu128`, reporting **LOW `CVE-2025-3000`**, fix **2.13.0**, concerning local `torch.jit.script` compilation ([upstream reproducer](https://github.com/pytorch/pytorch/issues/149623)). This improves coverage over the earlier PyPI scan that skipped CUDA wheels; metadata matching still does not provide exhaustive CUDA/native vulnerability coverage.

Outstanding security decisions are the media-parser findings, other native-library reachability, actual host-kernel assessment and remaining Python packaging/JIT findings. No broad dependency or distro upgrade was performed as part of this review. The built artifact remains a **functionally tested validation candidate**, with security and redistribution acceptance open.
