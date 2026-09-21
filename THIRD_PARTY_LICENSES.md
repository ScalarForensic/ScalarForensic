# Third-Party Licenses

ScalarForensic bundles the following third-party components.

---

## Alpine.js

**Location:** `src/scalar_forensic/web/static/alpine.js`  
**License:** MIT  
**Copyright:** Copyright (c) 2019-2023 Caleb Porzio and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## JetBrains Mono

**Location:** `src/scalar_forensic/web/static/fonts/JetBrainsMono-*.woff2`  
**License:** SIL Open Font License 1.1  
**Copyright:** Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono)

This Font Software is licensed under the SIL Open Font License, Version 1.1. The full license text is available at: https://scripts.sil.org/OFL

---

## Barlow Condensed

**Location:** `src/scalar_forensic/web/static/fonts/BarlowCondensed-*.ttf`  
**License:** SIL Open Font License 1.1  
**Copyright:** Copyright 2017 The Barlow Project Authors (https://github.com/jpt/barlow)

This Font Software is licensed under the SIL Open Font License, Version 1.1. The full license text is available at: https://scripts.sil.org/OFL

---

## OpenCV (opencv-python-headless)

**Location:** installed dependency of the optional `faces` dependency group  
**License:** Apache License 2.0  
**Copyright:** Copyright (c) 2000-2024, OpenCV team and contributors

Used by the optional face modality for face detection (`cv2.FaceDetectorYN`) and image
operations (`warpAffine`, `Laplacian`, `cvtColor`, `resize`). The face package never uses
OpenCV's video I/O — video decoding is PyAV's job throughout ScalarForensic.

Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

---

## ONNX Runtime (onnxruntime)

**Location:** installed dependency of the optional `faces` dependency group  
**License:** MIT  
**Copyright:** Copyright (c) Microsoft Corporation

Used by the optional face modality to run the operator-supplied face-recognition model on the
CPU execution provider.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## YuNet face detector (model weights)

**Location:** fetched on demand to `models/face_detection_yunet_2023mar.onnx` — **not bundled**  
**License:** MIT  
**Source:** OpenCV Zoo (https://github.com/opencv/opencv_zoo)

Fetched by `scripts/download_models.py --yunet` when the operator opts into the face modality.
The source repository does not track model weights. Airgap images copy the actual
`models/` directory and therefore can distribute DINOv2/SSCD, YuNet, or any
operator-added artifacts present there. Review that directory before release.

**Note on recognition weights:** ScalarForensic does not fetch a face-recognition model.
That model is operator-supplied, and its licence — commonly research-only in this model family
— is the operator's responsibility. See INSTALL.md, "Face modality (optional)".

## Release artifact inventory

The notices above are not a complete inventory of an airgap image. Each release
build emits `bundle-inventory.json` (wheel metadata, license declarations and file
hashes, model hashes, lock/export hashes) and embeds it in `/app`. The built image
also contains `/app/native-inventory.json` with OS package versions and native
library/executable hashes. Native license records remain in `/usr/share/doc` and
Python distribution metadata; retain those files during redistribution.

The image includes Python/PyTorch and transitive CUDA or ROCm libraries, PyAV and
its native libraries, system FFmpeg and dependencies, plus downloaded model
artifacts. Qdrant is a separately saved image. License declarations in wheel
metadata are evidence for review, not legal clearance. Model licenses and unclear
native/transitive rights are explicitly UNKNOWN pending release-maintainer review.
Do not claim a bundle is redistributable until its actual artifacts have been
reviewed. Keep advisory output and image IDs alongside the archive; see the
[current release procedure](docs/deployment.md#airgap-release-evidence).
