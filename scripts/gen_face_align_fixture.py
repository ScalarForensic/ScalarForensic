"""One-shot generator for the alignment golden fixture (run once, commit output).

Uses a deterministic synthetic 'face': gradient background + asymmetric
markers, warped from hand-picked plausible landmark positions.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from scalar_forensic.faces.align import align_face

out = Path("tests/fixtures/faces")
out.mkdir(parents=True, exist_ok=True)
# Smooth gradient base, not per-pixel noise: bilinear resampling of
# high-frequency noise is the worst case for the <=1-uint8 cross-build tolerance.
yy, xx = np.mgrid[0:480, 0:640]
img = np.stack([(xx * 255 // 640), (yy * 255 // 480), ((xx + yy) * 255 // 1120)], -1).astype(
    np.uint8
)
img[100:110, 200:210] = [255, 0, 0]  # asymmetric hard-edged marker: catches mirroring
lm = [[260.0, 210.0], [340.0, 205.0], [300.0, 260.0], [270.0, 310.0], [335.0, 305.0]]
Image.fromarray(img).save(out / "golden_source.png")
aligned = align_face(img, np.array(lm, dtype=np.float32))
Image.fromarray(aligned).save(out / "golden_aligned.png")
(out / "golden_landmarks.json").write_text(
    json.dumps({"source_png": "golden_source.png", "landmarks": lm}, indent=2)
)
print("fixture written")
