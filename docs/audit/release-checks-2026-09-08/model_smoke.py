import gc
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scalar_forensic.embedder import DINOv2Embedder, SSCDEmbedder
from scalar_forensic.faces.detect import YuNetDetector
from scalar_forensic.faces.embed import OnnxFaceEmbedder
assert os.getuid() == 10001
assert torch.cuda.is_available()
try:
    Path("/images/write-probe").write_text("synthetic permission probe")
except OSError as exc:
    assert exc.errno in (13, 30)
else:
    raise AssertionError("Evidence mount unexpectedly writable")
images = [Image.open('/images/synthetic-0.png').convert('RGB'), Image.open('/images/synthetic-1.png').convert('RGB')]
result = {'uid': os.getuid(), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'models': {}}
for name, factory, dim in [('dino', lambda: DINOv2Embedder('/app/models/dinov2-large', device='cuda', local_files_only=True), 1024), ('sscd', lambda: SSCDEmbedder('/app/models/sscd_disc_mixup.torchscript.pt', device='cuda'), 512)]:
    model = factory()
    v = np.asarray(model.embed_images(images))
    assert v.shape == (2, dim) and np.isfinite(v).all()
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=0.005)
    assert not np.allclose(v[0], v[1])
    result['models'][name] = {'shape': list(v.shape), 'compiled': model.compiled, 'finite_normalized_distinct': True}
    del model
    gc.collect()
    torch.cuda.empty_cache()
detector = YuNetDetector(Path('/app/models/face_detection_yunet_2023mar.onnx'), max_size=640)
result['models']['yunet'] = {'detections_on_synthetic_pattern': len(detector.detect(np.asarray(images[0])))}
face = OnnxFaceEmbedder(Path('/app/models/face_recognition_sface_2021dec.onnx'))
v = face.embed([np.asarray(im.resize((112,112))) for im in images])
assert v.shape[0] == 2 and np.isfinite(v).all()
assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=0.005)
result['models']['sface'] = {'shape': list(v.shape), 'finite_normalized': True, 'inputs': 'synthetic patterns, no face-accuracy claim'}
print(json.dumps(result, sort_keys=True))
