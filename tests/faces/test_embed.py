import json
from pathlib import Path

import numpy as np
import pytest

from scalar_forensic.faces.embed import OnnxFaceEmbedder, load_manifest


def test_manifest_loads_and_validates(tiny_onnx_model):
    m = load_manifest(tiny_onnx_model)
    assert m.embedding_dim == 512 and m.layout == "NCHW" and m.channel_order == "RGB"


def test_manifest_missing_field_is_actionable(tiny_onnx_model, tmp_path):
    bad = tmp_path / "m.onnx"
    bad.write_bytes(tiny_onnx_model.read_bytes())
    manifest = json.loads(Path(str(tiny_onnx_model) + ".manifest.json").read_text())
    del manifest["embedding_dim"]
    Path(str(bad) + ".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="embedding_dim"):
        load_manifest(bad)


def test_embed_returns_l2_normalised_and_records_norms(tiny_onnx_model):
    emb = OnnxFaceEmbedder(tiny_onnx_model)
    crops = [np.random.default_rng(i).integers(0, 255, (112, 112, 3), np.uint8) for i in range(3)]
    out = emb.embed(crops)
    assert out.shape == (3, 512) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert emb.embedding_norms.shape == (3,) and (emb.embedding_norms > 0).all()


def test_embed_is_deterministic(tiny_onnx_model):
    emb = OnnxFaceEmbedder(tiny_onnx_model)
    crop = [np.full((112, 112, 3), 90, np.uint8)]
    np.testing.assert_array_equal(emb.embed(crop), emb.embed(crop))


def test_session_validation_catches_dim_mismatch(tiny_onnx_model, tmp_path):
    lying = tmp_path / "lying.onnx"
    lying.write_bytes(tiny_onnx_model.read_bytes())
    manifest = json.loads(Path(str(tiny_onnx_model) + ".manifest.json").read_text())
    manifest["embedding_dim"] = 256  # model actually emits 512
    Path(str(lying) + ".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="embedding_dim"):
        OnnxFaceEmbedder(lying)


def test_identity_metadata(tiny_onnx_model):
    emb = OnnxFaceEmbedder(tiny_onnx_model)
    assert len(emb.model_hash) == 64 and len(emb.manifest_hash) == 64
    assert emb.normalization_id == "affine-127.5-128.0"


def test_embed_multiple_crops_with_fixed_batch1_model(tiny_onnx_model_batch1):
    """A model whose input declares a fixed batch dim of 1 (e.g. SFace) must
    still embed multi-face images — the embedder chunks instead of crashing."""
    emb = OnnxFaceEmbedder(tiny_onnx_model_batch1)
    crops = [np.random.default_rng(i).integers(0, 255, (112, 112, 3), np.uint8) for i in range(3)]
    out = emb.embed(crops)
    assert out.shape == (3, 512) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert emb.embedding_norms.shape == (3,) and (emb.embedding_norms > 0).all()


def test_fixed_batch1_chunking_matches_per_crop_calls(tiny_onnx_model_batch1):
    emb = OnnxFaceEmbedder(tiny_onnx_model_batch1)
    crops = [np.random.default_rng(i).integers(0, 255, (112, 112, 3), np.uint8) for i in range(2)]
    batched = emb.embed(crops)
    singles = np.vstack([emb.embed([c]) for c in crops])
    np.testing.assert_array_equal(batched, singles)
