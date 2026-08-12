from scalar_forensic.faces.provenance import PipelineConfig


def _cfg(**over):
    base = dict(
        detector_id="yunet",
        detector_model_hash="d" * 64,
        detector_score_threshold=0.5,
        detect_max_size=1600,
        embedder_model_name="emb.onnx",
        embedder_model_hash="e" * 64,
        manifest_hash="m" * 64,
        embedder_dim=512,
        alignment_version="arcface-112-v1",
        normalization_id="affine-127.5-128.0",
        min_conf=0.8,
        min_size=64,
        min_sharpness=25.0,
        max_clipped=0.6,
        max_pose=0.35,
        crop_dilation=0.15,
        sfn_version="1.0",
        cv2_version="4.10",
        ort_version="1.18",
    )
    base.update(over)
    return PipelineConfig(**base)


def test_hash_is_stable_and_hex():
    assert _cfg().config_hash == _cfg().config_hash
    assert len(_cfg().config_hash) == 64


def test_hash_changes_with_comparability_fields():
    assert _cfg().config_hash != _cfg(embedder_model_hash="f" * 64).config_hash
    assert _cfg().config_hash != _cfg(min_conf=0.9).config_hash


def test_hash_ignores_library_versions():
    assert (
        _cfg().config_hash
        == _cfg(cv2_version="9.9", ort_version="9.9", sfn_version="2.0").config_hash
    )


def test_payload_round_trip_contains_everything():
    p = _cfg().to_payload()
    assert p["pipeline_config_hash"] == _cfg().config_hash
    for key in ("detector_id", "embedder_model_hash", "alignment_version", "cv2_version"):
        assert key in p
