from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "faces" / "tiny_face.onnx"


@pytest.fixture(scope="session")
def tiny_onnx_model() -> Path:
    return _FIXTURE


@pytest.fixture(scope="session")
def tiny_onnx_model_batch1() -> Path:
    """Same net as tiny_face.onnx but exported with a fixed batch dim of 1,
    mirroring real embedder exports like SFace (input declared (1,3,112,112))."""
    return _FIXTURE.parent / "tiny_face_batch1.onnx"
