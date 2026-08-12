from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "faces" / "tiny_face.onnx"


@pytest.fixture(scope="session")
def tiny_onnx_model() -> Path:
    return _FIXTURE
