"""An expected content hash must identify exactly one selected model."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_models.py"
spec = importlib.util.spec_from_file_location("download_models_selection", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("flags", [["--dino", "--yunet"], ["--sscd", "--yunet"], ["--all"]])
def test_hash_rejects_multiple_models_before_any_download(flags, monkeypatch):
    downloaded = []
    for name in ("sscd", "dino", "yunet"):
        monkeypatch.setattr(module, f"download_{name}", lambda **kw: downloaded.append(kw))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *flags, "--hash", "a" * 64])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code != 0
    assert downloaded == []


def test_yunet_alone_uses_its_requested_hash(monkeypatch):
    downloaded = []
    monkeypatch.setattr(module, "download_yunet", lambda **kw: downloaded.append(kw))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--yunet", "--hash", "b" * 64])
    module.main()
    assert downloaded == [{"force": False, "expected_hash": "b" * 64}]
