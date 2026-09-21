"""Synthetic state restore rehearsal; real Qdrant/container recovery is separate."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_case_state.py"
spec = importlib.util.spec_from_file_location("verify_case_state", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_restore_preserves_reports_logs_and_referenced_artifacts(tmp_path):
    source = tmp_path / "source"
    restored = tmp_path / "restored"
    files = {
        "frames/video/frame.jpg": b"synthetic frame",
        "faces/chip.jpg": b"synthetic chip",
        "reports/run.csv": b"path,status\n/images/source.jpg,indexed\n",
        "reports/run.manifest.json": b'{"synthetic":true}\n',
        "face_audit.log": b'{"event":"index_run"}\n',
        "video_audit.log": b'{"event":"render"}\n',
        "sample-hits.json": json.dumps(
            {"frame": "frames/video/frame.jpg", "chip": "faces/chip.jpg"}
        ).encode(),
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = tmp_path / "inventory.json"
    subprocess.run([sys.executable, SCRIPT, "capture", source, manifest], check=True)
    shutil.copytree(source, restored)
    subprocess.run([sys.executable, SCRIPT, "verify", restored, manifest], check=True)
    for relative in json.loads((restored / "sample-hits.json").read_text()).values():
        assert (restored / relative).read_bytes() == files[relative]
    (restored / "face_audit.log").write_bytes(b"changed")
    result = subprocess.run([sys.executable, SCRIPT, "verify", restored, manifest])
    assert result.returncode == 1
    assert module.inventory(source) == json.loads(manifest.read_text())


def test_inventory_refuses_symlinks(tmp_path):
    (tmp_path / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="Symlinks"):
        module.inventory(tmp_path)
