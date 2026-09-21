"""Bundle metadata uses actual wheel contents and preserves unresolved rights."""

import importlib.util
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bundle_inventory.py"
spec = importlib.util.spec_from_file_location("bundle_inventory", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_inventory_wheel_native_member_model_and_lock(tmp_path):
    (tmp_path / "vendor").mkdir()
    wheel = tmp_path / "vendor/synthetic-1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "synthetic-1.dist-info/METADATA",
            "Name: synthetic\nVersion: 1\nLicense-Expression: MIT\n",
        )
        # Real setuptools wheels include metadata for vendored distributions.
        archive.writestr(
            "synthetic/_vendor/dependency-2.dist-info/METADATA",
            "Name: dependency\nVersion: 2\nLicense-Expression: BSD-3-Clause\n",
        )
        archive.writestr("synthetic/native.so", b"synthetic native bytes")
        archive.writestr("synthetic-1.dist-info/licenses/LICENSE", "MIT test fixture")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/unknown.onnx").write_bytes(b"synthetic weights")
    for name in (
        "uv.lock",
        "requirements.txt",
        "requirements-build.txt",
        "THIRD_PARTY_LICENSES.md",
    ):
        (tmp_path / name).write_text("synthetic\n")
    result = module.inventory(tmp_path)
    wheel_entry = next(a for a in result["artifacts"] if a["kind"] == "wheel")
    assert wheel_entry["license"] == "MIT"
    assert wheel_entry["native_members"][0]["path"] == "synthetic/native.so"
    assert len(wheel_entry["native_members"][0]["sha256"]) == 64
    model_entry = next(a for a in result["artifacts"] if a["kind"] == "model")
    assert model_entry["license"].startswith("UNKNOWN")
    assert next(a for a in result["artifacts"] if a["path"] == "uv.lock")["sha256"]


def test_advisory_scanner_failure_aborts_archive_release(tmp_path, monkeypatch):
    import os
    import shutil
    import subprocess

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(SCRIPT, scripts / SCRIPT.name)
    builder = SCRIPT.with_name("build_airgap_image.sh")
    shutil.copy(builder, scripts / builder.name)
    for name in (
        "uv.lock",
        "requirements.txt",
        "requirements-build.txt",
        "THIRD_PARTY_LICENSES.md",
    ):
        (tmp_path / name).write_text("synthetic\n")
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$1" >> docker-commands.txt\n')
    scanner = commands / "trivy"
    scanner.write_text('#!/bin/sh\nif [ "$1" = --version ]; then exit 0; fi\nexit 7\n')
    docker.chmod(0o755)
    scanner.chmod(0o755)
    monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        ["bash", str(scripts / builder.name), "--no-models", "--no-deps", "--scan"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert "save" not in (tmp_path / "docker-commands.txt").read_text().splitlines()
    assert not (tmp_path / "scalarforensic-latest.tar.gz").exists()


def test_unscanned_bundle_records_both_images_as_not_run(tmp_path, monkeypatch):
    for name in (
        "uv.lock",
        "requirements.txt",
        "requirements-build.txt",
        "THIRD_PARTY_LICENSES.md",
    ):
        (tmp_path / name).write_text("synthetic\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(SCRIPT, scripts / SCRIPT.name)
    builder = SCRIPT.with_name("build_airgap_image.sh")
    shutil.copy(builder, scripts / builder.name)
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
    subprocess.run(
        ["bash", str(scripts / builder.name), "--no-models", "--no-deps"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    for suffix in ("advisories", "qdrant-advisories"):
        report = json.loads((tmp_path / f"scalarforensic-latest.{suffix}.json").read_text())
        assert report["status"] == "not-run"
        assert report["reason"]
