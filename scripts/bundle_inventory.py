"""Inventory actual bundle inputs, with unknown licenses left explicit."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def member_digest(wheel: zipfile.ZipFile, name: str) -> str:
    with wheel.open(name) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> dict:
    artifacts = []
    for directory, kind in (("vendor", "wheel"), ("models", "model")):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file():
                continue
            entry = {
                "path": path.relative_to(root).as_posix(),
                "kind": kind,
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "license": "UNKNOWN: release maintainer must review redistribution rights",
            }
            if path.suffix == ".whl":
                with zipfile.ZipFile(path) as wheel:
                    # Only the root dist-info describes this wheel; vendored
                    # dependencies may carry nested dist-info directories.
                    metadata = [
                        n
                        for n in wheel.namelist()
                        if n.count("/") == 1 and n.endswith(".dist-info/METADATA")
                    ]
                    if len(metadata) != 1:
                        raise ValueError(f"Expected one wheel metadata file: {path}")
                    message = email.message_from_bytes(wheel.read(metadata[0]))
                    entry.update(
                        name=message["Name"],
                        version=message["Version"],
                        license=message["License-Expression"]
                        or message["License"]
                        or entry["license"],
                        license_files=[n for n in wheel.namelist() if "license" in n.lower()],
                        native_members=[
                            {"path": n, "sha256": member_digest(wheel, n)}
                            for n in wheel.namelist()
                            if ".so" in n or n.endswith((".dll", ".dylib"))
                        ],
                    )
            artifacts.append(entry)
    for name in (
        "uv.lock",
        "requirements.txt",
        "requirements-build.txt",
        "THIRD_PARTY_LICENSES.md",
    ):
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing release input: {path}")
        artifacts.append({"path": name, "kind": "release-input", "sha256": digest(path)})
    return {"schema": 1, "review_owner": "release maintainer", "artifacts": artifacts}


def native_inventory() -> dict:
    packages = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${Package}\t${Version}\n"], text=True
    ).splitlines()
    files = []
    for base in (Path("/usr/lib"), Path("/usr/local/lib")):
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.is_symlink() and ".so" in path.name:
                files.append({"path": str(path), "sha256": digest(path)})
    for executable in (Path("/usr/bin/ffmpeg"), Path("/usr/bin/ffprobe")):
        if executable.exists():
            files.append({"path": str(executable), "sha256": digest(executable)})
    return {
        "schema": 1,
        "packages": packages,
        "native_files": files,
        "license_records": "/usr/share/doc/*/copyright and installed *.dist-info/licenses",
        "review_owner": "release maintainer",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = native_inventory() if args.native else inventory(args.root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
