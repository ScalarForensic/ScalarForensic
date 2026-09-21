"""Create/verify a checksum inventory of quiescent application state.

This does not snapshot Qdrant, lock writers, or authenticate the inventory.
Keep the inventory outside the state directory and preserve it with the backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inventory(root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("State root must be a directory")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks require separate operator inventory: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        elif not path.is_dir():
            raise ValueError(f"Unsupported state entry: {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.manifest.resolve().is_relative_to(args.root.resolve()):
        parser.error("Keep the manifest outside the state root")
    actual = inventory(args.root)
    if args.action == "capture":
        with args.manifest.open("x", encoding="utf-8") as stream:
            json.dump(actual, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"Captured {len(actual)} files")
        return 0
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        changed = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
        print(json.dumps({"missing": missing, "extra": extra, "changed": changed}))
        return 1
    print(f"Verified {len(actual)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
