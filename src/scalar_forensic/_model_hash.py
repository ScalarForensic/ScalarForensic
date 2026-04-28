"""Shared DINOv2 model-hash primitives.

Imported by safeguards.py and embedder.py.  This module is intentionally
stdlib-only (no torch, no qdrant, no huggingface_hub) so it can be imported
in any context without heavyweight side-effects.

scripts/download_models.py mirrors _hash_dino_snapshot without importing the
package (to keep that script's dependency footprint minimal).  When changing
DINO_CONTENT_EXTENSIONS or _hash_dino_snapshot, update that script too.
"""

import hashlib
from pathlib import Path

# Only these file extensions carry model weights or configuration that determine
# embedding output.  huggingface_hub ≥ 0.20 populates the model directory with
# auxiliary files (*.metadata, CACHEDIR.TAG, .gitignore) whose presence varies
# by library version; the HuggingFace repository also includes .gitattributes
# and README.md that are irrelevant to inference.  Including any of these in
# the hash would make it impossible to reproduce the same value across
# deployment environments even when the actual model weights are byte-for-byte
# identical — which is exactly the property forensic integrity requires.
#
# scripts/download_models.py mirrors this value; keep both in sync.
DINO_CONTENT_EXTENSIONS: frozenset[str] = frozenset({".safetensors", ".bin", ".json"})


def hash_dino_snapshot(snapshot_path: Path) -> str:
    """SHA-256 over the content files of a resolved DINOv2 snapshot directory.

    Only files whose suffix is in ``DINO_CONTENT_EXTENSIONS`` contribute to
    the digest, in lexicographic order of their basename.  Subdirectory depth
    is irrelevant — only the basename (not the relative path) enters the hash,
    matching the structure of standard HuggingFace snapshots where all content
    files sit at the top level.
    """
    h = hashlib.sha256()
    for file in sorted(snapshot_path.rglob("*")):
        if not file.is_file() or file.suffix not in DINO_CONTENT_EXTENSIONS:
            continue
        h.update(file.name.encode())
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()
