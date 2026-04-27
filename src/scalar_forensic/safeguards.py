"""Embedding-configuration safeguards.

Centralises the consistency checks that gate ``sfn index`` and ``sfn-web``
startup against an existing Qdrant collection.  Embeddings are *not*
back-compatible across configuration changes — a collection populated with
``SFN_NORMALIZE_SIZE=224`` cannot be safely queried with ``=512`` because
cosine distances between the two embedding spaces are meaningless.

The functions in this module read one existing point per vector type and
compare its provenance payload (``{vn}_model_hash``, ``{vn}_normalize_size``,
``sscd_n_crops``) against the current settings.  Older indexes that lack
these fields are skipped — the absence of a field is treated as "unknown",
not "mismatch", to avoid retroactively breaking existing deployments.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Filter, HasVectorCondition

if TYPE_CHECKING:
    from scalar_forensic.config import Settings


class QdrantUnavailable(Exception):
    """Raised when Qdrant cannot be reached at startup-check time."""


def compute_dino_model_hash(model_name: str) -> str:
    """Hash a DINOv2 snapshot directory or local model directory.

    Mirrors :pyattr:`scalar_forensic.embedder.DinoV2Embedder.model_hash`
    *without* loading the model into memory, so it is cheap enough for
    web-server startup.  Each file's basename and content contribute to
    the digest, so renaming or replacing any snapshot file changes the hash.
    """
    local = Path(model_name)
    if local.is_dir():
        snapshot_path = local
    else:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(snapshot_download(model_name, local_files_only=True))

    h = hashlib.sha256()
    for file in sorted(snapshot_path.rglob("*")):
        if not file.is_file():
            continue
        h.update(file.name.encode())
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def compute_sscd_model_hash(model_path: str | Path) -> str:
    """Hash an SSCD TorchScript checkpoint file.

    Mirrors :pyattr:`scalar_forensic.embedder.SSCDEmbedder.model_hash`.
    """
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_remote_model_hash(endpoint: str, model_name: str, embedding_dim: int) -> str:
    """Hash a remote (OpenAI-compatible) embedder configuration.

    Mirrors :pyattr:`scalar_forensic.embedder.RemoteEmbedder.model_hash`.
    """
    h = hashlib.sha256()
    h.update(endpoint.rstrip("/").encode())
    h.update(model_name.encode())
    h.update(str(embedding_dim).encode())
    return h.hexdigest()


def _fetch_one_point_payload(
    client: QdrantClient,
    collection: str,
    vector_name: str,
    fields: list[str],
) -> dict | None:
    """Return the payload of one point that has *vector_name* set, or ``None``."""
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[HasVectorCondition(has_vector=vector_name)]),
        with_payload=fields,
        with_vectors=False,
        limit=1,
    )
    if not points:
        return None
    return points[0].payload or {}


def check_collection_compat(
    client: QdrantClient,
    collection: str,
    settings: Settings,
    *,
    expected_dino_hash: str | None = None,
    expected_sscd_hash: str | None = None,
    check_normalize_size: bool = True,
) -> list[str]:
    """Return a list of human-readable mismatch descriptions.

    An empty return value means *no detectable mismatch* — either everything
    matches, the collection does not exist yet, the collection has no points
    of that vector type, or the existing points pre-date the provenance
    payload fields.

    Phase-specific scope (caller-controlled):

    * **Indexing (``sfn index``)** must verify all three fields — mixing two
      embedding configurations in one collection corrupts the index
      irreversibly.  Pass the model hashes and leave ``check_normalize_size=True``.
    * **Web (``sfn-web``)** only verifies ``sscd_n_crops`` because that is the
      single setting whose mismatch causes Phase-2 (query) to silently produce
      wrong vectors at request time.  ``normalize_size`` and ``model_hash``
      drift are caught at the next index run, not at server startup, so the
      web caller passes ``expected_*_hash=None`` and
      ``check_normalize_size=False``.

    :raises QdrantUnavailable: when Qdrant is unreachable or returns an
        unexpected response.  Callers decide whether that is fatal.
    """
    try:
        existing = {c.name for c in client.get_collections().collections}
    except (ResponseHandlingException, UnexpectedResponse, ConnectionError, OSError) as exc:
        raise QdrantUnavailable(str(exc)) from exc

    if collection not in existing:
        return []

    try:
        info = client.get_collection(collection)
    except (ResponseHandlingException, UnexpectedResponse, ConnectionError, OSError) as exc:
        raise QdrantUnavailable(str(exc)) from exc

    vectors_cfg = info.config.params.vectors
    if not isinstance(vectors_cfg, dict):
        # Single-vector legacy collection — provenance fields not used.
        return []

    errors: list[str] = []

    expected_hash_by_vector: dict[str, str | None] = {
        "dino": expected_dino_hash,
        "sscd": expected_sscd_hash,
    }

    for vn in ("dino", "sscd"):
        if vn not in vectors_cfg:
            continue

        fields = [f"{vn}_model_hash", f"{vn}_normalize_size"]
        if vn == "sscd":
            fields.append("sscd_n_crops")

        try:
            payload = _fetch_one_point_payload(client, collection, vn, fields)
        except (
            ResponseHandlingException,
            UnexpectedResponse,
            ConnectionError,
            OSError,
        ) as exc:
            raise QdrantUnavailable(str(exc)) from exc

        if payload is None:
            continue

        # model_hash — only checked when the caller supplied an expected value.
        expected_hash = expected_hash_by_vector[vn]
        stored_hash = payload.get(f"{vn}_model_hash")
        if expected_hash is not None and stored_hash and stored_hash != expected_hash:
            errors.append(f"[{vn}] model_hash: stored={stored_hash}  current={expected_hash}")

        # normalize_size — SSCD's is fixed (288 px) regardless of SFN_NORMALIZE_SIZE,
        # so only DINOv2's is user-controlled and worth comparing here.
        if check_normalize_size and vn == "dino":
            stored_norm = payload.get(f"{vn}_normalize_size")
            if stored_norm is not None and stored_norm != settings.normalize_size:
                errors.append(
                    f"[{vn}] normalize_size: stored={stored_norm}"
                    f"  current={settings.normalize_size}"
                )

        if vn == "sscd":
            stored_crops = payload.get("sscd_n_crops")
            if stored_crops is not None and stored_crops != settings.sscd_n_crops:
                errors.append(
                    f"[sscd] sscd_n_crops: stored={stored_crops}  current={settings.sscd_n_crops}"
                )

    return errors


def expected_model_hashes_from_settings(
    settings: Settings,
    *,
    needed_vectors: set[str],
) -> dict[str, str]:
    """Compute model hashes the web server would produce for *settings*.

    *needed_vectors* names the vector types we want hashes for (subset of
    ``{"dino", "sscd"}``).  Only those are computed — hashing the DINOv2
    snapshot is cheap but not free, and we should skip it when no ``dino``
    vector is present in the collection.

    For remote embedders, the same ``endpoint+model+dim`` hash is returned
    for every requested vector type, mirroring how RemoteEmbedder works.
    """
    hashes: dict[str, str] = {}
    if settings.embedding_endpoint:
        if not settings.embedding_model or not settings.embedding_dim:
            return {}
        h = compute_remote_model_hash(
            settings.embedding_endpoint,
            settings.embedding_model,
            settings.embedding_dim,
        )
        for vn in needed_vectors:
            hashes[vn] = h
        return hashes

    if "dino" in needed_vectors:
        try:
            hashes["dino"] = compute_dino_model_hash(settings.model_dino)
        except Exception:  # noqa: BLE001
            # Model file is missing or HF resolution failed — separate error
            # path (offline_model_error) will surface that; don't crash the
            # safeguard with an unrelated message.
            pass
    if "sscd" in needed_vectors:
        try:
            hashes["sscd"] = compute_sscd_model_hash(settings.model_sscd)
        except (FileNotFoundError, OSError):
            pass
    return hashes
