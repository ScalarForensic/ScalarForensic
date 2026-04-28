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

    The returned string is prefixed with ``config:`` to make it unambiguous,
    both in stored Qdrant payloads and in the UI, that this hash identifies
    the remote endpoint configuration rather than actual model weights.
    """
    h = hashlib.sha256()
    h.update(endpoint.rstrip("/").encode())
    h.update(model_name.encode())
    h.update(str(embedding_dim).encode())
    return "config:" + h.hexdigest()


# Maximum points to inspect per vector type when looking for collection-internal
# provenance inconsistency.  Large enough to almost certainly catch a mixed
# corpus (any reasonable mix has the second config show up well before 1024
# points), small enough that the scan runs in tens of milliseconds at startup.
_INCONSISTENCY_SAMPLE_LIMIT = 1024
_SCROLL_PAGE = 256


def _sample_provenance_tuples(
    client: QdrantClient,
    collection: str,
    vector_name: str,
    fields: list[str],
) -> set[tuple]:
    """Return the set of distinct provenance tuples observed in a sample.

    Tuples are ``(model_hash, normalize_size, sscd_n_crops_or_None)``.  Stops
    early as soon as a *second* distinct tuple appears — we only need to
    distinguish "collection is internally consistent" from "collection has
    multiple configs".  Points whose provenance fields are entirely absent
    (older indexes that pre-date the payload schema) are skipped so they do
    not register as a phantom "(None, None, None)" tuple.
    """
    seen: set[tuple] = set()
    next_offset = None
    sampled = 0
    while sampled < _INCONSISTENCY_SAMPLE_LIMIT:
        page_limit = min(_SCROLL_PAGE, _INCONSISTENCY_SAMPLE_LIMIT - sampled)
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[HasVectorCondition(has_vector=vector_name)]),
            with_payload=fields,
            with_vectors=False,
            limit=page_limit,
            offset=next_offset,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            tup = (
                payload.get(f"{vector_name}_model_hash"),
                payload.get(f"{vector_name}_normalize_size"),
                payload.get("sscd_n_crops") if vector_name == "sscd" else None,
            )
            if tup == (None, None, None):
                # Pre-provenance index — skip rather than treat as a real tuple.
                continue
            seen.add(tup)
            if len(seen) > 1:
                return seen
        sampled += len(points)
        if next_offset is None:
            break
    return seen


def _format_provenance_tuple(vn: str, tup: tuple) -> str:
    """Human-readable line for one observed provenance tuple."""
    h, norm, crops = tup
    parts = [f"model_hash={h}" if h else "model_hash=<absent>"]
    if vn == "dino":
        parts.append(f"normalize_size={norm}" if norm is not None else "normalize_size=<absent>")
    if vn == "sscd":
        parts.append(f"sscd_n_crops={crops}" if crops is not None else "sscd_n_crops=<absent>")
    return "    " + ", ".join(parts)


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

    Both ``sfn index`` and ``sfn-web`` enforce all three fields by default —
    drift in any of them changes the embedding function (model_hash → different
    weights, normalize_size → different ViT input resolution, sscd_n_crops →
    different multi-crop average), and cosine similarity across two embedding
    spaces is mathematically undefined.

    Two distinct failure modes are detected:

    1. **Current-vs-stored mismatch** — current settings differ from the
       (uniform) provenance tuple recorded in the collection.
    2. **Collection-internal inconsistency** — a sample of up to
       ``_INCONSISTENCY_SAMPLE_LIMIT`` points reveals two or more distinct
       provenance tuples for the same vector type.  This means a previous
       run (probably with ``--ignore-config-mismatch``) merged two embedding
       configurations into one collection; nothing the caller does to .env
       will fix it — the collection itself must be re-indexed.

    The ``check_normalize_size`` flag exists for tests that want to assert
    individual checks in isolation, not as a phase-discrimination knob.

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
            seen = _sample_provenance_tuples(client, collection, vn, fields)
        except (
            ResponseHandlingException,
            UnexpectedResponse,
            ConnectionError,
            OSError,
        ) as exc:
            raise QdrantUnavailable(str(exc)) from exc

        if not seen:
            # Either no points of this vector type, or all sampled points
            # pre-date the provenance schema.  Nothing to compare.
            continue

        if len(seen) > 1:
            tuple_lines = "\n".join(_format_provenance_tuple(vn, t) for t in sorted(seen))
            errors.append(
                f"[{vn}] collection contains multiple embedding configurations "
                f"(found {len(seen)} distinct provenance tuples in a sample of "
                f"≤{_INCONSISTENCY_SAMPLE_LIMIT} points):\n{tuple_lines}\n"
                "    The collection itself is internally inconsistent — no .env "
                "change can make queries meaningful.  Re-index from scratch."
            )
            # Skip current-vs-stored comparison: with multiple stored configs
            # there is no single "stored" tuple to compare against.
            continue

        (stored_hash, stored_norm, stored_crops) = next(iter(seen))

        expected_hash = expected_hash_by_vector[vn]
        if expected_hash is not None and stored_hash and stored_hash != expected_hash:
            errors.append(f"[{vn}] model_hash: stored={stored_hash}  current={expected_hash}")

        # normalize_size — SSCD's is fixed (288 px) regardless of SFN_NORMALIZE_SIZE,
        # so only DINOv2's is user-controlled and worth comparing here.
        if check_normalize_size and vn == "dino":
            if stored_norm is not None and stored_norm != settings.normalize_size:
                errors.append(
                    f"[{vn}] normalize_size: stored={stored_norm}"
                    f"  current={settings.normalize_size}"
                )

        if vn == "sscd":
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
