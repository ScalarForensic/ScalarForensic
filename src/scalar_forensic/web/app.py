"""FastAPI web application for ScalarForensic Phase 2 query interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from scalar_forensic.config import ENV_ALLOW_ONLINE, Settings
from scalar_forensic.video_playback import router as video_playback_router
from scalar_forensic.web.routes import analyze as analyze_routes
from scalar_forensic.web.routes import faces as faces_routes
from scalar_forensic.web.routes import files as files_routes
from scalar_forensic.web.routes import tags as tags_routes
from scalar_forensic.web.routes import video as video_routes
from scalar_forensic.web.session import reap_idle_sessions

_STATIC_DIR = Path(__file__).parent / "static"


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = Settings()

    # Log effective batch size so operators know which value is in use.
    if settings.batch_size is not None:
        logging.getLogger(__name__).info("Batch size: %d (SFN_BATCH_SIZE)", settings.batch_size)
    else:
        from scalar_forensic.calibration import load_cached_batch_size

        cached = load_cached_batch_size()
        if cached is not None:
            logging.getLogger(__name__).info("Batch size: %d (calibration cache)", cached)
        else:
            logging.getLogger(__name__).info(
                "Batch size: 32 (default — run `sfn` once to auto-calibrate)"
            )

    # Face modality: warn once at startup rather than per request.  The
    # /api/faces/availability endpoint carries the same reason to the UI.
    if settings.faces_enabled:
        face_err = settings.face_startup_error()
        if face_err:
            logging.getLogger(__name__).warning("Face modality unavailable: %s", face_err)
        for note in settings.face_threshold_notes():
            logging.getLogger(__name__).warning("face threshold: %s", note)

    async def _reaper() -> None:
        while True:
            await asyncio.sleep(60)
            await reap_idle_sessions(settings.session_ttl_seconds)

    reaper_task = asyncio.create_task(_reaper())
    try:
        yield
    finally:
        reaper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper_task


app = FastAPI(title="ScalarForensic", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(analyze_routes.router)
app.include_router(files_routes.router)
app.include_router(video_routes.router)
app.include_router(video_playback_router)
app.include_router(tags_routes.router)
app.include_router(faces_routes.router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _check_collection_compat(settings: Settings, *, ignore_mismatch: bool = False) -> None:
    """Hard-fail if Phase-2 (query) would produce silently wrong results.

    Every Phase-2 query re-embeds the user input with the *current*
    ``SFN_MODEL_*``, ``SFN_NORMALIZE_SIZE`` and ``SFN_SSCD_N_CROPS`` and
    cosine-compares it against vectors in the collection that were produced
    under the *previous* values.  A drift in any of the three changes the
    embedding function — query and corpus vectors no longer share an
    embedding space, and similarity scores become silently meaningless
    (results still rank, calibrated thresholds are no longer calibrated).

    Skips silently when the collection does not exist or has no points (fresh
    install).  Payload fields absent in older indexes are skipped so as not to
    break existing deployments retroactively.

    Qdrant connectivity errors are reported as warnings — request handling will
    fail naturally if the database stays down — but never silently treated as
    "fresh install", which would mask real configuration drift.

    With ``ignore_mismatch=True`` the check still runs and logs a warning, but
    does not block startup.  This is the ``--ignore-config-mismatch`` escape
    hatch for read-only inspection of a known-incompatible collection; it
    must be opted into per invocation, never via the environment.
    """
    from scalar_forensic.safeguards import (
        QdrantUnavailable,
        check_collection_compat,
        expected_model_hashes_from_settings,
    )

    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    except (ConnectionError, OSError, ValueError) as exc:
        print(
            f"[WARN] Qdrant client construction failed during compatibility check: {exc}\n"
            "       Server will start; per-request errors will surface the issue.",
            file=sys.stderr,
        )
        return

    # Only hash the model files that correspond to vectors actually present
    # in the collection — avoids loading a DINOv2 snapshot when the collection
    # is SSCD-only, and vice versa.
    try:
        existing_collections = {c.name for c in client.get_collections().collections}
    except (ConnectionError, OSError) as exc:
        print(
            f"[WARN] Qdrant unreachable during compatibility check: {exc}\n"
            "       Server will start; request handlers will surface the issue.",
            file=sys.stderr,
        )
        return
    except Exception as exc:  # noqa: BLE001 — explicitly logged, not swallowed
        print(
            f"[WARN] Unexpected error contacting Qdrant during compatibility check: {exc}\n"
            "       Server will start; request handlers will surface the issue.",
            file=sys.stderr,
        )
        return

    if settings.collection not in existing_collections:
        return

    try:
        info = client.get_collection(settings.collection)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[WARN] Could not inspect collection {settings.collection!r}: {exc}\n"
            "       Server will start; request handlers will surface the issue.",
            file=sys.stderr,
        )
        return

    vectors_cfg = info.config.params.vectors
    needed_vectors: set[str] = set()
    if isinstance(vectors_cfg, dict):
        for vn in ("dino", "sscd"):
            if vn in vectors_cfg:
                needed_vectors.add(vn)

    expected = expected_model_hashes_from_settings(settings, needed_vectors=needed_vectors)

    try:
        errors = check_collection_compat(
            client,
            settings.collection,
            settings,
            expected_dino_hash=expected.get("dino"),
            expected_sscd_hash=expected.get("sscd"),
        )
    except QdrantUnavailable as exc:
        print(
            f"[WARN] Qdrant unreachable during compatibility check: {exc}\n"
            "       Server will start; request handlers will surface the issue.",
            file=sys.stderr,
        )
        return

    if not errors:
        return

    detail = "\n  ".join(errors)
    if ignore_mismatch:
        print(
            f"\n[WARN] Embedding configuration mismatch (--ignore-config-mismatch set):\n"
            f"\n  {detail}\n"
            f"\nQuery-time embeddings are produced with the current settings and\n"
            f"compared against vectors stored under the previous values.  Similarity\n"
            f"scores will be silently meaningless against this collection.  Use only\n"
            f"for read-only inspection of a known-incompatible index.\n",
            file=sys.stderr,
        )
        return

    print(
        f"\n[ERROR] Embedding configuration mismatch — server cannot start safely.\n"
        f"\n  {detail}\n"
        f"\nQuery-time embeddings are produced with the current settings and\n"
        f"compared against vectors stored under the previous values.  Cosine\n"
        f"scores would be silently meaningless if the server started.\n"
        f"Options:\n"
        f"  • Restore the original settings in .env to match the indexed collection, OR\n"
        f"  • Re-index the collection: sfn <input_dir> --sscd / --dino, OR\n"
        f"  • Download the exact model version recorded in this collection"
        f" (local DINOv2/SSCD only):\n"
        f"      uv run python scripts/download_models.py --dino --hash <stored_hash>\n"
        f"      uv run python scripts/download_models.py --sscd --hash <stored_hash>\n"
        f"    (use the stored= value from the mismatch detail above), OR\n"
        f"  • Pass --ignore-config-mismatch to start anyway (read-only; results will be\n"
        f"    silently wrong — never use for forensic conclusions)\n",
        file=sys.stderr,
    )
    sys.exit(1)


def start() -> None:
    parser = argparse.ArgumentParser(
        prog="sfn-web",
        description="ScalarForensic web UI",
    )
    parser.add_argument(
        "--allow-online",
        action="store_true",
        default=False,
        help=(
            "Allow outward internet connections (e.g. to HuggingFace Hub for first-time "
            "model downloads). Offline by default — see SFN_ALLOW_ONLINE in .env."
        ),
    )
    parser.add_argument(
        "--ignore-config-mismatch",
        action="store_true",
        default=False,
        help=(
            "Start the server even when the current SFN_MODEL_*, SFN_NORMALIZE_SIZE "
            "or SFN_SSCD_N_CROPS differ from the values used to populate the indexed "
            "collection.  Cosine similarity scores will be silently meaningless under "
            "drift; use only for read-only inspection of a known-incompatible index, "
            "never for forensic conclusions."
        ),
    )
    args = parser.parse_args()

    # Write back to os.environ before constructing Settings so that every
    # per-request Settings() instance created by FastAPI handlers also sees
    # allow_online=True — mutating the object here would have no effect on them.
    if args.allow_online:
        os.environ[ENV_ALLOW_ONLINE] = "true"

    settings = Settings()

    # Apply HuggingFace offline guard before any model loading occurs.
    # Qdrant / remote-embedder connections are unaffected.
    settings.apply_network_policy()

    # Pre-flight: always check DINOv2 — available modes aren't known until Qdrant
    # is queried, so we conservatively validate all potentially-used model configs.
    err = settings.offline_model_error(need_dino=True)
    if err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    # Pre-flight: reject mismatched embedding config before accepting any requests.
    _check_collection_compat(settings, ignore_mismatch=args.ignore_config_mismatch)

    uvicorn.run(
        "scalar_forensic.web.app:app",
        host=settings.web_host,
        port=settings.web_port,
        reload=False,
    )
