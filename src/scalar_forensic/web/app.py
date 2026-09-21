"""FastAPI web application for ScalarForensic Phase 2 query interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from scalar_forensic.config import ENV_ALLOW_ONLINE, Settings
from scalar_forensic.media_worker.client import WorkerUnavailable
from scalar_forensic.video_playback import router as video_playback_router
from scalar_forensic.video_playback.jobs import runner
from scalar_forensic.web.ingress import UploadLimitMiddleware
from scalar_forensic.web.routes import analyze as analyze_routes
from scalar_forensic.web.routes import faces as faces_routes
from scalar_forensic.web.routes import files as files_routes
from scalar_forensic.web.routes import tags as tags_routes
from scalar_forensic.web.routes import video as video_routes
from scalar_forensic.web.session import initialize_temp_root, reap_idle_sessions, shutdown_sessions

_STATIC_DIR = Path(__file__).parent / "static"


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = Settings()
    settings.apply_network_policy()
    inspection = bool(getattr(_app.state, "inspection_mode", False))
    await asyncio.to_thread(_check_collection_compat, settings, ignore_mismatch=inspection)
    if settings.reference_collection:
        reference_settings = copy.copy(settings)
        reference_settings.collection = settings.reference_collection
        await asyncio.to_thread(
            _check_collection_compat, reference_settings, ignore_mismatch=inspection
        )
    await asyncio.to_thread(initialize_temp_root, settings.session_temp_root)

    # Log effective batch size so operators know which value is in use.
    if settings.batch_size is not None:
        logging.getLogger(__name__).info("Batch size: %d (SFN_BATCH_SIZE)", settings.batch_size)
    else:
        # Request modes determine the combined model memory footprint. There is
        # no selected pipeline at startup, so its cache cannot be resolved yet.
        logging.getLogger(__name__).info(
            "Batch size: resolved per request from matching model calibration cache; default 32"
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
            runner.reap_completed()

    reaper_task = asyncio.create_task(_reaper())
    try:
        yield
    finally:
        reaper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper_task
        await shutdown_sessions()
        await runner.shutdown()


def _configured_upload_limits(app):
    settings = Settings()
    return UploadLimitMiddleware(
        app, max_bytes=settings.max_upload_bytes, max_concurrent=settings.max_concurrent_uploads
    )


app = FastAPI(title="ScalarForensic", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.exception_handler(WorkerUnavailable)
async def media_worker_unavailable(_request, _exc) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Media worker unavailable"})


app.add_middleware(_configured_upload_limits)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(analyze_routes.router)
app.include_router(files_routes.router)
app.include_router(video_routes.router)
app.include_router(video_playback_router)
app.include_router(tags_routes.router)
app.include_router(faces_routes.router)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


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
    install).  Populated indexes with missing model/preprocessing provenance require a rebuild.

    Qdrant connectivity errors are reported as warnings — request handling will
    fail naturally if the database stays down — but never silently treated as
    "fresh install", which would mask real configuration drift.

    With ``ignore_mismatch=True`` the check still runs and logs a warning, but
    does not block startup.  This is the ``--ignore-config-mismatch`` escape
    hatch for read-only inspection of a known-incompatible collection; it
    must be opted into per invocation, never via the environment.
    """

    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    except (ConnectionError, OSError, ValueError) as exc:
        print(
            f"[WARN] Qdrant client construction failed during compatibility check: {exc}\n"
            "       Server will start; per-request errors will surface the issue.",
            file=sys.stderr,
        )
        return

    try:
        _check_collection_compat_with_client(client, settings, ignore_mismatch=ignore_mismatch)
    finally:
        client.close()


def _check_collection_compat_with_client(client, settings, *, ignore_mismatch=False):
    from scalar_forensic.safeguards import (
        QdrantUnavailable,
        check_collection_compat,
        expected_model_hashes_from_settings,
    )

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
            f"scores would be silently meaningless against this collection.\n"
            f"Similarity is blocked.\n"
            f"Only metadata and original evidence may be inspected.\n",
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
        f"  • Pass --ignore-config-mismatch for metadata/original viewing only;\n"
        f"    incompatible similarity queries remain blocked.\n",
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
            "collection, for metadata and original-evidence viewing only. "
            "Incompatible similarity queries remain blocked."
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
    err = settings.offline_model_error(need_dino=not args.ignore_config_mismatch)
    if err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    # ASGI lifespan runs the same pre-flight for both console and direct ASGI launches.
    app.state.inspection_mode = args.ignore_config_mismatch

    uvicorn.run(
        "scalar_forensic.web.app:app",
        host=settings.web_host,
        port=settings.web_port,
        reload=False,
    )
