"""Phase 1 of the web pipeline: hashing and embedding uploaded files."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from scalar_forensic.config import Settings
from scalar_forensic.embedder import (
    _SSCD_SCALE,
    AnyEmbedder,
    effective_preprocessing_cap,
    hash_bytes,
    hash_bytes_md5,
    hash_file_both,
    preprocess_batch,
    preprocess_pil_batch,
)
from scalar_forensic.video import VIDEO_EXTENSIONS, extract_frames
from scalar_forensic.web.pipeline._embedders import _get_embedder
from scalar_forensic.web.session import FileEntry, Session, VideoFrameEntry

# ---------------------------------------------------------------------------
# Phase 1: Analysis
# ---------------------------------------------------------------------------


@dataclass
class ProgressEvent:
    type: str  # "progress" | "file_done" | "error" | "done"
    current: int = 0
    total: int | None = None
    filename: str = ""
    file_id: str = ""
    message: str = ""
    session_id: str = ""


def analyze_session(
    session: Session,
    modes: list[str],
    settings: Settings,
) -> Generator[ProgressEvent]:
    """Hash and embed every file in the session. Yields progress events."""
    need_sscd = "altered" in modes
    need_dino = "semantic" in modes

    embedders: dict[str, AnyEmbedder] = {}
    if need_sscd:
        try:
            embedders["sscd"] = _get_embedder("sscd", settings)
        except Exception as exc:  # noqa: BLE001
            yield ProgressEvent(type="error", message=f"SSCD model load failed: {exc}")
    if need_dino:
        try:
            embedders["dino"] = _get_embedder("dino", settings)
        except Exception as exc:  # noqa: BLE001
            yield ProgressEvent(type="error", message=f"DINOv2 model load failed: {exc}")

    total = len(session.files)
    for i, entry in enumerate(session.files):
        yield ProgressEvent(
            type="progress",
            current=i,
            total=total,
            filename=entry.filename,
            file_id=entry.file_id,
        )
        try:
            if Path(entry.filename).suffix.lower() in VIDEO_EXTENSIONS:
                yield from _analyze_video_file(entry, embedders, settings)
            else:
                _analyze_file(entry, embedders)
            yield ProgressEvent(
                type="file_done",
                current=i + 1,
                total=total,
                filename=entry.filename,
                file_id=entry.file_id,
            )
        except Exception as exc:  # noqa: BLE001
            entry.error = str(exc)
            yield ProgressEvent(
                type="error",
                current=i + 1,
                total=total,
                filename=entry.filename,
                file_id=entry.file_id,
                message=str(exc),
            )

    yield ProgressEvent(type="done", total=total, session_id=session.session_id)


def _analyze_file(entry: FileEntry, embedders: dict[str, AnyEmbedder]) -> None:
    data = entry.temp_path.read_bytes()
    entry.file_hash = hash_bytes(data)
    entry.file_hash_md5 = hash_bytes_md5(data)
    if not embedders:
        return
    _effective_cap = effective_preprocessing_cap(
        max((e.normalize_size for e in embedders.values()), default=_SSCD_SCALE)
    )
    pre_results = preprocess_batch([data], cap=_effective_cap)
    result = pre_results[0]
    if isinstance(result, Exception):
        raise result.with_traceback(result.__traceback__)
    pre_images = [result]
    for key, embedder in embedders.items():
        norm_images = embedder.normalize_batch_bytes(pre_images)
        emb = embedder.embed_images(norm_images)[0]
        if key == "sscd":
            entry.sscd_embedding = emb
        else:
            entry.dino_embedding = emb


def _video_frame_batch(settings: Settings) -> int:
    """Return the effective video-frame batch size.

    Resolution order: explicit ``SFN_BATCH_SIZE`` > calibration cache > 32.
    """
    if settings.batch_size is not None:
        return settings.batch_size
    from scalar_forensic.calibration import load_cached_batch_size

    return load_cached_batch_size() or 32


def _analyze_video_file(
    entry: FileEntry, embedders: dict[str, AnyEmbedder], settings: Settings
) -> Generator[ProgressEvent, None, None]:
    """Extract frames from an uploaded video temp file and embed each one.

    Frames are extracted via the seek-based generator and processed in batches
    whose size is resolved via :func:`_video_frame_batch` (explicit config,
    calibration cache, or default 32) so peak memory is bounded regardless of
    video length.  The video file is hashed in a streaming pass rather than
    loaded into RAM.

    Yields ``video_progress`` ProgressEvents after each batch so the web UI
    can show how many frames have been processed while a long video is running.
    """
    from itertools import batched as _batched

    # Single-pass chunked hash — avoids two separate reads of the upload file.
    entry.file_hash, entry.file_hash_md5 = hash_file_both(entry.temp_path)
    entry.is_video = True
    frame_entries: list[VideoFrameEntry] = []
    _batch_sz = _video_frame_batch(settings)

    try:
        gen = extract_frames(
            entry.temp_path, fps=settings.video_fps, max_frames=settings.video_max_frames
        )
        for raw_batch in _batched(gen, _batch_sz):
            frames = list(raw_batch)
            if not frames:
                continue

            _effective_cap = effective_preprocessing_cap(
                max((e.normalize_size for e in embedders.values()), default=_SSCD_SCALE)
            )
            pil_images = preprocess_pil_batch([f.image for f in frames], cap=_effective_cap)

            # Create / extend frame_entries for this batch
            batch_start = len(frame_entries)
            for frame in frames:
                frame_entries.append(
                    VideoFrameEntry(
                        frame_index=frame.frame_index,
                        timecode_ms=frame.timecode_ms,
                        frame_hash=frame.frame_hash,
                    )
                )

            for key, embedder in embedders.items():
                norm_images = embedder.normalize_batch_bytes(pil_images)
                embeddings = embedder.embed_images(norm_images)
                for j, emb in enumerate(embeddings):
                    fe = frame_entries[batch_start + j]
                    if key == "sscd":
                        fe.sscd_embedding = emb
                    else:
                        fe.dino_embedding = emb

            yield ProgressEvent(
                type="video_progress",
                current=len(frame_entries),
                total=settings.video_max_frames or None,
                filename=entry.filename,
                file_id=entry.file_id,
            )

    except Exception as exc:
        raise RuntimeError(f"Video frame extraction failed for {entry.filename}: {exc}") from exc

    if not frame_entries:
        raise RuntimeError("No frames could be extracted from the video")

    entry.video_frames = frame_entries

    # The first frame's embeddings serve as the entry's top-level vector.
    # query_session() iterates entry.video_frames to query with each frame.
    entry.sscd_embedding = frame_entries[0].sscd_embedding
    entry.dino_embedding = frame_entries[0].dino_embedding
