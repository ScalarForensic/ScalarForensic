"""Video frame extraction for forensic processing.

Treats video files as containers: extracts representative frames via uniform
temporal sampling and feeds each frame through the existing image embedding
pipeline unchanged.

Forensic reproducibility guarantee: given the same video file and the same
extraction parameters (fps, max_frames), this module always produces frames
with identical content, timecodes, and SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".3gp",
        ".ts",
        ".mts",
    }
)


@dataclass(frozen=True)
class ExtractedFrame:
    """A single frame extracted from a video file.

    All fields are immutable to ensure accidental mutation is caught early.
    ``frame_bytes`` is PNG-encoded (lossless, deterministic) and is the
    canonical byte representation used for SHA-256 hashing and thumbnail
    generation.  ``image`` is the decoded PIL RGB Image for embedding.
    """

    image: Image.Image
    timecode_ms: int  # position in video in milliseconds
    frame_index: int  # ordinal among yielded unique frames (0-based)
    frame_hash: str  # SHA-256 of PNG-encoded frame bytes
    frame_bytes: bytes  # PNG bytes used for hashing + thumbnail generation

    # Needed so the frozen dataclass can hold a non-hashable Image.Image
    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.frame_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractedFrame):
            return NotImplemented
        return self.frame_hash == other.frame_hash


def _png_encode(image: Image.Image) -> bytes:
    """Encode a PIL Image to PNG bytes (lossless, deterministic across platforms)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def extract_frames(
    video_path: Path,
    fps: float = 1.0,
    max_frames: int = 500,
) -> Iterator[ExtractedFrame]:
    """Yield representative frames from a video at the given sampling rate.

    Sampling strategy: uniform temporal sampling — one frame per ``1/fps``
    seconds.  Only the first decoded frame at or past each target timestamp is
    kept.  Within-video hash deduplication removes identical frames (e.g. a
    static title card or freeze frame) so that storage is not wasted on
    content-identical segments.

    Forensic reproducibility: the same video file + same (fps, max_frames)
    arguments always produces the same sequence of frames with identical
    content, timecodes, and SHA-256 hashes.

    Performance: uses seek-based extraction — for each target timestamp the
    container is seeked to the nearest keyframe before the target, then frames
    are decoded only until the target is reached.  This scales with the number
    of extracted frames rather than total video frames, which gives large
    speed-ups for long or high-frame-rate videos sampled at low fps.  Falls
    back to sequential decoding automatically for containers that do not
    support seeking (e.g. some live-stream recordings).

    Args:
        video_path: Absolute path to the video file (read-only access).
        fps: Frames to extract per second of video (default 1.0).
        max_frames: Hard cap on the number of yielded frames. 0 means no cap.

    Yields:
        :class:`ExtractedFrame` for each unique (non-duplicate) sampled frame.
    """
    import av  # local import: PyAV is optional at the module level

    frame_interval_s = 1.0 / fps
    seen_hashes: set[str] = set()

    try:
        container = av.open(str(video_path))
    except Exception as exc:
        raise RuntimeError(f"Cannot open video {video_path}: {exc}") from exc

    try:
        video_streams = container.streams.video
        if not video_streams:
            raise RuntimeError(f"No video stream found in {video_path}")

        video_stream = video_streams[0]
        time_base = float(video_stream.time_base)

        # Resolve total duration (seconds) — used as loop termination guard.
        duration_s: float | None = None
        if container.duration is not None:
            duration_s = container.duration / av.time_base
        elif video_stream.duration is not None:
            duration_s = float(video_stream.duration) * time_base

        # Probe whether the container supports reliable seeking by attempting
        # a single seek to t=0.  If it raises or the stream reports no_seek,
        # fall back to the sequential (full-decode) path.
        seekable = not getattr(video_stream, "no_seek", False)
        if seekable:
            try:
                container.seek(0, backward=True, any_frame=False, stream=video_stream)
            except Exception:
                seekable = False

        if seekable:
            yield from _extract_frames_seek(
                container, video_stream, time_base, duration_s,
                frame_interval_s, seen_hashes, max_frames,
            )
        else:
            yield from _extract_frames_sequential(
                container, video_stream, time_base,
                frame_interval_s, seen_hashes, max_frames,
            )
    finally:
        container.close()


def _make_frame(
    av_frame,  # av.VideoFrame
    time_base: float,
    frame_index: int,
) -> ExtractedFrame:
    """Decode, PNG-encode, and hash one PyAV video frame."""
    pil_image = av_frame.to_image()
    frame_bytes = _png_encode(pil_image)
    frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    pts_s = float(av_frame.pts) * time_base
    return ExtractedFrame(
        image=pil_image,
        timecode_ms=int(pts_s * 1000),
        frame_index=frame_index,
        frame_hash=frame_hash,
        frame_bytes=frame_bytes,
    )


def _extract_frames_seek(
    container,
    video_stream,
    time_base: float,
    duration_s: float | None,
    frame_interval_s: float,
    seen_hashes: set[str],
    max_frames: int,
) -> Iterator[ExtractedFrame]:
    """Seek-based frame extraction — O(extracted_frames) decoder work."""
    frame_index = 0
    target_s = 0.0

    while True:
        if max_frames > 0 and frame_index >= max_frames:
            break
        if duration_s is not None and target_s > duration_s + frame_interval_s:
            break

        target_pts = int(target_s / time_base)
        try:
            container.seek(target_pts, backward=True, any_frame=False, stream=video_stream)
        except Exception:
            break  # container stopped accepting seeks

        found: ExtractedFrame | None = None
        for av_frame in container.decode(video=0):
            if av_frame.pts is None:
                continue
            pts_s = float(av_frame.pts) * time_base
            if pts_s < target_s:
                continue  # still before target — decode forward from keyframe

            # First frame at or after target_s
            ef = _make_frame(av_frame, time_base, frame_index)
            if ef.frame_hash not in seen_hashes:
                seen_hashes.add(ef.frame_hash)
                yield ef
                frame_index += 1
            else:
                # Deduplicated — don't increment frame_index so ordinals stay
                # contiguous, but do advance the target.
                pass
            found = ef
            break

        if found is None:
            break  # no frame found past target → end of stream

        # Advance to the next target, skipping over the frame we just returned.
        target_s = max(target_s + frame_interval_s, found.timecode_ms / 1000.0 + frame_interval_s)


def _extract_frames_sequential(
    container,
    video_stream,
    time_base: float,
    frame_interval_s: float,
    seen_hashes: set[str],
    max_frames: int,
) -> Iterator[ExtractedFrame]:
    """Sequential fallback — decodes every frame; used when seeking is unavailable."""
    next_target_s = 0.0
    frame_index = 0

    for av_frame in container.decode(video=0):
        if av_frame.pts is None:
            continue
        pts_s = float(av_frame.pts) * time_base

        if pts_s < next_target_s:
            continue

        ef = _make_frame(av_frame, time_base, frame_index)

        if ef.frame_hash in seen_hashes:
            next_target_s = pts_s + frame_interval_s
            continue

        seen_hashes.add(ef.frame_hash)
        yield ef
        frame_index += 1

        if max_frames > 0 and frame_index >= max_frames:
            break

        next_target_s = pts_s + frame_interval_s


def extract_frame_at(video_path: Path, timecode_ms: int) -> Image.Image | None:
    """Seek to ``timecode_ms`` and return the nearest frame as a PIL RGB Image.

    Used by the web API to serve individual frames on demand without pre-extracting
    or caching anything on disk.

    Returns ``None`` if the video cannot be opened, the timecode is out of range,
    or no frame is found near the requested position.
    """
    import av  # local import

    target_s = timecode_ms / 1000.0
    try:
        container = av.open(str(video_path))
    except Exception:
        return None

    try:
        video_streams = container.streams.video
        if not video_streams:
            return None

        video_stream = video_streams[0]
        time_base = float(video_stream.time_base)

        # Seek to the nearest keyframe at or before the target
        target_pts = int(target_s / time_base)
        container.seek(target_pts, backward=True, any_frame=False, stream=video_stream)

        # Decode forward until we reach the target timestamp
        for av_frame in container.decode(video=0):
            if av_frame.pts is None:
                continue
            pts_s = av_frame.pts * time_base
            # Accept the first frame at or just before the target (±1 frame tolerance)
            if pts_s >= target_s - (1.0 / max(float(video_stream.average_rate or 30), 1)):
                return av_frame.to_image()
    except Exception:
        return None
    finally:
        container.close()

    return None


def get_video_info(video_path: Path) -> dict:
    """Return metadata about a video file (duration, codec, resolution, fps).

    Used for timeline rendering in the web UI and for forensic metadata display.
    Returns a dict with best-effort fields; never raises.
    """
    import av  # local import

    try:
        with av.open(str(video_path)) as container:
            info: dict = {}
            if container.duration is not None:
                info["duration_s"] = container.duration / av.time_base
            if container.bit_rate:
                info["bit_rate"] = container.bit_rate
            if container.streams.video:
                vs = container.streams.video[0]
                info["codec"] = vs.codec_context.name
                info["width"] = vs.width
                info["height"] = vs.height
                if vs.average_rate:
                    info["fps"] = float(vs.average_rate)
                if vs.frames:
                    info["frame_count"] = vs.frames
            return info
    except Exception as exc:
        return {"error": str(exc)}


def get_pyav_version() -> str:
    """Return the installed PyAV version string for provenance tracking."""
    try:
        return importlib.metadata.version("av")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def make_virtual_path(video_path: Path, frame: ExtractedFrame) -> str:
    """Return the virtual path string for a video frame.

    Format: ``/absolute/path/to/video.mp4::frame_000001_t=1000ms``

    The ``::`` separator is unambiguous in filesystem paths (colons are valid
    on Linux but ``::`` does not appear in normal paths).  The timecode suffix
    ``t={ms}ms`` is the authoritative re-extraction key — ``frame_index`` is
    informational only.
    """
    abs_path = str(video_path.resolve())
    return f"{abs_path}::frame_{frame.frame_index:06d}_t={frame.timecode_ms}ms"


def parse_virtual_path(virtual_path: str) -> tuple[Path, int] | None:
    """Parse a virtual video frame path into ``(video_path, timecode_ms)``.

    Returns ``None`` if the string is not a valid virtual frame path.
    """
    if "::" not in virtual_path:
        return None
    video_part, frame_spec = virtual_path.split("::", 1)
    # Parse "frame_000001_t=1000ms"
    if not frame_spec.startswith("frame_") or "_t=" not in frame_spec:
        return None
    try:
        t_part = frame_spec.rsplit("_t=", 1)[1]
        if not t_part.endswith("ms"):
            return None
        timecode_ms = int(t_part[:-2])
    except (ValueError, IndexError):
        return None
    return Path(video_part), timecode_ms
