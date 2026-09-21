"""Source-scoped frame planning and per-modality completion decisions."""

from dataclasses import dataclass
from pathlib import Path

from .models import _S_FAIL_EMB, _S_INDEXED, _S_SKIP_FRAME_DUP, _S_SKIP_IDX


@dataclass
class FramePlan:
    needs_per_spec: list[set[Path]]
    any_needs: set[Path]
    confirmed_by_spec: dict[int, dict[Path, int]]
    skipped_per_spec: list[int]


def plan_frames(
    paths: list[Path],
    metadata: dict[Path, dict],
    sources: dict[Path, Path],
    stored_per_spec: list[set[tuple[str, int]]],
) -> FramePlan:
    plan = FramePlan([set() for _ in stored_per_spec], set(), {}, [0] * len(stored_per_spec))
    for si, stored in enumerate(stored_per_spec):
        for path in paths:
            vm = metadata[path]
            if (vm["video_hash"], vm["frame_timecode_ms"]) in stored:
                source = sources[path]
                counts = plan.confirmed_by_spec.setdefault(si, {})
                counts[source] = counts.get(source, 0) + 1
                plan.skipped_per_spec[si] += 1
            else:
                plan.needs_per_spec[si].add(path)
                plan.any_needs.add(path)
    return plan


@dataclass(frozen=True)
class VideoCompletion:
    spec_index: int
    video_hash: str
    frame_count: int


def completion_plan(
    videos: list[Path],
    totals: dict[Path, int],
    hashes: dict[Path, str],
    already_done: dict[Path, set[int]],
    confirmed_by_spec: dict[int, dict[Path, int]],
    n_specs: int,
) -> list[VideoCompletion]:
    """Only acknowledge videos with every extracted slot durably represented."""
    return [
        VideoCompletion(si, hashes[video], total)
        for video in videos
        if (total := totals.get(video, 0)) > 0
        for si in range(n_specs)
        if si not in already_done.get(video, set())
        and confirmed_by_spec.get(si, {}).get(video, 0) == total
    ]


def _video_status(
    n_total: int, n_indexed: int, n_already: int, n_run_dup: int
) -> "tuple[str, str]":
    """Classify a video source file from the outcomes of its extracted frames.

    ``n_total`` frames were extracted; of those ``n_indexed`` produced new
    vectors, ``n_already`` were already in Qdrant, and ``n_run_dup`` were
    byte-identical to a frame of other media processed in the same run.
    Returns ``(status, reason)`` — both are court-facing, so each branch states
    the fact that holds, never a failure standing in for a duplicate.
    """
    if n_indexed > 0:
        return _S_INDEXED, f"{n_total} frames extracted"
    if n_already >= n_total:
        return _S_SKIP_IDX, f"all {n_total} extracted frames already indexed"
    if n_already + n_run_dup >= n_total:
        detail = (
            f"{n_run_dup} in-run duplicate(s), {n_already} already indexed"
            if n_already
            else "in-run duplicates"
        )
        return (
            _S_SKIP_FRAME_DUP,
            f"all {n_total} extracted frames were duplicates of frames from "
            f"other media in this run ({detail})",
        )
    return _S_FAIL_EMB, f"{n_total} frames extracted but no new vectors were indexed"


@dataclass
class VideoPlan:
    to_process: list[Path]
    already_done: dict[Path, set[int]]
    complete: set[Path]
    duplicates: set[Path]


def plan_videos(
    paths: list[Path],
    hashes: dict[Path, str],
    stored_info: list[dict[str, dict]],
    *,
    fps: float,
    max_frames: int,
    faces: bool,
) -> VideoPlan:
    """Each source digest owns one frame namespace, even if discovered at two paths."""
    plan = VideoPlan([], {}, set(), set())
    seen: set[str] = set()
    for path in paths:
        if (digest := hashes.get(path)) is None:
            continue
        if digest in seen:
            plan.duplicates.add(path)
            continue
        seen.add(digest)
        already = {
            si
            for si, info in enumerate(stored_info)
            if (stored := info.get(digest)) is not None
            and stored["extraction_fps"] == fps
            and stored["max_frames_cap"] == max_frames
            and stored["complete"]
        }
        if already:
            plan.already_done[path] = already
        if stored_info and len(already) == len(stored_info) and not faces:
            plan.complete.add(path)
        else:
            plan.to_process.append(path)
    return plan
