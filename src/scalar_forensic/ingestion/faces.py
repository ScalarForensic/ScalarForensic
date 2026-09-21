"""Face reconciliation with explicit commit results and reference-safe cleanup."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scalar_forensic.faces.indexing import FaceIndexResult, FacePipeline
    from scalar_forensic.faces.store import FaceStore


@dataclass(frozen=True)
class FaceCommitResult:
    observations: "FaceIndexResult"
    stale_points: list[dict]


@dataclass
class FaceRunState:
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "detected": 0,
            "kept": 0,
            "review_only": 0,
            "dropped_noncanon": 0,
            "failed": 0,
        }
    )
    rejected: dict[str, int] = field(default_factory=dict)
    review_reasons: dict[str, int] = field(default_factory=dict)
    stale_points: list[dict] = field(default_factory=list)

    def record(self, committed: FaceCommitResult) -> None:
        result = committed.observations
        self.stale_points.extend(committed.stale_points)
        for name, value in (
            ("detected", result.n_detected),
            ("kept", result.n_kept),
            ("review_only", result.n_review_only),
            ("dropped_noncanon", result.n_dropped_noncanonical),
        ):
            self.counts[name] += value
        for bucket, incoming in (
            (self.rejected, result.rejected),
            (self.review_reasons, result.review_only_reasons),
        ):
            for reason, count in incoming.items():
                bucket[reason] = bucket.get(reason, 0) + count


def commit_faces(
    pipeline: "FacePipeline",
    path: Path,
    image_hash: str,
    video_metadata: dict | None,
    result: "FaceIndexResult",
) -> FaceCommitResult:
    """Select stale observations before writing; acknowledge marker only after vector removal."""
    metadata = video_metadata or {}
    marker = pipeline.store.marker_point(
        image_hash,
        metadata.get("video_hash"),
        pipeline.cfg.config_hash,
        result.n_detected,
        result.n_kept,
        result.rejected,
        n_review_only=result.n_review_only,
        review_only_reasons=result.review_only_reasons,
        n_dropped_noncanonical=result.n_dropped_noncanonical,
        frame_timecode_ms=metadata.get("frame_timecode_ms"),
    )
    stale = pipeline.store.stale_face_points(
        image_hash,
        {str(point.id) for point in result.points},
        video_hash=metadata.get("video_hash"),
        frame_timecode_ms=metadata.get("frame_timecode_ms"),
    )
    source = metadata.get("video_path") or str(path)
    stale = [{**point, "source_path": source} for point in stale]
    # A review-only upsert cannot by itself be trusted to clear a previous
    # vector. Failure to clear must prevent the idempotency marker being stored.
    pipeline.store.upsert_faces(result.points)
    pipeline.store.clear_face_vector(result.review_only_point_ids)
    pipeline.store.upsert_faces([marker])
    return FaceCommitResult(result, stale)


def unlink_unreferenced_chips(store: "FaceStore", directory: Path | None, hashes: list[str]) -> int:
    """Reference-safe unlink; requires one writer and per-case directories."""
    if directory is None:
        return 0
    from scalar_forensic.faces.chips import chip_paths

    removed = 0
    for digest in store.unreferenced_chip_hashes(hashes):
        for path in chip_paths(directory, digest):
            if path.exists():
                path.unlink()
                removed += 1
    return removed


@dataclass(frozen=True)
class FaceRemoval:
    n_points: int
    n_chip_files: int


def remove_stale_faces(
    store: "FaceStore", points: list[dict], directory: Path | None
) -> FaceRemoval:
    """Apply an explicitly approved stale set, then collect only unreferenced artifacts."""
    result = store.delete_face_points([str(point["id"]) for point in points])
    return FaceRemoval(
        result.n_points, unlink_unreferenced_chips(store, directory, result.chip_hashes)
    )
