"""Discovery and content-level image deduplication, before writes."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .models import _S_SKIP_DUP, _S_SKIP_IDX, _S_UNSUPPORTED, _FileRecord


@dataclass
class DiscoveryResult:
    records: dict[Path, _FileRecord]
    images: list[Path]
    videos: list[Path]


def discover_files(entries: Iterable[tuple[Path, str]]) -> DiscoveryResult:
    result = DiscoveryResult({}, [], [])
    for path, kind in entries:
        rec = result.records[path] = _FileRecord(path=path)
        if kind == "image":
            result.images.append(path)
        elif kind == "video":
            result.videos.append(path)
            rec.status, rec.reason = _S_UNSUPPORTED, "video (pending)"
        else:
            rec.status = _S_UNSUPPORTED
            rec.reason = f"unsupported extension: {path.suffix.lower() or '(no extension)'}"
    return result


@dataclass
class ImagePlan:
    needs_per_spec: list[set[Path]]
    any_needs: set[Path]
    duplicates: set[Path]
    already_indexed: set[Path]

    def record(self, records: dict[Path, _FileRecord], skipped_counts: list[int]) -> None:
        for path in self.duplicates:
            records[path].status, records[path].reason = (
                _S_SKIP_DUP,
                "duplicate in run (same SHA-256)",
            )
        for path in self.already_indexed:
            records[path].status, records[path].reason = _S_SKIP_IDX, "already indexed in Qdrant"
        for si in range(len(self.needs_per_spec)):
            skipped_counts[si] += len(self.duplicates) + len(self.already_indexed)


def plan_images(
    paths: list[Path],
    hash_lookup: dict[Path, str],
    dedup_hashes: list[set[str]],
    dedup_paths: list[set[str]],
    n_specs: int,
) -> ImagePlan:
    """Plan content deduplication without changing caller records or counters."""
    unique_by_hash: dict[str, Path] = {}
    for path in paths:
        unique_by_hash.setdefault(hash_lookup[path], path)
    winners = set(unique_by_hash.values())
    needs = [
        {
            path
            for path in winners
            if hash_lookup[path] not in dedup_hashes[si]
            and str(path.resolve()) not in dedup_paths[si]
        }
        for si in range(n_specs)
    ]
    any_needs = set().union(*needs) if needs else set()
    return ImagePlan(needs, any_needs, set(paths) - winners, winners - any_needs)
