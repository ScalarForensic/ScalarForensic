"""Acknowledged vector writes, separated from embedding and reporting."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scalar_forensic.indexer import Indexer


@dataclass(frozen=True)
class CommittedBatch:
    spec_index: int
    paths: tuple[Path, ...]


@dataclass
class BatchWrite:
    spec_index: int
    indexer: "Indexer"
    paths: list[Path]
    hashes: list[str]
    md5_hashes: list[str]
    embeddings: list[list[float]]
    metadata: dict
    exif: dict[Path, dict] | None
    video_metadata: list[dict | None] | None

    def commit(self) -> CommittedBatch:
        """A failed write produces no success result and cannot count as complete."""
        self.indexer.upsert_batch(
            self.paths,
            self.hashes,
            self.embeddings,
            self.metadata,
            self.exif,
            self.md5_hashes,
            self.video_metadata,
        )
        return CommittedBatch(self.spec_index, tuple(self.paths))
