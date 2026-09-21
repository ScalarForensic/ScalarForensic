"""Records passed between ingestion stages."""

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scalar_forensic.embedder import ExifInfo

# file-level status codes
_S_INDEXED = "indexed"
_S_SKIP_DUP = "skipped_dup_batch"
_S_SKIP_IDX = "skipped_indexed"
# Video source file whose every extracted frame was a byte-identical duplicate
# of a frame from other media in the same run — nothing new to index, but not a
# failure either.  Distinct from _S_SKIP_DUP: the video file itself is not a
# duplicate, its frames are.
_S_SKIP_FRAME_DUP = "skipped_frames_dup_run"
_S_FAIL_READ = "failed_read"
_S_FAIL_PRE = "failed_preprocessing"
_S_FAIL_EMB = "failed_embedding"
_S_UNSUPPORTED = "unsupported"


@dataclass
class _FileRecord:
    path: Path
    status: str = "pending"
    reason: str = ""
    md5: str = ""
    sha256: str = ""
    is_frame: bool = False  # True for frame JPEGs written during slicing


@dataclass
class _BatchCtx:
    """Carries Phase-A results (read/hash/dedup) across one loop iteration.

    The preprocessing Future is submitted at the end of Phase A and resolved
    at the start of the next iteration's finish step — so CPU preprocessing
    of batch N+1 overlaps with GPU embedding of batch N.
    """

    batch_num: int
    batch_bytes: int
    read_s: float
    hash_s: float
    imgs_at_batch: int  # items_processed_so_far snapshot — used for the progress line
    path_hash_pairs: list[tuple[Path, str]]
    md5_by_sha256: dict[str, str]
    unique_pairs: list[tuple[Path, str]]
    to_embed_per_spec: list[list[tuple[Path, str]]]
    exif_data: "dict[Path, ExifInfo] | None"
    paths_to_pre: list[Path]
    # tuple[list[Image.Image | Exception], list[face results], float] — see
    # _timed_preprocess; face results are (path, sha256, vmeta,
    # FaceIndexResult | Exception) for this batch's in-loop face detection.
    pre_future: "Future[tuple[list, list, float]] | None"
