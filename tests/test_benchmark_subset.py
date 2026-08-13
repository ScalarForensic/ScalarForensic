"""Sampling logic tests for scripts/make_benchmark_subset.py.

Hermetic: every test builds its own corpus in tmp_path — no real corpus,
no Qdrant. The script is not an installed package, so it is loaded from
its file path via importlib.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "make_benchmark_subset.py"
_spec = importlib.util.spec_from_file_location("make_benchmark_subset", _SCRIPT)
mbs = importlib.util.module_from_spec(_spec)
sys.modules["make_benchmark_subset"] = mbs
_spec.loader.exec_module(mbs)


def _build_corpus(root: Path) -> dict[str, int]:
    """Create a fake corpus; return expected per-stratum totals."""
    root.mkdir(parents=True, exist_ok=True)
    counts = {"heic": 40, "jpg": 30, "other_image": 20, "video": 10}
    for i in range(counts["heic"]):
        (root / f"IMG_{i:04d}.HEIC").write_bytes(b"heic" + bytes([i % 256]))
    for i in range(counts["jpg"]):
        (root / f"IMG_{i:04d}.JPG").write_bytes(b"jpg" + bytes([i % 256]))
    for i in range(counts["other_image"]):
        (root / f"PIC_{i:04d}.png").write_bytes(b"png" + bytes([i % 256]))
    for i in range(counts["video"]):
        (root / f"VID_{i:04d}.MOV").write_bytes(b"mov" + bytes([i % 256]))
    # Non-media files must be skipped, never copied.
    (root / "IMG_0001.AAE").write_text("sidecar")
    (root / "report.csv").write_text("a,b\n")
    return counts


def test_classify_maps_extensions_case_insensitively():
    assert mbs.classify(Path("a.HEIC")) == "heic"
    assert mbs.classify(Path("a.jpeg")) == "jpg"
    assert mbs.classify(Path("a.PNG")) == "other_image"
    assert mbs.classify(Path("a.WebP")) == "other_image"
    assert mbs.classify(Path("a.mov")) == "video"
    assert mbs.classify(Path("a.MP4")) == "video"
    assert mbs.classify(Path("a.aae")) is None
    assert mbs.classify(Path("a.csv")) is None


def test_stratified_counts_and_proportions(tmp_path):
    corpus = tmp_path / "corpus"
    counts = _build_corpus(corpus)
    manifest, selected = mbs.build_selection(corpus, fraction=0.10, seed=1)

    for name, total in counts.items():
        assert manifest["strata"][name]["total"] == total
        expected = max(1, round(total * 0.10))
        assert manifest["strata"][name]["selected"] == expected
        assert len(selected[name]) == expected
    assert manifest["skipped_non_media"] == 2


def test_small_stratum_selects_at_least_one(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "only.mov").write_bytes(b"x")
    manifest, selected = mbs.build_selection(corpus, fraction=0.10, seed=1)
    assert selected["video"] == ["only.mov"]
    assert manifest["strata"]["video"]["selected"] == 1


def test_selection_is_deterministic_for_same_seed(tmp_path):
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    _, first = mbs.build_selection(corpus, fraction=0.10, seed=1)
    _, second = mbs.build_selection(corpus, fraction=0.10, seed=1)
    assert first == second


def test_selection_differs_for_different_seed(tmp_path):
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    _, a = mbs.build_selection(corpus, fraction=0.10, seed=1)
    _, b = mbs.build_selection(corpus, fraction=0.10, seed=2)
    assert a != b


def test_stratum_selection_independent_of_other_strata(tmp_path):
    """Adding files to one stratum must not change another's selection."""
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    _, before = mbs.build_selection(corpus, fraction=0.10, seed=1)
    for i in range(100, 120):
        (corpus / f"VID_{i:04d}.MOV").write_bytes(b"mov")
    _, after = mbs.build_selection(corpus, fraction=0.10, seed=1)
    assert before["heic"] == after["heic"]
    assert before["jpg"] == after["jpg"]
    assert before["other_image"] == after["other_image"]


def test_make_subset_copies_files_and_writes_manifest(tmp_path):
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    dest = tmp_path / "corpus_bench10"
    manifest = mbs.make_subset(corpus, dest, fraction=0.10, seed=1)

    manifest_path = dest / mbs.MANIFEST_NAME
    assert manifest_path.is_file()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["seed"] == 1
    assert on_disk["strata"] == manifest["strata"]

    for name in mbs.STRATA:
        for rel in manifest["strata"][name]["files"]:
            copy = dest / rel
            assert copy.is_file(), rel
            # Real copies with source content — not symlinks.
            assert not copy.is_symlink()
            assert copy.read_bytes() == (corpus / rel).read_bytes()

    copied = [p for p in dest.rglob("*") if p.is_file() and p != manifest_path]
    total_selected = sum(manifest["strata"][n]["selected"] for n in mbs.STRATA)
    assert len(copied) == total_selected
    # No sidecar / csv leaked into the subset.
    assert not [p for p in copied if p.suffix.lower() in (".aae", ".csv")]


def test_make_subset_preserves_subdirectory_layout(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "sub").mkdir(parents=True)
    (corpus / "sub" / "a.jpg").write_bytes(b"a")
    dest = tmp_path / "bench"
    mbs.make_subset(corpus, dest, fraction=1.0, seed=1)
    assert (dest / "sub" / "a.jpg").is_file()


def test_make_subset_refuses_dest_inside_source(tmp_path):
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    with pytest.raises(SystemExit, match="inside source"):
        mbs.make_subset(corpus, corpus / "bench10", fraction=0.10, seed=1)


def test_make_subset_refuses_non_empty_dest(tmp_path):
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    dest = tmp_path / "bench"
    dest.mkdir()
    (dest / "stale.txt").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        mbs.make_subset(corpus, dest, fraction=0.10, seed=1)


def test_make_subset_source_never_modified(tmp_path):
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    before = sorted(str(p.relative_to(corpus)) for p in corpus.rglob("*"))
    mbs.make_subset(corpus, tmp_path / "bench", fraction=0.10, seed=1)
    after = sorted(str(p.relative_to(corpus)) for p in corpus.rglob("*"))
    assert before == after
