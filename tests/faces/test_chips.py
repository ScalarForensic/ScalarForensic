import numpy as np
from PIL import Image

from scalar_forensic.faces.chips import (
    aligned_chip_hash,
    chip_paths,
    dilated_clamped_bbox,
    review_chip_hash,
    review_chip_paths,
    write_aligned_chips,
    write_review_chips,
    write_thumbnail,
)


def test_chip_hash_is_dimension_prefixed():
    a = np.zeros((112, 112, 3), np.uint8)
    b = np.zeros((112 * 112 * 3,), np.uint8).reshape(56, 224, 3)
    assert aligned_chip_hash(a) != aligned_chip_hash(b)  # same bytes, different dims


def test_hash_domains_are_separated():
    arr = np.full((112, 112, 3), 7, dtype=np.uint8)
    assert aligned_chip_hash(arr) != review_chip_hash(arr)


def test_hashes_are_dimension_sensitive():
    a = np.zeros((112, 112, 3), dtype=np.uint8)
    b = np.zeros((56, 224, 3), dtype=np.uint8)
    assert a.tobytes() == b.tobytes()  # only the prefix can separate these
    assert review_chip_hash(a) != review_chip_hash(b)


def test_hashes_separate_channel_count_and_dtype():
    # Byte-identical buffers that differ only in trailing shape or dtype.
    # Not reachable today (everything is HxWx3 uint8) but these digests
    # become filenames, so the prefix must pin them.
    assert review_chip_hash(np.zeros((6, 6), np.uint8)) != review_chip_hash(
        np.zeros((6, 6, 1), np.uint8)
    )
    assert review_chip_hash(np.zeros((2, 2, 4), np.uint8)) != review_chip_hash(
        np.zeros((2, 2, 1), np.uint32)
    )


def test_hash_is_insensitive_to_array_contiguity():
    # A crop is a non-contiguous view; it must hash as its own bytes.
    src = np.random.default_rng(7).integers(0, 255, (40, 40, 3), np.uint8)
    view = src[10:20, 5:15]
    assert review_chip_hash(view) == review_chip_hash(np.ascontiguousarray(view))


def test_paths_are_sharded(tmp_path):
    png, jpg, thumb = chip_paths(tmp_path, "abcd" + "0" * 60)
    assert png.parent.name == "ab" and png.suffix == ".png" and jpg.name.endswith(".review.jpg")
    assert thumb.name.endswith(".thumb.jpg")


def test_review_paths_match_the_shared_suffixes(tmp_path):
    # Both kinds resolve review artefacts by the same rule, so the review
    # pair must be byte-identical to chip_paths' last two entries.
    chash = "abcd" + "0" * 60
    _, jpg, thumb = chip_paths(tmp_path, chash)
    assert review_chip_paths(tmp_path, chash) == (jpg, thumb)


def test_dilation_clamps_to_image_bounds():
    # bbox at the corner: dilation must clamp, not go negative.
    x, y, w, h = dilated_clamped_bbox((0.0, 0.0, 100.0, 100.0), 0.15, img_w=640, img_h=480)
    assert (x, y) == (0, 0) and w <= 640 and h <= 480 and w > 100


def test_write_aligned_chips_round_trip_lossless_png(tmp_path):
    rng = np.random.default_rng(1)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    ahash, rhash = write_aligned_chips(
        tmp_path, aligned, source, bbox=(100, 100, 80, 80), dilation=0.15, thumb_size=256
    )
    png, _, _ = chip_paths(tmp_path, ahash)
    jpg, thumb = review_chip_paths(tmp_path, rhash)
    assert png.exists() and jpg.exists() and thumb.exists()
    np.testing.assert_array_equal(np.array(Image.open(png)), aligned)  # PNG is lossless


def test_write_aligned_chips_stores_review_under_review_hash(tmp_path):
    aligned = np.full((112, 112, 3), 3, dtype=np.uint8)
    src = np.random.default_rng(2).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    ahash, rhash = write_aligned_chips(
        tmp_path,
        aligned,
        src,
        bbox=(100.0, 100.0, 80.0, 80.0),
        dilation=0.15,
        thumb_size=256,
    )
    assert ahash != rhash
    png, aligned_review, _ = chip_paths(tmp_path, ahash)
    review, thumb = review_chip_paths(tmp_path, rhash)
    assert png.exists()
    assert review.exists() and thumb.exists()
    assert not aligned_review.exists()


def test_embedded_and_review_only_share_one_review_jpeg(tmp_path):
    # The same source crop reached by both paths must resolve to one file --
    # this is precisely why purge has to check for remaining references.
    rng = np.random.default_rng(9)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    src = rng.integers(0, 255, (400, 400, 3), np.uint8)
    bbox = (100.0, 100.0, 80.0, 80.0)
    _, rhash = write_aligned_chips(tmp_path, aligned, src, bbox=bbox, dilation=0.15, thumb_size=256)
    review, _ = review_chip_paths(tmp_path, rhash)
    before = review.read_bytes(), review.stat().st_mtime_ns
    only = write_review_chips(tmp_path, src, bbox=bbox, dilation=0.15, thumb_size=256)
    assert only == rhash
    assert len(list(tmp_path.rglob("*.review.jpg"))) == 1
    # Untouched, not just single: a rewrite would briefly truncate a file the
    # *other* surviving observation depends on.
    assert (review.read_bytes(), review.stat().st_mtime_ns) == before


def test_write_review_chips_writes_two_files_and_no_png(tmp_path):
    src = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    chash = write_review_chips(
        tmp_path, src, bbox=(100.0, 100.0, 80.0, 80.0), dilation=0.15, thumb_size=256
    )
    assert chash is not None
    review, thumb = review_chip_paths(tmp_path, chash)
    assert review.exists() and thumb.exists()
    png, _, _ = chip_paths(tmp_path, chash)
    assert not png.exists()


def test_write_review_chips_returns_none_for_offimage_bbox(tmp_path):
    src = np.zeros((100, 100, 3), dtype=np.uint8)
    assert (
        write_review_chips(
            tmp_path, src, bbox=(500.0, 500.0, 40.0, 40.0), dilation=0.15, thumb_size=256
        )
        is None
    )
    assert not any(tmp_path.rglob("*.jpg"))


def test_review_thumbnail_never_upscales(tmp_path):
    # Load-bearing: the review path's honesty rests on native resolution.
    src = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    chash = write_review_chips(
        tmp_path, src, bbox=(80.0, 80.0, 40.0, 40.0), dilation=0.15, thumb_size=256
    )
    review, thumb = review_chip_paths(tmp_path, chash)
    with Image.open(review) as r, Image.open(thumb) as t:
        assert t.size == r.size
        assert max(t.size) < 256


def test_thumbnail_caps_long_side_and_keeps_aspect(tmp_path):
    rng = np.random.default_rng(2)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (1200, 1600, 3), np.uint8)
    # Review crop is 800x400 before dilation -> long side well above thumb_size.
    _, rhash = write_aligned_chips(
        tmp_path, aligned, source, bbox=(100, 100, 800, 400), dilation=0.15, thumb_size=256
    )
    jpg, thumb = review_chip_paths(tmp_path, rhash)
    tw, th = Image.open(thumb).size
    rw, rh = Image.open(jpg).size
    assert max(tw, th) == 256
    assert abs((tw / th) - (rw / rh)) < 0.02  # aspect preserved


def test_small_review_chip_is_not_upscaled(tmp_path):
    rng = np.random.default_rng(3)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    _, rhash = write_aligned_chips(
        tmp_path, aligned, source, bbox=(10, 10, 50, 50), dilation=0.15, thumb_size=256
    )
    jpg, thumb = review_chip_paths(tmp_path, rhash)
    assert Image.open(thumb).size == Image.open(jpg).size


def test_write_thumbnail_regenerates_deleted_thumbnail(tmp_path):
    rng = np.random.default_rng(4)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    _, rhash = write_aligned_chips(
        tmp_path, aligned, source, bbox=(100, 100, 300, 300), dilation=0.15, thumb_size=128
    )
    jpg, thumb = review_chip_paths(tmp_path, rhash)
    thumb.unlink()
    write_thumbnail(jpg, thumb, thumb_size=128)
    assert thumb.exists() and max(Image.open(thumb).size) == 128


def test_aligned_chip_hash_ignores_thumbnail_presence(tmp_path):
    # The thumbnail is derived and non-evidentiary: it must not enter the hash.
    rng = np.random.default_rng(5)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    before = aligned_chip_hash(aligned)
    ahash, rhash = write_aligned_chips(
        tmp_path, aligned, source, bbox=(50, 50, 200, 200), dilation=0.15, thumb_size=256
    )
    _, thumb = review_chip_paths(tmp_path, rhash)
    thumb.unlink()
    assert ahash == before == aligned_chip_hash(aligned)


def test_aligned_chips_with_offimage_bbox_still_writes_png(tmp_path):
    # No review crop exists, so the review hash must be None rather than a
    # hash pointing at a file that was never written.
    aligned = np.full((112, 112, 3), 5, dtype=np.uint8)
    src = np.zeros((100, 100, 3), dtype=np.uint8)
    ahash, rhash = write_aligned_chips(
        tmp_path, aligned, src, bbox=(500.0, 500.0, 40.0, 40.0), dilation=0.15, thumb_size=256
    )
    png, _, _ = chip_paths(tmp_path, ahash)
    assert png.exists()
    assert rhash is None
