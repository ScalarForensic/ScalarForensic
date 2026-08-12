import numpy as np
from PIL import Image

from scalar_forensic.faces.chips import (
    chip_hash,
    chip_paths,
    dilated_clamped_bbox,
    write_chips,
    write_thumbnail,
)


def test_chip_hash_is_dimension_prefixed():
    a = np.zeros((112, 112, 3), np.uint8)
    b = np.zeros((112 * 112 * 3,), np.uint8).reshape(56, 224, 3)
    assert chip_hash(a) != chip_hash(b)  # same bytes, different dims


def test_paths_are_sharded(tmp_path):
    png, jpg, thumb = chip_paths(tmp_path, "abcd" + "0" * 60)
    assert png.parent.name == "ab" and png.suffix == ".png" and jpg.name.endswith(".review.jpg")
    assert thumb.name.endswith(".thumb.jpg")


def test_dilation_clamps_to_image_bounds():
    # bbox at the corner: dilation must clamp, not go negative.
    x, y, w, h = dilated_clamped_bbox((0.0, 0.0, 100.0, 100.0), 0.15, img_w=640, img_h=480)
    assert (x, y) == (0, 0) and w <= 640 and h <= 480 and w > 100


def test_write_chips_round_trip_lossless_png(tmp_path):
    rng = np.random.default_rng(1)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    chash = write_chips(
        tmp_path, aligned, source, bbox=(100, 100, 80, 80), dilation=0.15, thumb_size=256
    )
    png, jpg, thumb = chip_paths(tmp_path, chash)
    assert png.exists() and jpg.exists() and thumb.exists()
    np.testing.assert_array_equal(np.array(Image.open(png)), aligned)  # PNG is lossless


def test_thumbnail_caps_long_side_and_keeps_aspect(tmp_path):
    rng = np.random.default_rng(2)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (1200, 1600, 3), np.uint8)
    # Review crop is 800x400 before dilation -> long side well above thumb_size.
    chash = write_chips(
        tmp_path, aligned, source, bbox=(100, 100, 800, 400), dilation=0.15, thumb_size=256
    )
    _, jpg, thumb = chip_paths(tmp_path, chash)
    tw, th = Image.open(thumb).size
    rw, rh = Image.open(jpg).size
    assert max(tw, th) == 256
    assert abs((tw / th) - (rw / rh)) < 0.02  # aspect preserved


def test_small_review_chip_is_not_upscaled(tmp_path):
    rng = np.random.default_rng(3)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    chash = write_chips(
        tmp_path, aligned, source, bbox=(10, 10, 50, 50), dilation=0.15, thumb_size=256
    )
    _, jpg, thumb = chip_paths(tmp_path, chash)
    assert Image.open(thumb).size == Image.open(jpg).size


def test_write_thumbnail_regenerates_deleted_thumbnail(tmp_path):
    rng = np.random.default_rng(4)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    chash = write_chips(
        tmp_path, aligned, source, bbox=(100, 100, 300, 300), dilation=0.15, thumb_size=128
    )
    _, jpg, thumb = chip_paths(tmp_path, chash)
    thumb.unlink()
    write_thumbnail(jpg, thumb, thumb_size=128)
    assert thumb.exists() and max(Image.open(thumb).size) == 128


def test_chip_hash_ignores_thumbnail_presence(tmp_path):
    # The thumbnail is derived and non-evidentiary: it must not enter the hash.
    rng = np.random.default_rng(5)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    before = chip_hash(aligned)
    chash = write_chips(
        tmp_path, aligned, source, bbox=(50, 50, 200, 200), dilation=0.15, thumb_size=256
    )
    _, _, thumb = chip_paths(tmp_path, chash)
    thumb.unlink()
    assert chash == before == chip_hash(aligned)
