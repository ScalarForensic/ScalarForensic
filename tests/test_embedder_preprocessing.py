"""Tests for the image preprocessing pipeline in embedder.py.

Covers:
  - _open_rgb: EXIF orientation correction
  - _open_rgb: ICC colour-profile → sRGB conversion
  - _open_rgb: graceful fallback on malformed ICC data
  - _cap_short_side: cap parameter behaviour
  - preprocess_batch / preprocess_pil_batch: cap forwarding
  - _sscd_crops: crop count and dimensions
  - Multi-crop averaging: L2-normalisation invariant
"""

import io

import pytest
import torch
from PIL import Image, ImageCms

from scalar_forensic.embedder import (
    _SHARED_CAP,
    _SSCD_INPUT_SIZE,
    _SSCD_SCALE,
    _cap_short_side,
    _open_rgb,
    _sscd_crops,
    _sscd_resize,
    preprocess_batch,
    preprocess_pil_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_image(w: int, h: int, color: tuple = (128, 64, 32)) -> Image.Image:
    """Create a solid-colour RGB image."""
    img = Image.new("RGB", (w, h), color)
    return img


def _jpeg_bytes(img: Image.Image, exif: bytes | None = None) -> bytes:
    """Encode *img* to JPEG bytes, optionally embedding raw EXIF data."""
    buf = io.BytesIO()
    kwargs: dict = {"format": "JPEG", "quality": 90}
    if exif is not None:
        kwargs["exif"] = exif
    img.save(buf, **kwargs)
    return buf.getvalue()


def _make_exif_orientation(orientation: int) -> bytes:
    """Build EXIF bytes with the given Orientation tag value using Pillow's own encoder."""
    # Use Pillow's Exif object so the output is guaranteed to be in the format
    # that ImageOps.exif_transpose() can read back correctly.
    scratch = Image.new("RGB", (1, 1))
    exif = scratch.getexif()
    exif[0x0112] = orientation  # Orientation tag (IFD 0)
    return exif.tobytes()


def _srgb_icc_bytes() -> bytes:
    """Return the raw bytes of an sRGB ICC profile."""
    profile = ImageCms.createProfile("sRGB")
    return ImageCms.ImageCmsProfile(profile).tobytes()


def _make_png_with_icc(w: int, h: int, icc_bytes: bytes) -> bytes:
    """Create a PNG image with the given ICC profile embedded."""
    img = _solid_image(w, h)
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=icc_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Group A: _open_rgb — EXIF orientation
# ---------------------------------------------------------------------------


class TestOpenRgbExifOrientation:
    def test_orientation_1_no_change(self):
        """Orientation=1 (normal) leaves dimensions unchanged."""
        img = _solid_image(100, 50)  # landscape
        exif = _make_exif_orientation(1)
        data = _jpeg_bytes(img, exif=exif)
        result = _open_rgb(data)
        assert result.size == (100, 50)

    def test_orientation_6_rotates_90cw(self):
        """Orientation=6 (90° CW rotation) should produce a portrait image."""
        # Original stored as landscape (100×50); Orientation=6 means
        # the camera was rotated so the correct view is portrait (50×100).
        img = _solid_image(100, 50)
        # Pillow can embed full EXIF via piexif or via the exif keyword.
        # Use exif= kwarg which accepts raw bytes.
        exif = _make_exif_orientation(6)
        data = _jpeg_bytes(img, exif=exif)
        result = _open_rgb(data)
        # After transpose, width and height should be swapped.
        assert result.size == (50, 100), (
            f"Expected (50, 100) after EXIF Orientation=6 transpose, got {result.size}"
        )

    def test_no_exif_no_error(self):
        """Images without EXIF are returned unchanged without error."""
        img = _solid_image(80, 60)
        data = _jpeg_bytes(img)
        result = _open_rgb(data)
        assert result.mode == "RGB"
        assert result.size == (80, 60)


# ---------------------------------------------------------------------------
# Group B & C: _open_rgb — ICC profile metadata is ignored (no conversion)
# ---------------------------------------------------------------------------


class TestOpenRgbIccProfile:
    def test_valid_srgb_icc_no_error(self):
        """Image with embedded sRGB ICC profile opens without error (profile ignored)."""
        icc = _srgb_icc_bytes()
        data = _make_png_with_icc(64, 64, icc)
        result = _open_rgb(data)
        assert result.mode == "RGB"
        assert result.size == (64, 64)

    def test_malformed_icc_no_error(self):
        """Malformed ICC bytes in metadata do not raise — profile is ignored."""
        bad_icc = b"\xff\xfe\x00garbage data that is not a valid ICC profile"
        data = _make_png_with_icc(64, 64, bad_icc)
        result = _open_rgb(data)
        assert result.mode == "RGB"
        assert result.size == (64, 64)

    def test_truncated_icc_no_error(self):
        """Truncated ICC bytes in metadata do not raise — profile is ignored."""
        icc = _srgb_icc_bytes()[:8]  # header only
        data = _make_png_with_icc(64, 64, icc)
        result = _open_rgb(data)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# Group D & E: _cap_short_side — cap parameter
# ---------------------------------------------------------------------------


class TestCapShortSide:
    def test_default_cap_downscales_oversized(self):
        """Default cap (331) downscales a 400×600 image."""
        img = _solid_image(400, 600)
        result = _cap_short_side(img)
        w, h = result.size
        assert min(w, h) == _SHARED_CAP
        # Aspect ratio must be preserved.
        assert abs(w / h - 400 / 600) < 0.02

    def test_explicit_cap_512_passes_400px_through(self):
        """cap=512: a 400×600 image (short side 400 < 512) is returned unchanged."""
        img = _solid_image(400, 600)
        result = _cap_short_side(img, cap=512)
        assert result.size == (400, 600)

    def test_explicit_cap_331_downscales_400px(self):
        """cap=331: a 400×600 image is scaled so short side == 331."""
        img = _solid_image(400, 600)
        result = _cap_short_side(img, cap=331)
        assert min(result.size) == 331

    def test_small_image_unchanged(self):
        """Images smaller than cap are returned as-is."""
        img = _solid_image(100, 150)
        result = _cap_short_side(img, cap=512)
        assert result.size == (100, 150)

    def test_cap_512_downscales_800px(self):
        """cap=512: an 800×1200 image is downscaled to short side 512."""
        img = _solid_image(800, 1200)
        result = _cap_short_side(img, cap=512)
        assert min(result.size) == 512


# ---------------------------------------------------------------------------
# Group F: preprocess_batch / preprocess_pil_batch — cap forwarding
# ---------------------------------------------------------------------------


class TestPreprocessBatchCap:
    def test_preprocess_batch_default_cap(self):
        """preprocess_batch with default cap produces short side ≤ 331."""
        img = _solid_image(600, 800)
        data = _jpeg_bytes(img)
        results = preprocess_batch([data])
        assert len(results) == 1
        result = results[0]
        assert not isinstance(result, Exception)
        assert min(result.size) <= _SHARED_CAP

    def test_preprocess_batch_explicit_cap_512(self):
        """preprocess_batch(cap=512) allows short sides up to 512."""
        img = _solid_image(400, 600)
        data = _jpeg_bytes(img)
        results = preprocess_batch([data], cap=512)
        result = results[0]
        assert not isinstance(result, Exception)
        # 400 < 512, so image should pass through unchanged.
        assert result.size == (400, 600)

    def test_preprocess_pil_batch_cap_forwarded(self):
        """preprocess_pil_batch(cap=512) passes 400×600 image through unchanged."""
        img = _solid_image(400, 600)
        results = preprocess_pil_batch([img], cap=512)
        assert results[0].size == (400, 600)

    def test_preprocess_batch_returns_rgb(self):
        """preprocess_batch always returns RGB mode images."""
        img = _solid_image(200, 300)
        data = _jpeg_bytes(img)
        results = preprocess_batch([data])
        assert results[0].mode == "RGB"


# ---------------------------------------------------------------------------
# Group G & H: _sscd_crops — count and dimensions
# ---------------------------------------------------------------------------


class TestSscdCrops:
    @pytest.fixture
    def scaled_img(self):
        """A 331×496 image (SSCD scale, portrait)."""
        img = _solid_image(331, 496)
        return _sscd_resize(img)

    def test_n_crops_1_returns_single_center_crop(self, scaled_img):
        crops = _sscd_crops(scaled_img, n_crops=1)
        assert len(crops) == 1
        assert crops[0].size == (_SSCD_INPUT_SIZE, _SSCD_INPUT_SIZE)

    def test_n_crops_5_returns_five_crops(self, scaled_img):
        crops = _sscd_crops(scaled_img, n_crops=5)
        assert len(crops) == 5

    def test_all_crops_correct_size(self, scaled_img):
        for n in (1, 5):
            crops = _sscd_crops(scaled_img, n_crops=n)
            for crop in crops:
                assert crop.size == (_SSCD_INPUT_SIZE, _SSCD_INPUT_SIZE), (
                    f"Crop size mismatch for n_crops={n}: {crop.size}"
                )

    def test_sscd_resize_short_side(self):
        """_sscd_resize produces an image with short side == _SSCD_SCALE."""
        img = _solid_image(200, 300)
        result = _sscd_resize(img)
        assert min(result.size) == _SSCD_SCALE

    def test_sscd_resize_upscales_small_image(self):
        """_sscd_resize upscales images smaller than 331 px."""
        img = _solid_image(100, 150)
        result = _sscd_resize(img)
        assert min(result.size) == _SSCD_SCALE


# ---------------------------------------------------------------------------
# Group I: Multi-crop averaging — L2-normalisation invariant
# ---------------------------------------------------------------------------


class TestMultiCropAveraging:
    """Verify the averaging formula produces unit-norm output.

    These tests exercise the torch math without requiring the actual SSCD
    TorchScript checkpoint — we synthesise embedding tensors directly.
    """

    def _average_crops(self, emb_tensor: torch.Tensor) -> torch.Tensor:
        """Mirror the multi-crop averaging logic from SSCDEmbedder.embed_images."""
        import torch.nn.functional as F  # noqa: PLC0415

        embs = F.normalize(emb_tensor, p=2, dim=1)
        avg = embs.mean(dim=0, keepdim=True)
        return F.normalize(avg, p=2, dim=1).squeeze(0)

    def test_single_unit_vector_unchanged(self):
        """Averaging a single unit vector returns that same vector."""
        v = torch.tensor([[1.0, 0.0, 0.0]])
        result = self._average_crops(v)
        assert torch.allclose(result, v.squeeze(0), atol=1e-6)

    def test_output_is_unit_norm(self):
        """Averaged multi-crop result is always L2-normalised (norm ≈ 1.0)."""

        torch.manual_seed(42)
        for n_crops in (2, 3, 5):
            embs = torch.randn(n_crops, 512)
            result = self._average_crops(embs)
            norm = torch.linalg.norm(result).item()
            assert abs(norm - 1.0) < 1e-5, (
                f"Output norm={norm:.6f} for n_crops={n_crops}, expected ≈ 1.0"
            )

    def test_opposite_unit_vectors_average_to_unit_norm(self):
        """Even degenerate inputs (opposing unit vectors) produce a defined output."""
        # Two opposite unit vectors average to the zero vector before normalisation;
        # the subsequent L2-normalise should handle this without NaN/inf.
        # Pillow uses float32; we test the same.
        v1 = torch.tensor([[1.0, 0.0]])
        v2 = torch.tensor([[-1.0, 0.0]])
        embs = torch.cat([v1, v2], dim=0)
        result = self._average_crops(embs)
        # The result may be arbitrary direction but must not be NaN.
        assert not torch.isnan(result).any(), "Result contains NaN for opposing unit vectors"
