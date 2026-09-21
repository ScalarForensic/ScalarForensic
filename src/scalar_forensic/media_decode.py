"""Encoded-image decoding with no model, video or application imports."""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

import numpy as np
from PIL import Image, ImageOps

_MAX_IMAGE_PIXELS_ENV = "SFN_MAX_IMAGE_PIXELS"


def _max_image_pixels_from_env() -> int | None:
    """Read Pillow's decompression-bomb guard from an explicit env override.

    By default, leave Pillow's built-in MAX_IMAGE_PIXELS limit unchanged so web
    and other untrusted-image paths retain decompression-bomb protection.
    Set SFN_MAX_IMAGE_PIXELS to:
      - a positive integer to allow a larger finite pixel count; or
      - "none" / "disable" / "disabled" to turn the guard off explicitly for
        trusted ingestion runs.
    """
    raw_value = os.getenv(_MAX_IMAGE_PIXELS_ENV)
    if raw_value is None:
        return Image.MAX_IMAGE_PIXELS
    value = raw_value.strip()
    if not value:
        return Image.MAX_IMAGE_PIXELS
    if value.lower() in {"none", "disable", "disabled"}:
        return None
    try:
        max_pixels = int(value)
    except ValueError:
        raise ValueError(
            f"{_MAX_IMAGE_PIXELS_ENV} must be a positive integer or one of"
            f" 'none', 'disable', 'disabled'; got {raw_value!r}"
        ) from None
    if max_pixels <= 0:
        raise ValueError(
            f"{_MAX_IMAGE_PIXELS_ENV} must be a positive integer or one of"
            " 'none', 'disable', 'disabled'"
        )
    return max_pixels


def _configure_max_image_pixels_from_env() -> None:
    Image.MAX_IMAGE_PIXELS = _max_image_pixels_from_env()


_configure_max_image_pixels_from_env()

_EXIF_GPS_IFD_TAG = 34853  # PIL ExifTags.Base.GPSInfo


class ExifInfo(TypedDict):
    exif: bool
    exif_geo_data: bool


def extract_exif(data: bytes) -> ExifInfo:
    """Return EXIF presence flags from raw image bytes.

    Opens the image without converting to RGB — colour conversion strips EXIF on
    some formats. Falls back to exif=False on any error.
    """
    try:
        img = Image.open(io.BytesIO(data))
        exif_obj = img.getexif()
        return ExifInfo(exif=len(exif_obj) > 0, exif_geo_data=_EXIF_GPS_IFD_TAG in exif_obj)
    except Exception:  # noqa: BLE001
        return ExifInfo(exif=False, exif_geo_data=False)


def extract_exif_detailed(data: bytes) -> dict:
    """Return detailed file + EXIF metadata dict for display in the UI."""
    result: dict = {"exif": False, "exif_geo_data": False}
    try:
        img = Image.open(io.BytesIO(data))
        result["width"], result["height"] = img.size
        result["format"] = img.format or ""
        result["size_bytes"] = len(data)

        exif_obj = img.getexif()
        result["exif"] = len(exif_obj) > 0
        result["exif_geo_data"] = _EXIF_GPS_IFD_TAG in exif_obj

        if result["exif"]:
            # Common tags by ID (stable across Pillow versions)
            _TAG_NAMES = {271: "make", 272: "model", 306: "datetime", 305: "software"}
            for tag_id, key in _TAG_NAMES.items():
                val = exif_obj.get(tag_id)
                if val:
                    result[key] = str(val).strip("\x00").strip()

        if result["exif_geo_data"]:
            gps_ifd = exif_obj.get_ifd(_EXIF_GPS_IFD_TAG)
            try:
                lat_dms = gps_ifd.get(2)  # GPSLatitude
                lat_ref = gps_ifd.get(1)  # GPSLatitudeRef 'N'/'S'
                lon_dms = gps_ifd.get(4)  # GPSLongitude
                lon_ref = gps_ifd.get(3)  # GPSLongitudeRef 'E'/'W'
                if lat_dms and lon_dms:

                    def _dms(dms: tuple, ref: str) -> float:
                        d, m, s = [float(x) for x in dms]
                        v = d + m / 60.0 + s / 3600.0
                        return -v if ref in ("S", "W") else v

                    result["gps_lat"] = round(_dms(lat_dms, lat_ref or "N"), 6)
                    result["gps_lon"] = round(_dms(lon_dms, lon_ref or "E"), 6)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return result


_EXIF_ORIENTATION_TAG = 0x0112  # EXIF tag 274
_EXIF_ORIENTATION_FORMATS = frozenset({"JPEG", "TIFF", "WEBP", "MPO", "HEIF"})


def display_rgb(data: bytes) -> Image.Image:
    """Native display/chip decode without embedding draft or EXIF transforms."""
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def thumbnail_input(data: bytes, cap: int) -> Image.Image:
    """Preserve regeneration's resize-before-convert and grayscale JPEG input."""
    with Image.open(io.BytesIO(data)) as image:
        thumb = image.copy()
    thumb.thumbnail((cap, cap), Image.Resampling.LANCZOS)
    if thumb.mode not in ("RGB", "L"):
        thumb = thumb.convert("RGB")
    return thumb


def _open_rgb(data: bytes, cap: int = 331) -> Image.Image:
    """Decode image bytes to a normalised RGB PIL Image.

    ICC colour profiles are ignored — images are treated as sRGB regardless of
    any embedded profile.  Wide-gamut sources (AdobeRGB, ProPhoto) would
    produce slightly shifted pixel values, but such images are rare in forensic
    datasets and the semantic embedding models are robust to small colour shifts.

    EXIF orientation is applied so that phone photos stored with a non-trivial
    Orientation tag are embedded in the orientation the user sees, not the raw
    sensor orientation.  The check is skipped entirely for formats that never
    carry EXIF (PNG, BMP, GIF …) and short-circuits for the common Orientation=1
    case to avoid a full decode+rotate on every image.

    For JPEG inputs, ``draft()`` is called before any pixel decode to enable
    libjpeg's built-in shrink-on-load: the decoder picks the largest 1/N scale
    (N ∈ {2, 4, 8}) that still produces both dimensions ≥ *cap*, then
    ``_cap_short_side`` does the final resize in software.  A 4000×3000 JPEG
    shrunk to 1/8 (500×375) before Lanczos resizing to 331 px is decoded in a
    fraction of the time, with negligible quality difference for semantic
    embedding.  Images already close to *cap* decode at full size (no-op).
    """
    img = Image.open(io.BytesIO(data))
    # draft() must be called before any operation that triggers full pixel
    # decode (including exif_transpose and convert).  getexif() only reads
    # the file header so it is safe to call before draft().
    if img.format == "JPEG":
        img.draft("RGB", (cap, cap))
    if img.format in _EXIF_ORIENTATION_FORMATS:
        if img.getexif().get(_EXIF_ORIENTATION_TAG, 1) != 1:
            img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _cap_short_side(img: Image.Image, cap: int) -> Image.Image:
    """Scale down proportionally if the short side exceeds *cap*; leave smaller images untouched."""
    w, h = img.size
    short = min(w, h)
    if short <= cap:
        return img
    scale = cap / short
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def preprocess_batch(image_data: list[bytes], cap: int) -> list[Image.Image | Exception]:
    """Open RGB (with EXIF orientation correction), cap short side to *cap* px.

    Parallelised over CPU cores.  Returns one entry per input image: an
    ``Image.Image`` on success or the raised ``Exception`` on failure.
    Individual futures are used instead of ``pool.map()`` so that a single
    corrupt or oversized file does not abort the rest of the batch — callers
    should check each result with ``isinstance(result, Exception)``.

    Pass ``max(_SSCD_SCALE, settings.normalize_size)`` as *cap* so that
    DINOv2 receives images at its configured resolution.
    """

    def _process(data: bytes) -> Image.Image:
        return _cap_short_side(_open_rgb(data, cap), cap)

    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(_process, data) for data in image_data]
        results: list[Image.Image | Exception] = []
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results


def load_for_detection(data: bytes) -> np.ndarray:
    """Decode *data* to oriented full-resolution RGB uint8 (H, W, 3).

    Pillow's MAX_IMAGE_PIXELS decompression-bomb guard stays active here
    (deliberate: this path decodes at full resolution with no 331 px cap
    to protect it; SFN_MAX_IMAGE_PIXELS overrides for trusted ingestion,
    same as everywhere else).
    """
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def decode_shared(data: bytes, cap: int) -> tuple[np.ndarray, Image.Image | None]:
    """One native-resolution decode serving face detection AND the embed path.

    Returns ``(native_rgb_array, capped_pil_or_None)``.  The array is
    pixel-identical to :func:`scalar_forensic.faces.decode.load_for_detection`.
    The second element is the embedding-path input (what
    ``preprocess_batch`` would have produced from the same bytes) derived by
    downscaling the native decode — returned only when that derivation is
    byte-identical to the ``embedder._open_rgb`` + ``_cap_short_side`` path,
    so a faces-on run embeds exactly the same pixels as a faces-off run:

    - JPEG: ``None``.  ``_open_rgb`` uses libjpeg ``draft()`` shrink-on-load,
      whose pixels differ slightly from a full decode + Lanczos; the caller
      must decode the embed input separately (cheap for JPEG).
    - Non-JPEG with an EXIF Orientation != 1 in a format ``_open_rgb`` does
      not orient (e.g. PNG eXIf): ``None`` — detection sees the oriented
      image, the embed path deliberately keeps its unoriented behaviour.
    - Everything else (HEIC, PNG, WEBP, …): the derived image, identical.
    """
    pil = Image.open(io.BytesIO(data))
    fmt = pil.format
    orientation = pil.getexif().get(_EXIF_ORIENTATION_TAG, 1)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    arr = np.asarray(pil)
    if fmt == "JPEG" or (fmt not in _EXIF_ORIENTATION_FORMATS and orientation != 1):
        return arr, None
    return arr, _cap_short_side(Image.fromarray(arr), cap)


def register_heif() -> bool:
    """Register the optional Pillow HEIF plugin without importing the scanner."""
    try:
        import pillow_heif
    except ImportError:
        return False
    pillow_heif.register_heif_opener()
    return True
