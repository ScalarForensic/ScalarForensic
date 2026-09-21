"""Bounded source-byte/RGB protocol. Worker replies are untrusted input."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass, field
from typing import BinaryIO

from PIL import Image

VERSION = 2
MAX_HEADER = 65536
MAX_SOURCE = 128 * 1024 * 1024
MAX_RGB = 256 * 1024 * 1024
MAX_DIMENSION = 32768
OPERATIONS = frozenset(
    {
        "preprocess",
        "open_rgb",
        "native",
        "shared",
        "exif",
        "exif_detailed",
        "display_rgb",
        "thumbnail",
    }
)
_HEX = re.compile(r"[0-9a-f]+\Z")


class ProtocolError(ValueError):
    """Invalid or incomplete wire data; never retry using local decoding."""


class DecodeError(ValueError):
    """Worker reported a bounded decode failure, not a successful empty result."""


@dataclass(frozen=True)
class Request:
    request_id: str
    source_sha256: str
    operation: str
    cap: int
    source_bytes: int
    max_image_pixels: int | None = field(default_factory=lambda: Image.MAX_IMAGE_PIXELS)


@dataclass(frozen=True)
class RGBImage:
    role: str
    width: int
    height: int
    pixels: bytes
    mode: str = "RGB"


@dataclass(frozen=True)
class Response:
    generation: str
    images: tuple[RGBImage, ...]
    metadata: dict


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _hex(value, length):
    return type(value) is str and len(value) == length and _HEX.fullmatch(value) is not None


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ProtocolError("unexpected header fields")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ProtocolError("nonfinite JSON value")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError("nonfinite JSON value")
    return result


def read_exact(stream: BinaryIO, size: int) -> bytes:
    parts = bytearray()
    while len(parts) < size:
        chunk = stream.read(min(size - len(parts), 1024 * 1024))
        if not chunk:
            raise ProtocolError("truncated message")
        parts.extend(chunk)
    return bytes(parts)


def read_header(stream: BinaryIO) -> dict:
    size = struct.unpack("!I", read_exact(stream, 4))[0]
    if not 0 < size <= MAX_HEADER:
        raise ProtocolError("header exceeds limit")
    try:
        result = json.loads(
            read_exact(stream, size),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=_float,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("invalid JSON header") from exc
    if type(result) is not dict:
        raise ProtocolError("header must be an object")
    return result


def write_header(stream: BinaryIO, header: dict) -> None:
    try:
        data = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise ProtocolError("invalid outgoing header") from exc
    if not 0 < len(data) <= MAX_HEADER:
        raise ProtocolError("header exceeds limit")
    stream.write(struct.pack("!I", len(data)) + data)


def validate_request_header(header):
    _keys(
        header,
        (
            "version",
            "request_id",
            "source_sha256",
            "operation",
            "cap",
            "source_bytes",
            "max_image_pixels",
        ),
    )
    if (
        not _integer(header["version"], VERSION, VERSION)
        or not _hex(header["request_id"], 32)
        or not _hex(header["source_sha256"], 64)
        or type(header["operation"]) is not str
        or header["operation"] not in OPERATIONS
        or not _integer(header["cap"], 1, MAX_DIMENSION)
        or not _integer(header["source_bytes"], 1, MAX_SOURCE)
        or (
            header["max_image_pixels"] is not None
            and not _integer(header["max_image_pixels"], 1, 2**63 - 1)
        )
    ):
        raise ProtocolError("invalid request")
    return Request(
        *(
            header[name]
            for name in (
                "request_id",
                "source_sha256",
                "operation",
                "cap",
                "source_bytes",
                "max_image_pixels",
            )
        )
    )


def read_request(stream: BinaryIO) -> tuple[Request, bytes]:
    request = validate_request_header(read_header(stream))
    data = read_exact(stream, request.source_bytes)
    if stream.read(1):
        raise ProtocolError("trailing request data")
    if hashlib.sha256(data).hexdigest() != request.source_sha256:
        raise ProtocolError("source digest mismatch")
    return request, data


def write_request(stream: BinaryIO, request: Request, data: bytes) -> None:
    header = {
        "version": VERSION,
        "request_id": request.request_id,
        "source_sha256": request.source_sha256,
        "operation": request.operation,
        "cap": request.cap,
        "source_bytes": request.source_bytes,
        "max_image_pixels": request.max_image_pixels,
    }
    validate_request_header(header)
    if (
        len(data) != request.source_bytes
        or hashlib.sha256(data).hexdigest() != request.source_sha256
    ):
        raise ProtocolError("outgoing source does not match request")
    write_header(stream, header)
    stream.write(data)


def _metadata(meta, operation):
    if type(meta) is not dict:
        raise ProtocolError("invalid metadata")
    if operation not in ("exif", "exif_detailed"):
        if meta:
            raise ProtocolError("unsolicited metadata")
        return
    allowed = {"exif", "exif_geo_data"}
    if operation == "exif_detailed":
        allowed.update(
            (
                "width",
                "height",
                "format",
                "size_bytes",
                "make",
                "model",
                "datetime",
                "software",
                "gps_lat",
                "gps_lon",
            )
        )
    if not {"exif", "exif_geo_data"} <= meta.keys() or not meta.keys() <= allowed:
        raise ProtocolError("unexpected metadata fields")
    for key, value in meta.items():
        if key in ("exif", "exif_geo_data"):
            valid = type(value) is bool
        elif key in ("width", "height", "size_bytes"):
            valid = _integer(value, 1, MAX_SOURCE if key == "size_bytes" else MAX_DIMENSION)
        elif key in ("gps_lat", "gps_lon"):
            limit = 90 if key == "gps_lat" else 180
            valid = (
                type(value) in (int, float) and -limit <= value <= limit and math.isfinite(value)
            )
        else:
            valid = type(value) is str and len(value) <= 2048 and "\x00" not in value
        if not valid:
            raise ProtocolError("invalid metadata value")


def _response(header, request, generation):
    _keys(
        header,
        (
            "version",
            "request_id",
            "source_sha256",
            "generation",
            "status",
            "images",
            "metadata",
            "error",
        ),
    )
    if (
        not _integer(header["version"], VERSION, VERSION)
        or header["request_id"] != request.request_id
        or header["source_sha256"] != request.source_sha256
        or not _hex(generation, 32)
        or header["generation"] != generation
        or type(header["status"]) is not str
        or header["status"] not in ("ok", "error")
        or type(header["images"]) is not list
        or len(header["images"]) > 2
    ):
        raise ProtocolError("invalid response identity or status")
    if header["status"] == "error":
        if (
            header["images"]
            or header["metadata"] != {}
            or type(header["error"]) is not str
            or header["error"] not in ("invalid_image", "resource_limit", "worker_error")
        ):
            raise ProtocolError("invalid error response")
        return
    if header["error"] is not None:
        raise ProtocolError("success has error data")
    _metadata(header["metadata"], request.operation)
    roles = []
    total = 0
    for row in header["images"]:
        _keys(row, ("role", "mode", "width", "height", "length", "sha256"))
        if (
            type(row["role"]) is not str
            or row["role"] not in ("embedding", "native")
            or type(row["mode"]) is not str
            or row["mode"] not in ("RGB", "L")
            or (row["mode"] != "RGB" and request.operation != "thumbnail")
            or not _integer(row["width"], 1, MAX_DIMENSION)
            or not _integer(row["height"], 1, MAX_DIMENSION)
            or not _integer(row["length"], 1, MAX_RGB)
            or row["length"] != row["width"] * row["height"] * (3 if row["mode"] == "RGB" else 1)
            or not _hex(row["sha256"], 64)
        ):
            raise ProtocolError("invalid RGB description")
        roles.append(row["role"])
        total += row["length"]
    permitted = {
        "preprocess": [["embedding"]],
        "open_rgb": [["embedding"]],
        "native": [["native"]],
        "shared": [["native"], ["native", "embedding"]],
        "exif": [[]],
        "exif_detailed": [[]],
        "display_rgb": [["embedding"]],
        "thumbnail": [["embedding"]],
    }
    if roles not in permitted[request.operation] or total > MAX_RGB:
        raise ProtocolError("unexpected or oversized output images")


def read_response(stream: BinaryIO, request: Request, generation: str) -> Response:
    header = read_header(stream)
    _response(header, request, generation)
    images = []
    for row in header["images"]:
        pixels = read_exact(stream, row["length"])
        if hashlib.sha256(pixels).hexdigest() != row["sha256"]:
            raise ProtocolError("RGB transfer digest mismatch")
        images.append(RGBImage(row["role"], row["width"], row["height"], pixels, row["mode"]))
    if stream.read(1):
        raise ProtocolError("trailing response data")
    if header["status"] == "error":
        raise DecodeError(header["error"])
    return Response(generation, tuple(images), header["metadata"])


def write_response(
    stream: BinaryIO,
    request: Request,
    generation: str,
    images: tuple[RGBImage, ...] = (),
    metadata: dict | None = None,
    error: str | None = None,
) -> None:
    header = {
        "version": VERSION,
        "request_id": request.request_id,
        "source_sha256": request.source_sha256,
        "generation": generation,
        "status": "error" if error else "ok",
        "images": [],
        "metadata": {} if metadata is None else metadata,
        "error": error,
    }
    for image in images:
        header["images"].append(
            {
                "role": image.role,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "length": len(image.pixels),
                "sha256": hashlib.sha256(image.pixels).hexdigest(),
            }
        )
    _response(header, request, generation)
    write_header(stream, header)
    for image in images:
        stream.write(image.pixels)
