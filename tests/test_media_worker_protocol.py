"""Worker output is untrusted even when the original source is read-only."""

import hashlib
import io
import json
import struct
from dataclasses import replace

import pytest


def protocol():
    from scalar_forensic.media_worker import protocol as module

    return module


def framed(header, payload=b""):
    raw = json.dumps(header).encode()
    return io.BytesIO(struct.pack("!I", len(raw)) + raw + payload)


def request():
    return protocol().Request("a" * 32, hashlib.sha256(b"source").hexdigest(), "preprocess", 331, 6)


def response_header():
    return {
        "version": 2,
        "request_id": "a" * 32,
        "source_sha256": hashlib.sha256(b"source").hexdigest(),
        "generation": "b" * 32,
        "status": "ok",
        "images": [
            {
                "role": "embedding",
                "mode": "RGB",
                "width": 1,
                "height": 1,
                "length": 3,
                "sha256": hashlib.sha256(b"\x01\x02\x03").hexdigest(),
            }
        ],
        "metadata": {},
        "error": None,
    }


def test_raw_rgb_is_owned_and_exact():
    result = protocol().read_response(
        framed(response_header(), b"\x01\x02\x03"), request(), "b" * 32
    )
    assert result.images[0].pixels == b"\x01\x02\x03"
    assert result.images[0].width == result.images[0].height == 1
    assert result.generation == "b" * 32


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "c" * 32),
        ("source_sha256", "d" * 64),
        ("generation", "c" * 32),
        ("version", True),
        ("version", 1),
        ("version", 3),
        ("status", "surprise"),
        ("metadata", {"path": "/etc/passwd"}),
    ],
)
def test_rejects_wrong_identity_version_or_schema(field, value):
    header = response_header()
    header[field] = value
    with pytest.raises(protocol().ProtocolError):
        protocol().read_response(framed(header, b"\x01\x02\x03"), request(), "b" * 32)


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", True),
        ("height", 0),
        ("width", 1000000000),
        ("length", 2),
        ("length", 10**12),
        ("role", "filename"),
        ("sha256", "0" * 64),
    ],
)
def test_rejects_invalid_rgb_before_trusting_output(field, value):
    header = response_header()
    header["images"][0][field] = value
    with pytest.raises(protocol().ProtocolError):
        protocol().read_response(framed(header, b"\x01\x02\x03"), request(), "b" * 32)


@pytest.mark.parametrize("payload", [b"", b"\x01\x02", b"\x01\x02\x03extra"])
def test_rejects_truncated_or_trailing_payload(payload):
    with pytest.raises(protocol().ProtocolError):
        protocol().read_response(framed(response_header(), payload), request(), "b" * 32)


@pytest.mark.parametrize(
    "raw", [b'{"version":1,"version":1}', b'{"x":NaN}', b"[]", b'{"x":Infinity}', b'{"x":1e400}']
)
def test_strict_json_rejects_ambiguous_or_nonfinite_headers(raw):
    stream = io.BytesIO(struct.pack("!I", len(raw)) + raw)
    with pytest.raises(protocol().ProtocolError):
        protocol().read_header(stream)


def test_oversize_header_rejected_without_reading_body():
    class PrefixOnly:
        def __init__(self):
            self.reads = 0

        def read(self, size):
            self.reads += 1
            assert self.reads == 1, "read body before enforcing header size"
            assert size == 4
            return struct.pack("!I", 65537)

    with pytest.raises(protocol().ProtocolError):
        protocol().read_header(PrefixOnly())


def test_request_rejects_source_hash_mismatch():
    stream = io.BytesIO()
    protocol().write_request(stream, request(), b"source")
    wire = stream.getvalue()[:-6] + b"forged"
    with pytest.raises(protocol().ProtocolError):
        protocol().read_request(io.BytesIO(wire))


def test_request_does_not_accept_paths_or_extra_fields():
    header = {
        "version": 2,
        "request_id": "a" * 32,
        "source_sha256": hashlib.sha256(b"source").hexdigest(),
        "operation": "preprocess",
        "cap": 331,
        "source_bytes": 6,
        "max_image_pixels": 100,
        "path": "/etc/passwd",
    }
    with pytest.raises(protocol().ProtocolError):
        protocol().read_request(framed(header, b"source"))


@pytest.mark.parametrize("limit", [100, None])
def test_request_roundtrip_binds_the_pixel_limit(limit):
    outgoing = replace(request(), max_image_pixels=limit)
    stream = io.BytesIO()
    protocol().write_request(stream, outgoing, b"source")
    stream.seek(0)
    incoming, data = protocol().read_request(stream)
    assert incoming.max_image_pixels == limit
    assert data == b"source"


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, "disabled", 2**63])
def test_request_refuses_invalid_pixel_limit(limit):
    with pytest.raises(protocol().ProtocolError):
        protocol().write_request(
            io.BytesIO(), replace(request(), max_image_pixels=limit), b"source"
        )


def test_worker_error_cannot_carry_partial_success():
    header = response_header()
    header.update(status="error", error="invalid_image")
    with pytest.raises(protocol().ProtocolError):
        protocol().read_response(framed(header, b"\x01\x02\x03"), request(), "b" * 32)


def test_grayscale_thumbnail_retains_luma_encoding_input():
    req = protocol().Request("a" * 32, request().source_sha256, "thumbnail", 331, 6)
    header = response_header()
    header["images"][0].update(mode="L", length=1, sha256=hashlib.sha256(b"\x7f").hexdigest())
    result = protocol().read_response(framed(header, b"\x7f"), req, "b" * 32)
    assert result.images[0].mode == "L"
    assert result.images[0].pixels == b"\x7f"


def test_grayscale_cannot_be_substituted_for_embedding_rgb():
    header = response_header()
    header["images"][0].update(mode="L", length=1, sha256=hashlib.sha256(b"\x7f").hexdigest())
    with pytest.raises(protocol().ProtocolError):
        protocol().read_response(framed(header, b"\x7f"), request(), "b" * 32)


def test_combined_images_are_bounded_before_payload_read(monkeypatch):
    monkeypatch.setattr(protocol(), "MAX_RGB", 4)
    req = protocol().Request("a" * 32, request().source_sha256, "shared", 331, 6)
    header = response_header()
    native = dict(header["images"][0], role="native")
    header["images"].insert(0, native)
    with pytest.raises(protocol().ProtocolError, match="oversized"):
        protocol().read_response(framed(header), req, "b" * 32)
