"""Fixed, bounded video messages. Encoded artifacts are opaque byte streams."""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import PurePosixPath

MAX_HEADER = 65536
MAX_PAYLOAD = 256 * 1024 * 1024
MAX_DATA = 1024 * 1024


class ProtocolError(RuntimeError):
    """The peer sent an invalid or incomplete message."""


def _require(condition, message="invalid video message"):
    if not condition:
        raise ProtocolError(message)


def _keys(value, names):
    _require(type(value) is dict and set(value) == set(names.split()))


def _integer(value, low, high=2**63 - 1):
    _require(type(value) is int and low <= value <= high)


def _number(value, low, high, *, positive=False):
    _require(type(value) in (int, float))
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    _require(valid and (not positive or value > 0))


def _hex(value, length):
    _require(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value))


def _json(header):
    _require(type(header) is dict)
    try:
        encoded = json.dumps(header, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise ProtocolError("invalid JSON metadata") from exc
    _require(0 < len(encoded) <= MAX_HEADER, "JSON metadata exceeds limit")
    return encoded


def _read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        _require(type(chunk) is bytes and 0 < len(chunk) <= size - len(chunks), "truncated packet")
        chunks.extend(chunk)
    return bytes(chunks)


def _write_all(stream, data):
    view = memoryview(data)
    while view:
        count = stream.write(view)
        _require(type(count) is int and 0 < count <= len(view), "incomplete packet write")
        view = view[count:]


def send_packet(stream, header: dict, payload: bytes = b""):
    encoded = _json(header)
    _require(type(payload) is bytes and len(payload) <= MAX_PAYLOAD)
    _write_all(stream, struct.pack("!I", len(encoded)) + encoded + struct.pack("!I", len(payload)))
    _write_all(stream, payload)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def recv_packet(stream):
    length = struct.unpack("!I", _read_exact(stream, 4))[0]
    _require(0 < length <= MAX_HEADER, "JSON metadata exceeds limit")
    try:
        header = json.loads(_read_exact(stream, length), object_pairs_hook=_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("invalid JSON metadata") from exc
    _json(header)
    length = struct.unpack("!I", _read_exact(stream, 4))[0]
    _require(length <= MAX_PAYLOAD, "payload exceeds limit")
    if "type" in header:
        _validate_event(header, length)
    else:
        validate_request(header)
        _require(length == 0, "video requests cannot carry payloads")
    return header, _read_exact(stream, length)


def _identity(header):
    _require(type(header["version"]) is int and header["version"] == 1)
    _hex(header["request_id"], 32)
    _hex(header["generation"], 32)


def validate_request(header: dict):
    _json(header)
    _keys(header, "version request_id generation operation source parameters")
    _identity(header)
    op, source, params = header["operation"], header["source"], header["parameters"]
    _require(type(op) is str)
    if op in ("capability", "ping"):
        _require(source is None)
    else:
        _require(type(source) is str and 0 < len(source) <= 4096 and "\0" not in source)
        path = PurePosixPath(source)
        _require(not path.is_absolute() and ".." not in path.parts and path.parts)
    if op in ("info", "stream_report", "rewrap", "ping"):
        _keys(params, "")
    elif op == "frames":
        _keys(params, "fps max_frames")
        _number(params["fps"], 0, 1000, positive=True)
        _integer(params["max_frames"], 0, 1000000)
    elif op == "frame_at":
        _keys(params, "timecode_ms")
        _integer(params["timecode_ms"], 0, 10**12)
    elif op == "capability":
        _keys(params, "hwaccel")
    elif op == "encode":
        _keys(
            params,
            "output_height chunk_seconds hwaccel timeout hdr start duration "
            "has_audio watched threads nice limit_bytes",
        )
        for key, low, high in [
            ("output_height", 144, 16384),
            ("chunk_seconds", 1, 600),
            ("timeout", 1, 86400),
            ("nice", 0, 19),
        ]:
            _integer(params[key], low, high)
        for key in ("hdr", "has_audio", "watched"):
            _require(type(params[key]) is bool)
        for key, maximum in [("start", 10**9), ("duration", 86400)]:
            if params[key] is not None:
                _number(params[key], 0, maximum, positive=key == "duration")
        for key, maximum in [("threads", 256), ("limit_bytes", 2**50)]:
            if params[key] is not None:
                _integer(params[key], 1, maximum)
    else:
        raise ProtocolError("unknown video operation")
    if op in ("encode", "capability"):
        _require(type(params["hwaccel"]) is str and params["hwaccel"] in ("auto", "cuda", "none"))
    return header


def validate_event(header: dict, payload: bytes):
    _require(type(payload) is bytes)
    return _validate_event(header, len(payload))


def _validate_event(header: dict, payload_size: int):
    _json(header)
    _keys(header, "version request_id generation type body")
    _identity(header)
    kind, body = header["type"], header["body"]
    _require(type(kind) is str and type(body) is dict)
    if kind == "result":
        pass
    elif kind == "progress":
        _keys(body, "frames out_seconds written_bytes")
        _integer(body["frames"], 0)
        _integer(body["written_bytes"], 0)
        _number(body["out_seconds"], 0, 2**63 - 1)
    elif kind == "frame":
        _keys(body, "width height timecode_ms frame_index frame_hash")
        for key in ("width", "height"):
            _integer(body[key], 1, 32768)
        _integer(body["timecode_ms"], -(2**63))
        _integer(body["frame_index"], 0)
        _hex(body["frame_hash"], 64)
        _require(payload_size == body["width"] * body["height"] * 3 <= MAX_PAYLOAD)
    elif kind == "artifact":
        _keys(body, "size sha256")
        _integer(body["size"], 1, 2**50)
        _hex(body["sha256"], 64)
    elif kind == "data":
        _keys(body, "")
        _require(0 < payload_size <= MAX_DATA)
    elif kind == "error":
        _keys(body, "kind message stderr returncode timed_out written_bytes limit_bytes")
        _require(body["kind"] in ("decode", "encode", "ceiling", "internal"))
        for key, maximum in [("message", 4096), ("stderr", 65536)]:
            _require(type(body[key]) is str and len(body[key]) <= maximum)
        if body["returncode"] is not None:
            _integer(body["returncode"], -(2**63))
        _require(type(body["timed_out"]) is bool)
        for key in ("written_bytes", "limit_bytes"):
            _integer(body[key], 0)
    else:
        raise ProtocolError("unknown video event")
    if kind not in ("frame", "data"):
        _require(payload_size == 0)
    return header
