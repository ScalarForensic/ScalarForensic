import io
import json
import struct

import pytest

from scalar_forensic.media_worker import video_protocol as p


def request(operation="info", parameters=None, source="clip.mp4"):
    return dict(
        version=1,
        request_id="a" * 32,
        generation="b" * 32,
        operation=operation,
        source=source,
        parameters=parameters or {},
    )


class Partial(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))

    def write(self, data):
        return super().write(data[:3])


def test_packet_handles_partial_reads_and_writes():
    stream = Partial()
    header = dict(version=1, request_id="a" * 32, generation="b" * 32, type="data", body={})
    p.send_packet(stream, header, b"abcdef")
    stream.seek(0)
    assert p.recv_packet(stream) == (header, b"abcdef")


@pytest.mark.parametrize(
    "raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b"[]", b'{"a":1e999}', b"[" * 1000 + b"]" * 1000]
)
def test_invalid_json_is_rejected(raw):
    with pytest.raises(p.ProtocolError):
        p.recv_packet(io.BytesIO(struct.pack("!I", len(raw)) + raw + b"\0" * 4))


@pytest.mark.parametrize(
    "wire",
    [
        b"",
        b"\0\0",
        struct.pack("!I", 65537),
        b"\0\0\0\x02{}" + struct.pack("!I", 2**29),
        b"\0\0\0\x02{}\0\0\0\x02x",
    ],
)
def test_truncated_and_oversized_packets(wire):
    with pytest.raises(p.ProtocolError):
        p.recv_packet(io.BytesIO(wire))


@pytest.mark.parametrize("source", ["../x", "/x", "x/../y", "", "x\0y"])
def test_evidence_path_is_relative_and_bounded(source):
    with pytest.raises(p.ProtocolError):
        p.validate_request(request(source=source))


def test_parameters_cannot_supply_commands_or_boolean_numbers():
    for params in [
        {"argv": ["sh"]},
        {"fps": True, "max_frames": 1},
        {"fps": float("inf"), "max_frames": 1},
    ]:
        with pytest.raises(p.ProtocolError):
            p.validate_request(request("frames", params))


def test_frame_payload_matches_validated_geometry():
    event = dict(
        version=1,
        request_id="a" * 32,
        generation="b" * 32,
        type="frame",
        body=dict(width=2, height=1, timecode_ms=-10, frame_index=0, frame_hash="c" * 64),
    )
    assert p.validate_event(event, b"123456") == event
    with pytest.raises(p.ProtocolError):
        p.validate_event(event, b"123")


@pytest.mark.parametrize(
    "header",
    [
        {},
        request(),
        dict(version=1, request_id="a" * 32, generation="b" * 32, type="data", body={}),
        dict(version=1, request_id="a" * 32, generation="b" * 32, body={}),
    ],
)
def test_invalid_event_length_is_rejected_before_payload_read(header):
    raw = json.dumps(header).encode()
    wire = struct.pack("!I", len(raw)) + raw + struct.pack("!I", 2**28)

    class HeaderOnly(io.BytesIO):
        def read(self, size=-1):
            assert self.tell() < len(wire), "untrusted payload was read"
            return super().read(size)

    with pytest.raises(p.ProtocolError):
        p.recv_packet(HeaderOnly(wire))


@pytest.mark.parametrize(
    "key,value",
    [
        ("output_height", "144;touch /tmp/pwn"),
        ("timeout", True),
        ("threads", 0),
        ("start", -1),
        ("duration", 0),
        ("hwaccel", ["none"]),
        ("hdr", 1),
        ("limit_bytes", 2**51),
        ("argv", ["sh"]),
    ],
)
def test_encode_options_reject_injection_and_out_of_range_values(key, value):
    params = dict(
        output_height=144,
        chunk_seconds=5,
        hwaccel="none",
        timeout=30,
        hdr=False,
        start=None,
        duration=None,
        has_audio=True,
        watched=False,
        threads=None,
        nice=0,
        limit_bytes=None,
    )
    params[key] = value
    with pytest.raises(p.ProtocolError):
        p.validate_request(request("encode", params))
