"""Real local transports exercise process reuse and fail-closed retirement."""

import errno
import hashlib
import io
import multiprocessing
import os
import socket
import time
from contextlib import contextmanager

import pytest
from PIL import Image

from scalar_forensic.media_worker import protocol
from scalar_forensic.media_worker.client import WorkerClient, WorkerUnavailable

GENERATION = "a" * 32


def _source(fmt="PNG"):
    out = io.BytesIO()
    Image.new("RGB", (12, 8), (40, 80, 120)).save(out, format=fmt)
    return out.getvalue()


def _run(listener, settings):
    from scalar_forensic.media_worker.server import serve

    raise SystemExit(serve(listener, generation=GENERATION, **settings))


@contextmanager
def _worker(tmp_path, **settings):
    from scalar_forensic.media_worker.server import bound_listener

    with bound_listener(tmp_path / "worker.sock") as listener:
        process = multiprocessing.get_context("fork").Process(
            target=_run, args=(listener, settings)
        )
        process.start()
        try:
            yield WorkerClient(tmp_path / "worker.sock", GENERATION, timeout=3), process
        finally:
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join(3)


def test_sequential_requests_reuse_pid_and_preserve_pixels(tmp_path, monkeypatch):
    from scalar_forensic.media_worker import server

    receiver, sender = multiprocessing.Pipe(duplex=False)
    original_dispatch = server.dispatch

    def observed_dispatch(*args):
        sender.send(os.getpid())
        return original_dispatch(*args)

    monkeypatch.setattr(server, "dispatch", observed_dispatch)
    with _worker(tmp_path, max_requests=2) as (client, process):
        pid = process.pid
        for operation, geometry in [("native", (12, 8)), ("preprocess", (6, 4))]:
            result = client.decode(_source(), operation, cap=4).response.images[0]
            assert (result.width, result.height) == geometry
            assert result.pixels == bytes((40, 80, 120)) * (geometry[0] * geometry[1])
            assert receiver.poll(1)
            assert receiver.recv() == pid
        process.join(3)
        assert process.exitcode == 0
    receiver.close()
    sender.close()


@pytest.mark.parametrize("fmt,roles", [("JPEG", ["native"]), ("PNG", ["native", "embedding"])])
def test_shared_slots(tmp_path, fmt, roles):
    with _worker(tmp_path, max_requests=1) as (client, process):
        result = client.decode(_source(fmt), "shared", cap=4).response
        assert [image.role for image in result.images] == roles
        assert (result.images[0].width, result.images[0].height) == (12, 8)
        process.join(3)
        assert process.exitcode == 0


def test_invalid_image_does_not_retire(tmp_path):
    with _worker(tmp_path, max_requests=2) as (client, process):
        with pytest.raises(protocol.DecodeError, match="invalid_image"):
            client.decode(b"broken image", "native")
        assert client.decode(_source(), "exif").response.metadata == {
            "exif": False,
            "exif_geo_data": False,
        }
        process.join(3)
        assert process.exitcode == 0


@pytest.mark.parametrize("application_limit", [10, None])
def test_pixel_limit_mismatch_retires_worker_without_returning_pixels(
    tmp_path, monkeypatch, application_limit
):
    with _worker(tmp_path, max_requests=1) as (client, process):
        # The forked worker keeps its startup guard; only the caller changes.
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", application_limit)
        with pytest.raises(WorkerUnavailable):
            client.decode(_source(), "native")
        process.join(3)
        assert process.exitcode == 1


def test_matching_strict_pixel_limit_rejects_oversize_image(tmp_path, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with _worker(tmp_path, max_requests=1) as (client, process):
        with pytest.raises(protocol.DecodeError, match="invalid_image"):
            client.decode(_source(), "native")
        process.join(3)
        assert process.exitcode == 0


@pytest.mark.parametrize("failure", ["digest", "operation", "trailing", "timeout"])
def test_bad_request_retires_generation(tmp_path, failure):
    with _worker(tmp_path, request_timeout=0.15) as (client, process):
        with socket.socket(socket.AF_UNIX) as channel:
            channel.connect(str(client.socket_path))
            if failure != "timeout":
                source = _source()
                header = {
                    "version": 2,
                    "request_id": "b" * 32,
                    "source_sha256": "0" * 64
                    if failure == "digest"
                    else hashlib.sha256(source).hexdigest(),
                    "operation": "bad" if failure == "operation" else "native",
                    "cap": 4,
                    "source_bytes": len(source),
                    "max_image_pixels": Image.MAX_IMAGE_PIXELS,
                }
                stream = channel.makefile("wb", buffering=0)
                protocol.write_header(stream, header)
                stream.write(source + (b"x" if failure == "trailing" else b""))
                channel.shutdown(socket.SHUT_WR)
                stream.close()
            process.join(3)
            assert process.exitcode == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"request_timeout": float("nan")},
        {"max_seconds": float("inf")},
        {"concurrency": 0},
        {"max_queued": -1},
        {"max_requests": True},
        {"generation": "bad"},
    ],
)
def test_invalid_limits_rejected_before_serving(settings):
    from scalar_forensic.media_worker.server import serve

    with pytest.raises(ValueError):
        serve(None, **({"generation": GENERATION} | settings))


def test_socket_lifecycle_preserves_existing_and_replacement(tmp_path):
    from scalar_forensic.media_worker.server import bound_listener

    path = tmp_path / "worker.sock"
    path.write_text("original")
    with pytest.raises(FileExistsError), bound_listener(path):
        pass
    assert path.read_text() == "original"
    path.unlink()
    with bound_listener(path):
        assert path.stat().st_mode & 0o777 == 0o600
        path.unlink()
        path.write_text("replacement")
    assert path.read_text() == "replacement"


def test_runtime_guard_rejects_unrestricted_host(monkeypatch):
    from scalar_forensic.media_worker.server import validate_runtime

    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(RuntimeError, match="nonroot"):
        validate_runtime()


def test_lifetime_expires_without_requests(tmp_path):
    start = time.monotonic()
    with _worker(tmp_path, max_seconds=0.1) as (_, process):
        process.join(3)
        assert process.exitcode == 0
    assert time.monotonic() - start < 3


@pytest.mark.parametrize(
    "failure", [protocol.ProtocolError, TimeoutError, RuntimeError, MemoryError]
)
def test_unexpected_error_from_dispatch_retires(tmp_path, monkeypatch, failure):
    from scalar_forensic.media_worker import server

    def broken(*args):
        raise failure("bad decoder result")

    monkeypatch.setattr(server, "dispatch", broken)
    with _worker(tmp_path, max_requests=1) as (client, process):
        with pytest.raises(WorkerUnavailable):
            client.decode(_source(), "native")
        process.join(3)
        assert process.exitcode == 1


def test_decoder_hang_exits_process_by_deadline(tmp_path, monkeypatch):
    from scalar_forensic.media_worker import server

    def stuck(*args):
        time.sleep(20)

    monkeypatch.setattr(server, "dispatch", stuck)
    with _worker(tmp_path, request_timeout=0.15) as (client, process):
        with pytest.raises(WorkerUnavailable):
            client.decode(_source(), "native")
        process.join(2)
        assert process.exitcode == 1


def test_disconnected_reply_retires(tmp_path):
    with _worker(tmp_path) as (client, process):
        data = _source()
        request = protocol.Request(
            "b" * 32, hashlib.sha256(data).hexdigest(), "native", 4, len(data)
        )
        with socket.socket(socket.AF_UNIX) as channel:
            channel.connect(str(client.socket_path))
            stream = channel.makefile("wb", buffering=0)
            protocol.write_request(stream, request, data)
            stream.close()
        process.join(3)
        assert process.exitcode == 1


def test_saturated_admission_rejects_extra_connection(tmp_path):
    with _worker(tmp_path, concurrency=1, max_queued=1, request_timeout=1) as (client, process):
        channels = [socket.socket(socket.AF_UNIX) for _ in range(3)]
        try:
            for channel in channels:
                channel.connect(str(client.socket_path))
            channels[-1].settimeout(0.5)
            assert channels[-1].recv(1) == b""
        finally:
            for channel in channels:
                channel.close()
        process.join(3)
        assert process.exitcode == 1


@pytest.mark.parametrize(
    "status,interfaces,gpu",
    [
        ("NoNewPrivs: 0\nSeccomp: 2\nCapEff: 0", [(1, "lo")], False),
        ("NoNewPrivs: 1\nSeccomp: 0\nCapEff: 0", [(1, "lo")], False),
        ("NoNewPrivs: 1\nSeccomp: 2\nCapEff: 1", [(1, "lo")], False),
        ("NoNewPrivs: 1\nSeccomp: 2\nCapEff: 0", [(1, "lo"), (2, "eth0")], False),
        ("NoNewPrivs: 1\nSeccomp: 2\nCapEff: 0", [(1, "lo")], True),
        ("", [(1, "lo")], False),
    ],
)
def test_runtime_guard_fails_closed_for_each_restriction(monkeypatch, status, interfaces, gpu):
    from scalar_forensic.media_worker import server

    monkeypatch.setattr(os, "geteuid", lambda: 10001)
    monkeypatch.setattr(server.Path, "read_text", lambda *args: status)
    monkeypatch.setattr(socket, "if_nameindex", lambda: interfaces)
    monkeypatch.setattr(server.Path, "glob", lambda *args: iter(()))
    monkeypatch.setattr(server.Path, "exists", lambda *args: gpu)
    with pytest.raises(RuntimeError):
        server.validate_runtime()


def test_output_limit_checked_before_tobytes(monkeypatch):
    from scalar_forensic.media_worker import server

    class HugeImage(Image.Image):
        @property
        def size(self):
            return (32768, 32768)

        @property
        def mode(self):
            return "RGB"

        def tobytes(self):
            pytest.fail("oversize pixels were materialized")

    monkeypatch.setattr(server.media_decode, "_open_rgb", lambda *args: HugeImage())
    request = protocol.Request("b" * 32, "c" * 64, "open_rgb", 4, 1)
    with pytest.raises(server.ResourceLimit):
        server.dispatch(request, b"x", GENERATION)


def test_thumbnail_preserves_gray_and_display_keeps_full_geometry(tmp_path):
    data = io.BytesIO()
    Image.new("L", (12, 8), 63).save(data, format="PNG")
    with _worker(tmp_path, max_requests=2) as (client, process):
        thumb = client.decode(data.getvalue(), "thumbnail", cap=4).response.images[0]
        assert (thumb.width, thumb.height, thumb.mode, thumb.pixels) == (4, 3, "L", b"?" * 12)
        display = client.decode(data.getvalue(), "display_rgb", cap=4).response.images[0]
        assert (display.width, display.height, display.mode, display.pixels) == (
            12,
            8,
            "RGB",
            b"?" * 288,
        )
        process.join(3)
        assert process.exitcode == 0


def test_out_of_protocol_exif_cannot_become_silent_empty_metadata(tmp_path, monkeypatch):
    from scalar_forensic.media_worker import server

    monkeypatch.setattr(
        server.media_decode,
        "extract_exif_detailed",
        lambda data: {
            "exif": True,
            "exif_geo_data": True,
            "gps_lat": float("nan"),
            "gps_lon": 999,
            "make": "x" * 3000,
            "model": "valid",
            "software": "bad\x00value",
        },
    )
    with _worker(tmp_path, max_requests=1) as (client, process):
        with pytest.raises(WorkerUnavailable):
            client.decode(_source(), "exif_detailed")
        process.join(3)
        assert process.exitcode == 1


def test_combined_limit_checked_before_either_tobytes(monkeypatch):
    from scalar_forensic.media_worker import server

    class SmallImage(Image.Image):
        @property
        def size(self):
            return (5, 5)

        @property
        def mode(self):
            return "RGB"

        def tobytes(self):
            pytest.fail("combined limit must be checked before copying any image")

    monkeypatch.setattr(protocol, "MAX_RGB", 100)
    monkeypatch.setattr(
        server.media_decode, "decode_shared", lambda *args: (SmallImage(), SmallImage())
    )
    request = protocol.Request("b" * 32, "c" * 64, "shared", 4, 1)
    with pytest.raises(server.ResourceLimit):
        server.dispatch(request, b"x", GENERATION)


def test_exif_detailed_retains_exact_valid_fields(tmp_path):
    with _worker(tmp_path, max_requests=1) as (client, process):
        data = _source()
        result = client.decode(data, "exif_detailed").response
        assert result.images == ()
        assert result.metadata == {
            "exif": False,
            "exif_geo_data": False,
            "width": 12,
            "height": 8,
            "format": "PNG",
            "size_bytes": len(data),
        }
        process.join(3)
        assert process.exitcode == 0


def test_cli_guard_fails_before_opening_socket(tmp_path, monkeypatch):
    from scalar_forensic.media_worker import server

    monkeypatch.setattr(os, "geteuid", lambda: 0)
    path = tmp_path / "worker.sock"
    with pytest.raises(SystemExit) as error:
        server.main(["--socket", str(path), "--generation", GENERATION])
    assert error.value.code == 1
    assert not path.exists()


def test_own_socket_is_removed_after_listener_closes(tmp_path):
    from scalar_forensic.media_worker import server

    path = tmp_path / "worker.sock"
    with server.bound_listener(path):
        assert path.is_socket()
    assert not path.exists()


@pytest.mark.parametrize(
    "code,expected",
    [
        (errno.ENOMEM, "resource_limit"),
        (errno.EMFILE, "resource_limit"),
        (errno.ENFILE, "resource_limit"),
        (errno.ENOSPC, "resource_limit"),
        (errno.EIO, "worker_error"),
    ],
)
def test_runtime_oserror_returns_bounded_failure_and_retires(tmp_path, monkeypatch, code, expected):
    from scalar_forensic.media_worker import server

    def broken(*args):
        raise OSError(code, "private runtime diagnostics must not escape")

    monkeypatch.setattr(server, "dispatch", broken)
    with _worker(tmp_path, max_requests=1) as (client, process):
        data = _source()
        request = protocol.Request(
            "b" * 32, hashlib.sha256(data).hexdigest(), "native", 4, len(data)
        )
        with socket.socket(socket.AF_UNIX) as channel:
            channel.settimeout(2)
            channel.connect(str(client.socket_path))
            with channel.makefile("rwb", buffering=0) as stream:
                protocol.write_request(stream, request, data)
                channel.shutdown(socket.SHUT_WR)
                with pytest.raises(protocol.DecodeError, match=f"^{expected}$"):
                    protocol.read_response(stream, request, GENERATION)
        process.join(2)
        assert process.exitcode == 1


def test_output_overflow_retires_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "MAX_RGB", 1)
    with _worker(tmp_path, max_requests=1) as (client, process):
        with pytest.raises(WorkerUnavailable):
            client.decode(_source(), "native")
        process.join(2)
        assert process.exitcode == 1


def test_plain_image_oserror_allows_next_request(tmp_path, monkeypatch):
    from scalar_forensic.media_worker import server

    original = server.dispatch

    def decode(request, data, generation):
        if data == b"broken":
            raise OSError("malformed decoder stream")
        return original(request, data, generation)

    monkeypatch.setattr(server, "dispatch", decode)
    with _worker(tmp_path, max_requests=2) as (client, process):
        with pytest.raises(protocol.DecodeError, match="^invalid_image$"):
            client.decode(b"broken", "native")
        assert client.decode(_source(), "native").response.images[0].width == 12
        process.join(2)
        assert process.exitcode == 0
