import json
import socket
import struct
import threading
import time

import pytest

from scalar_forensic.media_worker import protocol


def client(path, **kwargs):
    from scalar_forensic.media_worker.client import WorkerClient

    return WorkerClient(path, "b" * 32, **kwargs)


def serve(path, transform, requests=1):
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(path))
    listener.listen(1)
    errors = []

    def run():
        try:
            with listener:
                for _ in range(requests):
                    conn, _ = listener.accept()
                    with conn, conn.makefile("rwb") as stream:
                        req, data = protocol.read_request(stream)
                        transform(stream, req, data)
                        stream.flush()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def test_client_roundtrip_returns_exact_rgb(tmp_path):
    path = tmp_path / "worker.sock"

    def reply(stream, req, data):
        assert data == b"source"
        protocol.write_response(
            stream, req, "b" * 32, (protocol.RGBImage("embedding", 1, 1, b"\x01\x02\x03"),)
        )

    thread, errors = serve(path, reply)
    result = client(path).decode(b"source", "preprocess", 331)
    thread.join(2)
    assert not thread.is_alive()
    assert not errors
    assert result.response.images[0].pixels == b"\x01\x02\x03"
    assert len(result.request.request_id) == 32


def test_unavailable_worker_fails_without_decoding(tmp_path):
    with pytest.raises(RuntimeError, match="worker"):
        client(tmp_path / "missing.sock").decode(b"not even an image", "preprocess", 331)


def test_wrong_generation_invalidates_client(tmp_path):
    path = tmp_path / "worker.sock"

    def reply(stream, req, data):
        protocol.write_response(
            stream, req, "c" * 32, (protocol.RGBImage("embedding", 1, 1, b"\x01\x02\x03"),)
        )

    thread, errors = serve(path, reply)
    worker = client(path)
    with pytest.raises(RuntimeError):
        worker.decode(b"source", "preprocess", 331)
    thread.join(2)
    assert not errors
    with pytest.raises(RuntimeError, match="invalidated"):
        worker.decode(b"source", "preprocess", 331)


def test_total_deadline_includes_wait_for_eof(tmp_path):
    path = tmp_path / "worker.sock"

    def reply(stream, req, data):
        protocol.write_response(
            stream, req, "b" * 32, (protocol.RGBImage("embedding", 1, 1, b"\x01\x02\x03"),)
        )
        stream.flush()
        time.sleep(0.25)

    thread, errors = serve(path, reply)
    before = time.monotonic()
    with pytest.raises(RuntimeError):
        client(path, timeout=0.05).decode(b"source", "preprocess", 331)
    assert time.monotonic() - before < 0.2
    thread.join(2)
    assert not errors


def test_decode_error_is_not_successful_empty_metadata(tmp_path):
    path = tmp_path / "worker.sock"

    def reply(stream, req, data):
        protocol.write_response(stream, req, "b" * 32, error="invalid_image")

    thread, errors = serve(path, reply)
    with pytest.raises(protocol.DecodeError, match="invalid_image"):
        client(path).decode(b"source", "exif", 331)
    thread.join(2)
    assert not errors


def test_request_body_is_validated_before_connecting(tmp_path):
    with pytest.raises(protocol.ProtocolError):
        client(tmp_path / "missing.sock").decode(b"", "preprocess", 331)


def test_failure_notice_prevents_another_client_submitting_to_failed_generation(tmp_path):
    from scalar_forensic.media_worker.client import WorkerUnavailable
    from scalar_forensic.media_worker.receipts import ReceiptLog

    notice = tmp_path / "failed"
    audit = tmp_path / "audit.jsonl"
    worker = client(
        tmp_path / "missing.sock",
        failure_notice=notice,
        receipt_log=ReceiptLog(audit, "case-a", "b" * 32),
    )
    notice.touch()
    with pytest.raises(WorkerUnavailable, match="invalidated"):
        worker.decode(b"source", "preprocess")
    assert not audit.exists()
    notice.unlink()
    with pytest.raises(WorkerUnavailable, match="invalidated"):
        worker.decode(b"source", "preprocess")


def test_failure_notice_during_response_discards_pixels_and_records_failure(tmp_path):
    from scalar_forensic.media_worker.client import WorkerUnavailable
    from scalar_forensic.media_worker.receipts import ReceiptLog

    path = tmp_path / "worker.sock"
    notice = tmp_path / "failed"
    audit = tmp_path / "audit.jsonl"

    def reply(stream, req, data):
        notice.touch()
        protocol.write_response(
            stream, req, "b" * 32, (protocol.RGBImage("embedding", 1, 1, b"\x01\x02\x03"),)
        )

    thread, errors = serve(path, reply)
    worker = client(path, failure_notice=notice, receipt_log=ReceiptLog(audit, "case-a", "b" * 32))
    try:
        with pytest.raises(WorkerUnavailable, match="invalidated"):
            worker.decode(b"source", "preprocess")
    finally:
        thread.join(2)
    assert not thread.is_alive()
    assert not errors
    assert [json.loads(line)["event"] for line in audit.read_text().splitlines()] == [
        "started",
        "worker_failed",
    ]


def test_client_does_not_allow_nonfinite_deadline(tmp_path):
    with pytest.raises(ValueError):
        client(tmp_path / "missing.sock", timeout=float("nan"))


def test_audit_records_failure_before_any_possible_result(tmp_path):
    from scalar_forensic.media_worker.receipts import ReceiptLog

    path = tmp_path / "audit.jsonl"
    worker = client(tmp_path / "missing.sock", receipt_log=ReceiptLog(path, "case-a", "b" * 32))
    with pytest.raises(RuntimeError):
        worker.decode(b"source", "preprocess", 331)
    events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    assert events == ["started", "worker_failed"]


def test_audit_failure_prevents_contacting_worker(tmp_path):
    from scalar_forensic.media_worker.receipts import ReceiptLog

    original = tmp_path / "original"
    original.write_bytes(b"evidence")
    path = tmp_path / "audit.jsonl"
    path.symlink_to(original)
    with pytest.raises(OSError):
        client(tmp_path / "missing.sock", receipt_log=ReceiptLog(path, "case-a", "b" * 32)).decode(
            b"source", "preprocess", 331
        )
    assert original.read_bytes() == b"evidence"


def test_huge_gps_integer_invalidates_generation_and_records_failure(tmp_path):
    from scalar_forensic.media_worker.client import WorkerUnavailable
    from scalar_forensic.media_worker.receipts import ReceiptLog

    path = tmp_path / "worker.sock"
    audit = tmp_path / "audit.jsonl"

    def reply(stream, req, data):
        header = {
            "version": 2,
            "request_id": req.request_id,
            "source_sha256": req.source_sha256,
            "generation": "b" * 32,
            "status": "ok",
            "images": [],
            "error": None,
            "metadata": {"exif": True, "exif_geo_data": True, "gps_lat": 10**1000},
        }
        raw = json.dumps(header).encode()
        stream.write(struct.pack("!I", len(raw)) + raw)

    thread, errors = serve(path, reply)
    worker = client(path, receipt_log=ReceiptLog(audit, "case-a", "b" * 32))
    try:
        with pytest.raises(WorkerUnavailable):
            worker.decode(b"source", "exif_detailed")
    finally:
        thread.join(2)
    assert not errors
    with pytest.raises(WorkerUnavailable, match="invalidated"):
        worker.decode(b"source", "exif_detailed")
    assert [json.loads(line)["event"] for line in audit.read_text().splitlines()] == [
        "started",
        "worker_failed",
    ]


@pytest.mark.parametrize("error", ["resource_limit", "worker_error"])
def test_worker_errors_invalidate_generation_and_record_failure(tmp_path, error):
    from scalar_forensic.media_worker.client import WorkerUnavailable
    from scalar_forensic.media_worker.receipts import ReceiptLog

    path = tmp_path / "worker.sock"
    audit = tmp_path / "audit.jsonl"

    def reply(stream, req, data):
        protocol.write_response(stream, req, "b" * 32, error=error)

    thread, errors = serve(path, reply)
    worker = client(path, receipt_log=ReceiptLog(audit, "case-a", "b" * 32))
    try:
        with pytest.raises(WorkerUnavailable):
            worker.decode(b"source", "preprocess")
    finally:
        thread.join(2)
    assert not errors
    with pytest.raises(WorkerUnavailable, match="invalidated"):
        worker.decode(b"source", "preprocess")
    assert [json.loads(line)["event"] for line in audit.read_text().splitlines()] == [
        "started",
        "worker_failed",
    ]


def test_transport_interruption_invalidates_and_records_before_propagating(tmp_path, monkeypatch):
    from scalar_forensic.media_worker.client import DeadlineStream, WorkerUnavailable
    from scalar_forensic.media_worker.receipts import ReceiptLog

    path = tmp_path / "worker.sock"
    audit = tmp_path / "audit.jsonl"
    channels = []

    def interrupted_read(stream, size):
        channels.append(stream.channel)
        raise KeyboardInterrupt("cancel decode")

    monkeypatch.setattr(DeadlineStream, "read", interrupted_read)
    # The worker consumes the full request and closes without a response.
    thread, errors = serve(path, lambda stream, req, data: None)
    worker = client(path, receipt_log=ReceiptLog(audit, "case-a", "b" * 32))
    try:
        with pytest.raises(KeyboardInterrupt, match="cancel decode"):
            worker.decode(b"source", "preprocess")
    finally:
        thread.join(2)
    assert not errors
    assert channels[0].fileno() == -1
    with pytest.raises(WorkerUnavailable, match="invalidated"):
        worker.decode(b"source", "preprocess")
    assert [json.loads(line)["event"] for line in audit.read_text().splitlines()] == [
        "started",
        "worker_failed",
    ]


def test_invalid_image_records_decode_failure_and_allows_next_request(tmp_path):
    from scalar_forensic.media_worker.receipts import ReceiptLog

    path = tmp_path / "worker.sock"
    audit = tmp_path / "audit.jsonl"

    def reply(stream, req, data):
        if data == b"invalid":
            protocol.write_response(stream, req, "b" * 32, error="invalid_image")
        else:
            protocol.write_response(
                stream, req, "b" * 32, (protocol.RGBImage("embedding", 1, 1, b"\x01\x02\x03"),)
            )

    thread, errors = serve(path, reply, requests=2)
    worker = client(path, receipt_log=ReceiptLog(audit, "case-a", "b" * 32))
    with pytest.raises(protocol.DecodeError, match="invalid_image"):
        worker.decode(b"invalid", "preprocess")
    assert worker.decode(b"source", "preprocess").response.images[0].pixels == b"\x01\x02\x03"
    thread.join(2)
    assert not thread.is_alive()
    assert not errors
    assert [json.loads(line)["event"] for line in audit.read_text().splitlines()] == [
        "started",
        "decode_failed",
        "started",
        "decoded",
    ]
