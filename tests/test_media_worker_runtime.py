import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scalar_forensic.media_worker.client import WorkerUnavailable


def runtime_module():
    from scalar_forensic.media_worker import runtime

    return runtime


def state(tmp_path, generation=None):
    runtime = runtime_module()
    generation = generation or uuid.uuid4().hex
    return runtime.RuntimeState(
        generation=generation,
        socket_path=tmp_path / "channels" / generation / "media.sock",
        case_id="case-a",
        image_id="sha256:" + "a" * 64,
        failure_notice=tmp_path / "failures" / generation,
    )


def test_state_rotation_publishes_new_identity_privately(tmp_path):
    runtime = runtime_module()
    target = tmp_path / "current.json"
    first, second = state(tmp_path), state(tmp_path)
    runtime.publish_state(target, first)
    assert runtime.read_state(target, "case-a") == first
    assert target.stat().st_mode & 0o777 == 0o600
    runtime.publish_state(target, second)
    assert runtime.read_state(target, "case-a") == second


def test_unavailable_or_wrong_case_state_fails_closed(tmp_path):
    runtime = runtime_module()
    target = tmp_path / "current.json"
    with pytest.raises(WorkerUnavailable):
        runtime.read_state(target, "case-a")
    runtime.publish_state(target, state(tmp_path))
    with pytest.raises(WorkerUnavailable):
        runtime.read_state(target, "case-b")


def test_state_rejects_worker_writable_failure_notice(tmp_path):
    runtime = runtime_module()
    current = state(tmp_path)
    unsafe = runtime.RuntimeState(
        generation=current.generation,
        socket_path=current.socket_path,
        case_id=current.case_id,
        image_id=current.image_id,
        failure_notice=current.socket_path.parent / "failed",
    )
    with pytest.raises(ValueError):
        runtime.publish_state(tmp_path / "current.json", unsafe)


def test_state_reader_bounds_input(tmp_path):
    runtime = runtime_module()
    target = tmp_path / "current.json"
    target.write_bytes(b" " * 8193)
    target.chmod(0o600)
    with pytest.raises(WorkerUnavailable):
        runtime.read_state(target, "case-a")


def configure(tmp_path, monkeypatch):
    runtime = runtime_module()
    target = tmp_path / "current.json"
    current = state(tmp_path)
    runtime.publish_state(target, current)
    monkeypatch.setenv("SFN_MEDIA_WORKER_STATE", str(target))
    monkeypatch.setenv("SFN_MEDIA_WORKER_CASE", "case-a")
    monkeypatch.setenv("SFN_MEDIA_WORKER_RECEIPTS", str(tmp_path / "receipts.jsonl"))
    return target, current


def test_managed_client_follows_rotation_and_records_runtime(tmp_path, monkeypatch):
    from scalar_forensic import media_access
    from scalar_forensic.media_worker.protocol import Request

    target, current = configure(tmp_path, monkeypatch)
    client = media_access.configured_worker()
    assert client.generation == current.generation
    client.receipt_log.append(Request("b" * 32, "c" * 64, "exif", 331, 4), "started")
    row = json.loads((tmp_path / "receipts.jsonl").read_text())
    assert row["runtime"] == {"image_id": "sha256:" + "a" * 64, "policy": "sfn-image-worker-v1"}
    replacement = state(tmp_path)
    runtime_module().publish_state(target, replacement)
    assert media_access.configured_worker().generation == replacement.generation


def test_managed_failure_notifies_supervisor_without_retry(tmp_path, monkeypatch):
    from scalar_forensic import media_access

    _, current = configure(tmp_path, monkeypatch)
    current.failure_notice.parent.mkdir()
    with pytest.raises(WorkerUnavailable):
        media_access.configured_worker().decode(b"invalid", "exif")
    assert current.failure_notice.is_file()
    rows = [json.loads(row) for row in (tmp_path / "receipts.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["started", "worker_failed"]


def test_managed_mode_cannot_fall_back_or_mix_static_configuration(tmp_path, monkeypatch):
    from scalar_forensic import media_access

    target, _ = configure(tmp_path, monkeypatch)
    target.unlink()
    with pytest.raises(WorkerUnavailable):
        media_access.configured_worker()
    monkeypatch.setenv("SFN_MEDIA_WORKER_SOCKET", "/static/socket")
    with pytest.raises(ValueError):
        media_access.configured_worker()


@pytest.fixture
def socket_root():
    with TemporaryDirectory(prefix="sfn-runtime-") as root:
        yield Path(root)


def test_queued_batch_items_use_replacement_generation(socket_root, monkeypatch):
    import socket

    from scalar_forensic import media_access
    from scalar_forensic.media_worker import protocol

    target, first = configure(socket_root, monkeypatch)
    second = state(socket_root)
    first.failure_notice.parent.mkdir()
    listeners = []
    for current in (first, second):
        current.socket_path.parent.mkdir(parents=True)
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(str(current.socket_path))
        listener.listen(1)
        listener.settimeout(3)
        listeners.append(listener)

    def serve(listener, current, pixels):
        with listener:
            channel, _ = listener.accept()
            with channel, channel.makefile("rwb", buffering=0) as stream:
                request, _ = protocol.read_request(stream)
                if current == first:
                    runtime_module().publish_state(target, second)
                protocol.write_response(
                    stream,
                    request,
                    current.generation,
                    [protocol.RGBImage("embedding", 1, 1, pixels)],
                    {},
                )
                channel.shutdown(socket.SHUT_WR)

    monkeypatch.setattr(
        media_access, "ThreadPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=1)
    )
    with ThreadPoolExecutor(max_workers=2) as servers:
        pending = [
            servers.submit(serve, listener, current, pixels)
            for listener, current, pixels in zip(
                listeners, (first, second), (b"\x01\x02\x03", b"\x04\x05\x06"), strict=True
            )
        ]
        images = media_access.preprocess_batch([b"one", b"two"], 1)
        assert all(not isinstance(item, Exception) for item in images)
        assert [item.tobytes() for item in images] == [b"\x01\x02\x03", b"\x04\x05\x06"]
        for future in pending:
            future.result()
