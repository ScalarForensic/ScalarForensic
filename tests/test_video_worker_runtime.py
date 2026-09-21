"""Managed video slots fail closed and keep control paths outside parser storage."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scalar_forensic.media_worker.client import WorkerUnavailable


def example(root):
    from scalar_forensic.media_worker.video_runtime import VideoRuntime

    return VideoRuntime(
        generation="a" * 32,
        socket_path=root / "channels" / ("a" * 32) / "video.sock",
        case_id="case",
        image_id="sha256:" + "b" * 64,
        policy="sfn-video-cpu-v1",
        failure_notice=root / "failures" / ("a" * 32),
        lock_path=root / "job.lock",
        evidence_root=Path("/evidence"),
        max_output_bytes=1024,
        job_timeout=60,
    )


def test_private_state_roundtrip_and_wrong_case_refusal(tmp_path):
    from scalar_forensic.media_worker.video_runtime import publish_state, read_state

    state = example(tmp_path)
    path = tmp_path / "current.json"
    publish_state(path, state)
    assert read_state(path, "case") == state
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(WorkerUnavailable):
        read_state(path, "different")


@pytest.mark.parametrize("field", ["failure_notice", "lock_path"])
def test_parser_writable_control_path_refused(tmp_path, field):
    from scalar_forensic.media_worker.video_runtime import publish_state

    state = example(tmp_path)
    bad = replace(state, **{field: state.socket_path.parent / "control"})
    with pytest.raises(ValueError):
        publish_state(tmp_path / "current.json", bad)


def test_public_state_and_unknown_policy_refused(tmp_path):
    from scalar_forensic.media_worker.video_runtime import publish_state, read_state

    state = example(tmp_path)
    path = tmp_path / "current.json"
    with pytest.raises(ValueError):
        publish_state(path, replace(state, policy="trust-me"))
    publish_state(path, state)
    path.chmod(0o644)
    with pytest.raises(WorkerUnavailable):
        read_state(path, "case")


def test_nonfinite_deadline_and_oversized_state_refused(tmp_path):
    from scalar_forensic.media_worker.video_runtime import publish_state, read_state

    path = tmp_path / "current.json"
    with pytest.raises(ValueError):
        publish_state(path, replace(example(tmp_path), job_timeout=float("nan")))
    path.write_bytes(b" " * 8193)
    path.chmod(0o600)
    with pytest.raises(WorkerUnavailable):
        read_state(path, "case")


def test_future_protocol_version_is_not_silently_accepted(tmp_path):
    from scalar_forensic.media_worker.video_runtime import publish_state, read_state

    path = tmp_path / "current.json"
    publish_state(path, example(tmp_path))
    row = json.loads(path.read_text())
    row["version"] = 2
    path.write_text(json.dumps(row))
    with pytest.raises(WorkerUnavailable):
        read_state(path, "case")


def test_explicit_upload_mapping_survives_state_publication(tmp_path):
    from scalar_forensic.media_worker.video_runtime import publish_state, read_state

    state = replace(example(tmp_path), uploads_root=Path("/case-uploads"))
    publish_state(tmp_path / "current.json", state)
    assert read_state(tmp_path / "current.json", "case").uploads_root == Path("/case-uploads")
