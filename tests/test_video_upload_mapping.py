"""Video queries use only an explicitly configured read-only upload root."""

import os
from dataclasses import replace

import pytest

from scalar_forensic.media_worker.video_protocol import ProtocolError


def test_worker_upload_mapping_and_escape_refusal(tmp_path):
    from scalar_forensic.media_worker.video_server import _source

    evidence, uploads = tmp_path / "evidence", tmp_path / "uploads"
    evidence.mkdir()
    uploads.mkdir()
    source = uploads / "query.mp4"
    source.write_bytes(b"synthetic bytes; no decoder needed for path selection")
    assert _source(evidence, "__uploads__/query.mp4", uploads) == source
    with pytest.raises(ProtocolError):
        _source(evidence, "__uploads__/query.mp4", None)
    (uploads / "escape").symlink_to(evidence, target_is_directory=True)
    (evidence / "outside.mp4").write_bytes(b"outside uploads")
    with pytest.raises(ProtocolError):
        _source(evidence, "__uploads__/escape/outside.mp4", uploads)


def test_managed_session_directories_grant_parser_group_read_only(tmp_path, monkeypatch):
    from scalar_forensic.media_worker.video_runtime import publish_state
    from scalar_forensic.web import session
    from tests.test_video_worker_runtime import example

    uploads = tmp_path / "uploads"
    uploads.mkdir(mode=0o700)
    state_path = tmp_path / "current.json"
    publish_state(state_path, replace(example(tmp_path), uploads_root=uploads))
    monkeypatch.setenv("SFN_VIDEO_CPU_STATE", str(state_path))
    monkeypatch.setenv("SFN_VIDEO_WORKER_CASE", "case")
    monkeypatch.setattr(session, "_temp_root", uploads)
    monkeypatch.setattr(session, "_owned_roots", {})
    try:
        session.initialize_temp_root(uploads)
        directory = session.session_temp_dir()
        for path in (uploads, directory.parent, directory):
            assert path.stat().st_mode & 0o777 == 0o750
        uploaded = directory / "query.mp4"
        uploaded.write_bytes(b"opaque encoded bytes")
        session.prepare_upload_permissions(uploaded)
        assert uploaded.stat().st_mode & 0o777 == 0o640
    finally:
        for _, fd in session._owned_roots.values():
            os.close(fd)


def test_managed_web_refuses_unmapped_upload_root(tmp_path, monkeypatch):
    from scalar_forensic.media_worker.video_runtime import publish_state
    from scalar_forensic.web import session
    from tests.test_video_worker_runtime import example

    state_path = tmp_path / "current.json"
    publish_state(state_path, example(tmp_path))
    monkeypatch.setenv("SFN_VIDEO_CPU_STATE", str(state_path))
    monkeypatch.setenv("SFN_VIDEO_WORKER_CASE", "case")
    monkeypatch.setattr(session, "_temp_root", tmp_path / "uploads")
    with pytest.raises(RuntimeError, match="upload"):
        session.initialize_temp_root(tmp_path / "uploads")
