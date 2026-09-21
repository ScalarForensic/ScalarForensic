"""Video slot rotation must respect active exports, but not a hung parser."""

import fcntl
import json
import os
from pathlib import Path

import pytest

from scalar_forensic.media_worker import supervisor
from tests.test_media_worker_supervisor import DockerBoundary


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from scalar_forensic.media_worker.video_supervisor import VideoSupervisor

    boundary = DockerBoundary()
    monkeypatch.setattr(supervisor.subprocess, "run", boundary)
    evidence = tmp_path / "case"
    evidence.mkdir()
    now = [0.0]
    manager = VideoSupervisor(
        image="local:test",
        state_dir=tmp_path / "state",
        case="CASE",
        role="renderer",
        evidence=evidence,
        client_evidence=Path("/app/evidence"),
        rotate_seconds=60,
        job_timeout=120,
        max_output_bytes=1024,
        scratch_bytes=4096,
        memory_bytes=8192,
        clock=lambda: now[0],
        probe=lambda path, generation: None,
    )
    monkeypatch.setattr(manager, "_seal_channel", lambda channel: None, raising=False)
    return manager, boundary, now


def state(manager):
    return json.loads((manager.state_dir / "current.json").read_text())


def test_busy_export_survives_rotation_but_idle_generation_is_replaced(setup):
    manager, boundary, now = setup
    with manager:
        manager.tick()
        first = state(manager)
        fd = os.open(first["lock_path"], os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            now[0] = 65
            manager.tick()
            assert state(manager)["generation"] == first["generation"]
            now[0] = 110
            manager.tick()
            assert len(boundary.containers) == 1
        finally:
            os.close(fd)
        manager.tick()
        assert state(manager)["generation"] != first["generation"]
        assert len(boundary.containers) == 1
    assert not boundary.containers


def test_failure_notice_replaces_exact_generation_even_while_job_lock_held(setup):
    manager, boundary, now = setup
    with manager:
        manager.tick()
        first = state(manager)
        fd = os.open(first["lock_path"], os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            Path(first["failure_notice"]).touch(mode=0o600)
            manager.tick()
            assert state(manager)["generation"] != first["generation"]
            assert len(boundary.containers) == 1
        finally:
            os.close(fd)


def test_external_deadline_retires_native_hang_without_python_cooperation(setup):
    manager, boundary, now = setup
    with manager:
        manager.tick()
        first = state(manager)
        fd = os.open(first["lock_path"], os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            active = Path(first["lock_path"]).parent / "active"
            active.write_text(
                json.dumps(
                    {"request_id": "a" * 32, "generation": first["generation"], "started": 0.0}
                )
            )
            active.chmod(0o600)
            manager.tick()
            now[0] = 126
            manager.tick()
            assert state(manager)["generation"] != first["generation"]
        finally:
            os.close(fd)


def test_abandoned_active_marker_reclaims_client_crash(setup):
    manager, boundary, now = setup
    with manager:
        manager.tick()
        first = state(manager)
        (Path(first["lock_path"]).parent / "active").touch(mode=0o600)
        manager.tick()
        assert state(manager)["generation"] != first["generation"]
    assert not (Path(first["lock_path"]).parent / "active").exists()


def test_back_to_back_jobs_get_individual_deadlines_without_observed_idle(setup):
    manager, boundary, now = setup
    with manager:
        manager.tick()
        first = state(manager)
        fd = os.open(first["lock_path"], os.O_RDWR)
        active = Path(first["lock_path"]).parent / "active"
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            active.write_text(
                json.dumps(
                    {"request_id": "a" * 32, "generation": first["generation"], "started": 0.0}
                )
            )
            active.chmod(0o600)
            manager.tick()
            now[0] = 110
            active.write_text(
                json.dumps(
                    {"request_id": "b" * 32, "generation": first["generation"], "started": 110.0}
                )
            )
            manager.tick()
            now[0] = 126
            manager.tick()
            assert state(manager)["generation"] == first["generation"]
        finally:
            os.close(fd)


@pytest.mark.parametrize("role", ["cpu", "renderer"])
def test_fixed_container_boundary_and_audit_policy(setup, role):
    manager, boundary, now = setup
    manager.role = role
    with manager:
        manager.tick()
        command = next(c for c in boundary.commands if c[1] == "create")
        assert command[command.index("--network") + 1] == "none"
        assert "--read-only" in command
        mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
        assert len(mounts) == 2
        assert any("dst=/evidence,readonly,bind-recursive=readonly" in m for m in mounts)
        assert not any("docker.sock" in m for m in mounts)
        scratch = [command[i + 1] for i, arg in enumerate(command) if arg == "--tmpfs"]
        assert any("/scratch:rw,noexec,nosuid,nodev,size=4096" in item for item in scratch)
        assert ("--gpus" in command) == (role == "renderer")
        current = state(manager)
        assert current["evidence_root"] == "/app/evidence"
        records = [
            json.loads(line)
            for line in (manager.state_dir / "generations.jsonl").read_text().splitlines()
        ]
        assert all(row["policy"] == current["policy"] for row in records)


def test_job_published_after_tick_timestamp_is_not_rejected(setup):
    manager, boundary, now = setup
    with manager:
        manager.tick()
        first = state(manager)
        fd = os.open(first["lock_path"], os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            active = Path(first["lock_path"]).parent / "active"
            active.write_text(
                json.dumps(
                    {"request_id": "a" * 32, "generation": first["generation"], "started": 0.1}
                )
            )
            active.chmod(0o600)
            stamps = iter([0.0, 0.2])
            manager.clock = lambda: next(stamps)
            manager.tick()
            assert state(manager)["generation"] == first["generation"]
        finally:
            os.close(fd)


def test_upload_mount_is_separate_readonly_and_publishes_app_mapping(setup, tmp_path):
    manager, boundary, now = setup
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    manager.uploads = uploads
    manager.client_uploads = Path("/case-uploads")
    with manager:
        manager.tick()
        command = next(c for c in boundary.commands if c[1] == "create")
        assert any(f"src={uploads},dst=/uploads,readonly" in arg for arg in command)
        assert state(manager)["uploads_root"] == "/case-uploads"
