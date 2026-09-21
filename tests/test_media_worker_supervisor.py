"""Host lifecycle tests: only Docker, probe and monotonic clock are substituted."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scalar_forensic.media_worker import supervisor

IMAGE = "sha256:" + "a" * 64


class DockerBoundary:
    def __init__(self):
        self.commands = []
        self.containers = {}
        self.fail_start = False
        self.image_volumes = None
        self.fail_remove = False
        self.missing_message = "No such object"

    def __call__(self, argv, **kwargs):
        assert kwargs["timeout"] == 15
        assert kwargs["check"] is True
        self.commands.append(argv)
        args = argv[1:]
        if args[:2] == ["image", "inspect"]:
            result = (
                IMAGE
                if "--format" in args
                else json.dumps([{"Id": IMAGE, "Config": {"Volumes": self.image_volumes}}])
            )
        elif args[0] == "ps":
            owner = args[args.index("--filter") + 1].split("=", 2)[2]
            result = "\n".join(
                cid
                for cid, row in self.containers.items()
                if row["Config"]["Labels"].get("sfn.media.owner") == owner
            )
        elif args[0] == "create":
            cid = f"{len(self.commands):064x}"
            labels = dict(args[i + 1].split("=", 1) for i, v in enumerate(args) if v == "--label")
            self.containers[cid] = {
                "Id": cid,
                "Image": IMAGE,
                "Name": args[args.index("--name") + 1],
                "Config": {"Labels": labels},
                "State": {"Running": False},
            }
            result = cid
        elif args[0] == "inspect":
            row = self.containers.get(args[-1]) or next(
                (row for row in self.containers.values() if row["Name"] == args[-1]), None
            )
            if row is None:
                raise subprocess.CalledProcessError(1, argv, stderr=self.missing_message)
            result = json.dumps([row])
        elif args[0] == "start":
            if self.fail_start:
                raise subprocess.TimeoutExpired(argv, 15)
            self.containers[args[-1]]["State"]["Running"] = True
            result = args[-1]
        elif args[:2] == ["rm", "--force"]:
            if self.fail_remove:
                raise subprocess.TimeoutExpired(argv, 15)
            del self.containers[args[-1]]
            result = args[-1]
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, result, "")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    docker = DockerBoundary()
    monkeypatch.setattr(supervisor.subprocess, "run", docker)
    now = [0.0]
    probes = []

    def probe(path, generation):
        probes.append((path, generation))

    manager = supervisor.Supervisor(
        image="local-worker:test",
        state_dir=tmp_path / "state",
        case="CASE",
        client_root=Path("/sfn-worker"),
        rotate_seconds=60,
        clock=lambda: now[0],
        probe=probe,
    )
    return manager, docker, now, probes


def current(manager):
    return json.loads((manager.state_dir / "current.json").read_text())


def records(manager):
    return [
        json.loads(line)
        for line in (manager.state_dir / "generations.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("setting,canonical", [("100", "100"), (" disabled ", "none")])
def test_pixel_guard_is_explicit_at_launch_and_recorded(setup, monkeypatch, setting, canonical):
    manager, docker, _, _ = setup
    monkeypatch.setenv("SFN_MAX_IMAGE_PIXELS", setting)
    manager = supervisor.Supervisor(
        image=manager.image,
        state_dir=manager.state_dir,
        case=manager.case,
        probe=lambda *args: None,
    )
    with manager:
        manager.tick()
        command = next(c for c in docker.commands if c[1] == "create")
        assert f"SFN_MAX_IMAGE_PIXELS={canonical}" in command
        expected = None if canonical == "none" else 100
        launched = [row for row in records(manager) if row["event"] == "started"]
        assert len(launched) == 1
        assert launched[0]["max_image_pixels"] == expected


def test_rotation_publishes_after_probe_and_drains_for_45_seconds(setup):
    manager, docker, now, probes = setup
    with manager:
        manager.tick()
        first = current(manager)["generation"]
        assert probes[0] == (manager.state_dir / "channels" / first / "media.sock", first)
        assert current(manager)["socket_path"] == f"/sfn-worker/channels/{first}/media.sock"
        assert current(manager)["image_id"] == IMAGE

        def next_probe(path, generation):
            assert current(manager)["generation"] == first
            assert len(docker.containers) == 2

        manager.probe = next_probe
        now[0] = 60
        manager.tick()
        second = current(manager)["generation"]
        assert second != first
        assert len(docker.containers) == 2
        now[0] = 104.99
        manager.tick()
        assert len(docker.containers) == 2
        now[0] = 105
        manager.tick()
        assert len(docker.containers) == 1
        assert (manager.state_dir / "channels" / first).is_dir()
        assert [r["event"] for r in records(manager) if r["generation"] == first] == [
            "started",
            "ready",
            "retired",
        ]
    assert not docker.containers
    assert not (manager.state_dir / "current.json").exists()
    assert len([c for c in docker.commands if c[1:3] == ["image", "inspect"]]) == 1


@pytest.mark.parametrize("failure", ["marker", "exit"])
def test_failure_removes_old_before_replacement_and_records_failure(setup, failure):
    manager, docker, now, probes = setup
    with manager:
        manager.tick()
        first = current(manager)["generation"]
        if failure == "marker":
            (manager.state_dir / "failures" / first).touch(mode=0o600)
        else:
            next(iter(docker.containers.values()))["State"]["Running"] = False

        def replacement_probe(path, generation):
            assert not (manager.state_dir / "current.json").exists()
            assert len(docker.containers) == 1

        manager.probe = replacement_probe
        manager.tick()
        assert current(manager)["generation"] != first
        assert any(r["event"] == "failed" and r["generation"] == first for r in records(manager))


def test_startup_failure_cleans_created_container_and_releases_lock(setup):
    manager, docker, now, probes = setup
    docker.fail_start = True
    with pytest.raises(subprocess.TimeoutExpired), manager:
        manager.tick()
    assert not docker.containers
    assert not (manager.state_dir / "current.json").exists()
    docker.fail_start = False
    with manager:
        manager.tick()


def test_failed_rotation_probe_exits_closed_and_cleans_both(setup):
    manager, docker, now, probes = setup
    with pytest.raises(RuntimeError, match="probe failed"), manager:
        manager.tick()
        now[0] = 60

        def fail(path, generation):
            raise RuntimeError("probe failed")

        manager.probe = fail
        manager.tick()
    assert not docker.containers
    assert not (manager.state_dir / "current.json").exists()
    assert any(r["event"] == "failed" for r in records(manager))


def test_runtime_command_restricts_worker_at_docker_boundary(setup):
    manager, docker, now, probes = setup
    with manager:
        manager.tick()
        command = next(c for c in docker.commands if c[1] == "create")
        for flag, value in {
            "--network": "none",
            "--cap-drop": "ALL",
            "--pids-limit": "256",
            "--memory": "4g",
            "--memory-swap": "4g",
            "--cpus": "4",
            "--security-opt": "no-new-privileges:true",
            "--pull": "never",
            "--user": f"{os.geteuid()}:{os.getegid()}",
            "--max-requests": "10000000",
            "--max-seconds": "150",
            "--concurrency": "4",
            "--max-queued": "4",
            "--request-timeout": "30",
        }.items():
            assert command[command.index(flag) + 1] == value
        assert "--read-only" in command
        assert "--no-healthcheck" in command
        assert command[command.index("--runtime") + 1] == "runc"
        assert "NVIDIA_VISIBLE_DEVICES=void" in command
        assert command.count("--mount") == 1
        assert command[command.index("--mount") + 1] == (
            f"type=bind,src={manager.state_dir}/channels/{current(manager)['generation']},dst=/ipc"
        )
        assert "--gpus" not in command
        assert IMAGE in command
        assert all(r["image_id"] == IMAGE and r["policy"] for r in records(manager))


def test_state_directory_is_private_and_exclusively_locked(setup):
    manager, docker, now, probes = setup
    with manager:
        assert manager.state_dir.stat().st_mode & 0o777 == 0o700
        other = supervisor.Supervisor(image="x", state_dir=manager.state_dir, case="CASE")
        with pytest.raises((BlockingIOError, RuntimeError)), other:
            pass


@pytest.mark.parametrize(
    "target",
    [
        "state",
        "channels",
        "failures",
        "current.json",
        "generations.jsonl",
        "owner",
        "supervisor.lock",
    ],
)
def test_rejects_symlink_state_targets(setup, tmp_path, target):
    manager, docker, now, probes = setup
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    if target == "state":
        manager.state_dir.symlink_to(outside)
    else:
        manager.state_dir.mkdir(mode=0o700)
        (manager.state_dir / target).symlink_to(outside)
    with pytest.raises((OSError, ValueError, RuntimeError)), manager:
        manager.tick()
    assert not docker.containers


def test_requires_nonroot_and_rotation_at_least_60(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        supervisor.Supervisor(image="x", state_dir=tmp_path, case="CASE", rotate_seconds=59)
    monkeypatch.setattr(supervisor.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError):
        supervisor.Supervisor(image="x", state_dir=tmp_path, case="CASE")


@pytest.mark.parametrize("message", ["No such object", "error: no such object"])
def test_disappeared_container_replaced(setup, message):
    manager, docker, now, probes = setup
    docker.missing_message = message
    with manager:
        manager.tick()
        first = current(manager)["generation"]
        docker.containers.clear()
        manager.tick()
        assert current(manager)["generation"] != first


def test_restart_cleans_only_owned_orphans_and_records_original_image(setup):
    manager, docker, now, probes = setup
    with manager:
        owner = manager.owner
    old_image = "sha256:" + "b" * 64
    old_generation = "c" * 32
    for cid, token in [("d" * 64, owner), ("e" * 64, "f" * 32)]:
        docker.containers[cid] = {
            "Id": cid,
            "Name": "old-" + cid,
            "Image": old_image,
            "Config": {
                "Labels": {"sfn.media.owner": token, "sfn.media.generation": old_generation}
            },
            "State": {"Running": True},
        }
    with manager:
        assert set(docker.containers) == {"e" * 64}
        assert records(manager)[0]["image_id"] == old_image
        assert records(manager)[0]["event"] == "failed"
        assert "max_image_pixels" not in records(manager)[0]


def test_ready_probe_decodes_fixed_synthetic_png(tmp_path, monkeypatch):
    import threading

    from scalar_forensic.media_worker.server import bound_listener, serve

    generation = "f" * 32
    path = tmp_path / "media.sock"
    with bound_listener(path) as listener:
        thread = threading.Thread(
            target=serve,
            args=(listener,),
            kwargs={
                "generation": generation,
                "max_requests": 1,
                "max_seconds": 5,
            },
        )
        thread.start()
        try:
            supervisor.ready_probe(path, generation)
        finally:
            thread.join(6)
        assert not thread.is_alive()


def test_image_declared_volumes_are_rejected_before_container_creation(setup):
    manager, docker, now, probes = setup
    docker.image_volumes = {"/app/data": {}}
    with pytest.raises(ValueError, match="volumes"), manager:
        manager.tick()
    assert not any(c[1] == "create" for c in docker.commands)


def test_audit_failure_still_removes_containers(setup, monkeypatch):
    manager, docker, now, probes = setup
    with pytest.raises(RuntimeError, match="cleanup failed"), manager:
        manager.tick()
        (manager.state_dir / "generations.jsonl").chmod(0o400)
    assert not docker.containers
    assert not (manager.state_dir / "current.json").exists()


def test_failed_removal_does_not_claim_retirement(setup):
    manager, docker, now, probes = setup
    with pytest.raises(RuntimeError, match="cleanup failed"), manager:
        manager.tick()
        docker.fail_remove = True
    assert not any(row["event"] == "retired" for row in records(manager))
    assert records(manager)[-1]["event"] == "failed"
    docker.fail_remove = False
    with supervisor.Supervisor(image="x", state_dir=manager.state_dir, case="CASE"):
        assert not docker.containers
