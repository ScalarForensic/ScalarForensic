"""Trusted host CLI for one case's ready worker and optional draining worker."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import math
import os
import re
import signal
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from scalar_forensic.media_worker.client import WorkerClient
from scalar_forensic.media_worker.runtime import POLICY, RuntimeState, publish_state

OWNER_LABEL = "sfn.media.owner"
GENERATION_LABEL = "sfn.media.generation"
# Fixed synthetic 1x1 PNG: readiness never submits case material.
PROBE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII="
)


def docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


def ready_probe(path: Path, generation: str) -> None:
    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("worker readiness socket deadline")
        time.sleep(0.05)
    WorkerClient(path, generation, timeout=max(0.001, deadline - time.monotonic())).decode(
        PROBE_PNG, "native"
    )


def private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError("state directories must be private, owned directories")


def private_file(path: Path, flags: int) -> int:
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
    ):
        os.close(fd)
        raise ValueError("state files must be private, singly linked owned files")
    return fd


@dataclass
class Generation:
    generation: str
    name: str
    container_id: str | None = None
    ready_at: float = 0
    drain_until: float = 0
    image_id: str = ""


class Supervisor:
    def __init__(
        self,
        *,
        image: str,
        state_dir: Path,
        case: str,
        client_root: Path | None = None,
        rotate_seconds: float = 3600,
        clock=time.monotonic,
        probe=ready_probe,
    ):
        if os.geteuid() == 0:
            raise ValueError("supervisor requires a nonroot operator")
        if not math.isfinite(rotate_seconds) or not 60 <= rotate_seconds <= 86310:
            raise ValueError("rotation must be between 60 and 86310 seconds")
        if not image or image.startswith("-") or not case or len(case) > 256:
            raise ValueError("image and bounded case identity are required")
        self.state_dir = Path(os.path.abspath(state_dir))
        if "," in str(self.state_dir):
            raise ValueError("state path cannot contain Docker mount separators")
        self.client_root = Path(client_root) if client_root is not None else self.state_dir
        if not self.client_root.is_absolute():
            raise ValueError("client root must be absolute")
        self.image, self.case = image, case
        from scalar_forensic.media_decode import _max_image_pixels_from_env

        self.max_image_pixels = _max_image_pixels_from_env()
        self.rotate_seconds, self.clock, self.probe = rotate_seconds, clock, probe
        self.lock_fd = None
        self.owner = ""
        self.image_id = ""
        self.ready: Generation | None = None
        self.draining: Generation | None = None
        self.owned: list[Generation] = []

    def __enter__(self):
        # Reject symlink ancestors as well as the directory itself.
        if self.state_dir.resolve() != self.state_dir:
            raise ValueError("state directory must not traverse symlinks")
        private_directory(self.state_dir)
        fd = private_file(self.state_dir / "supervisor.lock", os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        self.lock_fd = fd
        try:
            for name in ("channels", "failures"):
                private_directory(self.state_dir / name)
            for name in ("current.json", "generations.jsonl", "owner"):
                path = self.state_dir / name
                if os.path.lexists(path):
                    os.close(private_file(path, os.O_RDONLY))
            owner_path = self.state_dir / "owner"
            if not owner_path.exists():
                with os.fdopen(
                    private_file(owner_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "w"
                ) as handle:
                    handle.write(uuid.uuid4().hex)
                    handle.flush()
                    os.fsync(handle.fileno())
            self.owner = owner_path.read_text()
            if re.fullmatch(r"[0-9a-f]{32}", self.owner) is None:
                raise ValueError("invalid supervisor ownership token")
            images = json.loads(docker("image", "inspect", self.image))
            if len(images) != 1 or images[0]["Config"].get("Volumes"):
                raise ValueError("worker image must not declare implicit volumes")
            self.image_id = images[0]["Id"]
            if re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_id) is None:
                raise ValueError("image must resolve to an immutable local image ID")
            self.unpublish()
            # Recover only containers carrying this private directory's durable token.
            for cid in docker(
                "ps",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label={OWNER_LABEL}={self.owner}",
            ).splitlines():
                info = self.inspect(cid)
                generation = info["Config"]["Labels"].get(GENERATION_LABEL, "")
                if re.fullmatch(r"[0-9a-f]{32}", generation) is None:
                    raise RuntimeError("invalid orphan generation label")
                old = Generation(generation, info["Name"], cid, image_id=info["Image"])
                self.owned.append(old)
                self.retire(old, "failed", "supervisor restart")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def inspect(self, reference: str) -> dict:
        rows = json.loads(docker("inspect", reference))
        if len(rows) != 1:
            raise RuntimeError("ambiguous Docker identity")
        info = rows[0]
        if (
            re.fullmatch(r"[0-9a-f]{64}", info["Id"]) is None
            or info["Config"]["Labels"].get(OWNER_LABEL) != self.owner
        ):
            raise RuntimeError("container ownership mismatch")
        return info

    def record(self, generation: Generation, event: str, reason: str = "") -> None:
        row = {
            "generation": generation.generation,
            "image_id": generation.image_id,
            "policy": POLICY,
            "event": event,
            "at": datetime.now(UTC).isoformat(),
            "reason": reason[:160],
        }
        if event == "started":
            # Recovered orphans may have used another supervisor's settings.
            # Attribute the guard at launch, and join later events by generation.
            row["max_image_pixels"] = self.max_image_pixels
        with os.fdopen(
            private_file(
                self.state_dir / "generations.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT
            ),
            "a",
        ) as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def unpublish(self):
        path = self.state_dir / "current.json"
        if os.path.lexists(path):
            os.close(private_file(path, os.O_RDONLY))
            path.unlink()

    def remove(self, generation: Generation):
        reference = generation.container_id or generation.name
        try:
            info = self.inspect(reference)
        except subprocess.CalledProcessError as exc:
            if (
                "no such object" not in (exc.stderr or "").lower()
                and "no such container" not in (exc.stderr or "").lower()
            ):
                raise
        else:
            if info["Config"]["Labels"].get(GENERATION_LABEL) != generation.generation:
                raise RuntimeError("container generation mismatch")
            docker("rm", "--force", info["Id"])
        self.owned.remove(generation)

    def retire(self, generation: Generation, event: str, reason: str):
        if event == "failed":
            try:
                self.record(generation, event, reason)
            finally:
                self.remove(generation)
            return
        try:
            self.remove(generation)
        except Exception:
            self.record(generation, "failed", "container removal failed")
            raise
        self.record(generation, event, reason)

    def start(self) -> Generation:
        gen = uuid.uuid4().hex
        pixel_limit = "none" if self.max_image_pixels is None else str(self.max_image_pixels)
        channel = self.state_dir / "channels" / gen
        channel.mkdir(mode=0o700)  # Never reuse an IPC directory or generation.
        worker = Generation(gen, f"sfn-media-{self.owner}-{gen}", image_id=self.image_id)
        self.owned.append(worker)
        try:
            cid = docker(
                "create",
                "--name",
                worker.name,
                "--label",
                f"{OWNER_LABEL}={self.owner}",
                "--label",
                f"{GENERATION_LABEL}={gen}",
                "--pull",
                "never",
                "--network",
                "none",
                "--read-only",
                "--no-healthcheck",
                "--runtime",
                "runc",
                "--user",
                f"{os.geteuid()}:{os.getegid()}",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--pids-limit",
                "256",
                "--memory",
                "4g",
                "--memory-swap",
                "4g",
                "--cpus",
                "4",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                f"SFN_MAX_IMAGE_PIXELS={pixel_limit}",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
                "--log-driver",
                "local",
                "--log-opt",
                "max-size=10m",
                "--log-opt",
                "max-file=2",
                "--mount",
                f"type=bind,src={channel},dst=/ipc",
                "--entrypoint",
                "python",
                self.image_id,
                "-m",
                "scalar_forensic.media_worker.server",
                "--socket",
                "/ipc/media.sock",
                "--generation",
                gen,
                "--concurrency",
                "4",
                "--max-queued",
                "4",
                "--request-timeout",
                "30",
                "--max-requests",
                "10000000",
                "--max-seconds",
                f"{self.rotate_seconds + 90:g}",
            )
            if re.fullmatch(r"[0-9a-f]{64}", cid) is None:
                raise RuntimeError("invalid created container ID")
            worker.container_id = cid
            # Confirm identity before any operation on the returned identifier.
            info = self.inspect(cid)
            if info["Config"]["Labels"].get(GENERATION_LABEL) != gen:
                raise RuntimeError("created generation mismatch")
            docker("start", cid)
            self.record(worker, "started")
            self.probe(channel / "media.sock", gen)
            worker.ready_at = self.clock()
            self.record(worker, "ready")
            return worker
        except BaseException:
            self.record(worker, "failed", "worker startup or readiness failed")
            raise

    def failed(self, worker: Generation) -> bool:
        if os.path.lexists(self.state_dir / "failures" / worker.generation):
            return True
        try:
            return not self.inspect(worker.container_id)["State"]["Running"]
        except subprocess.CalledProcessError as exc:
            if (
                "no such object" in (exc.stderr or "").lower()
                or "no such container" in (exc.stderr or "").lower()
            ):
                return True
            raise

    def tick(self):
        now = self.clock()
        if self.draining is not None:
            if self.failed(self.draining):
                self.retire(self.draining, "failed", "draining worker failed")
                self.draining = None
            elif now >= self.draining.drain_until:
                self.retire(self.draining, "retired", "drain complete")
                self.draining = None
        if self.ready is not None and self.failed(self.ready):
            self.unpublish()
            self.retire(self.ready, "failed", "worker failure notice or unexpected exit")
            self.ready = None
        if self.ready is None or now >= self.ready.ready_at + self.rotate_seconds:
            if self.draining is not None:
                self.retire(self.draining, "retired", "replacement requires draining slot")
                self.draining = None
            new = self.start()
            state = RuntimeState(
                generation=new.generation,
                socket_path=self.client_root / "channels" / new.generation / "media.sock",
                case_id=self.case,
                image_id=self.image_id,
                failure_notice=self.client_root / "failures" / new.generation,
            )
            publish_state(self.state_dir / "current.json", state)
            if self.ready is not None:
                self.draining = self.ready
                self.draining.drain_until = self.clock() + 45
            self.ready = new

    def __exit__(self, exc_type, exc, traceback):
        if self.lock_fd is None:
            return
        errors = []
        try:
            try:
                self.unpublish()
            except Exception as error:
                errors.append(error)
            for worker in list(self.owned):
                try:
                    self.retire(worker, "retired", "supervisor shutdown")
                except Exception as error:
                    errors.append(error)
        finally:
            os.close(self.lock_fd)
            self.lock_fd = None
            self.ready = self.draining = None
        if errors:
            raise RuntimeError("supervisor cleanup failed; inspect owned containers") from errors[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--client-root", type=Path)
    parser.add_argument("--rotate-seconds", type=float, default=3600)
    args = vars(parser.parse_args(argv))

    def terminate(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with Supervisor(**args) as manager:
            while True:
                manager.tick()
                time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        parser.exit(
            1, "media worker supervisor failed; inspect generation history and owned containers\n"
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
