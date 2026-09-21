"""Host-owned lifecycle for one CPU parser or NVENC renderer slot.

Run separate slots for CPU work, chunks and full exports. Only this host process
has Docker access. The application holds job.lock; a parser never sees that lock
or its failure notices. Normal rotation waits for idle, failure does not.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import socket
import stat
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from scalar_forensic.media_worker.client import DeadlineStream
from scalar_forensic.media_worker.supervisor import (
    GENERATION_LABEL,
    OWNER_LABEL,
    Generation,
    Supervisor,
    docker,
    private_directory,
    private_file,
)
from scalar_forensic.media_worker.video_runtime import POLICIES, VideoRuntime, publish_state


def ready_probe(path: Path, generation: str) -> None:
    from scalar_forensic.media_worker.video_protocol import (
        recv_packet,
        send_packet,
        validate_event,
    )

    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("video readiness deadline")
        time.sleep(0.05)
    request_id = uuid.uuid4().hex
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(max(0.001, deadline - time.monotonic()))
        channel.connect(str(path))
        stream = DeadlineStream(channel, deadline)
        send_packet(
            stream,
            {
                "version": 1,
                "request_id": request_id,
                "generation": generation,
                "operation": "ping",
                "source": None,
                "parameters": {},
            },
        )
        header, payload = recv_packet(stream)
        validate_event(header, payload)
        if (
            header["request_id"] != request_id
            or header["generation"] != generation
            or header["type"] != "result"
            or header["body"] != {"ready": True}
            or stream.read(1)
        ):
            raise RuntimeError("invalid video readiness response")


class VideoSupervisor(Supervisor):
    """Reuse identity/recovery helpers, not the image worker's limits or draining."""

    def __init__(
        self,
        *,
        role: str,
        evidence: Path,
        client_evidence: Path | None = None,
        uploads: Path | None = None,
        client_uploads: Path | None = None,
        job_timeout: int = 3600,
        max_output_bytes: int = 4 * 1024**3,
        scratch_bytes: int = 8 * 1024**3,
        memory_bytes: int = 12 * 1024**3,
        cpus: float = 4,
        gpu: str = "0",
        probe=ready_probe,
        **kwargs,
    ):
        super().__init__(probe=probe, **kwargs)
        evidence = evidence.resolve(strict=True)
        if role not in POLICIES or not evidence.is_dir() or evidence == Path("/"):
            raise ValueError("a video role and scoped evidence directory are required")
        if "," in str(evidence) or not re.fullmatch(r"[0-9]+", gpu):
            raise ValueError("invalid evidence mount or GPU index")
        if not (1 <= job_timeout <= 86400 and 0 < max_output_bytes < scratch_bytes < memory_bytes):
            raise ValueError("output, scratch, memory and time limits must be bounded")
        if not 0 < cpus <= 256:
            raise ValueError("invalid CPU limit")
        self.role, self.evidence = role, evidence
        self.client_evidence = client_evidence or evidence
        if not self.client_evidence.is_absolute():
            raise ValueError("application evidence root must be absolute")
        self.uploads = uploads.resolve(strict=True) if uploads is not None else None
        self.client_uploads = client_uploads or self.uploads
        if self.uploads is not None and (
            not self.uploads.is_dir()
            or self.uploads == Path("/")
            or "," in str(self.uploads)
            or not self.client_uploads.is_absolute()
        ):
            raise ValueError("invalid scoped upload directory")
        if self.uploads is None and client_uploads is not None:
            raise ValueError("application upload mapping needs a host upload directory")
        self.job_timeout, self.max_output_bytes = job_timeout, max_output_bytes
        self.scratch_bytes, self.memory_bytes, self.cpus = scratch_bytes, memory_bytes, cpus
        self.gpu = gpu
        self.control = self.state_dir / "control"
        self.job_fd = None
        self.worker_uid = 65534
        if os.geteuid() == self.worker_uid:
            raise ValueError("supervisor and video parser must use different UIDs")

    def __enter__(self):
        super().__enter__()
        try:
            private_directory(self.control)
            self.job_fd = private_file(self.control / "job.lock", os.O_RDWR | os.O_CREAT)
            self.clear_active()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def clear_active(self):
        path = self.control / "active"
        if os.path.lexists(path):
            os.close(private_file(path, os.O_RDONLY))
            path.unlink()

    def record(self, generation: Generation, event: str, reason: str = "") -> None:
        row = {
            "generation": generation.generation,
            "image_id": generation.image_id,
            "policy": POLICIES[self.role],
            "event": event,
            "at": datetime.now(UTC).isoformat(),
            "reason": reason[:160],
        }
        with os.fdopen(
            private_file(
                self.state_dir / "generations.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT
            ),
            "a",
        ) as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _seal_channel(self, channel: Path):
        """Seal the host-owned directory before accepting any evidence requests."""
        node = channel / "video.sock"
        info = node.lstat()
        if (
            list(channel.iterdir()) != [node]
            or not stat.S_ISSOCK(info.st_mode)
            or info.st_uid != self.worker_uid
            or info.st_gid != os.getegid()
            or info.st_mode & 0o777 != 0o660
        ):
            raise RuntimeError("unexpected video IPC contents or ownership")
        channel.chmod(0o550)

    def remove(self, generation: Generation):
        # Only reclaim the socket after exact owned-container removal is confirmed.
        super().remove(generation)
        channel = self.state_dir / "channels" / generation.generation
        node = channel / "video.sock"
        if channel.exists():
            channel.chmod(0o700)
            if node.exists() and stat.S_ISSOCK(node.lstat().st_mode):
                node.unlink()
            if not list(channel.iterdir()):
                channel.rmdir()

    def _overdue(self) -> bool:
        try:
            with os.fdopen(private_file(self.control / "active", os.O_RDONLY), "rb") as stream:
                row = json.loads(stream.read(4097))
        except FileNotFoundError:
            return False  # Slot acquired, request not yet submitted.
        now = self.clock()  # The client may publish after tick sampled its clock.
        if (
            type(row) is not dict
            or set(row) != {"request_id", "generation", "started"}
            or type(row["request_id"]) is not str
            or re.fullmatch(r"[0-9a-f]{32}", row["request_id"]) is None
            or row["generation"] != self.ready.generation
            or type(row["started"]) not in (int, float)
            or not math.isfinite(row["started"])
            or not 0 <= row["started"] <= now
        ):
            raise ValueError("invalid trusted active video request")
        return now > row["started"] + self.job_timeout + 5

    def start(self) -> Generation:
        gen = uuid.uuid4().hex
        channel = self.state_dir / "channels" / gen
        channel.mkdir(mode=0o730)
        channel.chmod(0o730)  # Preserve bootstrap group write despite the host umask.
        worker = Generation(gen, f"sfn-video-{self.owner}-{gen}", image_id=self.image_id)
        self.owned.append(worker)
        args = [
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
            "--user",
            f"{self.worker_uid}:{os.getegid()}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "512",
            "--memory",
            str(self.memory_bytes),
            "--memory-swap",
            str(self.memory_bytes),
            "--cpus",
            str(self.cpus),
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=268435456,mode=1777",
            "--tmpfs",
            f"/scratch:rw,noexec,nosuid,nodev,size={self.scratch_bytes},mode=1777",
            "--log-driver",
            "local",
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=2",
            "--mount",
            f"type=bind,src={channel},dst=/ipc",
            "--mount",
            f"type=bind,src={self.evidence},dst=/evidence,readonly,"
            "bind-recursive=readonly,bind-propagation=rprivate",
        ]
        if self.uploads is not None:
            args += [
                "--mount",
                f"type=bind,src={self.uploads},dst=/uploads,readonly,"
                "bind-recursive=readonly,bind-propagation=rprivate",
            ]
        if self.role == "renderer":
            args += [
                "--gpus",
                f"device={self.gpu}",
                "--env",
                "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility",
            ]
        else:
            args += ["--runtime", "runc", "--env", "NVIDIA_VISIBLE_DEVICES=void"]
        args += [
            "--entrypoint",
            "python",
            self.image_id,
            "-m",
            "scalar_forensic.media_worker.video_server",
            "--socket",
            "/ipc/video.sock",
            "--generation",
            gen,
            "--role",
            self.role,
            "--evidence",
            "/evidence",
            "--scratch",
            "/scratch",
            "--max-output-bytes",
            str(self.max_output_bytes),
        ]
        if self.uploads is not None:
            args += ["--uploads", "/uploads"]
        try:
            worker.container_id = docker(*args)
            info = self.inspect(worker.container_id)
            if info["Config"]["Labels"].get(GENERATION_LABEL) != gen:
                raise RuntimeError("created video generation mismatch")
            docker("start", worker.container_id)
            self.record(worker, "started")
            self.probe(channel / "video.sock", gen)
            self._seal_channel(channel)
            worker.ready_at = self.clock()
            self.record(worker, "ready")
            return worker
        except BaseException:
            self.record(worker, "failed", "video startup or readiness failed")
            raise

    def tick(self):
        now = self.clock()
        idle = False
        try:
            try:
                fcntl.flock(self.job_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                idle = True
            except BlockingIOError:
                pass
            overdue = self.ready is not None and not idle and self._overdue()
            abandoned = idle and os.path.lexists(self.control / "active")
            if self.ready is not None and (self.failed(self.ready) or overdue or abandoned):
                self.unpublish()
                self.retire(self.ready, "failed", "video failure, abandoned client or deadline")
                self.ready = None
                self.clear_active()
            if self.ready is not None and idle and now >= self.ready.ready_at + self.rotate_seconds:
                self.unpublish()
                self.retire(self.ready, "retired", "idle rotation")
                self.ready = None
            if self.ready is None:
                new = self.start()
                publish_state(
                    self.state_dir / "current.json",
                    VideoRuntime(
                        generation=new.generation,
                        socket_path=self.client_root / "channels" / new.generation / "video.sock",
                        case_id=self.case,
                        image_id=self.image_id,
                        policy=POLICIES[self.role],
                        failure_notice=self.client_root / "failures" / new.generation,
                        lock_path=self.client_root / "control" / "job.lock",
                        evidence_root=self.client_evidence,
                        max_output_bytes=self.max_output_bytes,
                        job_timeout=self.job_timeout,
                        uploads_root=self.client_uploads,
                    ),
                )
                self.ready = new
        finally:
            if idle:
                fcntl.flock(self.job_fd, fcntl.LOCK_UN)

    def __exit__(self, exc_type, exc, traceback):
        try:
            super().__exit__(exc_type, exc, traceback)
        finally:
            if self.job_fd is not None:
                os.close(self.job_fd)
                self.job_fd = None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--role", choices=POLICIES, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--client-root", type=Path)
    parser.add_argument("--client-evidence", type=Path)
    parser.add_argument("--uploads", type=Path)
    parser.add_argument("--client-uploads", type=Path)
    parser.add_argument("--rotate-seconds", type=float, default=3600)
    parser.add_argument("--job-timeout", type=int, default=3600)
    parser.add_argument("--max-output-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--scratch-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--memory-bytes", type=int, default=12 * 1024**3)
    parser.add_argument("--cpus", type=float, default=4)
    parser.add_argument("--gpu", default="0")

    def terminate(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with VideoSupervisor(**vars(parser.parse_args(argv))) as manager:
            while True:
                manager.tick()
                time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        parser.exit(1, "video supervisor failed; inspect generation history and owned containers\n")
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
