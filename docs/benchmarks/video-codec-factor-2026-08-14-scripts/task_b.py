#!/usr/bin/env python3
"""Task B: chunk-encode latency while a full-video job runs, unniced vs niced+threaded.

The real §4.3 competitor is a single full-video job that holds a worker for
~51 min. This host's corpus has nothing that long, and re-encoding a
tens-of-minutes synthetic source is expensive on this box's tight /tmp
(tmpfs, shared with other sessions). Instead this harness SUSTAINS a
competitor: a background loop that immediately relaunches the full-copy job
on an 11.4-minute synthesized source (§3.5's own concat-copy method) the
instant one run finishes, for the whole span of the chunk-encode reps. This
is disclosed as a synthesized sustained competitor, not a single continuous
51-min job — the two are not proven equivalent, only that a competitor is
occupying a worker at every moment a chunk rep is timed.

Run with `PYTHONPATH=$PWD/src` from a repo checkout, e.g. `PYTHONPATH=$PWD/src
python docs/benchmarks/video-codec-factor-2026-08-14-scripts/task_b.py both`.
Needs the operator's dev corpus (`CHUNK_SRC` below) and a pre-built
concat-copy long source at `TASKB_FULL_SRC` (defaults to a path under this
script's own directory; build one with `ffmpeg -f concat -safe 0 -i list.txt
-c copy long_source.mov`, `list.txt` repeating one corpus file's path).
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from scalar_forensic.config import Settings
from scalar_forensic.video_playback.capability import Capability, select
from scalar_forensic.video_playback.encode import build_command

FULL_SRC = Path(os.environ.get("TASKB_FULL_SRC", str(Path(__file__).parent / "longsrc" / "long_source.mov")))
CHUNK_SRC = Path("/media/user01/SAM_870_SATA/Gitea_Backup/input_scalar/IMG_2668.MOV")
CHUNK_SRC_DURATION = 68.4
OUT = Path(__file__).parent / "taskb_out"
OUT.mkdir(exist_ok=True)

settings = Settings()
settings.video_output_height = 1080
cpu_cap = Capability(
    ffmpeg_path=settings.ffmpeg_path, ffmpeg_version="bench", encoder="libx264",
    hwaccel="none", tonemap_ok=True, notes=(),
)
pipeline = select(settings, cpu_cap, hdr=True)


def full_job_cmd(nice: bool, tag: str):
    dst = OUT / f"full_{tag}.mp4"
    cmd = build_command(settings, pipeline, FULL_SRC, dst, start=None, duration=None, has_audio=True)
    if nice:
        cmd = ["nice", "-n", "10"] + cmd[:1] + ["-threads", "12"] + cmd[1:]
    return cmd, dst


def chunk_cmd(start: float):
    assert start + 30.0 <= CHUNK_SRC_DURATION, (
        f"start={start} would run past CHUNK_SRC's {CHUNK_SRC_DURATION}s duration"
    )
    dst = OUT / f"chunk_{start:.0f}.mp4"
    cmd = build_command(settings, pipeline, CHUNK_SRC, dst, start=start, duration=30.0, has_audio=True)
    return cmd, dst


class Sustainer:
    """Keeps a full-job competitor running continuously by relaunching on exit."""

    def __init__(self, nice: bool, tag: str):
        self.nice = nice
        self.tag = tag
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.restarts = 0
        self.current_proc = None
        self._lock = threading.Lock()

    def _loop(self):
        while not self._stop.is_set():
            cmd, dst = full_job_cmd(self.nice, self.tag)
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self._lock:
                self.current_proc = proc
            proc.wait()
            self.restarts += 1
            if dst.exists():
                dst.unlink()
            if self._stop.is_set():
                break

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._lock:
            proc = self.current_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._thread.join(timeout=15)
        for tag_dst in OUT.glob(f"full_{self.tag}*"):
            tag_dst.unlink(missing_ok=True)

    def is_running(self) -> bool:
        with self._lock:
            proc = self.current_proc
        return proc is not None and proc.poll() is None


def measure(label: str, nice: bool, tag: str, n_chunks: int = 6, gap: float = 6.0):
    print(f"=== {label} ===", file=sys.stderr)
    sustainer = Sustainer(nice, tag)
    sustainer.start()
    time.sleep(4.0)  # let the first full-job launch ramp up before measuring
    results = []
    try:
        for i in range(n_chunks):
            running = sustainer.is_running()
            start = 10.0 + (i % 2) * 20.0
            cmd, dst = chunk_cmd(start)
            t0 = time.monotonic()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            wall = time.monotonic() - t0
            ok = r.returncode == 0 and dst.exists()
            results.append({"wall": wall, "competitor_running_at_start": running})
            print(
                f"  chunk {i} start={start} wall={wall:.2f}s ok={ok} "
                f"competitor_running_at_start={running} restarts_so_far={sustainer.restarts}",
                file=sys.stderr,
            )
            if dst.exists():
                dst.unlink()
            time.sleep(max(0, gap - wall))
    finally:
        sustainer.stop()
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    out = {}
    if mode in ("both", "unniced"):
        out["unniced"] = measure("unniced/unthreaded", nice=False, tag="unniced")
    if mode in ("both", "niced"):
        out["niced"] = measure("niced+threaded (nice 10, -threads 12)", nice=True, tag="niced")
    print(out)
