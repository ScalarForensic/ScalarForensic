#!/usr/bin/env python3
"""Task A: measure output_bytes / estimate_full_output_bytes() per source class.

Uses the repo's own build_command()/estimate_full_output_bytes()/_stream_report()
so the numbers reflect the real §8 pipeline and the real §6.3 estimator, not a
reimplementation of either.

Run with `PYTHONPATH=$PWD/src` from a repo checkout (the project's own
convention, CLAUDE.md "Contributing workflow"), e.g.:

    PYTHONPATH=$PWD/src uv run --no-project python \
        docs/benchmarks/video-codec-factor-2026-08-14-scripts/run_bench.py [class] [cpu|gpu]

or with the project's own venv activated. Requires the operator's dev corpus
at the `CORPUS` path in samples.py (not shipped with this repo — case
material, see the report for why).
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from scalar_forensic.config import Settings
from scalar_forensic.video_playback.codecs import _stream_report
from scalar_forensic.video_playback.cache import estimate_full_output_bytes
from scalar_forensic.video_playback.capability import Capability, HDR_TRANSFERS, select
from scalar_forensic.video_playback.encode import build_command

from samples import CORPUS, SAMPLES

OUTDIR = Path(__file__).parent / "outputs"
OUTDIR.mkdir(exist_ok=True)
RESULTS = Path(__file__).parent / "results.jsonl"

settings = Settings()
settings.video_output_height = 1080

CAPS = {
    "cpu": Capability(
        ffmpeg_path=settings.ffmpeg_path, ffmpeg_version="bench", encoder="libx264",
        hwaccel="none", tonemap_ok=True, notes=(),
    ),
    "gpu": Capability(
        ffmpeg_path=settings.ffmpeg_path, ffmpeg_version="bench", encoder="h264_nvenc",
        hwaccel="cuda", tonemap_ok=True, notes=(),
    ),
}


def run_one(cls: str, fname: str, row_kind: str) -> dict:
    src = Path(CORPUS) / fname
    info = _stream_report(src)
    hdr = info.get("video_color_trc") in HDR_TRANSFERS
    estimate = estimate_full_output_bytes(info, settings.video_output_height)
    pipeline = select(settings, CAPS[row_kind], hdr=hdr)
    dst = OUTDIR / f"{cls}__{fname}__{row_kind}.mp4"
    cmd = build_command(settings, pipeline, src, dst, start=None, duration=None, has_audio=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.monotonic() - t0
    rec = {
        "class": cls, "file": fname, "row_kind": row_kind,
        "src_bytes": src.stat().st_size,
        "src_codec": info.get("video_codec"), "src_pix_fmt": info.get("video_pix_fmt"),
        "src_width": info.get("video_width"), "src_height": info.get("video_height"),
        "src_bit_rate": info.get("bit_rate"), "src_duration_ms": info.get("duration_ms"),
        "src_color_trc": info.get("video_color_trc"), "hdr": hdr,
        "estimate_bytes": estimate,
        "command": cmd,
        "wall_s": round(wall, 3),
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-800:] if proc.returncode != 0 else "",
    }
    if proc.returncode == 0 and dst.exists():
        rec["output_bytes"] = dst.stat().st_size
        rec["ratio"] = rec["output_bytes"] / estimate if estimate else None
    else:
        rec["output_bytes"] = None
        rec["ratio"] = None
    return rec


def main():
    only_class = sys.argv[1] if len(sys.argv) > 1 else None
    only_row = sys.argv[2] if len(sys.argv) > 2 else None
    with RESULTS.open("a") as out:
        for cls, files in SAMPLES.items():
            if only_class and cls != only_class:
                continue
            for fname in files:
                for row_kind in (["cpu", "gpu"] if not only_row else [only_row]):
                    print(f"=== {cls} {fname} {row_kind} ===", file=sys.stderr)
                    try:
                        rec = run_one(cls, fname, row_kind)
                    except Exception as e:
                        rec = {"class": cls, "file": fname, "row_kind": row_kind, "error": str(e)}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"  ratio={rec.get('ratio')} wall={rec.get('wall_s')} out={rec.get('output_bytes')}", file=sys.stderr)
                    # reclaim disk immediately
                    dst = OUTDIR / f"{cls}__{fname}__{row_kind}.mp4"
                    if dst.exists():
                        dst.unlink()


if __name__ == "__main__":
    main()
