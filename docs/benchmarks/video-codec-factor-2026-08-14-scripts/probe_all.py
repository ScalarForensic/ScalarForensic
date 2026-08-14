#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

CORPUS = Path("/media/user01/SAM_870_SATA/Gitea_Backup/input_scalar")
OUT = Path(__file__).parent / "corpus_probe.tsv"

exts = {".mov", ".mp4", ".m4v"}
files = sorted(p for p in CORPUS.iterdir() if p.suffix.lower() in exts)
print(f"{len(files)} files", file=sys.stderr)

cols = [
    "file", "codec", "pix_fmt", "width", "height", "rotation",
    "duration_s", "bit_rate", "size_bytes", "color_transfer", "color_primaries", "fps",
]

with OUT.open("w") as out:
    out.write("\t".join(cols) + "\n")
    for i, f in enumerate(files):
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,duration,bit_rate,r_frame_rate,color_transfer,color_primaries:"
            "stream_side_data=rotation:format=duration,bit_rate,size",
            "-of", "json", str(f),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            d = json.loads(r.stdout)
        except Exception as e:
            print(f"FAIL {f}: {e}", file=sys.stderr)
            continue
        streams = d.get("streams", [{}])
        s = streams[0] if streams else {}
        fmt = d.get("format", {})
        rotation = ""
        for sd in s.get("side_data_list", []) or []:
            if "rotation" in sd:
                rotation = sd["rotation"]
        row = [
            str(f.relative_to(CORPUS)),
            s.get("codec_name", ""),
            s.get("pix_fmt", ""),
            s.get("width", ""),
            s.get("height", ""),
            str(rotation),
            fmt.get("duration", s.get("duration", "")),
            fmt.get("bit_rate", s.get("bit_rate", "")),
            fmt.get("size", ""),
            s.get("color_transfer", ""),
            s.get("color_primaries", ""),
            s.get("r_frame_rate", ""),
        ]
        out.write("\t".join(str(x) for x in row) + "\n")
        out.flush()
        if i % 50 == 0:
            print(f"{i}/{len(files)}", file=sys.stderr)
print("done", file=sys.stderr)
