#!/usr/bin/env python3
"""Create small, visibly synthetic media in the explicitly supplied empty directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--video', action='store_true')
a = p.parse_args()
a.directory.mkdir(parents=True, exist_ok=True)
if any(a.directory.iterdir()):
    raise SystemExit('Refusing to overwrite a nonempty fixture directory')
for n, color in enumerate(['#1455bb', '#bb5514']):
    im = Image.new('RGB', (640, 480), '#eeeeee')
    d = ImageDraw.Draw(im)
    d.rectangle((40 + 50*n, 70, 340, 330), fill=color)
    d.ellipse((370, 160, 560, 350), fill='#22aa55')
    d.text((40, 420), f'SYNTHETIC RELEASE FIXTURE {n}; NO REAL EVIDENCE', fill='black')
    im.save(a.directory / f'synthetic-{n}.png')
if a.video:
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                    '-f', 'lavfi', '-i', 'testsrc2=size=640x480:rate=10', '-t', '3',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(a.directory / 'synthetic-test-pattern.mp4')], check=True)
print(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(a.directory.iterdir())}, indent=2))
