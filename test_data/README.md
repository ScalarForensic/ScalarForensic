# `test_data/` — operator-supplied test clips

This directory is **tracked**. `data/` is gitignored, which is why the video
tests cannot look for fixtures there
(`docs/specs/video-playback-transcode.md` §14).

## What goes here

Nothing is required. The video-playback tests resolve their HDR fixture by a
three-source lookup, first hit wins:

1. `SFN_TEST_VIDEO_HDR` set to the path of a real file;
2. a clip found in **this directory**;
3. otherwise a clip generated at test time with `ffmpeg -f lavfi` into a tmp dir.

Dropping a clip here therefore needs **no code change** — the tests pick it up.

## What a dropped clip must be

| Property | Requirement |
|---|---|
| Name | `hdr_sample.<ext>` (`.mov`, `.mp4`, `.mkv`) |
| Bit depth | 10-bit (`yuv420p10le` or equivalent) |
| Transfer | HLG (`arib-std-b67`) or PQ (`smpte2084`), with `bt2020` primaries |
| Rotation | rotation side data present (e.g. `rotation=-90`) — this is what pins §3.1's GPU-path rotation defect |
| Size | small; a few seconds is enough |
| Licence | named in this file, in a line added below |

## What must never go here

**No footage from the operator's evidence corpus.** That is case material and it
does not go in a git repository. `docs/benchmarks/video-bench-2026-08-13.md`
names a corpus path as part of its methodology record; that is not permission to
copy the file.

A clip added here must be publicly redistributable, and its source and licence
recorded below.

## Clips present

_(none)_
