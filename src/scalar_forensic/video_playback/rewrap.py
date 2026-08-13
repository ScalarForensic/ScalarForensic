"""Lossless rewrap: the same coded packets, moved into an MP4 box structure.

Deliberately not part of ``encode.py`` (spec §11), which is the ffmpeg
re-encode path.  A rewrap is a PyAV stream copy: the video and audio bitstreams
that come out are bit-identical to the ones that went in, and nothing here may
ever grow into an encoder.
"""

from __future__ import annotations

from pathlib import Path

import av

from scalar_forensic.video_playback.cache import part_path, publish
from scalar_forensic.video_playback.codecs import _MP4_LEGAL_CODECS


def _repair_timestamps(packet: av.Packet, last_dts: int | None) -> bool:
    """Make *packet*'s timestamps muxable in place; True when something moved.

    Real iPhone .MOV files carry the occasional frame whose stored composition
    time lands *before* its decode time (measured on the corpus: 1 packet in 18
    on IMG_3743.MOV).  libavformat's muxer refuses such a packet outright with
    EINVAL, and refuses a decode time that fails to advance.  Both are repaired
    the minimal way — pull the decode stamp back to the composition stamp, or
    nudge it one tick past its predecessor.

    Timing metadata only.  The coded payload never changes, which is the whole
    point of a rewrap; the count of repairs is reported so the adjustment is
    never silent.
    """
    moved = False
    if packet.pts is not None and packet.dts is not None and packet.pts < packet.dts:
        packet.dts = packet.pts
        moved = True
    if last_dts is not None and packet.dts is not None and packet.dts <= last_dts:
        packet.dts = last_dts + 1
        if packet.pts is not None and packet.pts < packet.dts:
            packet.pts = packet.dts
        moved = True
    return moved


def _remux_to_mp4(src: Path, dst: Path) -> dict:
    """Stream-copy *src* into a faststart MP4 at *dst*; return a rewrap report.

    Not a re-encode: packets are demuxed and remuxed with their payloads
    untouched, so the video and audio bitstreams in *dst* are bit-identical to
    those in *src*.  Streams whose codec has no MP4 mapping are left behind and
    named in the report rather than silently dropped, and any timestamp repair
    (:func:`_repair_timestamps`) is counted there too.

    Writes to a sibling ``.part`` file and renames on success, so a reader can
    never observe a half-written viewing copy.
    """
    skipped: list[str] = []
    repaired = 0
    part = part_path(dst)
    part.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            av.open(str(src)) as inp,
            av.open(str(part), "w", format="mp4", options={"movflags": "+faststart"}) as out,
        ):
            mapping = {}
            for s in inp.streams:
                if s.type not in ("video", "audio"):
                    continue
                codec = s.codec_context.name
                if codec not in _MP4_LEGAL_CODECS:
                    skipped.append(f"{s.type}:{codec}")
                    continue
                out_stream = out.add_stream_from_template(s)
                # Carry the source's four-character codec tag across.  Left to
                # itself the muxer relabels Apple's hvc1 as hev1; both decode in
                # Chrome, but hvc1 is what the source says and the only one
                # QuickTime and Safari will open.
                if s.codec_tag:
                    out_stream.codec_tag = s.codec_tag
                mapping[s.index] = out_stream
            if not mapping:
                raise ValueError("no MP4-compatible stream in source")
            last_dts: dict[int, int] = {}
            for packet in inp.demux([s for s in inp.streams if s.index in mapping]):
                if packet.dts is None:  # flush packet from the demuxer
                    continue
                index = packet.stream.index
                if _repair_timestamps(packet, last_dts.get(index)):
                    repaired += 1
                last_dts[index] = packet.dts
                # Timestamps stay in the source time base; the muxer rescales
                # them onto its own when the packet is written.
                packet.stream = mapping[index]
                out.mux(packet)
        publish(part, dst)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return {"skipped_streams": skipped, "timestamp_repairs": repaired}
