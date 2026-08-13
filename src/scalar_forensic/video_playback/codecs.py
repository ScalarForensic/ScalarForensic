"""Container and codec classification: what the browser can do with a source.

Everything here inspects a container or a stream and decides what the browser
can do with it.  :func:`_stream_report` is the only function that opens the
file; the rest judge from what it reported.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av

_log = logging.getLogger(__name__)


# Codecs that are legal in an MP4 box structure.  A stream outside this set
# cannot be rewrapped (and must not be re-encoded), so it is left behind and
# named in the playback report rather than silently dropped.
_MP4_LEGAL_CODECS = frozenset(
    {
        "h264",
        "hevc",
        "av1",
        "vp9",
        "mpeg4",
        "mjpeg",
        "aac",
        "mp3",
        "ac3",
        "eac3",
        "alac",
        "opus",
        "flac",
    }
)

# ISO base-media brands are all playable in Chrome; "qt  " is the QuickTime
# brand that is not.  WebM is listed separately by extension because it carries
# no ftyp box at all — .mkv is deliberately absent, since Chrome opens WebM but
# not Matroska at large, so an .mkv takes the rewrap path.
_QUICKTIME_BRAND = "qt  "
_NATIVE_EXTENSIONS = frozenset({".webm"})


def _ftyp_brand(p: Path) -> str | None:
    """Return the ISO base-media major brand of *p*, or None if it has no ftyp box."""
    try:
        with p.open("rb") as fh:
            head = fh.read(12)
    except OSError:
        return None
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    return head[8:12].decode("latin-1")


def _needs_remux(p: Path) -> bool:
    """True when the browser cannot open *p*'s container as it stands.

    Container-level judgement only — no codec is inspected, because nothing here
    can fix a codec the browser lacks: a rewrap moves the same bitstream.
    """
    if p.suffix.lower() in _NATIVE_EXTENSIONS:
        return False
    brand = _ftyp_brand(p)
    return brand is None or brand == _QUICKTIME_BRAND


def _stream_report(p: Path) -> dict:
    """Container/codec summary used by the playback label and the rewrap filter.

    ``skipped_streams`` names the streams a rewrap would have to leave behind —
    Apple's Live-Photo .MOV files carry LPCM audio, which has no MP4 mapping and
    cannot be re-encoded here, so the viewing copy of one is silent.  The label
    says so rather than letting the operator infer a silent original.
    """
    info: dict = {
        "container_brand": _ftyp_brand(p),
        "video_codec": None,
        "audio_codec": None,
        "video_pix_fmt": None,
        "video_profile": None,
    }
    skipped: list[str] = []
    try:
        with av.open(str(p)) as container:
            info["format"] = container.format.name
            if container.duration is not None:
                info["duration_ms"] = int(container.duration / av.time_base * 1000)
            for s in container.streams:
                if s.type == "video" and info["video_codec"] is None:
                    info["video_codec"] = s.codec_context.name
                    info["video_codec_tag"] = s.codec_context.codec_tag
                    # Read from the container's codec parameters — no frame is
                    # decoded, which is what keeps opening a hit cheap (§5).
                    info["video_pix_fmt"] = s.codec_context.pix_fmt
                    info["video_profile"] = s.codec_context.profile
                elif s.type == "audio" and info["audio_codec"] is None:
                    info["audio_codec"] = s.codec_context.name
                if s.type in ("video", "audio") and s.codec_context.name not in _MP4_LEGAL_CODECS:
                    skipped.append(f"{s.type}:{s.codec_context.name}")
    except (av.FFmpegError, OSError) as exc:
        _log.debug("playback probe failed for %s: %s", p, exc)
        info["probe_error"] = str(exc)
    info["skipped_streams"] = skipped
    return info


# ---------------------------------------------------------------------------
# Codec allowlist (spec §5, §15.3, ruling §16)
# ---------------------------------------------------------------------------
#
# What the browser can decode is decided HERE, from the stream, and never from
# what the browser advertises: the operator's Chrome reports HEVC support and
# then fails to decode an iPhone HEVC file (ruling 2026-08-13).  The allowlist is
# therefore deliberately narrow — a codec is playable only in the pixel formats
# every target browser actually decodes.
#
# Phase 3 detects and reports.  It never encodes: a stream outside the allowlist
# is named as needing a transcode, and the analyst is pointed at Download
# original until the encode path lands (phase 4).

# codec name → maximum bit depth the browsers decode.  4:2:0 chroma is required
# for all of them; 4:2:2 and 4:4:4 have no browser decoder at any depth.
_PLAYABLE_CODECS: dict[str, int] = {
    "h264": 8,  # 8-bit 4:2:0 only — High 10 and 4:4:4 are not decodable
    "vp8": 8,
    "vp9": 10,  # profile 0 and profile 2
    "av1": 10,  # Main profile, 8- and 10-bit
}

# Display names for the reason string.  A label an analyst reads must name the
# codec the way the world names it, not the way libavcodec does.
_CODEC_DISPLAY_NAMES = {
    "h264": "H.264",
    "hevc": "HEVC",
    "vp8": "VP8",
    "vp9": "VP9",
    "av1": "AV1",
    "mpeg4": "MPEG-4 Part 2",
    "mpeg2video": "MPEG-2",
    "mjpeg": "Motion JPEG",
    "prores": "Apple ProRes",
    "dnxhd": "DNxHD",
    "vc1": "VC-1",
    "theora": "Theora",
}


def _codec_label(codec: str | None, bits: int | None, chroma: str | None) -> str:
    """Human name for a stream: ``"HEVC 10-bit"``, ``"H.264 4:4:4"``."""
    name = _CODEC_DISPLAY_NAMES.get(codec or "", codec or "unknown codec")
    parts = [name]
    if bits is not None and bits != 8:
        parts.append(f"{bits}-bit")
    if chroma is not None and chroma != "420":
        parts.append(f"{chroma[0]}:{chroma[1]}:{chroma[2]}")
    return " ".join(parts)


def _pixel_profile(pix_fmt: str | None, profile: str | None) -> tuple[int | None, str | None]:
    """Return ``(bit depth, chroma)`` for a stream, without decoding a frame.

    ``pix_fmt`` is what libavformat parsed out of the container's codec
    parameters, which is enough for every codec in the allowlist.  Where it is
    absent the profile name still settles the question for the two cases that
    matter (``"Main 10"``, ``"High 4:4:4 Predictive"``).  ``(None, None)`` means
    undetermined — reported as such, never guessed.
    """
    if pix_fmt:
        try:
            fmt = av.VideoFormat(pix_fmt)
            bits = max(c.bits for c in fmt.components if c.bits)
        except (ValueError, AttributeError):  # pragma: no cover - unknown format name
            bits = None
        for chroma in ("444", "422", "420", "411", "410"):
            if chroma in pix_fmt:
                return bits, chroma
        # Formats with no subsampling digits (gray, rgb…) are not 4:2:0 and no
        # browser decodes them in these codecs.
        return bits, None
    if profile:
        low = profile.lower()
        bits = 12 if "12" in low else 10 if "10" in low else None
        chroma = "444" if "4:4:4" in low else "422" if "4:2:2" in low else None
        if bits is not None or chroma is not None:
            return bits, chroma
    return None, None


def _decode_verdict(info: dict) -> tuple[bool | None, str]:
    """Can a browser decode this video stream?  ``(verdict, human reason)``.

    ``None`` is "cannot tell" — a stream that could not be probed, or a pixel
    format libavformat did not report.  It is a third answer on purpose: guessing
    "playable" hides a failure, guessing "transcode" claims work is needed on no
    evidence, and §5 forbids inventing a state that cannot be observed.
    """
    if info.get("probe_error"):
        return None, f"The container could not be probed ({info['probe_error']})."
    codec = info.get("video_codec")
    if not codec:
        return None, "No video stream was found in this container."
    bits, chroma = _pixel_profile(info.get("video_pix_fmt"), info.get("video_profile"))
    label = _codec_label(codec, bits, chroma)
    max_bits = _PLAYABLE_CODECS.get(codec)
    if max_bits is None:
        return False, f"{label}: no browser decoder for this codec."
    if bits is None and chroma is None:
        return None, f"{label}: the pixel format could not be read from the container."
    if chroma is not None and chroma != "420":
        return False, f"{label}: browsers decode 4:2:0 chroma only."
    if bits is not None and bits > max_bits:
        return False, f"{label}: this browser cannot decode it."
    return True, f"{label}: decodes natively."


def _playback_mode(info: dict, needs_remux: bool) -> tuple[str, str]:
    """Decide ``mode`` and the reason shown to the analyst.

    Codec first, container second: a rewrap moves the same bitstream, so it
    cannot rescue a stream the browser has no decoder for.
    """
    decodable, reason = _decode_verdict(info)
    if decodable is None:
        return "unknown", f"{reason} Playback cannot be judged from here — download the original."
    if decodable is False:
        return "transcode", (
            f"{reason} A transcoded viewing copy is required; "
            "encoding is not available yet — download the original to view it."
        )
    if needs_remux:
        return "rewrap", (
            f"{reason} The QuickTime container is not one browsers open, so the same "
            "packets are rewrapped into MP4 — no re-encode."
        )
    return "original", f"{reason} The file is streamed as it is stored on disk."
