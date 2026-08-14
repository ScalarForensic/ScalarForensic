"""Tests for in-browser playback of source videos.

Covered:
  GET /api/video-playback?path=…
  GET /api/video-playback-info?path=…
  GET /api/video-download?path=…
  the rewrap/cache/digest helpers in the scalar_forensic.video_playback package
  the SFN_VIDEO_CACHE_* settings

Fixtures are generated with PyAV at test time (a handful of 64×48 frames), so
the suite stays hermetic and needs neither Qdrant nor sample media on disk.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import errno
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from scalar_forensic import cli
from scalar_forensic.config import Settings
from scalar_forensic.embedder import hash_file
from scalar_forensic.video import VIDEO_EXTENSIONS
from scalar_forensic.video_playback import cache as vp_cache
from scalar_forensic.video_playback import capability as vp_capability
from scalar_forensic.video_playback import codecs as vp_codecs
from scalar_forensic.video_playback import digest as vp_digest
from scalar_forensic.video_playback import encode as vp_encode
from scalar_forensic.video_playback import jobs as vp_jobs
from scalar_forensic.video_playback import rewrap as vp_rewrap
from scalar_forensic.video_playback import routes as vp_routes
from scalar_forensic.video_playback import states as vp_states
from scalar_forensic.web.app import app
from scalar_forensic.web.routes import _shared as shared_routes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_clip(path: Path, container_format: str, *, frames: int = 8) -> Path:
    """Encode a tiny H.264 clip into *container_format* at *path*."""
    with av.open(str(path), "w", format=container_format) as c:
        stream = c.add_stream("libx264", rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            arr = np.full((48, 64, 3), (i * 24) % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    return path


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    """An input dir and a viewing-copy cache dir, wired through the environment."""
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    cache_dir = tmp_path / "video_cache"
    monkeypatch.setenv("SFN_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", str(cache_dir))
    # Keep the persistent hash cache inside the tmp dir: the default
    # (data/hash_cache.db, relative to CWD) would write into the checkout.
    monkeypatch.setenv("SFN_HASH_CACHE_PATH", str(tmp_path / "hash_cache.db"))
    vp_digest._reset_hash_cache()
    yield input_dir, cache_dir
    # The handle points into tmp_path; never let the next test inherit it.
    vp_digest._reset_hash_cache()


@pytest.fixture()
def mov(roots):
    input_dir, _ = roots
    return _write_clip(input_dir / "IMG_0001.MOV", "mov")


@pytest.fixture()
def mov_with_pcm(roots):
    """A .MOV carrying LPCM audio, as Apple's Live Photos do."""
    input_dir, _ = roots
    path = input_dir / "IMG_0002.MOV"
    with av.open(str(path), "w", format="mov") as c:
        vs = c.add_stream("libx264", rate=10)
        vs.width, vs.height = 64, 48
        vs.pix_fmt = "yuv420p"
        audio = c.add_stream("pcm_s16le", rate=8000)
        audio.layout = "mono"
        for i in range(8):
            arr = np.full((48, 64, 3), (i * 24) % 256, dtype=np.uint8)
            for packet in vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(packet)
            samples = np.zeros((1, 800), dtype=np.int16)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = 8000
            frame.pts = i * 800
            frame.time_base = Fraction(1, 8000)
            for packet in audio.encode(frame):
                c.mux(packet)
        for packet in vs.encode():
            c.mux(packet)
        for packet in audio.encode():
            c.mux(packet)
    return path


@pytest.fixture()
def mp4(roots):
    input_dir, _ = roots
    return _write_clip(input_dir / "clip.mp4", "mp4")


def _write_encoded(
    path: Path, container_format: str, encoder: str, pix_fmt: str, *, frames: int = 6
) -> Path:
    """Encode a tiny clip with *encoder* at *pix_fmt* — codec-detection fixtures."""
    with av.open(str(path), "w", format=container_format) as c:
        stream = c.add_stream(encoder, rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = pix_fmt
        for i in range(frames):
            arr = np.full((48, 64, 3), (i * 24) % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    return path


@pytest.fixture()
def hevc_10bit_mov(roots):
    """An HEVC Main-10 .MOV — the iPhone case the whole feature exists for."""
    input_dir, _ = roots
    return _write_encoded(input_dir / "IMG_0010.MOV", "mov", "libx265", "yuv420p10le")


# ---------------------------------------------------------------------------
# The HDR fixture (spec §14): env gate → tracked test_data/ → generated
# ---------------------------------------------------------------------------

FFMPEG = os.environ.get("SFN_FFMPEG_PATH", "ffmpeg")
FFPROBE = str(Path(FFMPEG).with_name("ffprobe")) if os.sep in FFMPEG else "ffprobe"

# Locally a missing ffmpeg is a skip; in CI it is a FAILURE.  ffmpeg is a
# declared dependency (§8) and `.github/workflows/ci.yml` installs it, so absent
# means the install step was dropped or the runner image changed — and a test
# that answers that by skipping puts the job back to green-by-skip, which is the
# state this guard exists to end.  A developer without ffmpeg still gets a skip.
_IN_CI = os.environ.get("CI", "").strip().lower() not in ("", "0", "false")


def _need_ffmpeg() -> None:
    if shutil.which(FFMPEG) is not None:
        return
    message = (
        f"ffmpeg ({FFMPEG!r}) is not installed. It is a declared dependency "
        "(docs/specs/video-playback-transcode.md §8) and CI installs it; if this "
        "fires in CI the install step is gone, not the requirement."
    )
    if _IN_CI:
        raise AssertionError(message)
    pytest.skip(message)


@pytest.fixture()
def ffmpeg_required() -> None:
    _need_ffmpeg()


requires_ffmpeg = pytest.mark.usefixtures("ffmpeg_required")

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "test_data"


def _operator_hdr_clip() -> Path | None:
    """A real HDR capture, if the operator supplied one.  Sources 1 and 2 of §14.

    Nothing from the evidence corpus is ever committed, so the repository can
    only ever hold source 3.  These two exist so a real clip can be dropped in
    without touching code — the honest assertions a synthetic picture cannot
    make are the whole reason §14 layers the lookup instead of picking one.
    """
    env = os.environ.get("SFN_TEST_VIDEO_HDR")
    if env and Path(env).is_file():
        return Path(env)
    if TEST_DATA_DIR.is_dir():
        for candidate in sorted(TEST_DATA_DIR.glob("hdr_sample.*")):
            if candidate.is_file():
                return candidate
    return None


# The ISO base-media display matrix for a −90° rotation, in the tkhd layout
# [a b u; c d v; x y w] — a/b/c/d/x/y in 16.16 and u/v/w in 2.30 fixed point.
# ffmpeg cannot *write* rotation side data on an output stream (the old
# `-metadata:s:v rotate=` is gone in 6.x), so the generated fixture gets a real
# display matrix patched into its tkhd box instead of a weakened assertion.
# The result is genuine side data: ffprobe reports it as "Display Matrix".
def _rotation_matrix(width: int) -> bytes:
    values = [0, -65536, 0, 65536, 0, 0, 0, width << 16, 1 << 30]
    return b"".join(struct.pack(">i", v) for v in values)


def _patch_display_matrix(path: Path, width: int) -> None:
    data = bytearray(path.read_bytes())
    box = data.find(b"tkhd")
    if box < 0:  # pragma: no cover - every mov/mp4 track has one
        raise AssertionError(f"no tkhd box in {path}")
    # box start is the 4 size bytes before the type; the matrix sits 48 bytes in
    # (version+flags, times, track id, duration, reserved, layer, group, volume).
    offset = box - 4 + 48
    data[offset : offset + 36] = _rotation_matrix(width)
    path.write_bytes(bytes(data))


def _generate_hdr_clip(dst: Path, *, rotated: bool = False) -> Path:
    """Source 3 of §14: synthesise a 10-bit HLG/bt2020 clip with ffmpeg.

    ffmpeg is already a §8 dependency, so this adds none.  What it is *not* is a
    substitute for a real capture — it carries the transfer, primaries, bit
    depth and (when asked) the display matrix that the pipeline has to handle,
    and nothing about real sensor noise, real HDR highlights or a real device's
    container quirks.
    """
    subprocess.run(  # noqa: S603 - fixed args, test-time fixture generation
        [
            FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x96:rate=10:duration=0.6,format=yuv420p10le,"
            "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=none",
            "-tag:v",
            "hvc1",
            "-y",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    if rotated:
        _patch_display_matrix(dst, 64)
    return dst


@pytest.fixture()
def hdr_hlg_mov(roots) -> Path:
    """A 10-bit HLG source in the input dir, by the §14 three-source lookup."""
    input_dir, _ = roots
    supplied = _operator_hdr_clip()
    if supplied is not None:
        dst = input_dir / f"hdr_sample{supplied.suffix}"
        shutil.copy(supplied, dst)
        return dst
    if shutil.which(FFMPEG) is None:
        pytest.skip("no HDR clip supplied and ffmpeg is absent (spec §14)")
    return _generate_hdr_clip(input_dir / "hdr_generated.mov")


@pytest.fixture()
def hdr_rotated_mov(roots) -> Path:
    """The same, carrying rotation side data — §3.1's defect needs it present.

    An operator-supplied clip is used only if it *actually* carries rotation:
    §14 requires it, and a clip silently missing it would turn the rotation
    test green while testing nothing.
    """
    input_dir, _ = roots
    supplied = _operator_hdr_clip()
    if supplied is not None and _display_rotation(supplied) is not None:
        dst = input_dir / f"hdr_rotated{supplied.suffix}"
        shutil.copy(supplied, dst)
        return dst
    if shutil.which(FFMPEG) is None:
        pytest.skip("no rotated HDR clip supplied and ffmpeg is absent (spec §14)")
    return _generate_hdr_clip(input_dir / "hdr_rotated.mov", rotated=True)


def _ffprobe_video_stream(path: Path) -> dict:
    """The first video stream as ffprobe describes it, side data included.

    ffprobe rather than PyAV: PyAV exposes no stream-level side data, so the
    display matrix — the exact thing §3.1's defect destroys — is invisible from
    Python.  ffprobe ships with the ffmpeg §8 already requires.
    """
    proc = subprocess.run(  # noqa: S603 - fixed args, test-time probe
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)["streams"][0]


def _display_rotation(path: Path) -> float | None:
    """The stream's rotation from its display matrix, or None if it carries none."""
    if shutil.which(FFPROBE) is None:
        return None
    for side in _ffprobe_video_stream(path).get("side_data_list", []):
        if "rotation" in side:
            return float(side["rotation"])
    return None


def _packet_payloads(path: Path) -> list[bytes]:
    """Raw bytes of every video packet in *path*, in demux order."""
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        return [bytes(p) for p in c.demux(stream) if p.dts is not None]


# ---------------------------------------------------------------------------
# Container classification
# ---------------------------------------------------------------------------


class TestContainerClassification:
    def test_quicktime_mov_needs_rewrap(self, mov):
        assert vp_codecs._ftyp_brand(mov) == "qt  "
        assert vp_codecs._needs_remux(mov) is True

    def test_mp4_is_served_as_is(self, mp4):
        assert vp_codecs._ftyp_brand(mp4) != "qt  "
        assert vp_codecs._needs_remux(mp4) is False

    def test_file_without_ftyp_box_needs_rewrap(self, tmp_path):
        p = tmp_path / "clip.avi"
        p.write_bytes(b"RIFF____AVI LIST")
        assert vp_codecs._ftyp_brand(p) is None
        assert vp_codecs._needs_remux(p) is True

    def test_webm_is_served_as_is(self, tmp_path):
        p = tmp_path / "clip.webm"
        p.write_bytes(b"\x1a\x45\xdf\xa3")
        assert vp_codecs._needs_remux(p) is False


# ---------------------------------------------------------------------------
# The rewrap itself — streams must survive bit for bit
# ---------------------------------------------------------------------------


class TestRemux:
    def test_rewrap_preserves_the_video_bitstream(self, mov, tmp_path):
        dst = tmp_path / "copy.mp4"
        report = vp_rewrap._remux_to_mp4(mov, dst)
        assert report == {"skipped_streams": [], "timestamp_repairs": 0}
        assert _packet_payloads(dst) == _packet_payloads(mov)

    def test_rewrap_keeps_the_codec_and_produces_an_mp4(self, mov, tmp_path):
        dst = tmp_path / "copy.mp4"
        vp_rewrap._remux_to_mp4(mov, dst)
        assert dst.read_bytes()[4:8] == b"ftyp"
        assert vp_codecs._ftyp_brand(dst) != "qt  "
        with av.open(str(mov)) as a, av.open(str(dst)) as b:
            assert a.streams.video[0].codec_context.name == b.streams.video[0].codec_context.name

    def test_the_source_codec_tag_survives(self, mov, tmp_path):
        """Left alone the muxer relabels Apple's hvc1 as hev1."""
        dst = tmp_path / "copy.mp4"
        vp_rewrap._remux_to_mp4(mov, dst)
        with av.open(str(mov)) as a, av.open(str(dst)) as b:
            assert b.streams.video[0].codec_context.codec_tag == (
                a.streams.video[0].codec_context.codec_tag
            )

    def test_moov_precedes_mdat_for_progressive_playback(self, mov, tmp_path):
        dst = tmp_path / "copy.mp4"
        vp_rewrap._remux_to_mp4(mov, dst)
        data = dst.read_bytes()
        assert 0 <= data.index(b"moov") < data.index(b"mdat")

    def test_no_part_file_survives_a_failed_rewrap(self, mov, tmp_path):
        dst = tmp_path / "copy.mp4"
        with (
            patch.object(vp_rewrap.av, "open", side_effect=OSError("boom")),
            pytest.raises(OSError),
        ):
            vp_rewrap._remux_to_mp4(mov, dst)
        assert list(tmp_path.glob("*.part")) == []
        assert not dst.exists()

    def test_lpcm_audio_is_named_not_dropped_silently(self, mov_with_pcm, tmp_path):
        """Apple Live-Photo .MOV carries LPCM, which has no MP4 mapping."""
        dst = tmp_path / "copy.mp4"
        report = vp_rewrap._remux_to_mp4(mov_with_pcm, dst)
        assert report["skipped_streams"] == ["audio:pcm_s16le"]
        with av.open(str(dst)) as c:
            assert len(c.streams.audio) == 0
        assert _packet_payloads(dst) == _packet_payloads(mov_with_pcm)

    def test_source_with_no_mp4_compatible_stream_raises(self, mov, tmp_path):
        with patch.object(vp_rewrap, "_MP4_LEGAL_CODECS", frozenset()):
            with pytest.raises(ValueError, match="no MP4-compatible stream"):
                vp_rewrap._remux_to_mp4(mov, tmp_path / "copy.mp4")


class TestTimestampRepair:
    """Real .MOV files carry frames the MP4 muxer refuses; repairs are counted."""

    def _packet(self, pts, dts):
        p = av.Packet(1)
        p.pts, p.dts = pts, dts
        return p

    def test_untouched_when_timestamps_are_already_muxable(self):
        p = self._packet(100, 90)
        assert vp_rewrap._repair_timestamps(p, 80) is False
        assert (p.pts, p.dts) == (100, 90)

    def test_decode_stamp_after_display_stamp_is_pulled_back(self):
        p = self._packet(600, 640)
        assert vp_rewrap._repair_timestamps(p, 560) is True
        assert (p.pts, p.dts) == (600, 600)

    def test_stalled_decode_stamp_is_nudged_past_its_predecessor(self):
        p = self._packet(500, 500)
        assert vp_rewrap._repair_timestamps(p, 500) is True
        assert p.dts == 501
        assert p.pts >= p.dts

    def test_repairs_are_counted_in_the_rewrap_report(self, mov, tmp_path):
        original = vp_rewrap._repair_timestamps
        with patch.object(vp_rewrap, "_repair_timestamps", side_effect=original) as spy:
            spy.side_effect = lambda packet, last: True
            report = vp_rewrap._remux_to_mp4(mov, tmp_path / "copy.mp4")
        assert report["timestamp_repairs"] == len(_packet_payloads(mov))


# ---------------------------------------------------------------------------
# Path safety and the playable-type set
# ---------------------------------------------------------------------------


class TestPathSafety:
    def test_relative_path_returns_400(self, client):
        r = client.get("/api/video-playback?path=relative/clip.mov")
        assert r.status_code == 400

    def test_non_video_extension_returns_400(self, client, roots):
        input_dir, _ = roots
        p = input_dir / "photo.jpg"
        p.write_bytes(b"x")
        r = client.get(f"/api/video-playback?path={p}")
        assert r.status_code == 400

    def test_path_outside_allowed_roots_returns_403(self, client, roots, tmp_path):
        outside = tmp_path / "elsewhere" / "clip.mov"
        outside.parent.mkdir()
        outside.write_bytes(b"x")
        r = client.get(f"/api/video-playback?path={outside}")
        assert r.status_code == 403

    def test_missing_file_returns_404(self, client, roots):
        input_dir, _ = roots
        r = client.get(f"/api/video-playback?path={input_dir / 'gone.mov'}")
        assert r.status_code == 404

    def test_every_scanner_video_extension_is_playable(self, roots):
        """The playable set is the scanner's, not a hand-copied list."""
        input_dir, _ = roots
        for ext in VIDEO_EXTENSIONS:
            p = input_dir / f"probe{ext}"
            p.write_bytes(b"x")
            assert shared_routes._resolve_video_path(str(p)) == p


# ---------------------------------------------------------------------------
# /api/video-playback
# ---------------------------------------------------------------------------


class TestVideoPlayback:
    def test_mp4_source_is_served_untouched(self, client, mp4):
        r = client.get(f"/api/video-playback?path={mp4}")
        assert r.status_code == 200
        assert r.headers["x-sfn-playback-mode"] == "original"
        assert r.headers["content-type"] == "video/mp4"
        assert r.content == mp4.read_bytes()

    def test_mov_source_is_served_as_a_rewrap(self, client, mov, roots):
        _, cache_dir = roots
        r = client.get(f"/api/video-playback?path={mov}")
        assert r.status_code == 200
        assert r.headers["x-sfn-playback-mode"] == "rewrap"
        assert r.headers["content-type"] == "video/mp4"
        assert r.content != mov.read_bytes()
        cached = list(cache_dir.glob("*/*.mp4"))
        assert len(cached) == 1
        assert cached[0].name == "rewrap.mp4"
        assert cached[0].parent.name == hash_file(mov)

    def test_second_request_serves_the_cached_copy(self, client, mov):
        assert client.get(f"/api/video-playback?path={mov}").status_code == 200
        with patch.object(
            vp_routes, "_remux_to_mp4", side_effect=AssertionError("rewrapped twice")
        ):
            r = client.get(f"/api/video-playback?path={mov}")
        assert r.status_code == 200

    def test_range_request_is_honoured(self, client, mov):
        full = client.get(f"/api/video-playback?path={mov}")
        assert full.headers["accept-ranges"] == "bytes"
        r = client.get(f"/api/video-playback?path={mov}", headers={"Range": "bytes=0-15"})
        assert r.status_code == 206
        assert r.content == full.content[:16]
        assert r.headers["content-range"] == f"bytes 0-15/{len(full.content)}"

    def test_disabled_cache_returns_503(self, client, mov, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", "")
        r = client.get(f"/api/video-playback?path={mov}")
        assert r.status_code == 503

    def test_unrewrappable_source_returns_422(self, client, roots):
        input_dir, _ = roots
        broken = input_dir / "broken.mov"
        broken.write_bytes(b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64)
        r = client.get(f"/api/video-playback?path={broken}")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/video-playback-info
# ---------------------------------------------------------------------------


class TestPlaybackInfo:
    def test_mov_info_reports_rewrap_and_the_source_digest(self, client, mov):
        r = client.get(f"/api/video-playback-info?path={mov}")
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "rewrap"
        assert body["video_codec"] == "h264"
        assert body["video_sha256"] == hash_file(mov)
        assert body["cached"] is False
        assert body["cache_enabled"] is True
        assert body["filename"] == "IMG_0001.MOV"

    def test_mp4_info_reports_original_and_the_digest(self, client, mp4):
        # The digest is reported in every mode: the label must describe the file
        # on disk, and an untouched original still needs its provenance stated.
        body = client.get(f"/api/video-playback-info?path={mp4}").json()
        assert body["mode"] == "original"
        assert body["video_sha256"] == hash_file(mp4)

    def test_info_reports_a_cached_copy(self, client, mov):
        client.get(f"/api/video-playback?path={mov}")
        assert client.get(f"/api/video-playback-info?path={mov}").json()["cached"] is True

    def test_unreadable_source_still_answers(self, client, roots):
        input_dir, _ = roots
        broken = input_dir / "broken.mov"
        broken.write_bytes(b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64)
        body = client.get(f"/api/video-playback-info?path={broken}").json()
        assert body["mode"] == "unknown"
        assert "probe_error" in body


# ---------------------------------------------------------------------------
# Codec allowlist and playback mode (spec §5, §15.3, ruling §16)
# ---------------------------------------------------------------------------


class TestPixelProfile:
    @pytest.mark.parametrize(
        ("pix_fmt", "expected"),
        [
            ("yuv420p", (8, "420")),
            ("yuvj420p", (8, "420")),
            ("yuv420p10le", (10, "420")),
            ("yuv422p10le", (10, "422")),
            ("yuv444p", (8, "444")),
            ("gray", (8, None)),
        ],
    )
    def test_bit_depth_and_chroma_are_read_from_the_pixel_format(self, pix_fmt, expected):
        assert vp_codecs._pixel_profile(pix_fmt, None) == expected

    def test_the_profile_settles_it_when_the_pixel_format_is_missing(self):
        assert vp_codecs._pixel_profile(None, "Main 10") == (10, None)
        assert vp_codecs._pixel_profile(None, "High 4:4:4 Predictive") == (None, "444")

    def test_neither_available_is_undetermined_not_a_guess(self):
        assert vp_codecs._pixel_profile(None, None) == (None, None)
        assert vp_codecs._pixel_profile(None, "High") == (None, None)


class TestDecodeVerdict:
    def _info(self, **kw):
        base = {"video_codec": "h264", "video_pix_fmt": "yuv420p", "video_profile": "High"}
        return {**base, **kw}

    def test_h264_8bit_is_playable(self):
        assert vp_codecs._decode_verdict(self._info())[0] is True

    def test_h264_10bit_is_not(self):
        verdict, reason = vp_codecs._decode_verdict(self._info(video_pix_fmt="yuv420p10le"))
        assert verdict is False
        assert "H.264 10-bit" in reason

    def test_h264_444_is_not(self):
        verdict, reason = vp_codecs._decode_verdict(self._info(video_pix_fmt="yuv444p"))
        assert verdict is False
        assert "4:4:4" in reason

    def test_vp9_10bit_is_playable(self):
        info = self._info(video_codec="vp9", video_pix_fmt="yuv420p10le")
        assert vp_codecs._decode_verdict(info)[0] is True

    def test_av1_and_vp8_are_on_the_allowlist(self):
        for codec in ("av1", "vp8"):
            assert vp_codecs._decode_verdict(self._info(video_codec=codec))[0] is True

    def test_hevc_is_not_on_the_allowlist_at_any_depth(self):
        # §16: Chrome advertises HEVC and then fails. The allowlist decides.
        verdict, reason = vp_codecs._decode_verdict(self._info(video_codec="hevc"))
        assert verdict is False
        assert "HEVC" in reason

    def test_prores_names_itself_in_the_reason(self):
        verdict, reason = vp_codecs._decode_verdict(self._info(video_codec="prores"))
        assert verdict is False
        assert "Apple ProRes" in reason

    def test_an_unprobeable_container_is_undetermined(self):
        verdict, reason = vp_codecs._decode_verdict({"probe_error": "moov atom not found"})
        assert verdict is None
        assert "moov atom not found" in reason

    def test_no_video_stream_is_undetermined(self):
        assert vp_codecs._decode_verdict({"video_codec": None})[0] is None

    def test_an_unreadable_pixel_format_is_undetermined_not_a_transcode(self):
        info = self._info(video_pix_fmt=None, video_profile=None)
        verdict, reason = vp_codecs._decode_verdict(info)
        assert verdict is None
        assert "pixel format" in reason


class TestPlaybackMode:
    def test_10bit_hevc_mov_reports_transcode(self, client, hevc_10bit_mov):
        body = client.get(f"/api/video-playback-info?path={hevc_10bit_mov}").json()
        assert body["mode"] == "transcode"
        assert "HEVC 10-bit" in body["mode_reason"]
        # No encoding happens in this phase; the escape route must be offered.
        assert body["download_url"]

    def test_h264_mp4_reports_original(self, client, mp4):
        body = client.get(f"/api/video-playback-info?path={mp4}").json()
        assert body["mode"] == "original"
        assert "H.264" in body["mode_reason"]

    def test_h264_in_a_quicktime_container_reports_rewrap(self, client, mov):
        # Right codec, wrong container: the packets move, nothing is re-encoded.
        body = client.get(f"/api/video-playback-info?path={mov}").json()
        assert body["mode"] == "rewrap"
        assert "no re-encode" in body["mode_reason"]

    def test_a_corrupt_container_reports_unknown_not_a_crash(self, client, roots):
        input_dir, _ = roots
        broken = input_dir / "broken.mov"
        broken.write_bytes(b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64)
        r = client.get(f"/api/video-playback-info?path={broken}")
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "unknown"
        assert "download the original" in body["mode_reason"]

    def test_the_codec_decides_before_the_container(self, client, roots):
        # A 10-bit HEVC already in an MP4 box structure still cannot be decoded,
        # so it must not be reported as playable-as-is.
        input_dir, _ = roots
        p = _write_encoded(input_dir / "hevc10.mp4", "mp4", "libx265", "yuv420p10le")
        assert client.get(f"/api/video-playback-info?path={p}").json()["mode"] == "transcode"

    def test_no_encoding_happens_on_open(self, client, hevc_10bit_mov, roots):
        _, cache_dir = roots
        client.get(f"/api/video-playback-info?path={hevc_10bit_mov}")
        assert not cache_dir.exists() or not list(cache_dir.iterdir())


# ---------------------------------------------------------------------------
# /api/video-download (spec §7.5)
# ---------------------------------------------------------------------------


class TestVideoDownload:
    def test_serves_the_source_bytes_untouched(self, client, mov):
        r = client.get(f"/api/video-download?path={mov}")
        assert r.status_code == 200
        assert r.content == mov.read_bytes()

    def test_a_mov_is_not_rewrapped_on_the_way_out(self, client, mov, roots):
        # The download is the escape route from the viewing copy, so it must be
        # the original container, not the cached MP4.
        _, cache_dir = roots
        client.get(f"/api/video-playback?path={mov}")  # populate the rewrap cache
        assert list(cache_dir.glob("*/*.mp4"))
        r = client.get(f"/api/video-download?path={mov}")
        assert r.content == mov.read_bytes()

    def test_content_disposition_names_the_original_file(self, client, mov):
        r = client.get(f"/api/video-download?path={mov}")
        disposition = r.headers["content-disposition"]
        assert disposition.startswith("attachment")
        assert "IMG_0001.MOV" in disposition

    def test_a_cached_digest_travels_with_the_bytes(self, client, mov):
        client.get(f"/api/video-playback-info?path={mov}")  # fills the HashCache
        r = client.get(f"/api/video-download?path={mov}")
        assert r.headers["x-sfn-source-sha256"] == hash_file(mov)

    def test_a_cold_cache_omits_the_header_rather_than_stalling(self, client, mov):
        # The escape route must start streaming now: a cold multi-GB source would
        # otherwise be read whole before the first byte moved.  An absent header
        # is "not computed", never "unverified".
        with patch.object(vp_digest, "hash_file", side_effect=AssertionError("hashed!")):
            with patch("scalar_forensic.embedder.hash_file_both", side_effect=AssertionError):
                r = client.get(f"/api/video-download?path={mov}")
        assert r.status_code == 200
        assert "x-sfn-source-sha256" not in r.headers
        assert r.content == mov.read_bytes()

    def test_a_disabled_hash_cache_still_serves_the_bytes(self, client, mov, monkeypatch):
        monkeypatch.setenv("SFN_HASH_CACHE_PATH", "")
        vp_digest._reset_hash_cache()
        r = client.get(f"/api/video-download?path={mov}")
        assert r.status_code == 200
        assert "x-sfn-source-sha256" not in r.headers

    def test_a_stale_cache_entry_is_not_served_as_the_digest(self, client, mov):
        client.get(f"/api/video-playback-info?path={mov}")  # fills the HashCache
        mov.write_bytes(mov.read_bytes() + b"tampered")
        r = client.get(f"/api/video-download?path={mov}")
        assert "x-sfn-source-sha256" not in r.headers

    def test_a_path_outside_the_allowed_roots_is_rejected(self, client, roots, tmp_path):
        outside = tmp_path / "elsewhere.mov"
        outside.write_bytes(b"\0" * 16)
        assert client.get(f"/api/video-download?path={outside}").status_code == 403

    def test_traversal_out_of_an_allowed_root_is_rejected(self, client, roots, tmp_path):
        input_dir, _ = roots
        outside = tmp_path / "secret.mov"
        outside.write_bytes(b"\0" * 16)
        traversal = f"{input_dir}/../{outside.name}"
        assert client.get(f"/api/video-download?path={traversal}").status_code == 403

    def test_a_relative_path_is_rejected(self, client, roots):
        assert client.get("/api/video-download?path=IMG_0001.MOV").status_code == 400

    def test_a_non_video_extension_is_rejected(self, client, roots):
        input_dir, _ = roots
        doc = input_dir / "notes.txt"
        doc.write_text("not a video")
        assert client.get(f"/api/video-download?path={doc}").status_code == 400

    def test_a_missing_file_is_404(self, client, roots):
        input_dir, _ = roots
        assert client.get(f"/api/video-download?path={input_dir / 'gone.mov'}").status_code == 404

    def test_a_directory_is_not_servable(self, client, roots):
        input_dir, _ = roots
        d = input_dir / "adir.mov"
        d.mkdir()
        assert client.get(f"/api/video-download?path={d}").status_code == 404

    def test_both_advertised_urls_escape_the_path(self, client, roots):
        # iPhone corpora carry '#' and spaces in filenames; an unescaped query
        # string truncates at the '#' and serves — or fails on — another file.
        input_dir, _ = roots
        awkward = _write_clip(input_dir / "IMG #7 &b.mov", "mov")
        body = client.get("/api/video-playback-info", params={"path": str(awkward)}).json()
        assert "#" not in body["download_url"]
        assert "#" not in body["playback_url"]
        assert client.get(body["download_url"]).content == awkward.read_bytes()
        assert client.get(body["playback_url"]).status_code == 200

    def test_playback_info_advertises_the_download(self, client, mov):
        body = client.get(f"/api/video-playback-info?path={mov}").json()
        assert body["download_url"].startswith("/api/video-download?path=")
        assert client.get(body["download_url"]).status_code == 200


# ---------------------------------------------------------------------------
# Digest correctness (spec §7.1)
# ---------------------------------------------------------------------------


class TestSourceDigest:
    def test_digest_matches_a_direct_hash(self, mov):
        assert vp_digest._source_digest(mov) == hash_file(mov)

    def test_second_call_is_served_from_the_cache(self, mov):
        vp_digest._source_digest(mov)
        with patch.object(vp_digest, "hash_file") as direct:
            with patch("scalar_forensic.embedder.hash_file_both") as both:
                assert vp_digest._source_digest(mov) == hash_file(mov)
        direct.assert_not_called()
        both.assert_not_called()

    def test_the_cache_survives_a_new_process(self, mov, roots):
        _, _ = roots
        vp_digest._source_digest(mov)
        vp_digest._reset_hash_cache()  # as a restart would
        with patch("scalar_forensic.embedder.hash_file_both") as both:
            assert vp_digest._source_digest(mov) == hash_file(mov)
        both.assert_not_called()

    def test_a_changed_file_is_rehashed(self, mov):
        first = vp_digest._source_digest(mov)
        os.utime(mov, (2_000_000, 2_000_000))
        mov.write_bytes(mov.read_bytes() + b"tampered")
        second = vp_digest._source_digest(mov)
        assert second != first
        assert second == hash_file(mov)

    def test_disabled_cache_still_answers(self, mov, monkeypatch):
        monkeypatch.setenv("SFN_HASH_CACHE_PATH", "")
        vp_digest._reset_hash_cache()
        assert vp_digest._hash_cache_for(Settings()) is None
        assert vp_digest._source_digest(mov) == hash_file(mov)

    def test_unwritable_db_falls_back_to_a_direct_hash(self, mov, tmp_path, monkeypatch):
        # A directory where the DB file should be: SQLite cannot open it.
        db = tmp_path / "unwritable.db"
        db.mkdir()
        monkeypatch.setenv("SFN_HASH_CACHE_PATH", str(db))
        vp_digest._reset_hash_cache()
        assert vp_digest._source_digest(mov) == hash_file(mov)

    def test_a_broken_cache_is_not_reopened_per_request(self, mov, tmp_path, monkeypatch):
        db = tmp_path / "unwritable.db"
        db.mkdir()
        monkeypatch.setenv("SFN_HASH_CACHE_PATH", str(db))
        vp_digest._reset_hash_cache()
        with patch.object(vp_digest, "HashCache", side_effect=OSError("nope")) as ctor:
            vp_digest._source_digest(mov)
            vp_digest._source_digest(mov)
        assert ctor.call_count == 1

    def test_a_failing_lookup_falls_back_instead_of_raising(self, mov):
        cache = MagicMock()
        cache.get_or_hash.side_effect = OSError("disk gone")
        with patch.object(vp_digest, "_hash_cache_for", return_value=cache):
            assert vp_digest._source_digest(mov) == hash_file(mov)

    def test_a_failing_flush_does_not_fail_the_digest(self, mov):
        cache = MagicMock()
        cache.get_or_hash.return_value = ("a" * 64, False)
        cache.flush.side_effect = OSError("read-only")
        with patch.object(vp_digest, "_hash_cache_for", return_value=cache):
            assert vp_digest._source_digest(mov) == "a" * 64

    def test_the_request_path_does_not_block_the_event_loop(self, client, mov):
        # The digest is computed in a worker thread, never inline in the handler.
        calls: list[str] = []
        real = vp_digest._source_digest

        def spy(p, settings=None):
            calls.append(threading.current_thread().name)
            return real(p, settings)

        with patch.object(vp_routes, "_source_digest", spy):
            assert client.get(f"/api/video-playback-info?path={mov}").status_code == 200
        assert calls and all(name != "MainThread" for name in calls)


# ---------------------------------------------------------------------------
# Stale evidence (spec §7.1)
# ---------------------------------------------------------------------------


class TestStaleEvidence:
    def test_matching_indexed_hash_clears_the_file(self, client, mov):
        url = f"/api/video-playback-info?path={mov}&video_hash={hash_file(mov)}"
        body = client.get(url).json()
        assert body["stale_evidence"] is False
        assert body["indexed_video_hash"] == hash_file(mov)
        assert body["stale_reason"] is None

    def test_a_differing_indexed_hash_is_reported_stale(self, client, mov):
        stale = "b" * 64
        body = client.get(f"/api/video-playback-info?path={mov}&video_hash={stale}").json()
        assert body["stale_evidence"] is True
        assert body["indexed_video_hash"] == stale
        # The computed digest is the one displayed; the indexed hash never
        # substitutes for it (spec §7.1).
        assert body["video_sha256"] == hash_file(mov)
        assert stale[:12] in body["stale_reason"]

    def test_a_file_edited_after_indexing_is_caught(self, client, mov):
        indexed = hash_file(mov)
        vp_digest._source_digest(mov)  # warm the cache, as a first view would
        mov.write_bytes(mov.read_bytes() + b"tampered")
        body = client.get(f"/api/video-playback-info?path={mov}&video_hash={indexed}").json()
        assert body["stale_evidence"] is True
        assert body["video_sha256"] == hash_file(mov)

    def test_no_indexed_hash_means_unchecked_not_clean(self, client, mov):
        body = client.get(f"/api/video-playback-info?path={mov}").json()
        assert body["stale_evidence"] is None
        assert body["indexed_video_hash"] is None

    def test_a_malformed_indexed_hash_is_treated_as_unchecked(self, client, mov):
        body = client.get(f"/api/video-playback-info?path={mov}&video_hash=nonsense").json()
        assert body["stale_evidence"] is None


# ---------------------------------------------------------------------------
# Cache bounding
# ---------------------------------------------------------------------------


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
FINGERPRINT = "f" * 64


@pytest.fixture(autouse=True)
def _clean_cache_state():
    """Module-level lease, pin, sweep, probe, admission and job state (CLAUDE.md).

    Autouse because every one of these is silent when it leaks: an inherited
    lease changes what a later eviction test evicts, an inherited probe answer
    decides a pipeline nobody selected, an admission counter left above zero
    turns a later chunk request into a spurious 503, and a full-video job left in
    the runner makes the next test's POST join it instead of starting one.  All
    of them read as flakes.
    """
    _reset_module_state()
    yield
    _reset_module_state()


def _reset_module_state() -> None:
    vp_cache.reset_leases()
    vp_cache.artifact_locks.reset()
    vp_cache._reset_sweep()
    vp_capability.reset_cache()
    vp_jobs.admission.reset()
    vp_jobs.runner.reset()
    vp_routes.reset_substitutions()


def _pipeline(**kw) -> vp_capability.Pipeline:
    base = vp_capability.select(Settings(), _capability(), hdr=False)
    return dataclasses.replace(base, **kw) if kw else base


def _video(cache_dir: Path, digest: str, *, size: int, mtime: float, name: str = "rewrap.mp4"):
    """Write one artifact under a per-video directory and date it."""
    p = cache_dir / digest / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)
    os.utime(p, (mtime, mtime))
    return p


class TestCacheKey:
    def test_key_is_the_hash_of_both_halves(self):
        expected = hashlib.sha256(f"source={DIGEST_A}\npipeline={FINGERPRINT}".encode()).hexdigest()
        assert vp_cache.cache_key(DIGEST_A, FINGERPRINT) == expected

    def test_two_sources_under_one_pipeline_are_two_keys(self):
        assert vp_cache.cache_key(DIGEST_A, FINGERPRINT) != vp_cache.cache_key(
            DIGEST_B, FINGERPRINT
        )

    def test_two_pipelines_over_one_source_are_two_keys(self):
        # The §6.1 defect this exists to prevent: one key holding two pictures.
        other = "e" * 64
        assert vp_cache.cache_key(DIGEST_A, FINGERPRINT) != vp_cache.cache_key(DIGEST_A, other)

    def test_the_two_halves_are_not_interchangeable(self):
        assert vp_cache.cache_key(DIGEST_A, FINGERPRINT) != vp_cache.cache_key(
            FINGERPRINT, DIGEST_A
        )

    @pytest.mark.parametrize("bad", ["", "zz", "A" * 64, "a" * 63])
    def test_a_non_digest_half_is_rejected(self, bad):
        with pytest.raises(ValueError):
            vp_cache.cache_key(bad, FINGERPRINT)
        with pytest.raises(ValueError):
            vp_cache.cache_key(DIGEST_A, bad)

    def test_a_real_pipeline_fingerprint_keys_an_artifact_dir(self, tmp_path):
        pipeline = _pipeline()
        d = vp_cache.artifact_dir(tmp_path, DIGEST_A, pipeline.fingerprint())
        assert d.parent == tmp_path / DIGEST_A
        assert d.name == vp_cache.cache_key(DIGEST_A, pipeline.fingerprint())

    def test_the_gpu_fallback_lands_in_a_different_artifact_dir(self, tmp_path):
        gpu = _pipeline(hwaccel="cuda", encoder="h264_nvenc")
        cpu = _pipeline()
        assert vp_cache.artifact_dir(
            tmp_path, DIGEST_A, gpu.fingerprint()
        ) != vp_cache.artifact_dir(tmp_path, DIGEST_A, cpu.fingerprint())


class TestPlaybackLease:
    def test_an_unregistered_video_is_none_not_expired(self):
        # Three-state on purpose: "never registered here" is not "the heartbeat
        # stopped", and a boolean would report both as "not being watched".
        assert vp_cache.lease_state(DIGEST_A).state == "none"

    def test_a_renewed_lease_is_held(self):
        state = vp_cache.renew_lease(DIGEST_A, 60)
        assert state.state == "held"
        assert 0 < state.seconds_remaining <= 60

    def test_a_lapsed_heartbeat_expires_the_lease(self, monkeypatch):
        base = time.monotonic()
        monkeypatch.setattr(vp_cache.time, "monotonic", lambda: base)
        vp_cache.renew_lease(DIGEST_A, 10)
        monkeypatch.setattr(vp_cache.time, "monotonic", lambda: base + 11)
        assert vp_cache.lease_state(DIGEST_A).state == "expired"
        assert DIGEST_A not in vp_cache.protected_videos()

    def test_release_drops_the_lease(self):
        vp_cache.renew_lease(DIGEST_A, 60)
        assert vp_cache.release_lease(DIGEST_A).state == "none"

    def test_a_zero_ttl_is_rejected(self):
        with pytest.raises(ValueError):
            vp_cache.renew_lease(DIGEST_A, 0)

    def test_a_pin_protects_and_unwinds(self):
        with vp_cache.pin(DIGEST_A):
            assert DIGEST_A in vp_cache.protected_videos()
        assert DIGEST_A not in vp_cache.protected_videos()

    def test_nested_pins_refcount(self):
        with vp_cache.pin(DIGEST_A), vp_cache.pin(DIGEST_A):
            pass
        assert vp_cache._pins == {}


class TestLeaseEndpoint:
    def test_the_endpoint_registers_and_refreshes(self, client, mov):
        r = client.post(f"/api/video-lease?path={mov}")
        assert r.status_code == 200
        body = r.json()
        assert body["video_sha256"] == hash_file(mov)
        assert body["state"] == "held"
        assert body["lease_seconds"] == 120

    def test_release_reports_none(self, client, mov):
        client.post(f"/api/video-lease?path={mov}")
        assert client.post(f"/api/video-lease?path={mov}&release=true").json()["state"] == "none"

    def test_a_path_outside_the_allowed_roots_is_rejected(self, client, roots, tmp_path):
        outside = tmp_path / "outside.mov"
        outside.write_bytes(b"\0")
        assert client.post(f"/api/video-lease?path={outside}").status_code == 403

    def test_playback_takes_a_lease_before_it_serves(self, client, mov):
        assert client.get(f"/api/video-playback?path={mov}").status_code == 200
        assert vp_cache.lease_state(hash_file(mov)).state == "held"


class TestCacheEviction:
    def test_oldest_videos_are_evicted_first(self, tmp_path):
        cache_dir = tmp_path / "cache"
        old = _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        mid = _video(cache_dir, DIGEST_B, size=100, mtime=2_000)
        new = _video(cache_dir, DIGEST_C, size=100, mtime=3_000)
        report = vp_cache.evict(cache_dir, 150, protect=set())
        assert report.videos_removed == 2
        assert report.bytes_freed == 200
        assert report.bytes_after == 100
        assert not old.exists() and not mid.exists() and new.exists()

    def test_a_whole_video_goes_never_one_chunk(self, tmp_path):
        # The §6.2 defect #148 contained rather than fixed: a *.mp4 glob picks
        # single files, and a video that loses one chunk mid-play is broken
        # playback rather than a freed byte.
        cache_dir = tmp_path / "cache"
        key = vp_cache.cache_key(DIGEST_A, FINGERPRINT)
        c0 = _video(cache_dir, DIGEST_A, size=100, mtime=1_000, name=f"{key}/c0.000.mp4")
        c1 = _video(cache_dir, DIGEST_A, size=100, mtime=1_000, name=f"{key}/c30.000.mp4")
        keeper = _video(cache_dir, DIGEST_B, size=100, mtime=3_000)
        vp_cache.evict(cache_dir, 150, protect=set())
        assert not c0.exists() and not c1.exists()
        assert not (cache_dir / DIGEST_A).exists()
        assert keeper.exists()

    def test_recency_is_the_newest_file_in_the_video(self, tmp_path):
        # One chunk served now makes the whole video recently played.
        cache_dir = tmp_path / "cache"
        stale = _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        _video(cache_dir, DIGEST_B, size=100, mtime=500, name="k/c0.000.mp4")
        fresh = _video(cache_dir, DIGEST_B, size=100, mtime=9_000, name="k/c30.000.mp4")
        vp_cache.evict(cache_dir, 250, protect=set())
        assert not stale.exists()
        assert fresh.exists()

    def test_a_leased_video_is_never_evicted(self, tmp_path):
        cache_dir = tmp_path / "cache"
        watched = _video(cache_dir, DIGEST_A, size=500, mtime=1_000)
        other = _video(cache_dir, DIGEST_B, size=100, mtime=2_000)
        vp_cache.renew_lease(DIGEST_A, 60)
        report = vp_cache.evict(cache_dir, 10)
        assert watched.exists()
        assert not other.exists()
        assert report.protected == (DIGEST_A,)
        assert report.over_ceiling is True

    def test_an_expired_lease_stops_protecting(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        watched = _video(cache_dir, DIGEST_A, size=500, mtime=1_000)
        base = time.monotonic()
        monkeypatch.setattr(vp_cache.time, "monotonic", lambda: base)
        vp_cache.renew_lease(DIGEST_A, 10)
        monkeypatch.setattr(vp_cache.time, "monotonic", lambda: base + 11)
        vp_cache.evict(cache_dir, 10)
        assert not watched.exists()

    def test_a_pinned_video_is_never_evicted(self, tmp_path):
        cache_dir = tmp_path / "cache"
        writing = _video(cache_dir, DIGEST_A, size=500, mtime=1_000)
        with vp_cache.pin(DIGEST_A):
            vp_cache.evict(cache_dir, 10)
        assert writing.exists()

    def test_part_files_count_against_the_ceiling(self, tmp_path):
        # §6.3: an in-flight .part is bytes on disk.  Left out of the accounting
        # the store here would measure 150 against a 950 ceiling and evict
        # nothing while holding 1000.
        cache_dir = tmp_path / "cache"
        old = _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        _video(cache_dir, DIGEST_B, size=50, mtime=2_000)
        _video(cache_dir, DIGEST_B, size=850, mtime=2_000, name="full.mp4.999999.part")
        report = vp_cache.evict(cache_dir, 950, protect=set())
        assert report.bytes_before == 1000
        assert report.videos_removed == 1
        assert not old.exists()

    def test_an_unevictable_overshoot_is_reported_not_forced(self, tmp_path):
        cache_dir = tmp_path / "cache"
        _video(cache_dir, DIGEST_A, size=500, mtime=1_000)
        vp_cache.renew_lease(DIGEST_A, 60)
        report = vp_cache.evict(cache_dir, 100)
        assert report.over_ceiling is True
        assert report.bytes_after == 500
        assert report.videos_removed == 0

    def test_zero_ceiling_disables_eviction(self, tmp_path):
        cache_dir = tmp_path / "cache"
        p = _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        report = vp_cache.evict(cache_dir, 0)
        assert report.videos_removed == 0
        assert report.over_ceiling is False
        assert p.exists()

    def test_a_legacy_top_level_rewrap_is_accounted_and_evictable(self, tmp_path):
        # The layout before the per-video directories.  Still a valid rewrap, so
        # it retires by LRU rather than being deleted on sight.
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        legacy = cache_dir / (DIGEST_A + ".mp4")
        legacy.write_bytes(b"\0" * 100)
        os.utime(legacy, (1_000, 1_000))
        keeper = _video(cache_dir, DIGEST_B, size=100, mtime=3_000)
        report = vp_cache.evict(cache_dir, 150, protect=set())
        assert report.bytes_before == 200
        assert not legacy.exists() and keeper.exists()

    def test_foreign_files_are_neither_counted_nor_deleted(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        stray = cache_dir / "notes.txt"
        stray.write_bytes(b"\0" * 5000)
        _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        report = vp_cache.evict(cache_dir, 50, protect=set())
        assert report.bytes_before == 100
        assert stray.exists()

    def test_a_missing_cache_dir_is_not_an_error(self, tmp_path):
        assert vp_cache.evict(tmp_path / "nope", 100).bytes_before == 0

    def test_cache_stays_under_the_ceiling_after_a_rewrap(self, client, mov, roots):
        _, cache_dir = roots
        filler = _video(cache_dir, DIGEST_A, size=4096, mtime=1_000)
        settings = MagicMock()
        settings.video_cache_dir = cache_dir
        settings.video_cache_max_bytes = 2048
        settings.video_lease_seconds = 120
        settings.hash_cache_path = cache_dir.parent / "hash_cache.db"
        with patch.object(vp_routes, "Settings", return_value=settings):
            r = client.get(f"/api/video-playback?path={mov}")
        assert r.status_code == 200
        assert not filler.exists()
        assert sum(f.stat().st_size for f in cache_dir.rglob("*.mp4")) <= 2048


class TestCeilingRefusal:
    """§6.3 with the §16 ruling: 50% of SFN_VIDEO_CACHE_MAX_BYTES."""

    def _settings(self, monkeypatch, ceiling: int, height: int = 1080) -> Settings:
        monkeypatch.setenv("SFN_VIDEO_CACHE_MAX_BYTES", str(ceiling))
        monkeypatch.setenv("SFN_VIDEO_OUTPUT_HEIGHT", str(height))
        return Settings()

    def test_estimate_scales_by_the_output_area(self):
        # The cap is min(ih, H), so a 2160p source loses three quarters of its
        # pixels at 1080 and a 720p source is passed through untouched.
        info = {"duration_ms": 10_000, "bit_rate": 8_000_000, "video_height": 2160}
        assert vp_cache.estimate_full_output_bytes(info, 1080) == 10_000_000 // 4
        assert (
            vp_cache.estimate_full_output_bytes({**info, "video_height": 720}, 1080) == 10_000_000
        )

    @pytest.mark.parametrize("missing", ["duration_ms", "bit_rate", "video_height"])
    def test_a_missing_input_yields_no_estimate_rather_than_a_guess(self, missing):
        info = {"duration_ms": 10_000, "bit_rate": 8_000_000, "video_height": 1080}
        assert vp_cache.estimate_full_output_bytes({**info, missing: None}, 1080) is None

    def test_a_job_that_fits_is_allowed(self, monkeypatch):
        s = self._settings(monkeypatch, 8 * 1024**3)
        info = {"duration_ms": 60_000, "bit_rate": 8_000_000, "video_height": 1080}
        verdict = vp_cache.check_ceiling(s, info)
        assert verdict.state == "fits"
        assert verdict.allowed is True
        assert verdict.limit_bytes == 4 * 1024**3

    def test_a_job_over_half_the_ceiling_is_refused_with_the_estimate(self, monkeypatch):
        s = self._settings(monkeypatch, 1024**3)
        info = {"duration_ms": 3_600_000, "bit_rate": 20_000_000, "video_height": 1080}
        verdict = vp_cache.check_ceiling(s, info)
        assert verdict.state == "refused"
        assert verdict.allowed is False
        assert verdict.estimate_bytes == 9_000_000_000
        assert "download the original" in verdict.reason.lower()

    def test_the_boundary_is_half_not_all_of_the_ceiling(self, monkeypatch):
        s = self._settings(monkeypatch, 1000)
        under = {"duration_ms": 1000, "bit_rate": 3_900, "video_height": 1080}
        over = {"duration_ms": 1000, "bit_rate": 4_100, "video_height": 1080}
        assert vp_cache.check_ceiling(s, under).state == "fits"
        assert vp_cache.check_ceiling(s, over).state == "refused"

    def test_an_unestimable_source_is_unknown_not_refused(self, monkeypatch):
        # Three-state: "this file would not say how big it is" is a different
        # sentence from "this video is too big for the cache".
        s = self._settings(monkeypatch, 8 * 1024**3)
        verdict = vp_cache.check_ceiling(s, {"duration_ms": None})
        assert verdict.state == "unknown"
        assert verdict.allowed is False
        assert verdict.estimate_bytes is None
        assert "cannot be estimated" in verdict.reason

    def test_no_ceiling_means_no_invariant_to_protect(self, monkeypatch):
        s = self._settings(monkeypatch, 0)
        info = {"duration_ms": 3_600_000, "bit_rate": 100_000_000, "video_height": 1080}
        assert vp_cache.check_ceiling(s, info).state == "fits"

    def test_the_stream_report_carries_the_estimate_inputs(self, mov):
        info = vp_codecs._stream_report(mov)
        assert info["video_height"] == 48
        assert info["video_width"] == 64
        assert info["duration_ms"] is not None


class TestPurge:
    def test_purge_one_media_leaves_the_rest(self, tmp_path):
        cache_dir = tmp_path / "cache"
        gone = _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        chunk = _video(cache_dir, DIGEST_A, size=50, mtime=1_000, name="k/c0.000.mp4")
        kept = _video(cache_dir, DIGEST_B, size=100, mtime=1_000)
        report = vp_cache.purge(cache_dir, media=DIGEST_A)
        assert report.videos == 1
        assert report.files == 2
        assert report.bytes_freed == 150
        assert report.digests == (DIGEST_A,)
        assert not gone.exists() and not chunk.exists() and kept.exists()

    def test_purge_all_empties_the_store(self, tmp_path):
        cache_dir = tmp_path / "cache"
        _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        _video(cache_dir, DIGEST_B, size=100, mtime=1_000)
        report = vp_cache.purge(cache_dir, all_=True)
        assert report.videos == 2
        assert report.bytes_freed == 200
        assert vp_cache.scan(cache_dir) == []

    def test_a_lease_does_not_stop_an_explicit_purge(self, tmp_path):
        # A lease bounds *automatic* eviction.  An examiner deleting a rendering
        # on purpose is the one act §6.4 keeps explicit precisely so it happens.
        cache_dir = tmp_path / "cache"
        p = _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        vp_cache.renew_lease(DIGEST_A, 60)
        assert vp_cache.purge(cache_dir, media=DIGEST_A).videos == 1
        assert not p.exists()

    def test_purging_an_unknown_media_is_a_no_op_not_an_error(self, tmp_path):
        cache_dir = tmp_path / "cache"
        _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        assert vp_cache.purge(cache_dir, media=DIGEST_B).videos == 0

    @pytest.mark.parametrize(
        "kw", [{}, {"media": DIGEST_A, "all_": True}, {"media": "not-a-digest"}]
    )
    def test_an_ambiguous_or_malformed_scope_is_rejected(self, tmp_path, kw):
        with pytest.raises(ValueError):
            vp_cache.purge(tmp_path, **kw)

    def test_the_cli_purges_one_media(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        p = _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", str(cache_dir))
        result = CliRunner().invoke(cli.video_app, ["purge", "--media", DIGEST_A])
        assert result.exit_code == 0, result.output
        assert DIGEST_A in result.output
        assert not p.exists()

    def test_the_cli_confirms_before_purging_everything(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        p = _video(cache_dir, DIGEST_A, size=100, mtime=1_000)
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", str(cache_dir))
        result = CliRunner().invoke(cli.video_app, ["purge", "--all"], input="n\n")
        assert result.exit_code == 1
        assert p.exists()

    def test_the_cli_rejects_both_scopes_at_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", str(tmp_path))
        result = CliRunner().invoke(cli.video_app, ["purge", "--media", DIGEST_A, "--all"])
        assert result.exit_code == 1
        assert "exactly one" in result.output

    def test_the_cli_reports_a_disabled_cache(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", "")
        result = CliRunner().invoke(cli.video_app, ["purge", "--media", DIGEST_A])
        assert result.exit_code == 1
        assert "SFN_VIDEO_CACHE_DIR" in result.output


class TestArtifactLocks:
    def test_the_lock_table_does_not_accumulate(self):
        # §10.4: locks must not accumulate unboundedly.  The dict this replaced
        # grew one entry per source digest ever seen and was never cleared.
        async def run():
            for i in range(50):
                async with vp_cache.artifact_locks.hold(f"{i:064x}"):
                    pass

        asyncio.run(run())
        assert len(vp_cache.artifact_locks) == 0

    def test_concurrent_holders_share_one_lock_and_serialise(self):
        order = []

        async def run():
            async def worker(n):
                async with vp_cache.artifact_locks.hold(DIGEST_A):
                    order.append(("in", n))
                    await asyncio.sleep(0)
                    order.append(("out", n))

            await asyncio.gather(*(worker(n) for n in range(3)))

        asyncio.run(run())
        assert [k for k, _ in order] == ["in", "out"] * 3
        assert len(vp_cache.artifact_locks) == 0

    def test_a_raising_holder_still_releases_the_entry(self):
        async def run():
            with contextlib.suppress(RuntimeError):
                async with vp_cache.artifact_locks.hold(DIGEST_A):
                    raise RuntimeError("boom")

        asyncio.run(run())
        assert len(vp_cache.artifact_locks) == 0


class TestAtomicPublication:
    def test_publish_renames_and_survives(self, tmp_path):
        part = tmp_path / "x.mp4.1.part"
        part.write_bytes(b"payload")
        dst = tmp_path / "x.mp4"
        vp_cache.publish(part, dst)
        assert not part.exists()
        assert dst.read_bytes() == b"payload"

    def test_a_part_file_from_a_dead_writer_is_swept(self, tmp_path):
        cache_dir = tmp_path / "cache"
        d = cache_dir / DIGEST_A
        d.mkdir(parents=True)
        dead = d / "full.mp4.999999.part"
        dead.write_bytes(b"\0")
        assert vp_cache.sweep_orphaned_parts(cache_dir) == 1
        assert not dead.exists()

    def test_a_part_file_from_a_live_writer_is_left_alone(self, tmp_path):
        cache_dir = tmp_path / "cache"
        d = cache_dir / DIGEST_A
        d.mkdir(parents=True)
        live = d / f"full.mp4.{os.getpid()}.part"
        live.write_bytes(b"\0")
        assert vp_cache.sweep_orphaned_parts(cache_dir) == 0
        assert live.exists()

    def test_a_published_artifact_is_not_swept(self, tmp_path):
        cache_dir = tmp_path / "cache"
        p = _video(cache_dir, DIGEST_A, size=10, mtime=1_000)
        assert vp_cache.sweep_orphaned_parts(cache_dir) == 0
        assert p.exists()

    def test_the_sweep_runs_once_per_process(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(vp_cache, "sweep_orphaned_parts", lambda d: calls.append(d))
        vp_cache.ensure_swept(tmp_path)
        vp_cache.ensure_swept(tmp_path)
        assert calls == [tmp_path]

    def test_playback_sweeps_the_store_on_first_use(self, client, mov, roots):
        _, cache_dir = roots
        d = cache_dir / DIGEST_A
        d.mkdir(parents=True)
        orphan = d / "full.mp4.999999.part"
        orphan.write_bytes(b"\0")
        assert client.get(f"/api/video-playback?path={mov}").status_code == 200
        assert not orphan.exists()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestPlaybackSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("SFN_VIDEO_CACHE_DIR", raising=False)
        monkeypatch.delenv("SFN_VIDEO_CACHE_MAX_BYTES", raising=False)
        monkeypatch.delenv("SFN_VIDEO_LEASE_SECONDS", raising=False)
        s = Settings()
        assert s.video_cache_dir is not None
        assert s.video_cache_dir.name == "video_cache"
        assert s.video_cache_max_bytes == 8 * 1024 * 1024 * 1024
        assert s.video_lease_seconds == 120

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", str(tmp_path / "elsewhere"))
        monkeypatch.setenv("SFN_VIDEO_CACHE_MAX_BYTES", "1024")
        s = Settings()
        assert s.video_cache_dir == tmp_path / "elsewhere"
        assert s.video_cache_max_bytes == 1024

    def test_empty_dir_disables_the_cache(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", "")
        assert Settings().video_cache_dir is None

    def test_negative_ceiling_is_rejected(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_CACHE_MAX_BYTES", "-1")
        with pytest.raises(ValueError, match="SFN_VIDEO_CACHE_MAX_BYTES"):
            Settings()

    @pytest.mark.parametrize("value", ["0", "3601"])
    def test_an_out_of_range_lease_is_rejected(self, monkeypatch, value):
        monkeypatch.setenv("SFN_VIDEO_LEASE_SECONDS", value)
        with pytest.raises(ValueError, match="SFN_VIDEO_LEASE_SECONDS"):
            Settings()


class TestTranscodeSettings:
    """The §12 defaults, each of which a §3.5 number is behind."""

    def test_defaults(self, monkeypatch):
        for name in (
            "SFN_FFMPEG_PATH",
            "SFN_VIDEO_HWACCEL",
            "SFN_VIDEO_OUTPUT_HEIGHT",
            "SFN_VIDEO_CHUNK_SECONDS",
            "SFN_VIDEO_MAX_WORKERS",
        ):
            monkeypatch.delenv(name, raising=False)
        s = Settings()
        assert s.ffmpeg_path == "ffmpeg"
        assert s.video_hwaccel == "auto"
        assert s.video_output_height == 1080
        assert s.video_chunk_seconds == 30
        assert s.video_max_workers == 2

    @pytest.mark.parametrize(
        ("name", "value", "match"),
        [
            ("SFN_VIDEO_HWACCEL", "vaapi", "SFN_VIDEO_HWACCEL"),
            ("SFN_VIDEO_OUTPUT_HEIGHT", "100", "SFN_VIDEO_OUTPUT_HEIGHT"),
            ("SFN_VIDEO_CHUNK_SECONDS", "0", "SFN_VIDEO_CHUNK_SECONDS"),
            ("SFN_VIDEO_CHUNK_SECONDS", "601", "SFN_VIDEO_CHUNK_SECONDS"),
            ("SFN_VIDEO_MAX_WORKERS", "0", "SFN_VIDEO_MAX_WORKERS"),
        ],
    )
    def test_invalid_values_are_rejected(self, monkeypatch, name, value, match):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=match):
            Settings()

    def test_hwaccel_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_HWACCEL", " CUDA ")
        assert Settings().video_hwaccel == "cuda"


# ---------------------------------------------------------------------------
# Capability probe and pipeline fingerprint (spec §6.1, §8)
# ---------------------------------------------------------------------------


def _capability(**kw) -> vp_capability.Capability:
    base = {
        "ffmpeg_path": "ffmpeg",
        "ffmpeg_version": "ffmpeg version 6.1.1-3ubuntu5",
        "encoder": "libx264",
        "hwaccel": "none",
        "tonemap_ok": True,
        "notes": (),
    }
    return vp_capability.Capability(**{**base, **kw})


class TestPipelineFingerprint:
    """§6.1: the half of the cache key that is not the source."""

    def test_every_field_is_hashed(self, monkeypatch):
        """A field added to Pipeline must change the fingerprint, or it is a lie.

        The point of this test is not the arithmetic — it is that adding a
        pixel-affecting setting to the pipeline without putting it in the key
        would let one cache entry serve two renderings under one label, which is
        exactly the defect §6.1 names.
        """
        pipeline = vp_capability.select(Settings(), _capability(), hdr=True)
        base = pipeline.fingerprint()
        for f in dataclasses.fields(pipeline):
            value = getattr(pipeline, f.name)
            if isinstance(value, tuple):
                mutated = (*value, "x")
            elif isinstance(value, int):
                mutated = value + 1
            else:
                mutated = f"{value}-x"
            other = dataclasses.replace(pipeline, **{f.name: mutated})
            assert other.fingerprint() != base, f"{f.name} is not in the fingerprint"

    def test_field_set_is_pinned(self):
        """Adding a field is a deliberate act, not a silent cache invalidation."""
        assert {f.name for f in dataclasses.fields(vp_capability.Pipeline)} == {
            "hwaccel",
            "decoder",
            "filter_chain",
            "encoder",
            "rate_control",
            "output_height",
            "chunk_seconds",
            "audio",
            "ffmpeg_version",
        }

    def test_fingerprint_is_stable_across_instances(self):
        s = Settings()
        a = vp_capability.select(s, _capability(), hdr=True)
        b = vp_capability.select(s, _capability(), hdr=True)
        assert a is not b
        assert a.fingerprint() == b.fingerprint()

    def test_hdr_and_sdr_are_different_pipelines(self):
        s = Settings()
        hdr = vp_capability.select(s, _capability(), hdr=True)
        sdr = vp_capability.select(s, _capability(), hdr=False)
        assert hdr.fingerprint() != sdr.fingerprint()
        assert vp_capability.TONEMAP_CHAIN in hdr.filter_chain
        assert vp_capability.TONEMAP_CHAIN not in sdr.filter_chain

    def test_gpu_fallback_is_a_different_key(self):
        """§8's job-time fallback must not reuse the GPU pipeline's cache entry."""
        s = Settings()
        gpu = vp_capability.select(s, _capability(encoder="h264_nvenc", hwaccel="cuda"), hdr=True)
        cpu = vp_capability.select(s, _capability(), hdr=True)
        assert gpu.fingerprint() != cpu.fingerprint()

    def test_chunk_length_moves_the_key(self, monkeypatch):
        s = Settings()
        monkeypatch.setenv("SFN_VIDEO_CHUNK_SECONDS", "15")
        assert (
            vp_capability.select(s, _capability(), hdr=True).fingerprint()
            != vp_capability.select(Settings(), _capability(), hdr=True).fingerprint()
        )

    def test_describe_carries_the_whole_pipeline(self):
        """§7.2: the label records what ran, not a summary of it."""
        described = vp_capability.select(Settings(), _capability(), hdr=True).describe()
        assert described["encoder"] == "libx264"
        assert described["tone_mapped"] is True
        assert described["ffmpeg_version"].startswith("ffmpeg version")
        assert len(described["fingerprint"]) == 64


class TestOutputResolutionPolicy:
    """§16's 1080p cap: a cap, never a target."""

    def test_scale_filter_never_upscales(self):
        assert vp_capability._scale_filter(1080) == "scale=-2:'min(ih,1080)'"

    def test_height_comes_from_settings(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_OUTPUT_HEIGHT", "720")
        pipeline = vp_capability.select(Settings(), _capability(), hdr=True)
        assert "min(ih,720)" in pipeline.filter_chain
        assert pipeline.output_height == 720

    def test_scale_precedes_the_tonemap(self):
        """Cheaper, and `ih` is post-autorotate — the property §3.1's GPU path lost."""
        chain = vp_capability.select(Settings(), _capability(), hdr=True).filter_chain
        assert chain.index("scale=-2") < chain.index("zscale")


class TestHdrDetection:
    def test_hlg_and_pq_are_hdr(self):
        assert vp_capability.is_hdr({"video_color_trc": "arib-std-b67"})
        assert vp_capability.is_hdr({"video_color_trc": "smpte2084"})

    def test_bt709_is_not(self):
        assert not vp_capability.is_hdr({"video_color_trc": "bt709"})

    def test_unknown_transfer_is_not_hdr(self):
        """Not-stated must not render as HDR: tone-mapping an SDR picture darkens it."""
        assert not vp_capability.is_hdr({})
        assert not vp_capability.is_hdr({"video_color_trc": None})

    def test_probe_reads_the_transfer_off_the_container(self, hdr_hlg_mov):
        info = vp_codecs._stream_report(hdr_hlg_mov)
        assert info["video_color_trc"] == "arib-std-b67"
        assert info["video_color_primaries"] == "bt2020"
        assert vp_capability.is_hdr(info)

    def test_an_sdr_clip_reports_no_hdr_transfer(self, mov):
        assert not vp_capability.is_hdr(vp_codecs._stream_report(mov))


class TestCapabilityRefusal:
    """Three-state (§5): "cannot tone-map" is not "encode it anyway"."""

    def test_no_encoder_is_unavailable(self):
        cap = _capability(encoder=None, tonemap_ok=False, notes=("libx264: not found",))
        assert cap.available is False
        assert "libx264: not found" in cap.unavailable_reason(hdr=False)
        with pytest.raises(RuntimeError):
            vp_capability.select(Settings(), cap, hdr=False)

    def test_a_build_without_tonemap_still_serves_sdr(self):
        cap = _capability(tonemap_ok=False, notes=("zscale not found",))
        assert cap.available is True
        assert cap.unavailable_reason(hdr=False) is None
        assert vp_capability.select(Settings(), cap, hdr=False).encoder == "libx264"

    def test_a_build_without_tonemap_refuses_hdr(self):
        """§3.1's second finding: 8-bit output still labelled HDR is the defect."""
        cap = _capability(tonemap_ok=False, notes=("zscale not found",))
        reason = cap.unavailable_reason(hdr=True)
        assert reason is not None
        assert "tone-map" in reason
        assert "Download the original" in reason
        with pytest.raises(RuntimeError, match="tone-map"):
            vp_capability.select(Settings(), cap, hdr=True)


class TestCapabilityProbe:
    """§8: believe a real encode, never an `-encoders` listing."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        vp_capability.reset_cache()
        yield
        vp_capability.reset_cache()

    def test_missing_binary_is_reported_not_raised(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SFN_FFMPEG_PATH", str(tmp_path / "no-such-ffmpeg"))
        cap = vp_capability.probe(Settings())
        assert cap.available is False
        assert cap.ffmpeg_version is None
        assert "not found" in cap.unavailable_reason(hdr=False)

    @requires_ffmpeg
    def test_real_probe_finds_a_working_pipeline(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_HWACCEL", "none")
        cap = vp_capability.probe(Settings())
        assert cap.available is True
        assert cap.encoder == "libx264"
        assert cap.hwaccel == "none"
        assert cap.ffmpeg_version.startswith("ffmpeg version")
        # The probe ran the tone-map chain, not just an encoder listing.
        assert cap.tonemap_ok is True

    @requires_ffmpeg
    def test_hwaccel_none_never_probes_the_gpu(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_HWACCEL", "none")
        calls: list[str] = []
        real = vp_capability._try_encode

        def spy(path, encoder, *, hdr):
            calls.append(encoder)
            return real(path, encoder, hdr=hdr)

        monkeypatch.setattr(vp_capability, "_try_encode", spy)
        vp_capability.probe(Settings())
        assert "h264_nvenc" not in calls

    @requires_ffmpeg
    def test_result_is_cached_per_process(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_HWACCEL", "none")
        settings = Settings()
        first = vp_capability.capability(settings)
        assert vp_capability.capability(settings) is first
        assert vp_capability.capability(settings, refresh=True) is not first

    @requires_ffmpeg
    def test_an_unknown_encoder_fails_the_probe(self, monkeypatch):
        """The probe's verdict is the exit code, so a bogus encoder must fail."""
        monkeypatch.setitem(vp_capability._RATE_CONTROL, "not_an_encoder", ())
        err = vp_capability._try_encode(Settings().ffmpeg_path, "not_an_encoder", hdr=False)
        assert err is not None and err.startswith("not_an_encoder:")


# ---------------------------------------------------------------------------
# Chunk encode (spec §4, §10) — the two defects §3.1 found
# ---------------------------------------------------------------------------


class TestEncodeCommand:
    """Pure argv construction — no ffmpeg needed, so these always run."""

    def _pipeline(self, **kw):
        return vp_capability.select(Settings(), _capability(**kw), hdr=kw.pop("hdr", True))

    def test_seek_is_an_input_option(self):
        """`-ss` before `-i` is the whole reason chunk cost is flat (§3.2, §3.5)."""
        cmd = vp_encode.build_command(
            Settings(), self._pipeline(), Path("/src.mov"), Path("/out.mp4"), start=90.0
        )
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-ss") + 1] == "90.000"

    def test_output_seeking_is_never_used(self):
        """A `-ss` after `-i` decodes and discards everything before the window."""
        cmd = vp_encode.build_command(
            Settings(), self._pipeline(), Path("/src.mov"), Path("/o.mp4"), start=5, duration=30
        )
        assert cmd.count("-ss") == 1
        assert cmd.index("-t") > cmd.index("-i")

    def test_zero_offset_omits_the_seek(self):
        cmd = vp_encode.build_command(
            Settings(), self._pipeline(), Path("/s.mov"), Path("/o.mp4"), start=0
        )
        assert "-ss" not in cmd

    def test_tonemapped_output_is_tagged_bt709(self):
        cmd = vp_encode.build_command(Settings(), self._pipeline(), Path("/s.mov"), Path("/o.mp4"))
        for flag in ("-colorspace", "-color_primaries", "-color_trc"):
            assert cmd[cmd.index(flag) + 1] == "bt709"

    def test_an_sdr_source_is_not_relabelled(self):
        """Stamping bt709 on an untouched SDR source would mislabel a BT.601 one."""
        pipeline = vp_capability.select(Settings(), _capability(), hdr=False)
        cmd = vp_encode.build_command(Settings(), pipeline, Path("/s.mp4"), Path("/o.mp4"))
        assert "-colorspace" not in cmd

    def test_a_source_without_audio_gets_an(self):
        cmd = vp_encode.build_command(
            Settings(), self._pipeline(), Path("/s.mov"), Path("/o.mp4"), has_audio=False
        )
        assert "-an" in cmd
        assert "-c:a" not in cmd

    def test_faststart_is_always_set(self):
        cmd = vp_encode.build_command(Settings(), self._pipeline(), Path("/s.mov"), Path("/o.mp4"))
        assert cmd[cmd.index("-movflags") + 1] == "+faststart"


@requires_ffmpeg
class TestChunkEncode:
    """Real encodes.  These are the tests §14 requires; they must not skip in CI."""

    @pytest.fixture()
    def cap(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_HWACCEL", "none")
        vp_capability.reset_cache()
        yield vp_capability.probe(Settings())
        vp_capability.reset_cache()

    def test_rotation_survives_the_encode(self, cap, hdr_rotated_mov, tmp_path):
        """§3.1's first finding, pinned: the GPU-filtered path lost rotation.

        The source carries a real display matrix (ffprobe reports it as side
        data).  The decided pipeline decodes and filters in software precisely
        so ffmpeg's autorotate runs, which turns the rotation into *geometry*:
        a 64×96 portrait source comes out 96×64.  If someone reintroduces
        `-hwaccel_output_format cuda`, or drops autorotate, the output stays
        64×96 and this fails.
        """
        assert _display_rotation(hdr_rotated_mov) is not None, "fixture carries no rotation"
        src = _ffprobe_video_stream(hdr_rotated_mov)
        assert (src["width"], src["height"]) == (64, 96)

        out = tmp_path / "chunk.mp4"
        vp_encode.encode(Settings(), cap, hdr_rotated_mov, out, hdr=True)

        result = _ffprobe_video_stream(out)
        assert (result["width"], result["height"]) == (96, 64), (
            "rotation was lost: the source's display matrix did not become geometry"
        )

    def test_output_is_tagged_bt709(self, cap, hdr_hlg_mov, tmp_path):
        """§3.1's second finding, pinned: 8-bit pixels under an HDR label.

        The source is 10-bit, bt2020 primaries, HLG transfer.  A naive
        conversion produces 8-bit output *still* tagged that way, which browsers
        render washed out with lifted blacks.  The tone-mapped path must emit
        bt709 on all three.
        """
        assert _ffprobe_video_stream(hdr_hlg_mov)["color_transfer"] == "arib-std-b67"

        out = tmp_path / "chunk.mp4"
        result = vp_encode.encode(Settings(), cap, hdr_hlg_mov, out, hdr=True)

        probed = _ffprobe_video_stream(out)
        assert probed["color_transfer"] == "bt709"
        assert probed["color_primaries"] == "bt709"
        assert probed["color_space"] == "bt709"
        assert probed["pix_fmt"] == "yuv420p"
        assert result.pipeline.describe()["tone_mapped"] is True

    def test_the_cap_downscales_but_never_upscales(self, cap, hdr_hlg_mov, tmp_path):
        """§16's cap: a 64×96 source is far under 1080 and must come out untouched."""
        out = tmp_path / "small.mp4"
        vp_encode.encode(Settings(), cap, hdr_hlg_mov, out, hdr=True)
        probed = _ffprobe_video_stream(out)
        assert max(probed["width"], probed["height"]) == 96

    def test_chunk_window_is_bounded_by_the_chunk_length(self, cap, hdr_hlg_mov, tmp_path):
        monkeypatch_free = Settings()
        out = tmp_path / "c0.mp4"
        result = vp_encode.encode_chunk(
            monkeypatch_free, cap, hdr_hlg_mov, out, hdr=True, start=0.0
        )
        assert out.exists()
        assert result.pipeline.chunk_seconds == 30
        # The 0.6 s fixture is shorter than the window; ffmpeg ends at the last
        # frame rather than padding, so the final chunk of a source is short.
        with av.open(str(out)) as c:
            assert c.duration / av.time_base < 30

    def test_a_later_chunk_starts_where_it_was_asked_to(self, cap, hdr_hlg_mov, tmp_path):
        """Input seeking must land on the window, not at the file's start."""
        monkeypatch_free = Settings()
        out = tmp_path / "c1.mp4"
        vp_encode.encode(monkeypatch_free, cap, hdr_hlg_mov, out, hdr=True, start=0.3, duration=0.3)
        with av.open(str(out)) as c:
            frames = sum(1 for _ in c.decode(video=0))
        # 0.3 s at 10 fps is ~3 frames; the whole 0.6 s clip would be ~6.
        assert 1 <= frames <= 4

    def test_a_negative_chunk_start_is_rejected(self, cap, hdr_hlg_mov, tmp_path):
        with pytest.raises(ValueError, match="chunk start"):
            vp_encode.encode_chunk(
                Settings(), cap, hdr_hlg_mov, tmp_path / "x.mp4", hdr=True, start=-1.0
            )

    def test_publication_is_atomic(self, cap, hdr_hlg_mov, tmp_path):
        """§10.2: a reader never sees a half-written viewing copy."""
        out = tmp_path / "atomic.mp4"
        vp_encode.encode(Settings(), cap, hdr_hlg_mov, out, hdr=True)
        assert out.exists()
        assert list(tmp_path.glob("*.part")) == []

    def test_a_failed_encode_leaves_no_part_file(self, cap, tmp_path, roots):
        """§10.1/§10.3: a failure must not leave a file that looks like an artifact."""
        input_dir, _ = roots
        broken = input_dir / "broken.mov"
        broken.write_bytes(b"not a video at all")
        out = tmp_path / "never.mp4"
        with pytest.raises(vp_encode.EncodeError):
            vp_encode.encode(Settings(), cap, broken, out, hdr=False)
        assert not out.exists()
        assert list(tmp_path.glob("*.part")) == []

    def test_a_timeout_kills_the_encoder(self, cap, hdr_hlg_mov, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_JOB_TIMEOUT", "1")
        real = subprocess.Popen

        def slow(cmd, **kw):
            return real(["sleep", "30"], **kw)

        monkeypatch.setattr(vp_encode.subprocess, "Popen", slow)
        with pytest.raises(vp_encode.EncodeError, match="timed out"):
            vp_encode.encode(Settings(), cap, hdr_hlg_mov, tmp_path / "t.mp4", hdr=False)
        assert list(tmp_path.glob("*.part")) == []

    def test_the_command_is_recorded_for_reproduction(self, cap, hdr_hlg_mov, tmp_path):
        """§7.2: `sfn-video render` must be able to print what produced a file."""
        result = vp_encode.encode(Settings(), cap, hdr_hlg_mov, tmp_path / "r.mp4", hdr=True)
        assert vp_capability.TONEMAP_CHAIN in " ".join(result.command)
        assert result.command[0] == Settings().ffmpeg_path
        assert result.wall_seconds > 0
        assert result.fell_back is False


class TestGpuFallback:
    """§8: a GPU failure at job time falls back to CPU and says so."""

    @requires_ffmpeg
    def test_fallback_reencodes_on_cpu_and_records_it(self, hdr_hlg_mov, tmp_path, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_HWACCEL", "none")
        cap = dataclasses.replace(
            vp_capability.probe(Settings()), encoder="h264_nvenc", hwaccel="cuda"
        )
        calls: list[list[str]] = []
        real_run = vp_encode._run

        def flaky(cmd, timeout):
            calls.append(cmd)
            if "h264_nvenc" in cmd:
                raise vp_encode.EncodeError("no NVENC capable devices found", command=cmd)
            return real_run(cmd, timeout)

        monkeypatch.setattr(vp_encode, "_run", flaky)
        out = tmp_path / "fallback.mp4"
        result = vp_encode.encode(Settings(), cap, hdr_hlg_mov, out, hdr=True)

        assert len(calls) == 2
        assert result.fell_back is True
        assert "NVENC" in result.fallback_reason
        assert result.pipeline.encoder == "libx264"
        assert out.exists()

    def test_a_cpu_failure_is_not_retried(self, tmp_path, monkeypatch):
        """Only the GPU path falls back; a CPU failure is the failure."""
        cap = _capability()
        calls = []

        def always_fails(cmd, timeout):
            calls.append(cmd)
            raise vp_encode.EncodeError("boom", command=cmd)

        monkeypatch.setattr(vp_encode, "_run", always_fails)
        with pytest.raises(vp_encode.EncodeError, match="boom"):
            vp_encode.encode(Settings(), cap, Path("/s.mov"), tmp_path / "o.mp4", hdr=True)
        assert len(calls) == 1

    def test_the_fallback_lands_under_a_different_cache_key(self, tmp_path, monkeypatch):
        """§6.1: the CPU encode is not the GPU encode, so it is not the same key."""
        gpu = vp_capability.select(
            Settings(), _capability(encoder="h264_nvenc", hwaccel="cuda"), hdr=True
        )
        cpu = vp_capability.select(Settings(), _capability(), hdr=True)
        assert gpu.fingerprint() != cpu.fingerprint()


# ---------------------------------------------------------------------------
# Phase 6 — chunk playback (§4.2), player states (§5), failure matrix (§10.1)
# ---------------------------------------------------------------------------


@pytest.fixture()
def hevc_long_mov(roots):
    """A 6 s HEVC 10-bit .MOV — long enough to hold several chunks."""
    input_dir, _ = roots
    return _write_encoded(input_dir / "IMG_0020.MOV", "mov", "libx265", "yuv420p10le", frames=60)


@pytest.fixture()
def short_chunks(monkeypatch):
    """2 s chunks, so a 6 s fixture is a multi-chunk video without a slow encode."""
    monkeypatch.setenv("SFN_VIDEO_CHUNK_SECONDS", "2")


class TestChunkSnapping:
    """A timecode names a chunk; it never names a new encode (§4.2)."""

    def test_a_timecode_maps_to_the_start_of_its_chunk(self):
        assert vp_routes.chunk_start_for(41.2, 30) == 30.0
        assert vp_routes.chunk_start_for(47.9, 30) == 30.0

    def test_the_first_chunk_starts_at_zero(self):
        assert vp_routes.chunk_start_for(0.0, 30) == 0.0
        assert vp_routes.chunk_start_for(29.999, 30) == 0.0

    def test_a_boundary_belongs_to_the_chunk_it_opens(self):
        assert vp_routes.chunk_start_for(30.0, 30) == 30.0

    def test_two_scrubs_inside_one_chunk_share_one_encode(self):
        # The property, not the arithmetic: this is why two analysts watching
        # the same moment do not produce two artifacts under two keys.
        assert vp_routes.chunk_start_for(41.2, 30) == vp_routes.chunk_start_for(47.9, 30)


@requires_ffmpeg
class TestChunkPlayback:
    def test_a_chunk_is_encoded_and_then_served(self, client, hevc_10bit_mov):
        r = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["player_state"] == "chunk-ready"
        assert body["cached"] is False
        assert body["chunk_start"] == 0.0

        served = client.get(body["chunk_url"])
        assert served.status_code == 200
        assert served.headers["content-type"] == "video/mp4"
        assert served.headers["X-SFN-Playback-Mode"] == "transcode"
        assert len(served.content) > 0

    def test_the_second_request_is_a_cache_hit_and_does_not_re_encode(self, client, hevc_10bit_mov):
        first = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()
        assert first["cached"] is False
        with patch.object(vp_encode, "_run", side_effect=AssertionError("re-encoded a hit")):
            second = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()
        assert second["cached"] is True
        assert second["encode_seconds"] is None
        assert second["chunk_url"] == first["chunk_url"]

    def test_a_get_never_encodes_a_missing_chunk(self, client, hevc_10bit_mov):
        # The double-buffered player fetches bytes with a plain GET; if that
        # could trigger an encode, a preload would start work nobody asked for.
        prepared = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()
        fp = prepared["pipeline_fingerprint"]
        r = client.get(f"/api/video-chunk?path={hevc_10bit_mov}&start=999.000&fp={fp}")
        assert r.status_code == 404
        assert "POST" in r.json()["detail"]

    def test_seeking_to_a_new_position_encodes_a_chunk_there(
        self, client, hevc_long_mov, short_chunks
    ):
        first = client.post(f"/api/video-chunk?path={hevc_long_mov}&t=0.5").json()
        assert first["chunk_start"] == 0.0
        seek = client.post(f"/api/video-chunk?path={hevc_long_mov}&t=4.3").json()
        assert seek["chunk_start"] == 4.0
        assert seek["cached"] is False
        assert seek["chunk_url"] != first["chunk_url"]
        assert client.get(seek["chunk_url"]).status_code == 200

    def test_the_next_chunk_start_is_stated_so_the_player_need_not_compute_it(
        self, client, hevc_long_mov, short_chunks
    ):
        body = client.post(f"/api/video-chunk?path={hevc_long_mov}&t=0").json()
        assert body["next_chunk_start"] == 2.0
        assert body["final_chunk"] is False

    def test_the_final_chunk_states_that_there_is_no_next_one(self, client, hevc_10bit_mov):
        # A prefetch of a chunk past the end would encode nothing and look like
        # a failure; the server says where the video stops instead.
        body = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()
        assert body["next_chunk_start"] is None
        assert body["final_chunk"] is True

    def test_a_chunk_request_holds_the_playback_lease(self, client, hevc_10bit_mov):
        body = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()
        assert vp_cache.lease_state(body["video_sha256"]).state == "held"

    def test_serving_a_chunk_renews_the_lease(self, client, hevc_10bit_mov):
        body = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()
        vp_cache.release_lease(body["video_sha256"])
        assert vp_cache.lease_state(body["video_sha256"]).state == "none"
        client.get(body["chunk_url"])
        assert vp_cache.lease_state(body["video_sha256"]).state == "held"

    def test_the_chunk_is_filed_under_the_pipeline_that_ran(self, client, hevc_10bit_mov):
        body = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()
        settings = Settings()
        expected = vp_cache.artifact_dir(
            settings.video_cache_dir, body["video_sha256"], body["pipeline_fingerprint"]
        ) / vp_cache.chunk_name(0.0)
        assert expected.is_file()

    def test_a_source_with_no_audio_track_is_not_a_failure(self, client, hevc_10bit_mov):
        # §10.1 lists "missing audio track"; the encode answers it with `-an`.
        assert (
            client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()["player_state"]
            == "chunk-ready"
        )


class TestChunkKeyingOnFallback:
    """§6.1: the artifact belongs to the pipeline that ran, not the one asked for."""

    def test_a_gpu_fallback_relocates_the_chunk_under_the_cpu_key(self, tmp_path):
        cache_dir = tmp_path / "cache"
        gpu_fp = "1" * 64
        cpu_fp = "2" * 64
        wrong = vp_cache.artifact_dir(cache_dir, DIGEST_A, gpu_fp) / "c0.000.mp4"
        wrong.parent.mkdir(parents=True)
        wrong.write_bytes(b"chunk")

        landed = vp_cache.relocate_to_pipeline_key(wrong, cache_dir, DIGEST_A, cpu_fp, "c0.000.mp4")

        assert landed == vp_cache.artifact_dir(cache_dir, DIGEST_A, cpu_fp) / "c0.000.mp4"
        assert landed.read_bytes() == b"chunk"
        assert not wrong.exists()

    def test_no_fallback_leaves_the_chunk_where_it_is(self, tmp_path):
        cache_dir = tmp_path / "cache"
        fp = "3" * 64
        p = vp_cache.artifact_dir(cache_dir, DIGEST_A, fp) / "c0.000.mp4"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"chunk")
        assert vp_cache.relocate_to_pipeline_key(p, cache_dir, DIGEST_A, fp, "c0.000.mp4") == p
        assert p.read_bytes() == b"chunk"


class TestPlayerStatesOnPlaybackInfo:
    """§5's states are decided by the server; the client never infers one."""

    def test_a_source_the_browser_decodes_is_playable(self, client, mp4):
        body = client.get(f"/api/video-playback-info?path={mp4}").json()
        assert body["mode"] == "original"
        assert body["player_state"] == "playable"

    def test_a_rewrapped_container_is_still_playable(self, client, mov):
        body = client.get(f"/api/video-playback-info?path={mov}").json()
        assert body["mode"] == "rewrap"
        assert body["player_state"] == "playable"

    def test_an_out_of_allowlist_codec_needs_a_transcode(self, client, hevc_10bit_mov):
        body = client.get(f"/api/video-playback-info?path={hevc_10bit_mov}").json()
        assert body["mode"] == "transcode"
        assert body["player_state"] == "needs-transcode"

    def test_an_unprobeable_container_is_unknown_and_not_needs_transcode(self, client, roots):
        # THE three-state case.  A file that will not open has said nothing
        # about whether it plays.  `#147` shipped "unknown displayed as
        # mismatch" in an evidence viewer; this is the same defect one layer up.
        input_dir, _ = roots
        broken = input_dir / "truncated.mov"
        broken.write_bytes(b"\x00" * 64)
        body = client.get(f"/api/video-playback-info?path={broken}").json()
        assert body["mode"] == "unknown"
        assert body["player_state"] == "unknown"
        assert body["player_state"] not in ("playable", "needs-transcode")

    def test_a_transcode_with_no_cache_configured_is_cache_disabled(
        self, client, hevc_10bit_mov, monkeypatch
    ):
        monkeypatch.delenv("SFN_VIDEO_CACHE_DIR", raising=False)
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", "")
        body = client.get(f"/api/video-playback-info?path={hevc_10bit_mov}").json()
        assert body["mode"] == "transcode"
        assert body["player_state"] == "cache-disabled"
        assert "SFN_VIDEO_CACHE_DIR" in body["player_state_reason"]

    def test_every_state_the_server_can_report_is_a_declared_state(self):
        # A state the UI has no branch for is a state the analyst never sees.
        assert set(vp_states.MODE_TO_STATE.values()) <= vp_states.PLAYER_STATES

    def test_the_full_job_states_are_declared_now_that_something_can_enter_them(self):
        # Phase 6 kept these out of PLAYER_STATES because §5 forbids inventing a
        # state nothing can reach.  Phase 7 built /api/video-full, so they are
        # ordinary states now — and each one is reachable, which the job tests
        # below exercise rather than assert about.
        assert vp_states.FULL_JOB_STATES == {
            "full-job-running",
            "full-job-done",
            "full-job-failed",
        }
        assert vp_states.FULL_JOB_STATES <= vp_states.PLAYER_STATES


class TestFailureMatrix:
    """Every §10.1 condition, mapped to a status, a §5 state and a retry rule."""

    def test_every_failure_maps_to_a_declared_player_state(self):
        rows = [
            v
            for k, v in vars(vp_states).items()
            if isinstance(v, vp_states.Failure) and k.isupper()
        ]
        assert rows, "the matrix is empty"
        for row in rows:
            assert row.state in vp_states.PLAYER_STATES, row.kind

    def test_a_retryable_failure_says_how_long_to_wait(self):
        # §10.1: "nothing may retry-storm".  A retryable failure without a delay
        # is an invitation to loop as fast as the network allows.
        rows = [
            v
            for k, v in vars(vp_states).items()
            if isinstance(v, vp_states.Failure) and k.isupper() and v.retryable
        ]
        assert rows
        for row in rows:
            assert row.retry_after_seconds, row.kind

    def test_a_non_retryable_failure_carries_no_retry_delay(self):
        assert vp_states.SOURCE_CHANGED.retry_after_seconds is None
        assert vp_states.SOURCE_CHANGED.retryable is False

    def test_the_response_body_names_the_condition_and_the_state(self):
        detail = vp_states.QUEUE_FULL.as_detail()
        assert detail["error"] == "queue-full"
        assert detail["player_state"] == "capacity-exhausted"
        assert detail["retryable"] is True
        assert detail["retry_after_seconds"] == 15

    def test_a_retryable_failure_sets_retry_after(self):
        assert vp_states.DISK_FULL.as_http().headers["Retry-After"] == "60"

    # --- the rows only ffmpeg can report ---------------------------------

    def test_ffmpeg_non_zero_exit_is_a_non_retryable_encode_failure(self):
        f = vp_states.classify(
            vp_encode.EncodeError("moov atom not found", command=["ffmpeg"], returncode=1)
        )
        assert (f.kind, f.status, f.state, f.retryable) == (
            "encode-failed",
            422,
            "chunk-failed",
            False,
        )
        assert "moov atom not found" in f.reason

    def test_a_job_timeout_is_a_504_and_retryable(self):
        f = vp_states.classify(
            vp_encode.EncodeError(
                "encode timed out after 3600s", command=["ffmpeg"], timed_out=True
            )
        )
        assert (f.kind, f.status, f.state, f.retryable) == (
            "job-timeout",
            504,
            "chunk-failed",
            True,
        )

    def test_an_oom_killed_encoder_is_capacity_exhausted_not_a_bad_file(self):
        # SIGKILL with no stderr is what the OOM killer leaves.  Reporting it as
        # "this file cannot be encoded" would tell the analyst to stop asking
        # about a video that is fine.
        f = vp_states.classify(
            vp_encode.EncodeError("ffmpeg exited -9", command=["ffmpeg"], returncode=-9)
        )
        assert (f.kind, f.status, f.state, f.retryable) == (
            "encoder-killed",
            507,
            "capacity-exhausted",
            True,
        )

    def test_another_signal_is_reported_by_name_and_not_as_an_oom(self):
        f = vp_states.classify(
            vp_encode.EncodeError("ffmpeg exited -15", command=["ffmpeg"], returncode=-15)
        )
        assert f.kind == "encode-failed"
        assert "SIGTERM" in f.reason

    def test_a_full_filesystem_is_capacity_exhausted_and_retryable(self):
        f = vp_states.classify(OSError(errno.ENOSPC, "No space left on device"))
        assert (f.kind, f.status, f.state, f.retryable) == (
            "disk-full",
            507,
            "capacity-exhausted",
            True,
        )

    def test_an_unwritable_cache_directory_is_cache_disabled_not_disk_full(self):
        f = vp_states.classify(OSError(errno.EACCES, "Permission denied"))
        assert (f.kind, f.status, f.state, f.retryable) == (
            "cache-unwritable",
            503,
            "cache-disabled",
            False,
        )

    def test_a_vanished_source_is_a_404(self):
        f = vp_states.classify(FileNotFoundError(errno.ENOENT, "No such file"))
        assert (f.kind, f.status, f.state) == ("source-disappeared", 404, "chunk-failed")

    def test_a_host_with_no_usable_pipeline_says_so_in_its_own_words(self):
        f = vp_states.classify(RuntimeError("This ffmpeg build cannot tone-map HDR"))
        assert (f.kind, f.status, f.retryable) == ("no-encode-pipeline", 503, False)
        assert "tone-map" in f.reason

    def test_an_unrecognised_exception_still_gets_a_row(self):
        f = vp_states.classify(ValueError("something nobody anticipated"))
        assert f.state in vp_states.PLAYER_STATES
        assert "something nobody anticipated" in f.reason

    def test_a_gpu_failure_is_not_in_the_matrix_because_it_falls_back(self):
        # §8: a GPU that fails at job time retries on CPU and produces a chunk.
        # A table that invented a `gpu-failed` state would show the analyst a
        # failure for a request that succeeded.
        kinds = {
            v.kind
            for k, v in vars(vp_states).items()
            if isinstance(v, vp_states.Failure) and k.isupper()
        }
        assert "gpu-failed" not in kinds
        assert "no-audio-track" not in kinds


class TestFailureMatrixOverHttp:
    """The same rows, reached through the endpoint rather than through classify."""

    def test_a_corrupt_container_is_422_and_unknown(self, client, roots):
        input_dir, _ = roots
        broken = input_dir / "broken.mov"
        broken.write_bytes(b"\x00" * 64)
        r = client.post(f"/api/video-chunk?path={broken}&t=0")
        assert r.status_code == 422
        assert r.json()["detail"]["player_state"] == "unknown"
        assert r.json()["detail"]["error"] == "corrupt-input"

    def test_a_source_that_needs_no_transcode_is_refused_with_409(self, client, mp4):
        r = client.post(f"/api/video-chunk?path={mp4}&t=0")
        assert r.status_code == 409
        assert r.json()["detail"]["error"] == "not-a-transcode"
        assert r.json()["detail"]["player_state"] == "playable"

    def test_a_timecode_past_the_end_is_refused_before_any_encode(self, client, hevc_10bit_mov):
        with patch.object(vp_encode, "_run", side_effect=AssertionError("encoded anyway")):
            r = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=99999")
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "timecode-out-of-range"

    def test_a_negative_timecode_is_refused(self, client, hevc_10bit_mov):
        r = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=-1")
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "timecode-out-of-range"

    def test_an_unset_cache_dir_is_503_cache_disabled(self, client, hevc_10bit_mov, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", "")
        r = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0")
        assert r.status_code == 503

    def test_a_changed_source_is_409_and_never_encoded(self, client, hevc_10bit_mov):
        with patch.object(vp_encode, "_run", side_effect=AssertionError("encoded a stale file")):
            r = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0&video_hash={'0' * 64}")
        assert r.status_code == 409
        assert r.json()["detail"]["error"] == "source-changed"

    def test_a_missing_source_is_404(self, client, roots):
        input_dir, _ = roots
        r = client.post(f"/api/video-chunk?path={input_dir / 'never_existed.mov'}&t=0")
        assert r.status_code == 404

    def test_a_path_outside_the_allowed_roots_is_403(self, client, roots, tmp_path):
        outside = tmp_path / "elsewhere.mov"
        outside.write_bytes(b"\x00")
        assert client.post(f"/api/video-chunk?path={outside}&t=0").status_code == 403
        assert (
            client.get(f"/api/video-chunk?path={outside}&start=0&fp={'a' * 64}").status_code == 403
        )

    def test_a_fingerprint_is_never_accepted_as_the_identity_of_a_file(
        self, client, hevc_10bit_mov
    ):
        # §9: a key never names a file.  A bad fp cannot reach outside the
        # video's own directory, and a non-hex one is refused outright.
        r = client.get(f"/api/video-chunk?path={hevc_10bit_mov}&start=0&fp=../../etc/passwd")
        assert r.status_code == 422

    def test_the_queue_refuses_rather_than_growing_without_limit(
        self, client, hevc_10bit_mov, monkeypatch
    ):
        monkeypatch.setenv("SFN_VIDEO_QUEUE_MAX", "1")
        monkeypatch.setenv("SFN_VIDEO_MAX_WORKERS", "1")
        vp_jobs.admission.admitted = 1  # one encode already in flight
        try:
            r = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0")
        finally:
            vp_jobs.admission.admitted = 0
        assert r.status_code == 503
        assert r.json()["detail"]["error"] == "queue-full"
        assert r.json()["detail"]["player_state"] == "capacity-exhausted"
        assert r.headers["Retry-After"] == "15"

    def test_the_admission_counter_is_released_after_a_failure(self, client, hevc_10bit_mov):
        with patch.object(
            vp_encode, "_run", side_effect=vp_encode.EncodeError("boom", command=["ffmpeg"])
        ):
            assert client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").status_code == 422
        assert vp_jobs.admission.admitted == 0


class TestEncodeFallbackLimits:
    """§8's CPU fallback answers a GPU fault — and only a GPU fault."""

    def test_a_timeout_on_the_gpu_is_not_retried_on_the_cpu(self, tmp_path):
        cap = _capability(encoder="h264_nvenc", hwaccel="cuda")
        calls = []

        def _boom(cmd, timeout):
            calls.append(cmd)
            raise vp_encode.EncodeError("timed out", command=cmd, timed_out=True)

        with patch.object(vp_encode, "_run", _boom), pytest.raises(vp_encode.EncodeError):
            vp_encode.encode(Settings(), cap, tmp_path / "src.mov", tmp_path / "out.mp4", hdr=False)
        assert len(calls) == 1, "the CPU path is slower; retrying spends the timeout twice"

    def test_an_oom_kill_on_the_gpu_is_not_retried_on_the_cpu(self, tmp_path):
        cap = _capability(encoder="h264_nvenc", hwaccel="cuda")
        calls = []

        def _boom(cmd, timeout):
            calls.append(cmd)
            raise vp_encode.EncodeError("killed", command=cmd, returncode=-9)

        with patch.object(vp_encode, "_run", _boom), pytest.raises(vp_encode.EncodeError):
            vp_encode.encode(Settings(), cap, tmp_path / "src.mov", tmp_path / "out.mp4", hdr=False)
        assert len(calls) == 1, "a second encoder under memory pressure makes it worse"


class TestFallbackCacheLookup:
    """A host whose GPU fails at job time must still hit its own cache."""

    def test_a_chunk_produced_by_the_fallback_is_found_again(self, tmp_path):
        cache_dir = tmp_path / "cache"
        gpu = _pipeline(encoder="h264_nvenc", hwaccel="cuda")
        cpu = _pipeline()
        name = vp_cache.chunk_name(0.0)
        landed = vp_cache.artifact_dir(cache_dir, DIGEST_A, cpu.fingerprint()) / name
        landed.parent.mkdir(parents=True)
        landed.write_bytes(b"chunk")

        # Nothing recorded yet: selecting the GPU pipeline misses, which is what
        # made every chunk re-encode forever before the substitution table.
        assert vp_routes._cached_chunk(cache_dir, DIGEST_A, gpu, name) is None

        vp_routes._substitutions[gpu.fingerprint()] = cpu
        hit = vp_routes._cached_chunk(cache_dir, DIGEST_A, gpu, name)
        assert hit is not None
        assert hit.path == landed

    def test_the_hit_is_labelled_with_the_pipeline_that_produced_it(self, tmp_path):
        # §7.2: the label names the encoder that ran. Reporting the GPU's fields
        # over a file libx264 wrote is the defect the fingerprint exists to stop.
        cache_dir = tmp_path / "cache"
        gpu = _pipeline(encoder="h264_nvenc", hwaccel="cuda")
        cpu = _pipeline()
        name = vp_cache.chunk_name(0.0)
        p = vp_cache.artifact_dir(cache_dir, DIGEST_A, cpu.fingerprint()) / name
        p.parent.mkdir(parents=True)
        p.write_bytes(b"chunk")
        vp_routes._substitutions[gpu.fingerprint()] = cpu

        hit = vp_routes._cached_chunk(cache_dir, DIGEST_A, gpu, name)
        assert hit.fingerprint == cpu.fingerprint()
        assert hit.describe["encoder"] == "libx264"
        assert hit.describe["fingerprint"] == cpu.fingerprint()

    def test_a_substitution_never_serves_another_videos_chunk(self, tmp_path):
        cache_dir = tmp_path / "cache"
        gpu = _pipeline(encoder="h264_nvenc", hwaccel="cuda")
        cpu = _pipeline()
        name = vp_cache.chunk_name(0.0)
        p = vp_cache.artifact_dir(cache_dir, DIGEST_B, cpu.fingerprint()) / name
        p.parent.mkdir(parents=True)
        p.write_bytes(b"chunk")
        vp_routes._substitutions[gpu.fingerprint()] = cpu
        assert vp_routes._cached_chunk(cache_dir, DIGEST_A, gpu, name) is None

    def test_the_direct_key_wins_when_both_exist(self, tmp_path):
        # The GPU recovered: the next genuine miss encodes on it again, and the
        # substitution must not keep the analyst on the older rendering.
        cache_dir = tmp_path / "cache"
        gpu = _pipeline(encoder="h264_nvenc", hwaccel="cuda")
        cpu = _pipeline()
        name = vp_cache.chunk_name(0.0)
        for pipe, payload in ((gpu, b"gpu"), (cpu, b"cpu")):
            q = vp_cache.artifact_dir(cache_dir, DIGEST_A, pipe.fingerprint()) / name
            q.parent.mkdir(parents=True)
            q.write_bytes(payload)
        vp_routes._substitutions[gpu.fingerprint()] = cpu
        hit = vp_routes._cached_chunk(cache_dir, DIGEST_A, gpu, name)
        assert hit.path.read_bytes() == b"gpu"
        assert hit.fingerprint == gpu.fingerprint()


# ---------------------------------------------------------------------------
# The double-buffered player, read the way the browser reads it
# ---------------------------------------------------------------------------
#
# These cannot execute the JS — there is no JS test harness in this repository —
# so they pin the wiring and the call sites that a live defect would come from,
# in the style of `tests/test_static_wiring_web.py`.  Behaviour was verified in
# a real browser; that check is recorded in the PR, not here.

STATIC = Path(__file__).resolve().parents[1] / "src" / "scalar_forensic" / "web" / "static"
PLAYER_JS = STATIC / "js" / "video_playback" / "player.js"


class TestPlayerWiring:
    def test_the_player_part_loads_before_the_assembler(self):
        # A part registered after app.js has already run is a part that never
        # reaches the component (CLAUDE.md).
        html = (STATIC / "index.html").read_text()
        assert html.index("/static/js/video_playback/player.js") < html.index("/static/app.js")

    def test_the_player_registers_itself_as_a_part_and_is_never_object_assigned(self):
        js = PLAYER_JS.read_text()
        assert "window.__sfnParts" in js
        assert "Object.assign(" not in js, "it evaluates getters instead of copying them"

    def test_both_video_buffers_exist_in_the_markup(self):
        # §4.2 is two elements. One element with a swapped src is a black frame
        # at every boundary while the browser reloads.
        html = (STATIC / "index.html").read_text()
        assert 'x-ref="chunkA"' in html
        assert 'x-ref="chunkB"' in html
        assert html.count("advanceToNextChunk(") == 2

    def test_the_hidden_buffer_holds_the_prefetched_chunk(self):
        html = (STATIC / "index.html").read_text()
        assert "chunk.buffer === 0 ? chunk.url : chunk.preload.url" in html
        assert "chunk.buffer === 1 ? chunk.url : chunk.preload.url" in html

    def test_the_player_never_decides_a_state_for_itself(self):
        # Every §5 state comes off the wire, so the §10.1 matrix has exactly one
        # implementation. A client that inferred a state from a status code
        # would be the second, and the two would drift.
        js = PLAYER_JS.read_text()
        assert "b.player_state" in js
        assert "d?.player_state" in js

    def test_the_encoding_state_shows_elapsed_time_and_no_percentage(self):
        # §5: "spinner with elapsed time; no fabricated percentage".
        js = PLAYER_JS.read_text()
        assert "elapsedS" in js
        assert "%" not in js.split("chunkElapsedLabel")[1].split("},")[0]

    def test_nothing_retries_on_its_own(self):
        # §10.1: "nothing may retry-storm". The countdown ticks; the analyst
        # clicks. A setInterval that called playChunkAt would be the storm.
        js = PLAYER_JS.read_text()
        countdown = js.split("_startChunkRetryCountdown() {")[-1].split("\n  retryChunk")[0]
        assert "playChunkAt" not in countdown
        assert "_requestChunk" not in countdown

    def test_a_failed_prefetch_does_not_change_what_the_analyst_sees(self):
        # Speculative work (§4.2). Reporting it would put a failure on screen
        # for a request nobody made.
        js = PLAYER_JS.read_text()
        body = js.split("async _prefetchNextChunk()")[1].split("\n  //")[0]
        assert "chunk.state" not in body
        assert "prefetchFailed" in body

    def test_the_lease_beats_well_inside_its_ttl_and_is_released_on_close(self):
        js = PLAYER_JS.read_text()
        assert "Math.floor(ttl / 4)" in js, "three lost beats must not drop the lease"
        assert "_beatChunkLease(true)" in js, "closing must release, not wait out the ttl"
        assert "keepalive: true" in js

    def test_closing_the_player_releases_the_lease_before_dropping_the_payload(self):
        # The release needs source_path, which lives on the payload.
        js = (STATIC / "js" / "evidence.js").read_text()
        body = js.split("closeVideoPlayback()")[1].split("\n    },")[0]
        assert body.index("closeChunkPlayback()") < body.index("this.videoPlayback = null")

    def test_a_stale_response_cannot_overwrite_a_newer_seek(self):
        js = PLAYER_JS.read_text()
        assert "_chunkRequestId" in js
        assert "if (id !== this._chunkRequestId) return;" in js

    def test_the_player_does_not_reach_for_media_source_extensions(self):
        # §4.1, twice reviewed and rejected. Phase 6 is where the temptation
        # returns hardest, so it is pinned rather than trusted.
        js = PLAYER_JS.read_text()
        for banned in ("MediaSource", "SourceBuffer", "appendBuffer", ".m3u8", "Hls("):
            assert banned not in js, banned

    def test_the_lease_interval_comes_from_the_server_not_from_a_constant(self):
        # A deployment that lowered SFN_VIDEO_LEASE_SECONDS must not lose its
        # lease mid-playback because the client hard-coded the default.
        assert "lease_seconds" in PLAYER_JS.read_text()

    def test_playback_info_reports_the_lease_and_chunk_length(self, client, hevc_10bit_mov):
        body = client.get(f"/api/video-playback-info?path={hevc_10bit_mov}").json()
        assert body["lease_seconds"] == Settings().video_lease_seconds
        assert body["chunk_seconds"] == Settings().video_chunk_seconds


class TestRemainingFailureRowsOverHttp:
    """The two §10.1 rows that need a container the fixtures cannot write."""

    def test_a_container_with_no_video_track_is_422(self, client, hevc_10bit_mov):
        report = vp_codecs._stream_report(hevc_10bit_mov)
        report["video_codec"] = None
        with patch.object(vp_routes, "_stream_report", return_value=report):
            r = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0")
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "no-video-track"

    def test_a_container_with_no_duration_is_refused_before_any_encode(
        self, client, hevc_10bit_mov
    ):
        # A timecode cannot be checked against a duration that is not there, and
        # a chunk cannot be bounded — so this refuses rather than encoding into
        # the dark and finding out.
        report = vp_codecs._stream_report(hevc_10bit_mov)
        report["duration_ms"] = None
        with (
            patch.object(vp_routes, "_stream_report", return_value=report),
            patch.object(vp_encode, "_run", side_effect=AssertionError("encoded anyway")),
        ):
            r = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0")
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "malformed-duration"

    def test_the_matrix_covers_every_condition_section_10_1_names(self):
        # The list in §10.1, as kinds. Two conditions are deliberately absent
        # because they are not failures by the time a caller sees them; they are
        # asserted absent by TestFailureMatrix instead.
        kinds = {
            v.kind
            for k, v in vars(vp_states).items()
            if isinstance(v, vp_states.Failure) and k.isupper()
        }
        assert {
            "corrupt-input",
            "no-video-track",
            "no-encode-pipeline",
            "encode-failed",
            "job-timeout",
            "encoder-killed",
            "disk-full",
            "cache-unwritable",
            "cache-unset",
            "source-disappeared",
            "source-changed",
            "malformed-duration",
            "queue-full",
        } <= kinds


# ---------------------------------------------------------------------------
# The full-video job (spec §4.3, §5, §6.3, §9, §10) — phase 7
# ---------------------------------------------------------------------------


def _await_terminal(c: TestClient, path: Path, *, timeout: float = 30.0) -> dict:
    """Poll the status endpoint until the job stops running.

    Polling and not a hook into the runner: a test that reached inside the job
    would pass against a runner the browser cannot observe, and the status
    endpoint is the only thing the player has.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = c.get(f"/api/video-job-status?path={path}").json()
        if body.get("state") == "none" or body.get("player_state") != "full-job-running":
            return body
        time.sleep(0.05)
    raise AssertionError("the job never left full-job-running")


class TestFullJobCeiling:
    """§6.3: the ceiling is *enforced* on the job path, not merely reported."""

    def test_a_refused_estimate_never_starts_an_encode(self, client, hevc_10bit_mov, monkeypatch):
        # 1 byte of cache: every estimate is over half of it.
        monkeypatch.setenv("SFN_VIDEO_CACHE_MAX_BYTES", "1")
        with patch.object(vp_jobs, "encode_full", side_effect=AssertionError("encoded anyway")):
            r = client.post(f"/api/video-full?path={hevc_10bit_mov}")
        assert r.status_code == 507
        detail = r.json()["detail"]
        assert detail["error"] == "full-copy-refused"
        assert detail["player_state"] == "capacity-exhausted"
        assert "download the original" in detail["reason"].lower()

    def test_an_unknown_estimate_refuses_the_job_too(self, client, hevc_10bit_mov):
        # §6.3: "this file would not say how big it is" is not permission to find
        # out by filling the cache.
        report = vp_codecs._stream_report(hevc_10bit_mov)
        report.pop("bit_rate", None)
        with (
            patch.object(vp_routes, "_stream_report", return_value=report),
            patch.object(vp_jobs, "encode_full", side_effect=AssertionError("encoded anyway")),
        ):
            r = client.post(f"/api/video-full?path={hevc_10bit_mov}")
        assert r.status_code == 507
        assert r.json()["detail"]["error"] == "full-copy-unknown"

    def test_playback_info_reports_the_estimate_without_starting_anything(
        self, client, hevc_10bit_mov
    ):
        body = client.get(f"/api/video-playback-info?path={hevc_10bit_mov}").json()
        assert body["full_copy"]["state"] in ("fits", "refused", "unknown")
        assert body["full_job"] is None


class TestFullCopyOverride:
    """§6.3, ruling 2026-08-14: the refusal is the examiner's to set aside.

    The estimate was measured wrong in both directions (over-reading HEVC 10-bit
    HDR on the CPU pipeline on 8 of 8 samples, one by 8×), so it can refuse an
    export whose real output would have fitted.  What the override may do and
    what it may never do are both pinned here: it bypasses the **forecast**, and
    `limit_bytes` — the number the `.part` watch aborts on — is untouched.
    """

    @staticmethod
    def _recorder():
        """A stand-in encoder that records its call and fails the job at once."""
        calls: list[dict] = []

        def _encode(*args, **kw):
            calls.append(kw)
            raise vp_encode.EncodeError("stopped by the test", command=[], returncode=1)

        return calls, _encode

    def _refusing(self, monkeypatch) -> None:
        # Half of 2 bytes is 1: every estimate this suite can produce is over it,
        # and the limit stays non-zero so the `.part` watch has a number.
        monkeypatch.setenv("SFN_VIDEO_CACHE_MAX_BYTES", "2")

    def test_an_override_starts_the_job_the_estimate_refused(self, hevc_10bit_mov, monkeypatch):
        self._refusing(monkeypatch)
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        calls, encode = self._recorder()
        with TestClient(app) as c, patch.object(vp_jobs, "encode_full", encode):
            refused = c.post(f"/api/video-full?path={hevc_10bit_mov}")
            assert refused.status_code == 507
            started = c.post(f"/api/video-full?path={hevc_10bit_mov}&override=true")
            assert started.status_code == 200, started.text
            _await_terminal(c, hevc_10bit_mov)
        assert len(calls) == 1

    def test_the_override_never_relaxes_the_limit_the_part_watch_aborts_on(
        self, hevc_10bit_mov, monkeypatch
    ):
        # Constraint 4 of the ruling, and the reason `limit_bytes` is read from
        # the setting rather than from a falsy check: half of a 2-byte ceiling is
        # 1, and an overridden job must still carry it into the encoder.
        self._refusing(monkeypatch)
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        calls, encode = self._recorder()
        with TestClient(app) as c, patch.object(vp_jobs, "encode_full", encode):
            c.post(f"/api/video-full?path={hevc_10bit_mov}&override=true")
            _await_terminal(c, hevc_10bit_mov)
        assert calls[0]["limit_bytes"] == 1

    def test_a_ceiling_too_small_to_halve_is_still_a_watched_limit(
        self, hevc_10bit_mov, monkeypatch
    ):
        # 50% of 1 byte floors to 0.  Zero is a limit an override can reach, and
        # reading it as "unbounded" would hand the one job that got past the
        # forecast the one encode with no `.part` watch at all.
        monkeypatch.setenv("SFN_VIDEO_CACHE_MAX_BYTES", "1")
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        calls, encode = self._recorder()
        with TestClient(app) as c, patch.object(vp_jobs, "encode_full", encode):
            c.post(f"/api/video-full?path={hevc_10bit_mov}&override=true")
            _await_terminal(c, hevc_10bit_mov)
        assert calls[0]["limit_bytes"] == 0

    def test_an_unattributable_override_is_refused_and_encodes_nothing(
        self, client, hevc_10bit_mov, monkeypatch
    ):
        self._refusing(monkeypatch)
        monkeypatch.delenv("SFN_EXAMINER_ID", raising=False)
        with patch.object(vp_jobs, "encode_full", side_effect=AssertionError("encoded anyway")):
            r = client.post(f"/api/video-full?path={hevc_10bit_mov}&override=true")
        assert r.status_code == 403
        assert r.json()["detail"] == vp_states.OVERRIDE_UNATTRIBUTED.as_detail()

    def test_the_audit_line_names_the_examiner_and_the_estimate_it_overrode(
        self, hevc_10bit_mov, monkeypatch, caplog
    ):
        self._refusing(monkeypatch)
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        estimate = vp_cache.check_ceiling(
            Settings(), vp_codecs._stream_report(hevc_10bit_mov)
        ).estimate_bytes
        _, encode = self._recorder()
        with (
            caplog.at_level(logging.WARNING, logger="scalar_forensic.video_playback.routes"),
            TestClient(app) as c,
            patch.object(vp_jobs, "encode_full", encode),
        ):
            c.post(f"/api/video-full?path={hevc_10bit_mov}&override=true")
            _await_terminal(c, hevc_10bit_mov)
        line = next(r for r in caplog.records if "OVERRIDDEN" in r.getMessage())
        message = line.getMessage()
        assert line.levelno == logging.WARNING
        assert "examiner-7" in message
        assert f"estimate_bytes={estimate}" in message
        assert "limit_bytes=1" in message
        assert "verdict=refused" in message
        assert str(hevc_10bit_mov) in message

    def test_the_job_view_discloses_the_override_for_the_browser(self, hevc_10bit_mov, monkeypatch):
        self._refusing(monkeypatch)
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        _, encode = self._recorder()
        with TestClient(app) as c, patch.object(vp_jobs, "encode_full", encode):
            body = c.post(f"/api/video-full?path={hevc_10bit_mov}&override=true").json()
            _await_terminal(c, hevc_10bit_mov)
        assert body["override"]["examiner_id"] == "examiner-7"
        assert body["override"]["verdict"] == "refused"
        assert body["override"]["limit_bytes"] == 1
        assert body["override"]["estimate_bytes"] == body["estimate_bytes"]
        # The constant, never a paraphrase of it.
        assert body["override"]["notice"] == vp_jobs.OVERRIDE_NOTICE

    def test_a_job_nobody_overrode_discloses_nothing(self, hevc_10bit_mov, monkeypatch):
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        _, encode = self._recorder()
        with TestClient(app) as c, patch.object(vp_jobs, "encode_full", encode):
            body = c.post(f"/api/video-full?path={hevc_10bit_mov}").json()
            _await_terminal(c, hevc_10bit_mov)
        assert body["override"] is None

    def test_the_override_applies_to_one_request_and_is_never_a_default(
        self, hevc_10bit_mov, monkeypatch
    ):
        # Constraint 1: a later refusal must not be silent because an earlier one
        # was set aside.  The second POST is the same client, the same video, the
        # same process — and is refused again.
        self._refusing(monkeypatch)
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        _, encode = self._recorder()
        with TestClient(app) as c, patch.object(vp_jobs, "encode_full", encode):
            assert c.post(f"/api/video-full?path={hevc_10bit_mov}&override=true").status_code == 200
            _await_terminal(c, hevc_10bit_mov)
            again = c.post(f"/api/video-full?path={hevc_10bit_mov}")
        assert again.status_code == 507
        assert again.json()["detail"]["error"] == "full-copy-refused"

    def test_an_unknown_verdict_is_overridable_too(self, hevc_10bit_mov, monkeypatch):
        # `unknown` says what the container would not report; it is not a claim
        # about the video, and `limit_bytes` does not need the estimate to exist.
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        report = vp_codecs._stream_report(hevc_10bit_mov)
        report.pop("bit_rate", None)
        calls, encode = self._recorder()
        with (
            TestClient(app) as c,
            patch.object(vp_routes, "_stream_report", return_value=report),
            patch.object(vp_jobs, "encode_full", encode),
        ):
            body = c.post(f"/api/video-full?path={hevc_10bit_mov}&override=true").json()
            _await_terminal(c, hevc_10bit_mov)
        assert body["override"]["verdict"] == "unknown"
        assert body["override"]["estimate_bytes"] is None
        assert len(calls) == 1

    def test_a_verdict_that_fits_has_nothing_to_override(self, hevc_10bit_mov, monkeypatch):
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        verdict = vp_cache.check_ceiling(Settings(), vp_codecs._stream_report(hevc_10bit_mov))
        assert verdict.state == "fits"
        assert verdict.overridable is False
        _, encode = self._recorder()
        with TestClient(app) as c, patch.object(vp_jobs, "encode_full", encode):
            body = c.post(f"/api/video-full?path={hevc_10bit_mov}&override=true").json()
            _await_terminal(c, hevc_10bit_mov)
        assert body["override"] is None

    def test_playback_info_says_whether_an_override_would_be_honoured(
        self, client, hevc_10bit_mov, monkeypatch
    ):
        self._refusing(monkeypatch)
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        attributable = client.get(f"/api/video-playback-info?path={hevc_10bit_mov}").json()
        assert attributable["full_copy"]["state"] == "refused"
        assert attributable["full_copy"]["overridable"] is True
        # No examiner identity, no attributable record, so the UI must not offer
        # a button this server would refuse with a 403.
        monkeypatch.delenv("SFN_EXAMINER_ID", raising=False)
        anonymous = client.get(f"/api/video-playback-info?path={hevc_10bit_mov}").json()
        assert anonymous["full_copy"]["overridable"] is False

    def test_a_fitting_video_is_never_advertised_as_overridable(
        self, client, hevc_10bit_mov, monkeypatch
    ):
        monkeypatch.setenv("SFN_EXAMINER_ID", "examiner-7")
        body = client.get(f"/api/video-playback-info?path={hevc_10bit_mov}").json()
        assert body["full_copy"]["state"] == "fits"
        assert body["full_copy"]["overridable"] is False


@requires_ffmpeg
class TestFullJobPartOvershoot:
    """§6.3: the estimate is a screen; the growing `.part` is the guarantee."""

    def test_an_overshooting_part_is_killed_and_leaves_nothing_behind(
        self, client, hevc_10bit_mov, monkeypatch
    ):
        real = vp_jobs.encode_full

        def _one_byte_limit(*args, **kw):
            kw["limit_bytes"] = 1  # any real encode passes this immediately
            return real(*args, **kw)

        with TestClient(app) as c, patch.object(vp_jobs, "encode_full", _one_byte_limit):
            assert c.post(f"/api/video-full?path={hevc_10bit_mov}").status_code == 200
            body = _await_terminal(c, hevc_10bit_mov)
        assert body["player_state"] == "full-job-failed"
        assert body["error"]["error"] == "full-copy-overshoot"
        assert body["error"]["retryable"] is False
        cache_dir = Settings().video_cache_dir
        assert not list(cache_dir.rglob("*.part"))
        assert not list(cache_dir.rglob(vp_cache.FULL_NAME))

    def test_the_overshoot_is_classified_without_a_second_table(self):
        exc = vp_encode.CeilingExceeded("too big", written_bytes=3, limit_bytes=2)
        row = vp_states.classify(exc)
        assert row.kind == "full-copy-overshoot"
        assert row.status == 507
        assert vp_states.classify_full_job(exc).state == "full-job-failed"


class TestFullJobFailureMapping:
    """§10.1: the same rows, landing in `full-job-failed` — not a parallel matrix."""

    def test_every_row_keeps_its_kind_status_and_retry_rule(self):
        rows = [
            v
            for k, v in vars(vp_states).items()
            if isinstance(v, vp_states.Failure) and k.isupper()
        ]
        for row in rows:
            mapped = vp_states.classify_full_job(
                vp_encode.EncodeError("x", command=[], returncode=1)
            )
            assert mapped.state == "full-job-failed"
        for exc, kind in [
            (vp_encode.EncodeError("t", command=[], timed_out=True), "job-timeout"),
            (OSError(errno.ENOSPC, "no space"), "disk-full"),
            (FileNotFoundError(), "source-disappeared"),
            (vp_encode.EncodeError("k", command=[], returncode=-9), "encoder-killed"),
        ]:
            mapped = vp_states.classify_full_job(exc)
            assert mapped.kind == kind
            assert mapped.state == "full-job-failed"
            assert mapped.status == vp_states.classify(exc).status
            assert mapped.retryable == vp_states.classify(exc).retryable

    def test_a_full_job_failure_state_is_a_declared_player_state(self):
        assert "full-job-failed" in vp_states.PLAYER_STATES


class TestFullJobRefcounts:
    """§10.4: a cancel by one analyst must never kill a job another is waiting on."""

    def _request(self, digest="d" * 64) -> vp_jobs.JobRequest:
        return vp_jobs.JobRequest(
            source=Path("/nowhere.MOV"),
            digest=digest,
            duration_seconds=10.0,
            hdr=False,
            has_audio=False,
            capability=_capability(),
            pipeline=_pipeline(),
            cache_dir=Path("/tmp"),
            limit_bytes=1,
            estimate_bytes=1,
        )

    def test_the_second_claim_joins_the_running_job(self):
        async def _go():
            runner = vp_jobs.JobRunner()
            with patch.object(runner, "_run", new=_never_finishing):
                first = runner.start(self._request(), Settings())
                second = runner.start(self._request(), Settings())
                assert first is second
                assert first.waiters == 2
                assert runner.cancel(self._request().digest) == "released"
                assert first.cancelled is False
                assert runner.cancel(self._request().digest) == "cancelled"
                assert first.cancelled is True
                runner.reset()

        asyncio.run(_go())

    def test_a_cancel_that_races_the_encoder_still_stops_it(self):
        # The window is between `Popen` returning and the runner being handed
        # the handle: a DELETE landing inside it sets a flag with nobody left to
        # read it, and a 51-minute encode runs on after the analyst was told it
        # had stopped.  `attach` re-checks the flag, which is what closes it.
        job = vp_jobs.FullJob(self._request(), Settings())
        job.cancel()  # no process yet — this must not be lost
        assert job.cancelled is True
        proc = MagicMock()
        with patch.object(vp_jobs, "_kill_group") as killed:
            job.attach(proc)
        assert killed.call_args.args == (proc,)

    def test_cancelling_a_job_nobody_started_is_a_404(self, client, hevc_10bit_mov):
        r = client.request("DELETE", f"/api/video-full?path={hevc_10bit_mov}")
        assert r.status_code == 404
        assert r.json()["detail"]["error"] == "no-such-job"

    def test_a_status_call_for_no_job_says_none_and_not_finished(self, client, hevc_10bit_mov):
        body = client.get(f"/api/video-job-status?path={hevc_10bit_mov}").json()
        assert body["state"] == "none"
        assert body["player_state"] is None


async def _never_finishing(job) -> None:
    """A job that runs until cancelled — the refcount is what is under test."""
    while not job.cancelled:
        await asyncio.sleep(0.01)


@requires_ffmpeg
class TestFullJobEndToEnd:
    def test_a_finished_job_publishes_a_full_copy_under_the_pipeline_key(self, hevc_10bit_mov):
        with TestClient(app) as c:
            started = c.post(f"/api/video-full?path={hevc_10bit_mov}")
            assert started.status_code == 200, started.text
            assert started.json()["player_state"] == "full-job-running"
            assert started.json()["contention_notice"]
            body = _await_terminal(c, hevc_10bit_mov)
            assert body["player_state"] == "full-job-done", body
            assert body["fraction"] is not None
            assert body["contention_notice"] is None
            served = c.get(body["full_url"])
            assert served.status_code == 200
            assert served.headers["X-SFN-Full-Copy"] == "true"
            assert len(served.content) > 0

    def test_a_get_never_encodes_a_missing_full_copy(self, client, hevc_10bit_mov):
        r = client.get(f"/api/video-full?path={hevc_10bit_mov}&fp={'a' * 64}")
        assert r.status_code == 404
        assert "POST" in r.json()["detail"]

    def test_a_fingerprint_is_never_accepted_as_the_identity_of_a_full_copy(
        self, client, hevc_10bit_mov
    ):
        r = client.get(f"/api/video-full?path={hevc_10bit_mov}&fp=../../etc/passwd")
        assert r.status_code == 422

    def test_cancelling_a_running_job_stops_it_and_leaves_no_part(self, hevc_long_mov):
        with TestClient(app) as c:
            assert c.post(f"/api/video-full?path={hevc_long_mov}").status_code == 200
            c.request("DELETE", f"/api/video-full?path={hevc_long_mov}")
            body = _await_terminal(c, hevc_long_mov)
        assert body["cancelled"] is True
        # A cancelled job is not a §5 state of its own: nothing is running, so
        # the video is back to needing a transcode.
        assert body["player_state"] == "needs-transcode"
        assert not list(Settings().video_cache_dir.rglob("*.part"))


@requires_ffmpeg
class TestFullJobContentionDisclosure:
    """§4.3 remedy (b): the degradation is disclosed, not left invisible."""

    def test_a_chunk_response_says_a_full_export_is_slowing_it_down(self, hevc_long_mov):
        with TestClient(app) as c:
            c.post(f"/api/video-full?path={hevc_long_mov}")
            body = c.post(f"/api/video-chunk?path={hevc_long_mov}&t=0").json()
            notice = body["contention_notice"]
            _await_terminal(c, hevc_long_mov, timeout=60)
        # The wording is the module's one disclosure sentence; what the test
        # pins is that a chunk answered under an export says the wait is worse,
        # not that it uses any particular adjective.
        assert notice == vp_jobs.CONTENTION_NOTICE
        assert "longer" in notice.lower()

    def test_a_chunk_with_no_export_running_claims_no_contention(self, client, hevc_10bit_mov):
        body = client.post(f"/api/video-chunk?path={hevc_10bit_mov}&t=0").json()
        assert body["contention_notice"] is None


class TestFullJobYield:
    """§4.3 remedy (a): the export loses the contention on purpose."""

    def test_the_job_runs_niced_and_thread_capped_and_a_chunk_does_not(self):
        settings = Settings()
        chunk = vp_encode.build_command(
            settings, _pipeline(), Path("/in.MOV"), Path("/out.mp4"), start=0.0, duration=30.0
        )
        assert "-threads" not in chunk
        assert "-progress" not in chunk
        full = vp_encode.build_command(
            settings,
            _pipeline(),
            Path("/in.MOV"),
            Path("/out.mp4"),
            threads=vp_encode.job_threads(settings),
            progress=True,
        )
        assert full[full.index("-threads") + 1] == str(vp_encode.job_threads(settings))
        assert full[full.index("-progress") + 1] == "pipe:1"
        assert settings.video_job_nice == 10

    def test_the_thread_cap_leaves_half_the_box_for_chunk_work(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_JOB_THREADS", "0")
        with patch.object(os, "cpu_count", return_value=8):
            assert vp_encode.job_threads(Settings()) == 4
        with patch.object(os, "cpu_count", return_value=1):
            assert vp_encode.job_threads(Settings()) == 1
        monkeypatch.setenv("SFN_VIDEO_JOB_THREADS", "3")
        assert vp_encode.job_threads(Settings()) == 3

    def test_the_renice_reaches_every_encoding_thread_and_not_just_the_pid(self):
        # Niceness on Linux is a *per-thread* attribute: `PRIO_PROCESS` renices
        # the one thread that called `Popen`, and every thread ffmpeg spawns to
        # do the encoding keeps the parent's priority — the yield would then be
        # measured on the one thread that does no work.  `PRIO_PGRP` is what
        # makes it reach them, and it is why the child gets its own session.
        with patch.object(os, "setpriority") as sp:
            vp_encode._lower_priority(4321, 10)
        assert sp.call_args.args == (os.PRIO_PGRP, 4321, 10)
        with patch.object(os, "setpriority") as sp:
            vp_encode._lower_priority(4321, 0)
        assert sp.call_count == 0

    def test_the_settings_are_validated_like_every_other(self, monkeypatch):
        monkeypatch.setenv("SFN_VIDEO_JOB_NICE", "20")
        with pytest.raises(ValueError, match="SFN_VIDEO_JOB_NICE"):
            Settings()
        monkeypatch.setenv("SFN_VIDEO_JOB_NICE", "10")
        monkeypatch.setenv("SFN_VIDEO_JOB_THREADS", "-1")
        with pytest.raises(ValueError, match="SFN_VIDEO_JOB_THREADS"):
            Settings()


class TestFullJobProgress:
    """§4.3: progress from ffmpeg's own output; `#139`'s labelling, not a promise."""

    def test_out_time_is_read_from_the_unambiguous_field(self):
        # `out_time_ms` has carried *microseconds* under a millisecond name for
        # years; reading it would report a 51-minute export as three seconds.
        assert vp_encode._out_seconds("00:01:30.500000") == pytest.approx(90.5)
        assert vp_encode._out_seconds("N/A") == 0.0
        assert vp_encode._out_seconds(None) == 0.0

    def test_the_watcher_reads_out_time_and_not_out_time_ms(self, tmp_path):
        # The parser above is only half the guarantee: the other half is *which*
        # field `_run_watched` hands it.  A progress block that reports the same
        # instant in both fields is the only thing that can tell those apart, so
        # this drives a real child emitting both — `out_time_ms` carries
        # microseconds, and reading it as milliseconds would say 90500 s.
        emitter = tmp_path / "emit.py"
        emitter.write_text(
            "print('frame=42')\n"
            "print('out_time_ms=90500000')\n"
            "print('out_time=00:01:30.500000')\n"
            "print('progress=continue')\n"
        )
        seen: list[vp_encode.Progress] = []
        vp_encode._run_watched([sys.executable, str(emitter)], timeout=30, on_progress=seen.append)
        assert [p.out_seconds for p in seen] == [pytest.approx(90.5)]

    def test_the_eta_is_labelled_as_an_extrapolation_and_never_as_a_time(self):
        assert vp_jobs.eta_label(None) is None
        assert vp_jobs.eta_label(45) == "~45 s remaining at current rate"
        assert vp_jobs.eta_label(240) == "~4 min remaining at current rate"
        assert "at current rate" in vp_jobs.eta_label(7200)
        for seconds in (10, 240, 7200):
            label = vp_jobs.eta_label(seconds)
            assert "±" not in label and "confidence" not in label

    def test_no_eta_is_offered_from_a_single_observation(self):
        request = vp_jobs.JobRequest(
            source=Path("/x.MOV"),
            digest="e" * 64,
            duration_seconds=100.0,
            hdr=False,
            has_audio=False,
            capability=_capability(),
            pipeline=_pipeline(),
            cache_dir=Path("/tmp"),
            limit_bytes=None,
            estimate_bytes=None,
        )
        job = vp_jobs.FullJob(request, Settings())
        assert job.eta_seconds is None
        job.observe(vp_encode.Progress(frames=10, out_seconds=1.0, written_bytes=100))
        assert job.eta_seconds is None  # one observation is not a rate
        time.sleep(0.01)
        job.observe(vp_encode.Progress(frames=20, out_seconds=2.0, written_bytes=200))
        assert job.eta_seconds is not None
        assert job.view()["fraction"] == pytest.approx(0.02)
        assert job.view()["eta_label"].endswith("at current rate")
