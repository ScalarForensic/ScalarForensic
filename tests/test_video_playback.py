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

import dataclasses
import json
import os
import shutil
import struct
import subprocess
import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from scalar_forensic.config import Settings
from scalar_forensic.embedder import hash_file
from scalar_forensic.video import VIDEO_EXTENSIONS
from scalar_forensic.video_playback import cache as vp_cache
from scalar_forensic.video_playback import capability as vp_capability
from scalar_forensic.video_playback import codecs as vp_codecs
from scalar_forensic.video_playback import digest as vp_digest
from scalar_forensic.video_playback import encode as vp_encode
from scalar_forensic.video_playback import rewrap as vp_rewrap
from scalar_forensic.video_playback import routes as vp_routes
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


def _write_encoded(path: Path, container_format: str, encoder: str, pix_fmt: str) -> Path:
    """Encode a tiny clip with *encoder* at *pix_fmt* — codec-detection fixtures."""
    with av.open(str(path), "w", format=container_format) as c:
        stream = c.add_stream(encoder, rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = pix_fmt
        for i in range(6):
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
        cached = list(cache_dir.glob("*.mp4"))
        assert len(cached) == 1
        assert cached[0].stem == hash_file(mov)

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
        assert list(cache_dir.glob("*.mp4"))
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


class TestCacheEviction:
    def _entry(self, cache_dir: Path, name: str, size: int, mtime: float) -> Path:
        p = cache_dir / name
        p.write_bytes(b"\0" * size)
        os.utime(p, (mtime, mtime))
        return p

    def test_oldest_entries_are_evicted_first(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        old = self._entry(cache_dir, "a" * 64 + ".mp4", 100, 1_000)
        mid = self._entry(cache_dir, "b" * 64 + ".mp4", 100, 2_000)
        new = self._entry(cache_dir, "c" * 64 + ".mp4", 100, 3_000)
        deleted = vp_cache._evict_cache(cache_dir, 150, keep=new)
        assert deleted == 2
        assert not old.exists() and not mid.exists() and new.exists()

    def test_the_file_being_served_is_never_evicted(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        keep = self._entry(cache_dir, "d" * 64 + ".mp4", 500, 1_000)
        other = self._entry(cache_dir, "e" * 64 + ".mp4", 100, 2_000)
        vp_cache._evict_cache(cache_dir, 10, keep=keep)
        assert keep.exists()
        assert not other.exists()

    def test_only_this_functions_own_rewraps_are_candidates(self, tmp_path):
        # Later phases park chunks and full copies in the same store; a bare
        # *.mp4 glob would delete a video's own artifacts mid-play (spec §6.2).
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        rewrap = self._entry(cache_dir, "a" * 64 + ".mp4", 100, 1_000)
        full = self._entry(cache_dir, "full.mp4", 100, 500)
        init = self._entry(cache_dir, "init.mp4", 100, 500)
        per_video = cache_dir / ("b" * 64)
        per_video.mkdir()
        chunk = self._entry(per_video, "c0.mp4", 100, 500)
        vp_cache._evict_cache(cache_dir, 10, keep=cache_dir / "nothing.mp4")
        assert not rewrap.exists()
        assert full.exists() and init.exists() and chunk.exists()

    def test_zero_ceiling_disables_eviction(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        p = self._entry(cache_dir, "a" * 64 + ".mp4", 100, 1_000)
        assert vp_cache._evict_cache(cache_dir, 0, keep=cache_dir / "x.mp4") == 0
        assert p.exists()

    def test_cache_stays_under_the_ceiling_after_a_rewrap(self, client, mov, roots):
        _, cache_dir = roots
        cache_dir.mkdir(parents=True, exist_ok=True)
        filler = cache_dir / ("f" * 64 + ".mp4")
        filler.write_bytes(b"\0" * 4096)
        settings = MagicMock()
        settings.video_cache_dir = cache_dir
        settings.video_cache_max_bytes = 2048
        settings.hash_cache_path = tmp_path_db = cache_dir.parent / "hash_cache.db"
        assert tmp_path_db.parent.exists()
        with patch.object(vp_routes, "Settings", return_value=settings):
            r = client.get(f"/api/video-playback?path={mov}")
        assert r.status_code == 200
        assert not filler.exists()
        assert sum(f.stat().st_size for f in cache_dir.glob("*.mp4")) <= 2048


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestPlaybackSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("SFN_VIDEO_CACHE_DIR", raising=False)
        monkeypatch.delenv("SFN_VIDEO_CACHE_MAX_BYTES", raising=False)
        s = Settings()
        assert s.video_cache_dir is not None
        assert s.video_cache_dir.name == "video_cache"
        assert s.video_cache_max_bytes == 8 * 1024 * 1024 * 1024

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
