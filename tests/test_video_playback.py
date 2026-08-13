"""Tests for in-browser playback of source videos.

Covered:
  GET /api/video-playback?path=…
  GET /api/video-playback-info?path=…
  the rewrap/cache helpers in scalar_forensic.web.routes.video
  the SFN_VIDEO_CACHE_* settings

Fixtures are generated with PyAV at test time (a handful of 64×48 frames), so
the suite stays hermetic and needs neither Qdrant nor sample media on disk.
"""

from __future__ import annotations

import os
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
from scalar_forensic.web.app import app
from scalar_forensic.web.routes import video as video_routes

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
    video_routes._reset_hash_cache()
    yield input_dir, cache_dir
    # The handle points into tmp_path; never let the next test inherit it.
    video_routes._reset_hash_cache()


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
        assert video_routes._ftyp_brand(mov) == "qt  "
        assert video_routes._needs_remux(mov) is True

    def test_mp4_is_served_as_is(self, mp4):
        assert video_routes._ftyp_brand(mp4) != "qt  "
        assert video_routes._needs_remux(mp4) is False

    def test_file_without_ftyp_box_needs_rewrap(self, tmp_path):
        p = tmp_path / "clip.avi"
        p.write_bytes(b"RIFF____AVI LIST")
        assert video_routes._ftyp_brand(p) is None
        assert video_routes._needs_remux(p) is True

    def test_webm_is_served_as_is(self, tmp_path):
        p = tmp_path / "clip.webm"
        p.write_bytes(b"\x1a\x45\xdf\xa3")
        assert video_routes._needs_remux(p) is False


# ---------------------------------------------------------------------------
# The rewrap itself — streams must survive bit for bit
# ---------------------------------------------------------------------------


class TestRemux:
    def test_rewrap_preserves_the_video_bitstream(self, mov, tmp_path):
        dst = tmp_path / "copy.mp4"
        report = video_routes._remux_to_mp4(mov, dst)
        assert report == {"skipped_streams": [], "timestamp_repairs": 0}
        assert _packet_payloads(dst) == _packet_payloads(mov)

    def test_rewrap_keeps_the_codec_and_produces_an_mp4(self, mov, tmp_path):
        dst = tmp_path / "copy.mp4"
        video_routes._remux_to_mp4(mov, dst)
        assert dst.read_bytes()[4:8] == b"ftyp"
        assert video_routes._ftyp_brand(dst) != "qt  "
        with av.open(str(mov)) as a, av.open(str(dst)) as b:
            assert a.streams.video[0].codec_context.name == b.streams.video[0].codec_context.name

    def test_the_source_codec_tag_survives(self, mov, tmp_path):
        """Left alone the muxer relabels Apple's hvc1 as hev1."""
        dst = tmp_path / "copy.mp4"
        video_routes._remux_to_mp4(mov, dst)
        with av.open(str(mov)) as a, av.open(str(dst)) as b:
            assert b.streams.video[0].codec_context.codec_tag == (
                a.streams.video[0].codec_context.codec_tag
            )

    def test_moov_precedes_mdat_for_progressive_playback(self, mov, tmp_path):
        dst = tmp_path / "copy.mp4"
        video_routes._remux_to_mp4(mov, dst)
        data = dst.read_bytes()
        assert 0 <= data.index(b"moov") < data.index(b"mdat")

    def test_no_part_file_survives_a_failed_rewrap(self, mov, tmp_path):
        dst = tmp_path / "copy.mp4"
        with (
            patch.object(video_routes.av, "open", side_effect=OSError("boom")),
            pytest.raises(OSError),
        ):
            video_routes._remux_to_mp4(mov, dst)
        assert list(tmp_path.glob("*.part")) == []
        assert not dst.exists()

    def test_lpcm_audio_is_named_not_dropped_silently(self, mov_with_pcm, tmp_path):
        """Apple Live-Photo .MOV carries LPCM, which has no MP4 mapping."""
        dst = tmp_path / "copy.mp4"
        report = video_routes._remux_to_mp4(mov_with_pcm, dst)
        assert report["skipped_streams"] == ["audio:pcm_s16le"]
        with av.open(str(dst)) as c:
            assert len(c.streams.audio) == 0
        assert _packet_payloads(dst) == _packet_payloads(mov_with_pcm)

    def test_source_with_no_mp4_compatible_stream_raises(self, mov, tmp_path):
        with patch.object(video_routes, "_MP4_LEGAL_CODECS", frozenset()):
            with pytest.raises(ValueError, match="no MP4-compatible stream"):
                video_routes._remux_to_mp4(mov, tmp_path / "copy.mp4")


class TestTimestampRepair:
    """Real .MOV files carry frames the MP4 muxer refuses; repairs are counted."""

    def _packet(self, pts, dts):
        p = av.Packet(1)
        p.pts, p.dts = pts, dts
        return p

    def test_untouched_when_timestamps_are_already_muxable(self):
        p = self._packet(100, 90)
        assert video_routes._repair_timestamps(p, 80) is False
        assert (p.pts, p.dts) == (100, 90)

    def test_decode_stamp_after_display_stamp_is_pulled_back(self):
        p = self._packet(600, 640)
        assert video_routes._repair_timestamps(p, 560) is True
        assert (p.pts, p.dts) == (600, 600)

    def test_stalled_decode_stamp_is_nudged_past_its_predecessor(self):
        p = self._packet(500, 500)
        assert video_routes._repair_timestamps(p, 500) is True
        assert p.dts == 501
        assert p.pts >= p.dts

    def test_repairs_are_counted_in_the_rewrap_report(self, mov, tmp_path):
        original = video_routes._repair_timestamps
        with patch.object(video_routes, "_repair_timestamps", side_effect=original) as spy:
            spy.side_effect = lambda packet, last: True
            report = video_routes._remux_to_mp4(mov, tmp_path / "copy.mp4")
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
            assert video_routes._resolve_video_path(str(p)) == p


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
            video_routes, "_remux_to_mp4", side_effect=AssertionError("rewrapped twice")
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
        assert video_routes._pixel_profile(pix_fmt, None) == expected

    def test_the_profile_settles_it_when_the_pixel_format_is_missing(self):
        assert video_routes._pixel_profile(None, "Main 10") == (10, None)
        assert video_routes._pixel_profile(None, "High 4:4:4 Predictive") == (None, "444")

    def test_neither_available_is_undetermined_not_a_guess(self):
        assert video_routes._pixel_profile(None, None) == (None, None)
        assert video_routes._pixel_profile(None, "High") == (None, None)


class TestDecodeVerdict:
    def _info(self, **kw):
        base = {"video_codec": "h264", "video_pix_fmt": "yuv420p", "video_profile": "High"}
        return {**base, **kw}

    def test_h264_8bit_is_playable(self):
        assert video_routes._decode_verdict(self._info())[0] is True

    def test_h264_10bit_is_not(self):
        verdict, reason = video_routes._decode_verdict(self._info(video_pix_fmt="yuv420p10le"))
        assert verdict is False
        assert "H.264 10-bit" in reason

    def test_h264_444_is_not(self):
        verdict, reason = video_routes._decode_verdict(self._info(video_pix_fmt="yuv444p"))
        assert verdict is False
        assert "4:4:4" in reason

    def test_vp9_10bit_is_playable(self):
        info = self._info(video_codec="vp9", video_pix_fmt="yuv420p10le")
        assert video_routes._decode_verdict(info)[0] is True

    def test_av1_and_vp8_are_on_the_allowlist(self):
        for codec in ("av1", "vp8"):
            assert video_routes._decode_verdict(self._info(video_codec=codec))[0] is True

    def test_hevc_is_not_on_the_allowlist_at_any_depth(self):
        # §16: Chrome advertises HEVC and then fails. The allowlist decides.
        verdict, reason = video_routes._decode_verdict(self._info(video_codec="hevc"))
        assert verdict is False
        assert "HEVC" in reason

    def test_prores_names_itself_in_the_reason(self):
        verdict, reason = video_routes._decode_verdict(self._info(video_codec="prores"))
        assert verdict is False
        assert "Apple ProRes" in reason

    def test_an_unprobeable_container_is_undetermined(self):
        verdict, reason = video_routes._decode_verdict({"probe_error": "moov atom not found"})
        assert verdict is None
        assert "moov atom not found" in reason

    def test_no_video_stream_is_undetermined(self):
        assert video_routes._decode_verdict({"video_codec": None})[0] is None

    def test_an_unreadable_pixel_format_is_undetermined_not_a_transcode(self):
        info = self._info(video_pix_fmt=None, video_profile=None)
        verdict, reason = video_routes._decode_verdict(info)
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
        with patch.object(video_routes, "hash_file", side_effect=AssertionError("hashed!")):
            with patch("scalar_forensic.embedder.hash_file_both", side_effect=AssertionError):
                r = client.get(f"/api/video-download?path={mov}")
        assert r.status_code == 200
        assert "x-sfn-source-sha256" not in r.headers
        assert r.content == mov.read_bytes()

    def test_a_disabled_hash_cache_still_serves_the_bytes(self, client, mov, monkeypatch):
        monkeypatch.setenv("SFN_HASH_CACHE_PATH", "")
        video_routes._reset_hash_cache()
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
        assert video_routes._source_digest(mov) == hash_file(mov)

    def test_second_call_is_served_from_the_cache(self, mov):
        video_routes._source_digest(mov)
        with patch.object(video_routes, "hash_file") as direct:
            with patch("scalar_forensic.embedder.hash_file_both") as both:
                assert video_routes._source_digest(mov) == hash_file(mov)
        direct.assert_not_called()
        both.assert_not_called()

    def test_the_cache_survives_a_new_process(self, mov, roots):
        _, _ = roots
        video_routes._source_digest(mov)
        video_routes._reset_hash_cache()  # as a restart would
        with patch("scalar_forensic.embedder.hash_file_both") as both:
            assert video_routes._source_digest(mov) == hash_file(mov)
        both.assert_not_called()

    def test_a_changed_file_is_rehashed(self, mov):
        first = video_routes._source_digest(mov)
        os.utime(mov, (2_000_000, 2_000_000))
        mov.write_bytes(mov.read_bytes() + b"tampered")
        second = video_routes._source_digest(mov)
        assert second != first
        assert second == hash_file(mov)

    def test_disabled_cache_still_answers(self, mov, monkeypatch):
        monkeypatch.setenv("SFN_HASH_CACHE_PATH", "")
        video_routes._reset_hash_cache()
        assert video_routes._hash_cache_for(Settings()) is None
        assert video_routes._source_digest(mov) == hash_file(mov)

    def test_unwritable_db_falls_back_to_a_direct_hash(self, mov, tmp_path, monkeypatch):
        # A directory where the DB file should be: SQLite cannot open it.
        db = tmp_path / "unwritable.db"
        db.mkdir()
        monkeypatch.setenv("SFN_HASH_CACHE_PATH", str(db))
        video_routes._reset_hash_cache()
        assert video_routes._source_digest(mov) == hash_file(mov)

    def test_a_broken_cache_is_not_reopened_per_request(self, mov, tmp_path, monkeypatch):
        db = tmp_path / "unwritable.db"
        db.mkdir()
        monkeypatch.setenv("SFN_HASH_CACHE_PATH", str(db))
        video_routes._reset_hash_cache()
        with patch.object(video_routes, "HashCache", side_effect=OSError("nope")) as ctor:
            video_routes._source_digest(mov)
            video_routes._source_digest(mov)
        assert ctor.call_count == 1

    def test_a_failing_lookup_falls_back_instead_of_raising(self, mov):
        cache = MagicMock()
        cache.get_or_hash.side_effect = OSError("disk gone")
        with patch.object(video_routes, "_hash_cache_for", return_value=cache):
            assert video_routes._source_digest(mov) == hash_file(mov)

    def test_a_failing_flush_does_not_fail_the_digest(self, mov):
        cache = MagicMock()
        cache.get_or_hash.return_value = ("a" * 64, False)
        cache.flush.side_effect = OSError("read-only")
        with patch.object(video_routes, "_hash_cache_for", return_value=cache):
            assert video_routes._source_digest(mov) == "a" * 64

    def test_the_request_path_does_not_block_the_event_loop(self, client, mov):
        # The digest is computed in a worker thread, never inline in the handler.
        calls: list[str] = []
        real = video_routes._source_digest

        def spy(p, settings=None):
            calls.append(threading.current_thread().name)
            return real(p, settings)

        with patch.object(video_routes, "_source_digest", spy):
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
        video_routes._source_digest(mov)  # warm the cache, as a first view would
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
        deleted = video_routes._evict_cache(cache_dir, 150, keep=new)
        assert deleted == 2
        assert not old.exists() and not mid.exists() and new.exists()

    def test_the_file_being_served_is_never_evicted(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        keep = self._entry(cache_dir, "d" * 64 + ".mp4", 500, 1_000)
        other = self._entry(cache_dir, "e" * 64 + ".mp4", 100, 2_000)
        video_routes._evict_cache(cache_dir, 10, keep=keep)
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
        video_routes._evict_cache(cache_dir, 10, keep=cache_dir / "nothing.mp4")
        assert not rewrap.exists()
        assert full.exists() and init.exists() and chunk.exists()

    def test_zero_ceiling_disables_eviction(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        p = self._entry(cache_dir, "a" * 64 + ".mp4", 100, 1_000)
        assert video_routes._evict_cache(cache_dir, 0, keep=cache_dir / "x.mp4") == 0
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
        with patch.object(video_routes, "Settings", return_value=settings):
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
