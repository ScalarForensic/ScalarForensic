"""What this ffmpeg build can actually do, and the fingerprint of what it did.

Two jobs, kept in one module because the second is a function of the first:

* **The capability probe** (spec §8) decides which encode pipeline is available
  on this host.  It does not read ``ffmpeg -encoders``: decoder, transfer
  function, filter, pixel format and encoder each fail independently across
  driver and build combinations, and a listed encoder that cannot be fed is a
  false positive.  The probe therefore runs a real
  decode → tone-map → encode → mux of a few frames and believes the exit code.

* **The pipeline fingerprint** (spec §6.1) is the half of the artifact cache key
  that is not the source.  ``key = sha256(source identity ‖ pipeline
  fingerprint)``, so anything that changes the rendered pixels has to be inside
  it or the cache will hold two pipelines' output under one label.

Nothing here encodes anything an analyst sees; :mod:`.encode` does that.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, fields
from pathlib import Path

from scalar_forensic.config import Settings

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter chains
# ---------------------------------------------------------------------------

# The tone-map chain measured in §3.1, unchanged.  It is what makes the output
# carry bt709/bt709 instead of an 8-bit picture still labelled bt2020/HLG —
# §3.1's second finding, which browsers render with lifted blacks.  It costs
# ~2.3× and is not optional for an HDR source.
TONEMAP_CHAIN = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)

# An SDR source needs no interpretation, only a browser-safe pixel format.
SDR_CHAIN = "format=yuv420p"

# Transfer characteristics that mean "this is HDR and must be tone-mapped".
HDR_TRANSFERS = frozenset({"arib-std-b67", "smpte2084"})


def _scale_filter(output_height: int) -> str:
    """Downscale to *output_height*, never upscale, keep the aspect ratio.

    ``-2`` keeps width even (H.264 4:2:0 requires it) and ``min(ih,H)`` is what
    makes this a cap rather than a target: a 720p source stays 720p, because
    upscaling invents detail in an evidence viewer (§7.4, §16).

    Placed *before* the tone-map so the expensive part of the chain works on
    fewer pixels.  ``ih`` is the height after ffmpeg's autorotate, which is why
    the CPU filter path keeps portrait clips upright where §3.1's
    ``-hwaccel_output_format cuda`` path lost them.
    """
    return f"scale=-2:'min(ih,{output_height})'"


def _video_filter(output_height: int, *, hdr: bool) -> str:
    return f"{_scale_filter(output_height)},{TONEMAP_CHAIN if hdr else SDR_CHAIN}"


# ---------------------------------------------------------------------------
# The pipeline and its fingerprint
# ---------------------------------------------------------------------------

# Rate control per encoder.  Recorded as data rather than inlined at the call
# site because it is a fingerprint member: changing a CRF changes the pixels.
_RATE_CONTROL: dict[str, tuple[str, ...]] = {
    "libx264": ("-preset", "medium", "-crf", "23"),
    "h264_nvenc": ("-preset", "p4", "-rc", "vbr", "-cq", "23"),
}

AUDIO_ARGS: tuple[str, ...] = ("-c:a", "aac", "-b:a", "128k")


@dataclass(frozen=True)
class Pipeline:
    """Everything that decides what the rendered pixels look like.

    **Every field of this dataclass is in the fingerprint.**  That is the rule,
    not an implementation detail: a field added here without thought silently
    changes every cache key, and a pixel-affecting setting left *out* of here
    silently shares one key between two renderings.  ``test_video_playback.py``
    pins the field set so adding one is a deliberate act.

    Deliberately **not** fields, because none of them changes a pixel:
    ``SFN_VIDEO_MAX_WORKERS``, the cache directory and its ceiling, the queue
    and timeout settings, the examiner id, the source path, and the timecode a
    chunk starts at (that is source identity and chunk arithmetic, not
    pipeline).

    ``ffmpeg_version`` **is** a field.  An encoder's output changes between
    builds, so two ffmpeg versions are two pipelines; the cost is that upgrading
    ffmpeg invalidates the cache, which is the conservative direction for a
    rendering that carries a label naming its pipeline (§7.2).

    ``hwaccel`` is the pipeline that *ran*, not the one that was selected.  §8
    lets a GPU failure at job time fall back to CPU; that fallback produces a
    different ``Pipeline``, therefore a different fingerprint, therefore a
    different cache key — which is the point, since the two encoders do not
    produce the same picture.
    """

    hwaccel: str
    decoder: str
    filter_chain: str
    encoder: str
    rate_control: tuple[str, ...]
    output_height: int
    chunk_seconds: int
    audio: tuple[str, ...]
    ffmpeg_version: str

    def canonical(self) -> str:
        """The exact bytes that get hashed — printable, so a label can show it."""
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = " ".join(value)
            parts.append(f"{f.name}={value}")
        return "\n".join(parts)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def describe(self) -> dict:
        """The §7.2 label payload: what actually ran, in full.

        Derived from ``fields(self)`` rather than hand-listed, for the same
        reason :meth:`canonical` is: a field added to the pipeline would change
        the fingerprint while a hand-written label quietly kept describing the
        old one — a label naming a pipeline it does not fully describe, in an
        evidence viewer.  The two computed extras stay explicit because they are
        not fields.
        """
        described = {
            f.name: " ".join(getattr(self, f.name))
            if isinstance(getattr(self, f.name), tuple)
            else getattr(self, f.name)
            for f in fields(self)
        }
        described["fingerprint"] = self.fingerprint()
        described["tone_mapped"] = TONEMAP_CHAIN in self.filter_chain
        return described


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    """What the probe found.  Three-state on purpose (§5): unknown is not false."""

    ffmpeg_path: str
    ffmpeg_version: str | None
    encoder: str | None
    hwaccel: str
    tonemap_ok: bool
    notes: tuple[str, ...]

    @property
    def available(self) -> bool:
        """True when an SDR source can be encoded at all."""
        return self.encoder is not None

    def unavailable_reason(self, *, hdr: bool) -> str | None:
        """Why an encode cannot run, phrased for an analyst, or None."""
        if self.encoder is None:
            return (
                f"No usable ffmpeg encode pipeline: {'; '.join(self.notes)}. "
                "Download the original to view it."
            )
        if hdr and not self.tonemap_ok:
            return (
                "This ffmpeg build cannot tone-map HDR (zscale/tonemap failed: "
                f"{'; '.join(self.notes)}). Encoding anyway would produce an 8-bit "
                "picture still labelled HDR, which browsers render washed out, so "
                "it is refused. Download the original to view it."
            )
        return None


# A 10-bit HLG/bt2020 source synthesised by lavfi.  It is not a substitute for
# a real capture (§14) — it exists to make the probe exercise the transfer
# function and pixel format a real source would, so a build without libzimg
# fails here at startup instead of at an analyst's first click.
_PROBE_SOURCE = (
    "testsrc2=size=320x240:rate=30:duration=0.2,format=yuv420p10le,"
    "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc"
)

_PROBE_TIMEOUT = 30


def _ffmpeg_version(ffmpeg_path: str) -> str | None:
    """First line of ``ffmpeg -version``, or None when there is no binary to ask."""
    if shutil.which(ffmpeg_path) is None and not Path(ffmpeg_path).is_file():
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - operator-configured binary, fixed args
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    first = proc.stdout.splitlines()[0].strip() if proc.stdout else ""
    return first or None


def _try_encode(ffmpeg_path: str, encoder: str, *, hdr: bool) -> str | None:
    """Run a real few-frame encode; return None on success or the error text.

    Six frames through the whole chain — lavfi decode, scale, tone-map (when
    *hdr*), encoder, MP4 mux — because that is the only question worth asking.
    An encoder that appears in ``-encoders`` and then cannot be fed a frame is
    exactly the false positive §8 forbids relying on.
    """
    chain = _video_filter(1080, hdr=hdr)
    with tempfile.TemporaryDirectory(prefix="sfn-vprobe-") as tmp:
        out = Path(tmp) / "probe.mp4"
        cmd = [
            ffmpeg_path,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            _PROBE_SOURCE,
            "-vf",
            chain,
            "-c:v",
            encoder,
            *_RATE_CONTROL[encoder],
            "-frames:v",
            "6",
            "-f",
            "mp4",
            "-y",
            str(out),
        ]
        try:
            proc = subprocess.run(  # noqa: S603 - operator-configured binary, fixed args
                cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT, check=False
            )
        except subprocess.TimeoutExpired:
            return f"{encoder}: probe timed out after {_PROBE_TIMEOUT}s"
        except OSError as exc:
            return f"{encoder}: {exc}"
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip().splitlines()
            return f"{encoder}: {detail[-1] if detail else f'exit {proc.returncode}'}"
        if not out.exists() or out.stat().st_size == 0:
            return f"{encoder}: produced no output"
    return None


def probe(settings: Settings) -> Capability:
    """Run the §8 capability probe.  Never raises; a failure is a reported state."""
    ffmpeg_path = settings.ffmpeg_path
    version = _ffmpeg_version(ffmpeg_path)
    if version is None:
        return Capability(
            ffmpeg_path=ffmpeg_path,
            ffmpeg_version=None,
            encoder=None,
            hwaccel="none",
            tonemap_ok=False,
            notes=(f"ffmpeg not found or not runnable at {ffmpeg_path!r}",),
        )

    notes: list[str] = []
    encoder: str | None = None
    hwaccel = "none"

    if settings.video_hwaccel in ("auto", "cuda"):
        err = _try_encode(ffmpeg_path, "h264_nvenc", hdr=True)
        if err is None:
            encoder, hwaccel = "h264_nvenc", "cuda"
        else:
            notes.append(err)
            if settings.video_hwaccel == "cuda":
                # Asked for explicitly and unavailable.  Falling back is still
                # right — §8 says a GPU failure never fails the request outright
                # — but it is recorded loudly rather than absorbed.
                _log.warning("SFN_VIDEO_HWACCEL=cuda requested but unusable: %s", err)

    if encoder is None:
        err = _try_encode(ffmpeg_path, "libx264", hdr=True)
        if err is None:
            encoder = "libx264"
        else:
            notes.append(err)

    # The HDR probes above include the tone-map chain, so reaching here with an
    # encoder means tone-mapping works.  If they all failed, retry without it:
    # that separates "no encoder at all" from "an encoder, but no libzimg", and
    # the second is a build that can still serve SDR sources honestly.
    tonemap_ok = encoder is not None
    if encoder is None:
        for candidate in ("h264_nvenc", "libx264"):
            if candidate == "h264_nvenc" and settings.video_hwaccel == "none":
                continue
            err = _try_encode(ffmpeg_path, candidate, hdr=False)
            if err is None:
                encoder = candidate
                hwaccel = "cuda" if candidate == "h264_nvenc" else "none"
                notes.append(f"{candidate} encodes SDR but the tone-map chain failed")
                break

    return Capability(
        ffmpeg_path=ffmpeg_path,
        ffmpeg_version=version,
        encoder=encoder,
        hwaccel=hwaccel,
        tonemap_ok=tonemap_ok,
        notes=tuple(notes),
    )


_cached: Capability | None = None
_cache_lock = threading.Lock()


def capability(settings: Settings, *, refresh: bool = False) -> Capability:
    """Process-wide cached :func:`probe` result.

    Cached because the probe spawns ffmpeg several times and the answer cannot
    change without a restart — a driver upgrade does change it, which is why
    ``ffmpeg_version`` is a fingerprint member and the cache key moves with it.
    """
    global _cached
    with _cache_lock:
        if _cached is None or refresh:
            _cached = probe(settings)
        return _cached


def reset_cache() -> None:
    """Drop the cached probe result (tests, and a future explicit re-probe)."""
    global _cached
    with _cache_lock:
        _cached = None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def is_hdr(info: dict) -> bool:
    """True when a :func:`~.codecs._stream_report` describes an HDR source.

    Judged from the transfer characteristics, which is what the tone-map chain
    actually interprets.  An unreadable or absent transfer is **not** treated as
    HDR: tone-mapping an SDR picture darkens it, and inventing that
    interpretation is worse than declining to.
    """
    transfer = (info.get("video_color_trc") or "").lower()
    return transfer in HDR_TRANSFERS


def select(settings: Settings, cap: Capability, *, hdr: bool) -> Pipeline:
    """The :class:`Pipeline` that will run, given the host and the source.

    Raises :class:`RuntimeError` when there is none — callers turn that into the
    player's transcode-unavailable state rather than a 500.
    """
    reason = cap.unavailable_reason(hdr=hdr)
    if reason is not None or cap.encoder is None or cap.ffmpeg_version is None:
        raise RuntimeError(reason or "no usable ffmpeg encode pipeline")
    return Pipeline(
        hwaccel=cap.hwaccel,
        # Software decode throughout.  §3.1 measured the alternative: with
        # `-hwaccel_output_format cuda` the frames never reach ffmpeg's
        # autorotate and every portrait clip plays on its side.  The GPU is used
        # for the encoder only, which is the row §3.1 and §3.5 both found
        # correct on orientation *and* colour.
        decoder="software",
        filter_chain=_video_filter(settings.video_output_height, hdr=hdr),
        encoder=cap.encoder,
        rate_control=_RATE_CONTROL[cap.encoder],
        output_height=settings.video_output_height,
        chunk_seconds=settings.video_chunk_seconds,
        audio=AUDIO_ARGS,
        ffmpeg_version=cap.ffmpeg_version,
    )
