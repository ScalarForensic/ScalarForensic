"""Public entry points must route before invoking a local evidence parser."""

from types import SimpleNamespace

import pytest

from scalar_forensic import video, video_access
from scalar_forensic.media_worker.client import WorkerUnavailable


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "SFN_VIDEO_CPU_STATE",
        "SFN_VIDEO_CHUNK_STATE",
        "SFN_VIDEO_FULL_STATE",
        "SFN_VIDEO_WORKER_CASE",
        "SFN_VIDEO_WORKER_RECEIPTS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_partial_configuration_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("SFN_VIDEO_CPU_STATE", str(tmp_path / "state"))
    with pytest.raises(WorkerUnavailable):
        video.get_video_info(tmp_path / "file")


def test_unconfigured_local():
    assert not video_access.configured()


def test_offline_full_slot_does_not_block_cpu(monkeypatch, tmp_path):
    from scalar_forensic.media_worker.video_runtime import VideoRuntime, publish_state

    for slot in ("cpu", "chunk", "full"):
        root = tmp_path / slot
        root.mkdir()
        monkeypatch.setenv(f"SFN_VIDEO_{slot.upper()}_STATE", str(root / "current.json"))
    monkeypatch.setenv("SFN_VIDEO_WORKER_CASE", "case")
    monkeypatch.setenv("SFN_VIDEO_WORKER_RECEIPTS", str(tmp_path / "receipts"))
    cpu = tmp_path / "cpu"
    state = VideoRuntime(
        "a" * 32,
        cpu / "ipc/socket",
        "case",
        "sha256:" + "b" * 64,
        "sfn-video-cpu-v1",
        cpu / "failed",
        cpu / "control/job.lock",
        tmp_path,
        1000,
        2,
    )
    publish_state(cpu / "current.json", state)
    assert video_access.client().state_path == cpu / "current.json"
    assert video_access.client().queue_timeout == 5
    chunk = tmp_path / "chunk"
    from dataclasses import replace

    publish_state(
        chunk / "current.json",
        replace(state, policy="sfn-video-nvenc-v1", lock_path=chunk / "control/job.lock"),
    )
    assert video_access.client("chunk").queue_timeout == 30
    monkeypatch.setenv("SFN_VIDEO_FULL_STATE", str(cpu / "other.json"))
    with pytest.raises(WorkerUnavailable, match="distinct"):
        video_access.client()


def test_public_cpu_routing(monkeypatch, tmp_path):
    from scalar_forensic.video_playback import codecs, rewrap

    calls = []
    monkeypatch.setattr(video_access, "configured", lambda: True)
    monkeypatch.setattr(
        video_access,
        "client",
        lambda: SimpleNamespace(
            call=lambda operation, *args, **kwargs: calls.append(operation) or {"remote": True}
        ),
    )
    monkeypatch.setattr(video_access, "frame_at", lambda *args: "raw")
    monkeypatch.setattr(video_access, "frames", lambda *args: iter(["raw"]))
    assert video.get_video_info(tmp_path / "missing") == {"remote": True}
    assert codecs._stream_report(tmp_path / "missing") == {"remote": True}
    assert rewrap._remux_to_mp4(tmp_path / "missing", tmp_path / "out") == {"remote": True}
    assert video.extract_frame_at(tmp_path / "missing", 0) == "raw"
    assert list(video.extract_frames(tmp_path / "missing")) == ["raw"]
    assert calls == ["info", "stream_report", "rewrap"]


@pytest.mark.parametrize("watched,expected_slot", [(False, "chunk"), (True, "full")])
def test_renderer_slot_routing(monkeypatch, tmp_path, watched, expected_slot):
    from scalar_forensic.video_playback.encode import Watch

    calls = []

    def choose(slot):
        calls.append(slot)
        return SimpleNamespace(
            call=lambda *a, **kw: dict(
                pipeline=None, command=[], wall_seconds=0, fell_back=False, fallback_reason=None
            )
        )

    monkeypatch.setattr(video_access, "client", choose)
    settings = SimpleNamespace(
        video_output_height=1080, video_chunk_seconds=15, video_hwaccel="auto", video_job_timeout=60
    )
    attached = []
    watch = Watch(on_start=attached.append) if watched else None
    video_access.encode(
        settings,
        tmp_path / "source",
        tmp_path / "dst",
        hdr=False,
        start=None,
        duration=None,
        has_audio=True,
        watch=watch,
    )
    assert calls == [expected_slot]
    if watched:
        from scalar_forensic.media_worker.video_client import Cancellation

        assert isinstance(attached[0], Cancellation)
