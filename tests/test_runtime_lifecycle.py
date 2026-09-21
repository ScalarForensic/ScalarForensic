"""Regressions for bounded encoder and session resource lifetimes."""

import sys
import time

import pytest

from scalar_forensic.video_playback.encode import CeilingExceeded, EncodeError, _run_watched


def test_silent_encoder_deadline_reaps_child():
    children = []
    start = time.monotonic()
    with pytest.raises(EncodeError, match="timed out") as error:
        _run_watched(
            [sys.executable, "-c", "import time; time.sleep(0.8)"],
            0.1,
            on_start=children.append,
        )
    assert error.value.timed_out
    assert time.monotonic() - start < 0.6
    assert children[0].poll() is not None


def test_silent_growing_encoder_hits_ceiling(tmp_path):
    part = tmp_path / "copy.part"
    with pytest.raises(CeilingExceeded):
        _run_watched(
            [
                sys.executable,
                "-c",
                "import pathlib,time,sys; "
                "pathlib.Path(sys.argv[1]).write_bytes(b'x'*256); time.sleep(0.8)",
                str(part),
            ],
            2,
            part=part,
            limit_bytes=32,
        )


def test_progress_callback_exception_reaps_child():
    children = []

    def fail(progress):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        _run_watched(
            [
                sys.executable,
                "-c",
                "import time; print('progress=continue', flush=True); time.sleep(0.8)",
            ],
            2,
            on_start=children.append,
            on_progress=fail,
        )
    assert children[0].poll() is not None


def test_completed_job_history_has_ttl_and_count_bound():
    from types import SimpleNamespace

    from scalar_forensic.video_playback.jobs import JobRunner

    runner = JobRunner()
    settings = SimpleNamespace(video_completed_job_ttl_seconds=10, video_completed_job_max=2)
    now = time.monotonic()
    for i, age in enumerate((30, 3, 2, 1)):
        runner._jobs[str(i)] = SimpleNamespace(finished_at=now - age, settings=settings)
    runner._jobs["running"] = SimpleNamespace(finished_at=None, settings=settings)
    runner.reap_completed()
    assert set(runner._jobs) == {"2", "3", "running"}


@pytest.mark.parametrize("endpoint", ["semantic_stats", "hit_provenance"])
def test_audit_database_call_does_not_block_event_loop(monkeypatch, endpoint):
    import asyncio
    from types import SimpleNamespace

    from scalar_forensic.web.routes import analyze as routes

    def delayed(*args):
        time.sleep(0.15)
        if endpoint == "semantic_stats":
            return SimpleNamespace(count=1), None
        return {"found": True}

    monkeypatch.setattr(routes, "get_session", lambda _: object())
    monkeypatch.setattr(
        routes,
        "query_semantic_stats" if endpoint == "semantic_stats" else "get_hit_qdrant_provenance",
        delayed,
    )

    async def run():
        work = (
            routes.semantic_stats("session", "file")
            if endpoint == "semantic_stats"
            else routes.hit_provenance("a" * 64)
        )
        task = asyncio.create_task(work)
        await asyncio.sleep(0.03)
        assert not task.done(), "synchronous database call blocked the heartbeat"
        assert (await task).status_code == 200

    asyncio.run(run())


def test_analysis_disconnect_stops_at_checkpoint_and_defers_cleanup(monkeypatch, tmp_path):
    import asyncio
    import io
    import threading
    from types import SimpleNamespace

    from fastapi import UploadFile

    from scalar_forensic.web import session as sessions
    from scalar_forensic.web.routes import analyze as routes

    release = threading.Event()
    entered = threading.Event()
    stopped = threading.Event()
    seen = []

    def analysis(session, *args):
        seen.append(session)
        try:
            yield SimpleNamespace(type="progress")
            entered.set()
            release.wait(2)
            yield SimpleNamespace(type="progress")
            raise AssertionError("analysis advanced beyond cancellation checkpoint")
        finally:
            stopped.set()

    monkeypatch.setattr(routes, "analyze_session", analysis)
    monkeypatch.setattr(routes, "session_temp_dir", lambda: tmp_path)

    async def run():
        response = await routes.analyze(
            [UploadFile(filename="test.jpg", file=io.BytesIO(b"test"))], modes="exact"
        )
        stream = response.body_iterator
        await anext(stream)
        while not entered.is_set():
            await asyncio.sleep(0.01)
        await stream.aclose()
        assert sessions.get_session(seen[0].session_id) is None
        assert tmp_path.exists()
        release.set()
        for _ in range(100):
            if stopped.is_set() and not tmp_path.exists():
                break
            await asyncio.sleep(0.01)
        assert stopped.is_set()
        assert not tmp_path.exists()

    asyncio.run(run())


def test_cancelled_job_keeps_pin_and_slot_until_encoder_reaped(monkeypatch, tmp_path):
    import asyncio
    import threading
    from types import SimpleNamespace

    from scalar_forensic.config import Settings
    from scalar_forensic.video_playback import cache, jobs

    attached = threading.Event()
    reaped = threading.Event()
    children = []
    digest = "c" * 64

    def encode(*args, on_start, **kwargs):
        def started(proc):
            children.append(proc)
            on_start(proc)
            attached.set()

        try:
            _run_watched([sys.executable, "-c", "import time; time.sleep(5)"], 10, on_start=started)
        finally:
            assert digest in cache.protected_videos()
            assert jobs.admission.admitted == 1
            reaped.set()

    monkeypatch.setattr(jobs, "encode_full", encode)
    monkeypatch.setattr(jobs.audit, "record_transcode", lambda *a, **kw: None)
    monkeypatch.setenv("SFN_VIDEO_CACHE_DIR", str(tmp_path))
    settings = Settings()
    request = SimpleNamespace(
        source=tmp_path / "source.mp4",
        digest=digest,
        pipeline=SimpleNamespace(fingerprint=lambda: "f" * 64),
        cache_dir=tmp_path,
        capability=None,
        hdr=False,
        has_audio=False,
        limit_bytes=None,
        started_by=None,
        override=None,
    )

    async def run():
        runner = jobs.JobRunner()
        job = runner.start(request, settings, claim_id="test-claim")
        while not attached.is_set():
            await asyncio.sleep(0.01)
        job.task.cancel()
        await job.task
        assert reaped.is_set()
        assert children[0].poll() is not None
        assert jobs.admission.admitted == 0
        assert job.cancelled
        assert job.finished_at is not None
        # The read lease intentionally remains until its normal expiry; the
        # writer pin must be gone now that no encoder can touch the directory.
        cache.release_lease(digest)
        assert digest not in cache.protected_videos()

    jobs.admission.reset()
    cache.reset_leases()
    try:
        asyncio.run(run())
    finally:
        jobs.admission.reset()
        cache.reset_leases()
