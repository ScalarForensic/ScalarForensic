import io
import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from scalar_forensic.cli import faces_app, index
from scalar_forensic.faces.store import PurgeResult

runner = CliRunner()


def _typer_app(fn):
    import typer

    app = typer.Typer()
    app.command()(fn)
    return app


def test_index_requires_at_least_one_modality(tmp_path, monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    result = runner.invoke(_typer_app(index), [str(tmp_path)])
    assert result.exit_code == 1
    assert "--dino, --sscd or --faces" in result.output


def test_faces_flag_without_env_is_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    result = runner.invoke(_typer_app(index), [str(tmp_path), "--faces"])
    assert result.exit_code == 1
    assert "SFN_FACES_ENABLED" in result.output


def test_faces_flag_enabled_without_models_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.delenv("SFN_FACE_DETECTOR_MODEL", raising=False)
    monkeypatch.delenv("SFN_FACE_EMBEDDER_MODEL", raising=False)
    result = runner.invoke(_typer_app(index), [str(tmp_path), "--faces"])
    assert result.exit_code == 1
    assert "SFN_FACE_DETECTOR_MODEL" in result.output


def _enable_faces(monkeypatch, tmp_path):
    det = tmp_path / "yunet.onnx"
    det.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"
    emb.write_bytes(b"x")
    (tmp_path / "emb.onnx.manifest.json").write_text("{}")
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_FACE_DETECTOR_MODEL", str(det))
    monkeypatch.setenv("SFN_FACE_EMBEDDER_MODEL", str(emb))
    monkeypatch.setenv("SFN_EXAMINER_ID", "ex1")
    monkeypatch.setenv("SFN_FACE_STORE_DIR", str(tmp_path / "faces"))


@pytest.fixture
def run_faces_cli(tmp_path, monkeypatch):
    """Drive ``index --faces`` over one image with a scripted FaceIndexResult.

    The pipeline is a mock except for its AuditLog, which is real and writes
    into *tmp_path* — the audit record is what several of these tests assert on,
    and a mocked appender would assert only that the call was made, not that the
    record on disk reconciles.
    """
    from PIL import Image

    from scalar_forensic.faces.audit import AuditLog
    from scalar_forensic.faces.indexing import FaceIndexResult

    img_dir = tmp_path / "media"
    img_dir.mkdir()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(img_dir / "a.jpg")

    def _run(
        *,
        detected: int,
        kept: int,
        review_only: int = 0,
        rejected: dict[str, int] | None = None,
        review_reasons: dict[str, int] | None = None,
        dropped_noncanonical: int = 0,
        review_only_point_ids: list[str] | None = None,
        clear_raises: Exception | None = None,
        stale: list[dict] | None = None,
        unreferenced: list[str] | None = None,
        confirm: str | None = None,
    ):
        _enable_faces(monkeypatch, tmp_path)
        pipeline = MagicMock()
        pipeline.audit = AuditLog(tmp_path / "face_audit.log")
        pipeline.cfg.config_hash = "cfg1"
        pipeline.store.collection_is_new.return_value = False
        pipeline.store.check_compat.return_value = []
        pipeline.store.processed_hashes.return_value = set()
        pipeline.store.stale_face_points.return_value = list(stale or [])
        pipeline.store.delete_face_points.return_value = PurgeResult(
            n_points=len(stale or []),
            chip_hashes=[h for s in (stale or []) for h in (s.get("review_chip_hash"),) if h],
        )
        pipeline.store.unreferenced_chip_hashes.return_value = list(unreferenced or [])
        if clear_raises is not None:
            pipeline.store.clear_face_vector.side_effect = clear_raises
        pipeline.process_image.return_value = FaceIndexResult(
            n_detected=detected,
            n_kept=kept,
            n_review_only=review_only,
            rejected=dict(rejected or {}),
            review_only_reasons=dict(review_reasons or {}),
            n_dropped_noncanonical=dropped_noncanonical,
            review_only_point_ids=list(review_only_point_ids or []),
        )
        with patch(
            "scalar_forensic.faces.indexing.FacePipeline.from_settings", return_value=pipeline
        ):
            # --report keeps the ingestion CSV out of the repo's data/reports/.
            result = runner.invoke(
                _typer_app(index),
                [str(img_dir), "--faces", "--report", str(tmp_path / "ingestion.csv")],
                input=confirm,
            )
        # A run that dies inside the face pass never reaches the index_run
        # append, so the log may legitimately not exist.
        log = tmp_path / "face_audit.log"
        events = (
            [json.loads(line) for line in log.read_text().splitlines() if line]
            if log.exists()
            else []
        )
        return result, events, pipeline

    return _run


def test_cli_summary_reconciles_counts(run_faces_cli):
    result, _, _pipeline = run_faces_cli(
        detected=6,
        kept=1,
        review_only=3,
        review_reasons={"size": 3},
        rejected={"confidence": 2},
    )
    assert result.exit_code == 0, result.output
    assert "6 detected" in result.output
    assert "1 comparable" in result.output
    assert "3 retained for review" in result.output
    assert "2 rejected" in result.output


def test_audit_index_run_records_review_only(run_faces_cli):
    _, events, _pipeline = run_faces_cli(
        detected=6,
        kept=1,
        review_only=3,
        review_reasons={"size": 3},
        rejected={"confidence": 2},
    )
    ev = [e for e in events if e["event"] == "index_run"][-1]
    assert ev["n_review_only"] == 3
    assert ev["review_only_reasons"] == {"size": 3}
    # Deliberately no reconciliation sum here: process_image is mocked, so the
    # numbers are this test's own literals.  The invariant is enforced against
    # a real run in tests/faces/test_indexing.py; what matters here is that the
    # record carries every field needed to check it.
    assert {"n_detected", "n_kept", "n_review_only", "n_rejected"} <= ev.keys()


def test_audit_index_run_records_noncanonical_drops(run_faces_cli):
    # A non-canonical drop is subtracted from n_detected before any gate sees
    # the face; without it in the record the reconciliation above is unfalsifiable.
    _, events, _pipeline = run_faces_cli(detected=2, kept=2, dropped_noncanonical=4)
    ev = [e for e in events if e["event"] == "index_run"][-1]
    assert ev["n_dropped_noncanonical"] == 4


def test_cli_summary_omits_review_clause_when_none(run_faces_cli):
    result, _, _pipeline = run_faces_cli(detected=2, kept=2)
    assert "retained for review" not in result.output
    assert "2 comparable" in result.output


def test_cli_clears_vectors_for_every_review_only_point(run_faces_cli):
    # The one line of production code that keeps a demoted observation out of
    # the search space.  Deleting it must fail the suite, not just the skipped
    # live-Qdrant test.
    ids = ["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"]
    _, _, pipeline = run_faces_cli(
        detected=3, kept=1, review_only=2, review_reasons={"size": 2}, review_only_point_ids=ids
    )
    pipeline.store.clear_face_vector.assert_called_once_with(ids)


def test_cli_writes_the_marker_only_after_the_vector_clear(run_faces_cli):
    # The marker is the idempotency record: once written for this config hash
    # the medium is never reprocessed.  Committing it before the clear would
    # make a failed demotion permanent and invisible — the payload would say
    # review-only while the vector stayed live in the index.
    ids = ["11111111-1111-1111-1111-111111111111"]
    _, _, pipeline = run_faces_cli(
        detected=2, kept=1, review_only=1, review_reasons={"size": 1}, review_only_point_ids=ids
    )
    order = [
        c[0] for c in pipeline.store.method_calls if c[0] in ("upsert_faces", "clear_face_vector")
    ]
    assert order == ["upsert_faces", "clear_face_vector", "upsert_faces"]
    marker_call = pipeline.store.upsert_faces.call_args_list[-1]
    assert marker_call.args[0] == [pipeline.store.marker_point.return_value]


def test_cli_marker_is_not_written_when_the_clear_fails(run_faces_cli):
    # delete_vectors 404s on an unknown id (verified live), so this is reachable.
    # The medium must stay unprocessed so a re-run can finish the demotion.
    ids = ["11111111-1111-1111-1111-111111111111"]
    _, _, pipeline = run_faces_cli(
        detected=2,
        kept=1,
        review_only=1,
        review_reasons={"size": 1},
        review_only_point_ids=ids,
        clear_raises=RuntimeError("Not found: no point with id"),
    )
    written = [c.args[0] for c in pipeline.store.upsert_faces.call_args_list]
    assert [pipeline.store.marker_point.return_value] not in written


def _stale(pid, **over):
    rec = {
        "id": pid,
        "observation_key": f"{'a' * 64}::10:20:30:40",
        "embedding_status": "embedded",
        "pipeline_config_hash": "oldcfg",
        "indexed_at": "2026-08-01T00:00:00+00:00",
        "review_chip_hash": f"r{pid}",
    }
    rec.update(over)
    return rec


def test_stale_observations_are_shown_before_anything_is_deleted(run_faces_cli):
    # The operator approves a deletion of biometric data; they must be told
    # what it is first — how many, of which kind, and under which config hash.
    result, _, pipeline = run_faces_cli(
        detected=1,
        kept=1,
        stale=[_stale("s1"), _stale("s2", embedding_status="review_only")],
        confirm="n\n",
    )
    assert "2 stale face observation" in result.output
    assert "1 embedded" in result.output and "1 review-only" in result.output
    assert "oldcfg" in result.output
    pipeline.store.delete_face_points.assert_not_called()


def test_declining_leaves_the_stale_observations_in_place(run_faces_cli):
    result, events, pipeline = run_faces_cli(
        detected=1, kept=1, stale=[_stale("s1")], confirm="n\n"
    )
    assert result.exit_code == 0, result.output
    pipeline.store.delete_face_points.assert_not_called()
    ev = [e for e in events if e["event"] == "index_run"][-1]
    assert ev["n_stale_detected"] == 1
    assert ev["n_stale_removed"] == 0
    # Declining must not be silent: the run continues, but the collection is
    # knowingly left holding observations the current config would not produce.
    assert "still present" in result.output.lower() or "remain" in result.output.lower()


def test_confirming_deletes_and_reports_in_summary_and_audit(run_faces_cli):
    result, events, pipeline = run_faces_cli(
        detected=1, kept=1, stale=[_stale("s1"), _stale("s2")], confirm="y\n"
    )
    assert result.exit_code == 0, result.output
    pipeline.store.delete_face_points.assert_called_once_with(["s1", "s2"])
    assert "2 stale removed" in result.output
    ev = [e for e in events if e["event"] == "index_run"][-1]
    assert ev["n_stale_detected"] == 2 and ev["n_stale_removed"] == 2


def test_stale_chip_files_go_through_the_reference_check(run_faces_cli):
    # A stale observation may share its review chip with a surviving one.
    _, _, pipeline = run_faces_cli(
        detected=1, kept=1, stale=[_stale("s1")], unreferenced=[], confirm="y\n"
    )
    pipeline.store.unreferenced_chip_hashes.assert_called_once_with(["rs1"])


def test_non_interactive_run_never_deletes(run_faces_cli, monkeypatch):
    # A scripted run must not abort at the end (every marker is already
    # written) and must not infer consent from a missing tty.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    result, events, pipeline = run_faces_cli(detected=1, kept=1, stale=[_stale("s1")])
    assert result.exit_code == 0, result.output
    pipeline.store.delete_face_points.assert_not_called()
    assert "Non-interactive" in result.output
    assert [e for e in events if e["event"] == "index_run"][-1]["n_stale_removed"] == 0


def test_no_prompt_and_no_stale_fields_when_nothing_is_stale(run_faces_cli):
    result, events, pipeline = run_faces_cli(detected=1, kept=1)
    # Substring-specific: the tmp_path in this test's own output contains the
    # word "stale" because of the test name.
    assert "stale face observation" not in result.output
    assert "stale removed" not in result.output
    pipeline.store.delete_face_points.assert_not_called()
    ev = [e for e in events if e["event"] == "index_run"][-1]
    assert ev["n_stale_detected"] == 0


def test_stale_detection_is_scoped_to_this_run_s_produced_ids(run_faces_cli):
    ids = ["11111111-1111-1111-1111-111111111111"]
    _, _, pipeline = run_faces_cli(
        detected=2,
        kept=1,
        review_only=1,
        review_reasons={"size": 1},
        review_only_point_ids=ids,
        stale=[],
    )
    # Called with this medium's hash and the ids the run actually wrote, or it
    # would report freshly written points as stale and offer to delete them.
    call = pipeline.store.stale_face_points.call_args
    assert len(call.args[0]) == 64  # the medium's sha256
    assert isinstance(call.args[1], set)


def test_purge_requires_exactly_one_scope(monkeypatch, tmp_path):
    _enable_faces(monkeypatch, tmp_path)
    result = runner.invoke(faces_app, ["purge"])
    assert result.exit_code == 1
    assert "exactly one" in result.output


def test_purge_media_deletes_chips_and_audits(monkeypatch, tmp_path):
    _enable_faces(monkeypatch, tmp_path)
    store_dir = tmp_path / "faces"
    chash = "ab" + "0" * 62
    from scalar_forensic.faces.chips import chip_paths

    png, jpg, thumb = chip_paths(store_dir, chash)
    png.parent.mkdir(parents=True, exist_ok=True)
    for p in (png, jpg, thumb):
        p.write_bytes(b"x")

    store = MagicMock()
    store.purge_media.return_value = PurgeResult(n_points=2, chip_hashes=[chash])
    store.unreferenced_chip_hashes.return_value = [chash]
    with (
        patch("scalar_forensic.faces.store.FaceStore", return_value=store),
        patch("qdrant_client.QdrantClient", MagicMock()),
    ):
        result = runner.invoke(faces_app, ["purge", "--media", "h1"])

    assert result.exit_code == 0, result.output
    assert "Purged 2 face point(s) and 3 chip file(s)." in result.output
    assert not png.exists() and not jpg.exists() and not thumb.exists()

    import json

    events = [
        json.loads(line) for line in (tmp_path / "face_audit.log").read_text().splitlines() if line
    ]
    assert events[-1]["event"] == "purge"
    assert events[-1]["examiner_id"] == "ex1" and events[-1]["n_points"] == 2


def test_purge_routes_freed_hashes_through_the_reference_check(monkeypatch, tmp_path):
    # Wiring only: the reference *decision* is the store's, and is covered by
    # test_store.py's scroll-projection test plus the live integration test.
    # What this pins is that the CLI asks before unlinking and unlinks exactly
    # what it was told — a purge that skipped the call would delete a chip a
    # surviving observation still authenticates.
    _enable_faces(monkeypatch, tmp_path)
    store_dir = tmp_path / "faces"
    from scalar_forensic.faces.chips import chip_paths

    gone = "ab" + "0" * 62
    shared = "cd" + "1" * 62
    for chash in (gone, shared):
        png, jpg, thumb = chip_paths(store_dir, chash)
        png.parent.mkdir(parents=True, exist_ok=True)
        for p in (png, jpg, thumb):
            p.write_bytes(b"x")

    store = MagicMock()
    store.purge_media.return_value = PurgeResult(n_points=2, chip_hashes=[gone, shared])
    store.unreferenced_chip_hashes.return_value = [gone]
    with (
        patch("scalar_forensic.faces.store.FaceStore", return_value=store),
        patch("qdrant_client.QdrantClient", MagicMock()),
    ):
        result = runner.invoke(faces_app, ["purge", "--media", "h1"])

    assert result.exit_code == 0, result.output
    store.unreferenced_chip_hashes.assert_called_once_with([gone, shared])
    assert not any(p.exists() for p in chip_paths(store_dir, gone))
    assert all(p.exists() for p in chip_paths(store_dir, shared))
    assert "3 chip file(s)" in result.output


def test_purge_unlinks_review_only_chip_pair(monkeypatch, tmp_path):
    # A review-only observation has no aligned PNG: the unlink loop must reach
    # the review pair through review_chip_paths, and must not count a missing PNG.
    _enable_faces(monkeypatch, tmp_path)
    store_dir = tmp_path / "faces"
    from scalar_forensic.faces.chips import review_chip_paths

    chash = "ef" + "2" * 62
    jpg, thumb = review_chip_paths(store_dir, chash)
    jpg.parent.mkdir(parents=True, exist_ok=True)
    for p in (jpg, thumb):
        p.write_bytes(b"x")

    store = MagicMock()
    store.purge_media.return_value = PurgeResult(n_points=1, chip_hashes=[chash])
    store.unreferenced_chip_hashes.return_value = [chash]
    with (
        patch("scalar_forensic.faces.store.FaceStore", return_value=store),
        patch("qdrant_client.QdrantClient", MagicMock()),
    ):
        result = runner.invoke(faces_app, ["purge", "--media", "h1"])

    assert result.exit_code == 0, result.output
    assert not jpg.exists() and not thumb.exists()
    # Two files, not four: chip_paths and review_chip_paths overlap on the
    # review pair, and the absent PNG must not inflate the audited count.
    assert "2 chip file(s)" in result.output


def test_purge_all_aborts_without_confirmation(monkeypatch, tmp_path):
    _enable_faces(monkeypatch, tmp_path)
    store = MagicMock()
    with (
        patch("scalar_forensic.faces.store.FaceStore", return_value=store),
        patch("qdrant_client.QdrantClient", MagicMock()),
    ):
        result = runner.invoke(faces_app, ["purge", "--all"], input="n\n")
    assert result.exit_code == 1
    assert "Aborted." in result.output
    store.purge_all.assert_not_called()
