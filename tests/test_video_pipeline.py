"""Tests for video-specific pipeline logic: _group_video_hits().

Forensic correctness requirements verified here:
- Each score on a grouped hit is an exact Qdrant similarity from one real 1:1
  comparison; nothing is computed or averaged across comparisons.
- The grouped hit is visible under all mode filters for which any dataset frame
  matched, not only the modes present on the highest-scoring frame.
- The representative frame (best score) provides the thumbnail/image_hash.
- matched_frames contains every dataset frame, sorted by timecode.
- query_timecodes is preserved unchanged from the representative.
"""

from __future__ import annotations

import pytest

from scalar_forensic.web.pipeline import Hit, _group_video_hits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIDEO_PATH = "/evidence/clip.mp4"


def _frame_hit(
    *,
    timecode_ms: int,
    scores: dict,
    video_path: str = VIDEO_PATH,
    frame_hash: str | None = None,
    query_timecodes: list[int] | None = None,
    model_provenance: dict | None = None,
) -> Hit:
    fhash = frame_hash or f"hash_{timecode_ms}"
    virtual = f"{video_path}::frame_{timecode_ms:06d}_t={timecode_ms}ms"
    return Hit(
        path=virtual,
        scores=scores,
        image_hash=fhash,
        is_video_frame=True,
        video_path=video_path,
        frame_timecode_ms=timecode_ms,
        query_timecodes=query_timecodes or [0],
        model_provenance=model_provenance or {},
    )


# ---------------------------------------------------------------------------
# Non-video hits pass through unchanged
# ---------------------------------------------------------------------------


def test_non_video_hits_pass_through():
    image_hit = Hit(path="/evidence/img.jpg", scores={"semantic": 0.8})
    result = _group_video_hits([image_hit])
    assert len(result) == 1
    assert result[0] is image_hit


def test_empty_list_returns_empty():
    assert _group_video_hits([]) == []


# ---------------------------------------------------------------------------
# Single video frame produces a single grouped hit
# ---------------------------------------------------------------------------


def test_single_frame_grouped_hit_scores_preserved():
    frame = _frame_hit(timecode_ms=1000, scores={"semantic": 0.9})
    result = _group_video_hits([frame])
    assert len(result) == 1
    hit = result[0]
    assert hit.scores == {"semantic": 0.9}
    assert hit.video_path == VIDEO_PATH
    assert hit.is_video_frame is True


def test_single_frame_matched_frames_list():
    frame = _frame_hit(timecode_ms=2000, scores={"altered": 0.7}, frame_hash="abc123")
    result = _group_video_hits([frame])
    hit = result[0]
    assert len(hit.matched_frames) == 1
    mf = hit.matched_frames[0]
    assert mf.timecode_ms == 2000
    assert mf.frame_hash == "abc123"
    assert mf.scores == {"altered": 0.7}


# ---------------------------------------------------------------------------
# Representative = highest-scoring frame
# ---------------------------------------------------------------------------


def test_representative_is_highest_scoring_frame():
    low = _frame_hit(timecode_ms=1000, scores={"semantic": 0.6}, frame_hash="low")
    high = _frame_hit(timecode_ms=3000, scores={"semantic": 0.95}, frame_hash="high")
    med = _frame_hit(timecode_ms=2000, scores={"semantic": 0.75}, frame_hash="med")

    result = _group_video_hits([low, high, med])
    assert len(result) == 1
    hit = result[0]
    assert hit.image_hash == "high", "representative must be the highest-scoring frame"
    assert hit.frame_timecode_ms == 3000


# ---------------------------------------------------------------------------
# matched_frames sorted by timecode regardless of input order
# ---------------------------------------------------------------------------


def test_matched_frames_sorted_by_timecode():
    frames = [
        _frame_hit(timecode_ms=3000, scores={"semantic": 0.7}),
        _frame_hit(timecode_ms=1000, scores={"semantic": 0.9}),
        _frame_hit(timecode_ms=2000, scores={"semantic": 0.8}),
    ]
    result = _group_video_hits(frames)
    timecodes = [mf.timecode_ms for mf in result[0].matched_frames]
    assert timecodes == [1000, 2000, 3000]


# ---------------------------------------------------------------------------
# Mode union: grouped hit must surface all modes present in any dataset frame
# ---------------------------------------------------------------------------


def test_grouped_hit_includes_alter_mode_from_non_representative_frame():
    """If the best SEMAN frame differs from the best ALTER frame, the grouped
    hit must still carry both modes so both mode filters show the video."""
    alter_frame = _frame_hit(
        timecode_ms=1000,
        scores={"altered": 0.85},
        model_provenance={"altered": {"name": "sscd", "hash": "sscdv1"}},
    )
    seman_frame = _frame_hit(
        timecode_ms=3000,
        scores={"semantic": 0.95},
        model_provenance={"semantic": {"name": "dino", "hash": "dinov1"}},
    )

    result = _group_video_hits([alter_frame, seman_frame])
    hit = result[0]

    assert "altered" in hit.scores, "ALTER mode from non-representative frame must be in scores"
    assert "semantic" in hit.scores, "SEMAN mode from representative frame must be in scores"


def test_grouped_hit_mode_scores_are_exact_best_per_mode():
    """Each score in the grouped hit must be the best (highest) score for that
    mode across all dataset frames — an exact Qdrant similarity, not an average."""
    frames = [
        _frame_hit(timecode_ms=1000, scores={"altered": 0.80}),
        _frame_hit(timecode_ms=2000, scores={"altered": 0.92}),  # best ALTER
        _frame_hit(timecode_ms=3000, scores={"semantic": 0.95}),  # best SEMAN, highest overall
    ]

    result = _group_video_hits(frames)
    hit = result[0]

    assert hit.scores.get("altered") == pytest.approx(0.92), (
        "ALTER score must be 0.92 — the best single ALTER comparison"
    )
    assert hit.scores.get("semantic") == pytest.approx(0.95), (
        "SEMAN score must be 0.95 — the best single SEMAN comparison"
    )


def test_grouped_hit_model_provenance_follows_best_score_per_mode():
    """model_provenance for a mode must come from the frame that holds the
    best score for that mode, not from a lower-scoring frame."""
    low_alter = _frame_hit(
        timecode_ms=1000,
        scores={"altered": 0.70},
        model_provenance={"altered": {"name": "sscd", "hash": "old"}},
    )
    high_alter = _frame_hit(
        timecode_ms=2000,
        scores={"altered": 0.92},
        model_provenance={"altered": {"name": "sscd", "hash": "new"}},
    )

    result = _group_video_hits([low_alter, high_alter])
    prov = result[0].model_provenance.get("altered", {})
    assert prov.get("hash") == "new", "provenance must come from the best-scoring ALTER frame"


# ---------------------------------------------------------------------------
# query_timecodes preserved
# ---------------------------------------------------------------------------


def test_query_timecodes_preserved_from_representative():
    frame_a = _frame_hit(timecode_ms=1000, scores={"semantic": 0.7}, query_timecodes=[500])
    frame_b = _frame_hit(timecode_ms=2000, scores={"semantic": 0.95}, query_timecodes=[500])

    result = _group_video_hits([frame_a, frame_b])
    assert result[0].query_timecodes == [500]


# ---------------------------------------------------------------------------
# Multiple different dataset videos stay separate
# ---------------------------------------------------------------------------


def test_two_different_videos_produce_two_grouped_hits():
    clip1 = _frame_hit(timecode_ms=1000, scores={"semantic": 0.8}, video_path="/clips/a.mp4")
    clip2 = _frame_hit(timecode_ms=2000, scores={"semantic": 0.7}, video_path="/clips/b.mp4")

    result = _group_video_hits([clip1, clip2])
    assert len(result) == 2
    video_paths = {h.video_path for h in result}
    assert video_paths == {"/clips/a.mp4", "/clips/b.mp4"}


# ---------------------------------------------------------------------------
# Mixed video + image hits
# ---------------------------------------------------------------------------


def test_mixed_hits_video_and_image():
    img = Hit(path="/img/photo.jpg", scores={"semantic": 0.85})
    frame_a = _frame_hit(timecode_ms=500, scores={"semantic": 0.9})
    frame_b = _frame_hit(timecode_ms=1500, scores={"altered": 0.8})

    result = _group_video_hits([img, frame_a, frame_b])

    # One image hit + one grouped video hit
    assert len(result) == 2
    image_hits = [h for h in result if not h.is_video_frame]
    video_hits = [h for h in result if h.is_video_frame]
    assert len(image_hits) == 1
    assert len(video_hits) == 1

    grouped = video_hits[0]
    assert "semantic" in grouped.scores
    assert "altered" in grouped.scores
    assert len(grouped.matched_frames) == 2
