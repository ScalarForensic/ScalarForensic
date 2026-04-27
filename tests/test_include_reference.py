"""Tests for include_reference behavior in query_session (Copilot review item).

Verifies:
- _query_vector is called against the reference collection when include_reference=True
  and settings.reference_collection is configured
- Reference hits carry is_reference=True
- Reference hits are never merged with case hits, even when unify=True
- No reference queries are issued when include_reference=False
- No reference queries are issued when reference_collection is None
- Both altered and semantic modes each trigger a reference collection query
- unify=False keeps reference hits as separate rows from case hits
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scalar_forensic.web.pipeline import Hit, query_session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CASE_COLLECTION = "sfn"
REF_COLLECTION = "sfn_reference"

PATH_CASE = "/case/photo.jpg"
PATH_REF = "/reference/known.jpg"


def _make_entry(
    file_id: str = "f1",
    filename: str = "photo.jpg",
    dino: bool = True,
    sscd: bool = False,
) -> MagicMock:
    entry = MagicMock()
    entry.file_id = file_id
    entry.filename = filename
    entry.error = None
    entry.file_hash = "sha256abc"
    entry.file_hash_md5 = "md5abc"
    entry.dino_embedding = [0.1] * 1024 if dino else None
    entry.sscd_embedding = [0.2] * 512 if sscd else None
    entry.is_video = False
    entry.video_frames = None
    return entry


def _make_session(entries: list) -> MagicMock:
    session = MagicMock()
    session.files = entries
    return session


def _mock_settings(reference_collection: str | None = REF_COLLECTION) -> MagicMock:
    s = MagicMock()
    s.qdrant_url = "http://localhost:6333"
    s.qdrant_api_key = None
    s.collection = CASE_COLLECTION
    s.reference_collection = reference_collection
    return s


def _case_hit(path: str = PATH_CASE, mode: str = "semantic") -> Hit:
    return Hit(path=path, scores={mode: 0.90}, is_reference=False)


def _ref_hit(path: str = PATH_REF, mode: str = "semantic") -> Hit:
    return Hit(path=path, scores={mode: 0.85}, is_reference=True)


def _dispatch_query_vector(
    client,
    collection,
    vector,
    mode,
    threshold,
    limit,
    vector_name="dino",
    is_reference_result=False,
):
    """Return a case or reference hit depending on which collection is queried."""
    if collection == REF_COLLECTION:
        return ([Hit(path=PATH_REF, scores={mode: 0.85}, is_reference=True)], [])
    return ([Hit(path=PATH_CASE, scores={mode: 0.90})], [])


# ---------------------------------------------------------------------------
# _query_vector is called with the reference collection
# ---------------------------------------------------------------------------


def test_reference_collection_queried_when_include_reference_true():
    """_query_vector must be invoked with the reference collection name."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            side_effect=_dispatch_query_vector,
        ) as mock_qv,
    ):
        query_session(session, ["semantic"], 0.75, 0.55, 10, settings, include_reference=True)

    collections_queried = {c.kwargs.get("collection") or c.args[1] for c in mock_qv.call_args_list}
    assert REF_COLLECTION in collections_queried, (
        "reference collection must be queried when include_reference=True"
    )


def test_reference_collection_not_queried_when_include_reference_false():
    """When include_reference=False, no queries against the reference collection."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            side_effect=_dispatch_query_vector,
        ) as mock_qv,
    ):
        query_session(session, ["semantic"], 0.75, 0.55, 10, settings, include_reference=False)

    collections_queried = {c.kwargs.get("collection") or c.args[1] for c in mock_qv.call_args_list}
    assert REF_COLLECTION not in collections_queried, (
        "reference collection must not be queried when include_reference=False"
    )


def test_reference_collection_not_queried_when_not_configured():
    """When reference_collection is None, no reference queries even if include_reference=True."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings(reference_collection=None)

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            return_value=([_case_hit()], []),
        ) as mock_qv,
    ):
        query_session(session, ["semantic"], 0.75, 0.55, 10, settings, include_reference=True)

    # Only the case collection should have been queried
    collections_queried = {c.kwargs.get("collection") or c.args[1] for c in mock_qv.call_args_list}
    assert REF_COLLECTION not in collections_queried


# ---------------------------------------------------------------------------
# Reference hits carry is_reference=True
# ---------------------------------------------------------------------------


def test_reference_hits_have_is_reference_true():
    """Hits returned from the reference collection overlay must have is_reference=True."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            side_effect=_dispatch_query_vector,
        ),
    ):
        results, _ = query_session(
            session, ["semantic"], 0.75, 0.55, 10, settings, include_reference=True
        )

    ref_hits = [h for h in results[0].hits if h.path == PATH_REF]
    assert ref_hits, "at least one reference hit must be present"
    assert all(h.is_reference for h in ref_hits), "all reference hits must have is_reference=True"


def test_case_hits_have_is_reference_false():
    """Case collection hits must not be tainted with is_reference=True."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            side_effect=_dispatch_query_vector,
        ),
    ):
        results, _ = query_session(
            session, ["semantic"], 0.75, 0.55, 10, settings, include_reference=True
        )

    case_hits = [h for h in results[0].hits if h.path == PATH_CASE]
    assert case_hits, "at least one case hit must be present"
    assert all(not h.is_reference for h in case_hits), "case hits must not have is_reference=True"


# ---------------------------------------------------------------------------
# Reference hits are never merged with case hits (unify=True)
# ---------------------------------------------------------------------------


def test_reference_hits_not_merged_with_case_hits_unify_true():
    """With unify=True, reference collection hits must still appear as distinct
    rows alongside case collection hits — they must not be silently dropped or
    absorbed into the case result rows.  Reference hits have distinct paths from
    case hits (design invariant), so they can never be keyed to the same merge
    bucket; this test verifies both kinds are present in the final output."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            side_effect=_dispatch_query_vector,
        ),
    ):
        results, _ = query_session(
            session,
            ["semantic"],
            0.75,
            0.55,
            10,
            settings,
            unify=True,
            include_reference=True,
        )

    hits = results[0].hits
    case_hits = [h for h in hits if not h.is_reference]
    ref_hits = [h for h in hits if h.is_reference]
    assert case_hits, "case hits must be present even when include_reference=True"
    assert ref_hits, "reference overlay hits must survive the unify merge pass"
    # Scores must not bleed across collections
    assert all(not h.is_reference for h in case_hits)
    assert all(h.is_reference for h in ref_hits)


# ---------------------------------------------------------------------------
# Both modes (altered + semantic) query the reference collection
# ---------------------------------------------------------------------------


def test_both_modes_query_reference_collection():
    """When both altered and semantic are active, _query_vector must be called
    against the reference collection for each mode."""
    session = _make_session([_make_entry(dino=True, sscd=True)])
    settings = _mock_settings()

    ref_calls: list[str] = []

    def _tracking_dispatch(
        client,
        collection,
        vector,
        mode,
        threshold,
        limit,
        vector_name="dino",
        is_reference_result=False,
    ):
        if collection == REF_COLLECTION:
            ref_calls.append(mode)
        return (
            [
                Hit(
                    path=PATH_CASE if collection == CASE_COLLECTION else PATH_REF,
                    scores={mode: 0.90},
                )
            ],
            [],
        )

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch("scalar_forensic.web.pipeline._query_vector", side_effect=_tracking_dispatch),
    ):
        query_session(
            session,
            ["altered", "semantic"],
            0.75,
            0.55,
            10,
            settings,
            include_reference=True,
        )

    assert "altered" in ref_calls, "altered mode must query the reference collection"
    assert "semantic" in ref_calls, "semantic mode must query the reference collection"


# ---------------------------------------------------------------------------
# unify=False: reference hits still appear as separate rows from case hits
# ---------------------------------------------------------------------------


def test_reference_hits_separate_from_case_hits_unify_false():
    """With unify=False, reference hits appear as their own rows, distinct from
    the case collection rows for the same mode."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            side_effect=_dispatch_query_vector,
        ),
    ):
        results, _ = query_session(
            session,
            ["semantic"],
            0.75,
            0.55,
            10,
            settings,
            unify=False,
            include_reference=True,
        )

    hits = results[0].hits
    ref_hits = [h for h in hits if h.is_reference]
    case_hits = [h for h in hits if not h.is_reference]
    assert ref_hits, "reference hits must be present in unify=False results"
    assert case_hits, "case hits must be present in unify=False results"
    ref_paths = {h.path for h in ref_hits}
    case_paths = {h.path for h in case_hits}
    assert ref_paths.isdisjoint(case_paths), (
        "reference and case paths must not overlap in unify=False results"
    )


# ---------------------------------------------------------------------------
# Reference overlay merges altered+semantic onto one row when unify=True
# ---------------------------------------------------------------------------


def test_reference_modes_merged_on_same_path_unify_true():
    """Mirrors the case-collection behaviour: a reference path that matches in
    both altered and semantic must collapse to a single Hit whose scores dict
    carries both modes when unify=True."""
    session = _make_session([_make_entry(dino=True, sscd=True)])
    settings = _mock_settings()

    def _dispatch(
        client,
        collection,
        vector,
        mode,
        threshold,
        limit,
        vector_name="dino",
        is_reference_result=False,
    ):
        # Reference collection matches the same PATH_REF for both modes.
        if collection == REF_COLLECTION:
            score = 0.85 if mode == "semantic" else 0.80
            return ([Hit(path=PATH_REF, scores={mode: score}, is_reference=True)], [])
        return ([Hit(path=PATH_CASE, scores={mode: 0.90})], [])

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch("scalar_forensic.web.pipeline._query_vector", side_effect=_dispatch),
    ):
        results, _ = query_session(
            session,
            ["altered", "semantic"],
            0.75,
            0.55,
            10,
            settings,
            unify=True,
            include_reference=True,
        )

    ref_hits = [h for h in results[0].hits if h.is_reference]
    assert len(ref_hits) == 1, (
        f"reference path matching both modes must collapse to one row under "
        f"unify=True, got {len(ref_hits)}"
    )
    assert set(ref_hits[0].scores.keys()) == {"altered", "semantic"}, (
        "merged reference hit must carry both modes' scores"
    )


# ---------------------------------------------------------------------------
# Reference overlay propagates query_timecodes from video-query frames
# ---------------------------------------------------------------------------


def test_reference_hits_carry_query_timecodes_for_video_query():
    """When the query is a video, each reference hit must carry the timecode of
    the query frame that matched it, matching the case-collection behaviour."""
    # Build a video session with two frames at distinct timecodes.
    entry = _make_entry(dino=True)
    entry.is_video = True
    frame_a = MagicMock()
    frame_a.timecode_ms = 1000
    frame_a.dino_embedding = [0.1] * 1024
    frame_a.sscd_embedding = None
    frame_b = MagicMock()
    frame_b.timecode_ms = 2000
    frame_b.dino_embedding = [0.2] * 1024
    frame_b.sscd_embedding = None
    entry.video_frames = [frame_a, frame_b]
    # When is_video is True the top-level embeddings are ignored in favour of
    # per-frame ones; clear them for clarity.
    entry.dino_embedding = None
    entry.sscd_embedding = None

    session = _make_session([entry])
    settings = _mock_settings()

    def _dispatch(
        client,
        collection,
        vector,
        mode,
        threshold,
        limit,
        vector_name="dino",
        is_reference_result=False,
    ):
        if collection == REF_COLLECTION:
            return ([Hit(path=PATH_REF, scores={mode: 0.85}, is_reference=True)], [])
        return ([Hit(path=PATH_CASE, scores={mode: 0.90})], [])

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch("scalar_forensic.web.pipeline._query_vector", side_effect=_dispatch),
    ):
        results, _ = query_session(
            session,
            ["semantic"],
            0.75,
            0.55,
            10,
            settings,
            unify=True,
            include_reference=True,
        )

    ref_hits = [h for h in results[0].hits if h.is_reference]
    assert ref_hits, "reference hits must be present for video query"
    # After the final unify pass the same reference path from both qtcs merges
    # into one hit whose query_timecodes holds both 1000 and 2000.
    all_qtcs: set[int] = set()
    for h in ref_hits:
        for tc in h.query_timecodes or []:
            all_qtcs.add(tc)
    assert {1000, 2000} <= all_qtcs, (
        f"reference hits must carry query_timecodes from every matching query frame, got {all_qtcs}"
    )


# ---------------------------------------------------------------------------
# Path-overlap safety: reference and case hits stay distinct even at same path
# ---------------------------------------------------------------------------


def test_reference_and_case_hits_isolated_on_path_overlap():
    """If a reference image happens to share a path with a case image, the two
    must still appear as independent rows — the is_reference flag is the
    chain-of-custody boundary."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    shared_path = "/overlap/shared.jpg"

    def _dispatch(
        client,
        collection,
        vector,
        mode,
        threshold,
        limit,
        vector_name="dino",
        is_reference_result=False,
    ):
        if collection == REF_COLLECTION:
            return ([Hit(path=shared_path, scores={mode: 0.85}, is_reference=True)], [])
        return ([Hit(path=shared_path, scores={mode: 0.90})], [])

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch("scalar_forensic.web.pipeline._query_vector", side_effect=_dispatch),
    ):
        results, _ = query_session(
            session,
            ["semantic"],
            0.75,
            0.55,
            10,
            settings,
            unify=True,
            include_reference=True,
        )

    hits = results[0].hits
    ref_hits = [h for h in hits if h.is_reference]
    case_hits = [h for h in hits if not h.is_reference]
    assert len(ref_hits) == 1, "reference hit on shared path must not be absorbed into case hit"
    assert len(case_hits) == 1, "case hit on shared path must not be absorbed into reference hit"
