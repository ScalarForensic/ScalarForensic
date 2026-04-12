"""Tests for per-mode independent result behavior (Batch 4 / issue #9).

Verifies:
- Same image appearing in two modes yields two separate Hit rows
- Deselecting Exact does not remove images from Semantic results (no exclude_hash)
- Per-mode limit: each active mode independently caps at `limit` hits
- Altered mode produces independent rows from Semantic for the same path
- _query_vector no longer accepts exclude_hash (signature regression guard)
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from scalar_forensic.web.pipeline import (
    Hit,
    _query_vector,
    query_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATH_A = "/evidence/photo.jpg"
PATH_B = "/evidence/other.jpg"


def _make_entry(
    file_id: str = "f1",
    filename: str = "photo.jpg",
    file_hash: str = "sha256abc",
    dino: bool = True,
    sscd: bool = False,
) -> MagicMock:
    entry = MagicMock()
    entry.file_id = file_id
    entry.filename = filename
    entry.error = None
    entry.file_hash = file_hash
    entry.file_hash_md5 = "md5abc"
    entry.dino_embedding = [0.1] * 1024 if dino else None
    entry.sscd_embedding = [0.2] * 512 if sscd else None
    return entry


def _make_session(entries: list) -> MagicMock:
    session = MagicMock()
    session.files = entries
    return session


def _mock_settings() -> MagicMock:
    s = MagicMock()
    s.qdrant_url = "http://localhost:6333"
    s.qdrant_api_key = None
    s.collection_sscd = "sfn_sscd"
    s.collection_dino = "sfn_dino"
    return s


def _exact_hit(path: str) -> Hit:
    return Hit(path=path, scores={"exact": 1.0})


def _semantic_hit(path: str, score: float = 0.95) -> Hit:
    return Hit(path=path, scores={"semantic": score})


def _altered_hit(path: str, score: float = 0.90) -> Hit:
    return Hit(path=path, scores={"altered": score})


# ---------------------------------------------------------------------------
# Signature guard — exclude_hash must not exist on _query_vector
# ---------------------------------------------------------------------------


def test_query_vector_has_no_exclude_hash_parameter():
    sig = inspect.signature(_query_vector)
    assert "exclude_hash" not in sig.parameters, (
        "_query_vector must not accept exclude_hash; self-matches are now always included"
    )


# ---------------------------------------------------------------------------
# Same path from multiple modes is merged into one unified row
# ---------------------------------------------------------------------------


def test_same_image_both_modes_yields_one_merged_row():
    """When the same path appears in exact and semantic results, scores are merged
    into a single Hit so the UI can show all scores in one row."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_exact",
            return_value=([_exact_hit(PATH_A)], []),
        ),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            return_value=([_semantic_hit(PATH_A, score=1.0)], []),
        ),
    ):
        results, _ = query_session(session, ["exact", "semantic"], 0.75, 0.55, 10, settings)

    assert len(results) == 1
    hits = results[0].hits
    assert len(hits) == 1, f"expected 1 merged row for same path in two modes, got {hits}"

    hit = hits[0]
    assert hit.path == PATH_A
    assert "exact" in hit.scores
    assert "semantic" in hit.scores
    assert hit.scores["exact"] == 1.0
    assert hit.scores["semantic"] == 1.0


# ---------------------------------------------------------------------------
# Deselecting Exact does not remove image from Semantic results
# ---------------------------------------------------------------------------


def test_semantic_only_includes_self_match():
    """With modes=['semantic'] only, the query image must still appear in results."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            return_value=([_semantic_hit(PATH_A, score=1.0)], []),
        ),
    ):
        results, _ = query_session(session, ["semantic"], 0.75, 0.55, 10, settings)

    hits = results[0].hits
    paths = [h.path for h in hits]
    assert PATH_A in paths, "query image must appear in semantic-only results"


def test_exact_deselected_does_not_drop_semantic_hit():
    """Even without Exact mode, a matching image appears via Semantic."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            return_value=([_semantic_hit(PATH_A, score=0.95)], []),
        ),
    ):
        results_no_exact, _ = query_session(session, ["semantic"], 0.75, 0.55, 10, settings)

    hits = results_no_exact[0].hits
    assert any(h.path == PATH_A for h in hits)
    assert all("exact" not in h.scores for h in hits)


# ---------------------------------------------------------------------------
# Per-mode limit: each mode caps at `limit` independently
# ---------------------------------------------------------------------------


def test_per_mode_limit_respected():
    """Each mode returns at most `limit` hits; combined total can exceed `limit`."""
    limit = 3
    session = _make_session([_make_entry(dino=True, sscd=True)])
    settings = _mock_settings()

    many_semantic = [_semantic_hit(f"/img/s{i}.jpg", score=0.9 - i * 0.01) for i in range(10)]
    many_altered = [_altered_hit(f"/img/a{i}.jpg", score=0.85 - i * 0.01) for i in range(10)]

    def _fake_query_vector(client, collection, vector, mode, threshold, limit):
        if mode == "semantic":
            return (many_semantic[:limit], [])
        return (many_altered[:limit], [])

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_exact",
            return_value=([_exact_hit("/img/exact.jpg")], []),
        ),
        patch("scalar_forensic.web.pipeline._query_vector", side_effect=_fake_query_vector),
    ):
        results, _ = query_session(
            session, ["exact", "altered", "semantic"], 0.75, 0.55, limit, settings
        )

    hits = results[0].hits
    semantic_hits = [h for h in hits if "semantic" in h.scores]
    altered_hits = [h for h in hits if "altered" in h.scores]
    exact_hits = [h for h in hits if "exact" in h.scores]

    assert len(semantic_hits) <= limit
    assert len(altered_hits) <= limit
    assert len(exact_hits) <= limit
    # Combined total can be up to 3 × limit
    assert len(hits) <= 3 * limit


# ---------------------------------------------------------------------------
# Altered + Semantic for same path merges into one row with both scores
# ---------------------------------------------------------------------------


def test_altered_and_semantic_same_path_yields_one_merged_row():
    """When altered and semantic both match the same path, they are merged into
    a single Hit with both scores so the UI can display a unified row."""
    session = _make_session([_make_entry(dino=True, sscd=True)])
    settings = _mock_settings()

    def _fake_query_vector(client, collection, vector, mode, threshold, limit):
        return ([Hit(path=PATH_A, scores={mode: 0.92})], [])

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch("scalar_forensic.web.pipeline._query_vector", side_effect=_fake_query_vector),
    ):
        results, _ = query_session(session, ["altered", "semantic"], 0.75, 0.55, 10, settings)

    hits = results[0].hits
    assert len(hits) == 1, f"expected 1 merged row, got {hits}"
    hit = hits[0]
    assert "altered" in hit.scores
    assert "semantic" in hit.scores
    assert hit.scores["altered"] == 0.92
    assert hit.scores["semantic"] == 0.92


# ---------------------------------------------------------------------------
# Per-mode dedup keeps highest-scoring hit, not first occurrence
# ---------------------------------------------------------------------------


def test_per_mode_dedup_keeps_highest_score():
    """If Qdrant returns the same path twice within one mode, keep the best score."""
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    # Qdrant returns PATH_A twice: lower score first, higher score second
    dup_hits = [
        Hit(path=PATH_A, scores={"semantic": 0.70}),
        Hit(path=PATH_A, scores={"semantic": 0.95}),
        Hit(path=PATH_B, scores={"semantic": 0.80}),
    ]

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch("scalar_forensic.web.pipeline._query_vector", return_value=(dup_hits, [])),
    ):
        results, _ = query_session(session, ["semantic"], 0.75, 0.55, 10, settings)

    hits = results[0].hits
    path_a_hits = [h for h in hits if h.path == PATH_A]
    assert len(path_a_hits) == 1, "duplicate path within same mode must be deduped to one row"
    assert path_a_hits[0].scores["semantic"] == 0.95, "must keep highest-scoring hit, not first"


# ---------------------------------------------------------------------------
# Combined sort: Exact hits (score 1.0) rank above vector hits
# ---------------------------------------------------------------------------


def test_exact_hits_rank_first_in_combined_sort():
    session = _make_session([_make_entry(dino=True)])
    settings = _mock_settings()

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_exact",
            return_value=([_exact_hit(PATH_A)], []),
        ),
        patch(
            "scalar_forensic.web.pipeline._query_vector",
            return_value=([_semantic_hit(PATH_B, score=0.80)], []),
        ),
    ):
        results, _ = query_session(session, ["exact", "semantic"], 0.75, 0.55, 10, settings)

    hits = results[0].hits
    assert hits[0].scores == {"exact": 1.0}, "exact hit must sort first"


# ---------------------------------------------------------------------------
# unify=False: per-mode independent rows
# ---------------------------------------------------------------------------


def _altered_hit(path: str, score: float = 0.90) -> Hit:
    return Hit(path=path, scores={"altered": score})


def test_unify_false_same_path_yields_separate_rows():
    """With unify=False the same path from altered and semantic appears as two rows."""
    session = _make_session([_make_entry(dino=True, sscd=True)])
    settings = _mock_settings()

    def _fake_qv(client, collection, vector, mode, threshold, limit):
        return ([Hit(path=PATH_A, scores={mode: 0.92})], [])

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch("scalar_forensic.web.pipeline._query_vector", side_effect=_fake_qv),
    ):
        results, _ = query_session(
            session, ["altered", "semantic"], 0.75, 0.55, 10, settings, unify=False
        )

    hits = results[0].hits
    assert len(hits) == 2, "unify=False must keep one row per mode even for the same path"
    modes = {next(iter(h.scores)) for h in hits}
    assert modes == {"altered", "semantic"}


def test_unify_false_sort_order_exact_altered_semantic():
    """With unify=False rows sort exact → altered → semantic."""
    session = _make_session([_make_entry(dino=True, sscd=True)])
    settings = _mock_settings()

    def _fake_qv(client, collection, vector, mode, threshold, limit):
        return ([Hit(path=PATH_A, scores={mode: 0.92})], [])

    with (
        patch("scalar_forensic.web.pipeline.QdrantClient"),
        patch(
            "scalar_forensic.web.pipeline._query_exact",
            return_value=([_exact_hit(PATH_B)], []),
        ),
        patch("scalar_forensic.web.pipeline._query_vector", side_effect=_fake_qv),
    ):
        results, _ = query_session(
            session, ["exact", "altered", "semantic"], 0.75, 0.55, 10, settings, unify=False
        )

    hits = results[0].hits
    row_modes = [next(iter(h.scores)) for h in hits]
    assert row_modes == ["exact", "altered", "semantic"], (
        f"expected exact→altered→semantic ordering, got {row_modes}"
    )


def test_merge_hit_max_score_never_downgrades():
    """_merge_hit must keep the higher score when the same mode appears twice."""
    from scalar_forensic.web.pipeline import _merge_hit

    high = Hit(path=PATH_A, scores={"semantic": 0.95})
    low = Hit(path=PATH_A, scores={"semantic": 0.70})

    dest: dict = {}
    _merge_hit(high, dest)
    _merge_hit(low, dest)  # lower score — must NOT overwrite 0.95

    assert dest[PATH_A].scores["semantic"] == 0.95, (
        "later lower score must not overwrite earlier higher score"
    )


def test_merge_hit_provenance_setdefault():
    """_merge_hit must keep the first-seen provenance for a mode, not replace it."""
    from scalar_forensic.web.pipeline import _merge_hit

    first = Hit(
        path=PATH_A,
        scores={"altered": 0.9},
        model_provenance={"altered": {"name": "v1", "hash": "aaa"}},
    )
    second = Hit(
        path=PATH_A,
        scores={"altered": 0.8},
        model_provenance={"altered": {"name": "v2", "hash": "bbb"}},
    )

    dest: dict = {}
    _merge_hit(first, dest)
    _merge_hit(second, dest)

    assert dest[PATH_A].model_provenance["altered"]["hash"] == "aaa", (
        "first-seen provenance must be preserved on repeated merge"
    )
