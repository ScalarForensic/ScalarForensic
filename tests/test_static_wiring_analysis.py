"""Static wiring checks for the analysis part of the sfn() Alpine component.

These read the shipped JS the way the browser does.  They cannot execute it, so
they pin the *call sites* that a live defect was traced back to — enough to stop
the same wrong endpoint from being wired in again.
"""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "scalar_forensic" / "web" / "static"


def _fn_body(js: str, name: str) -> str:
    """Return the source of one component method, up to the next method."""
    start = js.index(f"{name}(")
    rest = js[start:]
    # Methods are separated by a dedented `},` at two-space indentation.
    end = rest.index("\n    },")
    return rest[:end]


def test_selecting_a_video_frame_hit_shows_the_stored_frame_not_the_thumbnail():
    # Operator defect 2026-08-13: selecting a video hit put /api/thumbnail in
    # the big match pane — the 128x96 index thumb upscaled over a stored frame
    # that is full video resolution.  hit.path is the stored frame JPEG, and
    # /api/hit-image serves it, which is what every other selection path uses.
    js = (STATIC / "js" / "analysis.js").read_text()
    body = _fn_body(js, "async selectHit")
    assert "/api/hit-image?path=" in body
    assert "/api/thumbnail/" not in body


@pytest.mark.xfail(
    strict=True,
    reason=(
        "selectMatchedFrame still uses /api/thumbnail: MatchedVideoFrame carries "
        "only timecode_ms/frame_hash/scores, so the stored frame's path is not in "
        "the payload (pipeline/query.py:26-29, routes/analyze.py:210-218). Needs a "
        "path field on MatchedVideoFrame; delete this marker when that lands."
    ),
)
def test_every_big_pane_source_is_a_full_resolution_endpoint():
    # The thumbnail endpoint belongs in the hit *list*, never in matchSrc: the
    # big pane is where the examiner actually compares two images.
    js = (STATIC / "js" / "analysis.js").read_text()
    for assignment in re.findall(r"this\.matchSrc\s*=\s*([^;]+);", js, re.S):
        assert "/api/thumbnail/" not in assignment, assignment
