"""Static wiring checks for the shipped web page and the sfn() Alpine parts.

These read the statics the way the browser does.  They cannot execute them, so
they pin the *call sites* and the *markup shapes* that live defects were traced
back to — enough to stop the same mistake from being wired in again.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

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


def test_selecting_a_matched_frame_shows_that_frames_own_stored_jpeg():
    # MatchedVideoFrame.path carries each frame's own image_path, so the pane
    # shows the frame that was actually scored — not the representative's file
    # and not a thumbnail.  A missing path shows the placeholder: an upscaled
    # thumb would look like the frame without being it.
    js = (STATIC / "js" / "analysis.js").read_text()
    body = _fn_body(js, "async selectMatchedFrame")
    assert "mf.path" in body
    assert "/api/hit-image?path=" in body
    assert "/api/thumbnail/" not in body
    assert "vector-fallback.svg" in body


def test_every_big_pane_source_is_a_full_resolution_endpoint():
    # The thumbnail endpoint belongs in the hit *list*, never in matchSrc: the
    # big pane is where the examiner actually compares two images.
    js = (STATIC / "js" / "analysis.js").read_text()
    for assignment in re.findall(r"this\.matchSrc\s*=\s*([^;]+);", js, re.S):
        assert "/api/thumbnail/" not in assignment, assignment


# Elements the HTML parser never sees a closing tag for.
_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


class _XForRoots(HTMLParser):
    """Count the element children of every ``<template x-for>`` in the page.

    Alpine's x-for renders ``template.content.children[0]`` and warns that
    "additional elements will be ignored" — so a second root element is markup
    that silently never reaches the DOM.  Comments and text do not count.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[dict] = []
        self.findings: list[tuple[int, list[str]]] = []

    def handle_starttag(self, tag, attrs):
        if self.stack and self.stack[-1]["depth"] == 0:
            self.stack[-1]["roots"].append(tag)
        for frame in self.stack:
            frame["depth"] += 1
        if tag == "template":
            is_for = any(a[0].startswith("x-for") for a in attrs)
            self.stack.append({"depth": 0, "roots": [], "line": self.getpos()[0], "for": is_for})
        elif tag in _VOID:
            for frame in self.stack:
                frame["depth"] -= 1

    def handle_startendtag(self, tag, attrs):
        if self.stack and self.stack[-1]["depth"] == 0:
            self.stack[-1]["roots"].append(tag)

    def handle_endtag(self, tag):
        if tag == "template" and self.stack:
            frame = self.stack.pop()
            if frame["for"] and len(frame["roots"]) != 1:
                self.findings.append((frame["line"], frame["roots"]))
        for frame in self.stack:
            frame["depth"] -= 1


def _xfor_multi_roots(markup: str) -> list[tuple[int, list[str]]]:
    parser = _XForRoots()
    parser.feed(markup)
    return parser.findings


def test_the_checker_itself_catches_a_second_root():
    # Guards the guard: a checker that silently passes everything is worse than
    # no checker, and this one is hand-rolled around HTMLParser.
    bad = '<template x-for="x in xs"><span>a</span><span>b</span></template>'
    assert _xfor_multi_roots(bad) == [(1, ["span", "span"])]
    good = '<template x-for="x in xs"><span><b>a</b><i>b</i></span></template>'
    assert _xfor_multi_roots(good) == []
    voids = '<template x-for="x in xs"><div><img src="a"><br></div></template>'
    assert _xfor_multi_roots(voids) == []


def test_every_x_for_template_has_exactly_one_root_element():
    # Operator console showed the Alpine warning twice on 2026-08-13; the cause
    # was a matched-frame score template whose badge and value pill were two
    # roots, so every frame rendered the mode and dropped the number.
    html = (STATIC / "index.html").read_text()
    assert _xfor_multi_roots(html) == []


# ---------------------------------------------------------------------------
# Source-video playback (viewing copy)
# ---------------------------------------------------------------------------


def test_the_player_is_labelled_as_a_viewing_copy():
    # Forensic legibility is a merge requirement: the operator must never be
    # able to mistake a rewrapped stream for the evidential artifact.  Same
    # rule the uncalibrated face cosine is displayed under.
    html = (STATIC / "index.html").read_text()
    assert "VIEWING COPY" in html
    assert "streams unmodified" in html
    assert "authoritative artifacts" in html


def test_the_player_source_is_the_playback_endpoint_and_seeks_to_the_hit():
    html = (STATIC / "index.html").read_text()
    assert ':src="videoPlaybackSrc"' in html
    assert '@loadedmetadata="seekVideoToHit($el)"' in html


def test_playback_getters_live_in_computed_js():
    # Part files are merged by property descriptor; a getter defined in a
    # non-computed part still works, but the convention keeps them findable.
    computed = (STATIC / "js" / "computed.js").read_text()
    for name in (
        "get videoPlaybackSrc",
        "get activeHitTimecodeMs",
        "get videoPlaybackDroppedNotice",
        "get videoPlaybackProvenance",
        "get videoPlaybackDownloadUrl",
    ):
        assert name in computed, name
    for part in ("state.js", "evidence.js", "analysis.js", "triage.js"):
        assert "get videoPlayback" not in (STATIC / "js" / part).read_text(), part


def test_opening_the_player_asks_what_would_be_served_first():
    # The label has to be on screen with the first frame, so the info request
    # precedes the bytes rather than following them.
    js = (STATIC / "js" / "evidence.js").read_text()
    body = _fn_body(js, "async openVideoPlayback")
    assert "/api/video-playback-info?path=" in body


def test_the_indexed_hash_is_sent_so_the_server_can_judge_staleness():
    # Only the server can hash the file as it is now; the client's job is to
    # hand over the video_hash the index recorded and render the verdict.
    js = (STATIC / "js" / "evidence.js").read_text()
    body = _fn_body(js, "async openVideoPlayback")
    assert "video_hash=" in body


def test_unknown_provenance_is_never_rendered_as_a_mismatch():
    # The bug this pins: an absent indexed hash used to render the tooltip
    # "does NOT match the video_hash recorded for this frame" — unknown stated
    # as a finding, in an evidence viewer.  Three states, decided server-side.
    computed = (STATIC / "js" / "computed.js").read_text()
    body = _fn_body(computed, "get videoPlaybackProvenance")
    assert "stale_evidence" in body
    assert "'unchecked'" in body
    assert "videoPlaybackDigestMatchesHit" not in computed
    html = (STATIC / "index.html").read_text()
    assert "videoPlaybackDigestMatchesHit" not in html


def test_a_stale_file_is_called_out_prominently_not_in_a_tooltip():
    html = (STATIC / "index.html").read_text()
    assert "videoPlaybackProvenance === 'stale'" in html
    assert "videoPlaybackStaleReason" in html


def test_the_player_offers_the_download_escape_route():
    html = (STATIC / "index.html").read_text()
    assert "videoPlaybackDownloadUrl" in html
    computed = (STATIC / "js" / "computed.js").read_text()
    assert "/api/video-download?path=" in _fn_body(computed, "get videoPlaybackDownloadUrl")


def test_selecting_another_hit_closes_the_open_player():
    js = (STATIC / "js" / "analysis.js").read_text()
    assert "closeVideoPlayback()" in _fn_body(js, "async selectHit")


def test_query_face_chips_use_the_server_stamped_url():
    # The regression that actually bit (2026-08-13): queryFaceChipUrl rebuilt
    # the URL from the *current* selectedFileId, so a response that resolved
    # late aimed the earlier file's face indices at the file now selected —
    # 404 when it had fewer faces, another person's crop when it had more.
    # The server stamps chip_url with the file and the detection generation;
    # the client's only correct move is to use it verbatim.
    js = (STATIC / "js" / "faces.js").read_text()
    body = _fn_body(js, "queryFaceChipUrl")
    assert "chip_url" in body
    assert "selectedFileId" not in body
    assert "query-chip" not in body  # no hand-built URL anywhere in it


def test_a_superseded_query_faces_response_is_dropped():
    # Two files in one session means two POSTs in flight and the first can
    # resolve last.  loadQueryFaces pins the file and a request ordinal, and
    # applies nothing — faces, error, basket or follow-up searches — once a
    # newer request or a different selection has superseded it.
    js = (STATIC / "js" / "faces.js").read_text()
    body = _fn_body(js, "async loadQueryFaces")
    assert "_queryFacesSeq" in body
    # The response is gated before it is applied, not after.
    assert body.index("if (!current()) return;") < body.index("this.queryFaces = Array")
    assert "_queryFacesSeq" in (STATIC / "js" / "state.js").read_text()
