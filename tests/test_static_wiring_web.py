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
