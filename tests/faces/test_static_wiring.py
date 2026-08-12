from pathlib import Path

STATIC = Path("src/scalar_forensic/web/static")


def test_faces_part_is_loaded_before_app_js():
    html = (STATIC / "index.html").read_text()
    assert html.index("/static/js/faces.js") < html.index("/static/app.js")


def test_faces_part_registers_via_sfn_parts_not_object_assign():
    js = (STATIC / "js" / "faces.js").read_text()
    assert "__sfnParts" in js
    assert "Object.assign" not in js  # would evaluate getters instead of copying


def test_face_grid_uses_the_review_hash_domain():
    # chip_hash no longer exists in the payload, and the aligned hash names no
    # review artefact: binding either one renders a broken image for every face.
    html = (STATIC / "index.html").read_text()
    assert "face.chip_hash" not in html
    assert "faceThumbUrl(face.review_chip_hash)" in html
    assert "faceReviewUrl(face.review_chip_hash)" in html


def test_face_grid_labels_review_only_observations():
    html = (STATIC / "index.html").read_text()
    assert "face-chip-review-only" in html
    assert "faceStatusLabel(face)" in html


def test_faces_panel_loads_for_the_already_selected_hit():
    # $watch fires only on *change*.  selectedHit is already set when the panel
    # mounts, so a watcher alone never issues the by-image request and the panel
    # reads "0 in this image" for every hit — with a single hit, the face
    # browser is unreachable.  The panel must also load once at init.
    html = (STATIC / "index.html").read_text()
    panel = html[html.index('<div class="faces-panel"') :].split(">")[0]
    assert "$watch('selectedHit'" in panel
    watch_at = panel.index("$watch('selectedHit'")
    assert "loadFacesForHit(selectedHit?.image_hash)" in panel[:watch_at]


def test_review_only_chips_are_not_upscaled_or_cropped():
    # The blanket .face-chip img rule upscales and centre-crops; on a 40 px
    # crop that fabricates display pixels and hides part of the evidence,
    # contradicting the spec's native-resolution claim.
    css = (STATIC / "style.css").read_text()
    block = css[css.index(".face-chip-review-only img") :].split("}")[0]
    assert "object-fit: contain" in block
    assert "max-width: 72px" in block and "width: auto" in block


def test_query_face_strip_selects_only_searchable_faces():
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="query-faces"')
    block = html[start : start + 2500]
    assert "toggleQueryFace(" in block
    assert "face.searchable" in block  # review-only chips are not selectable
    assert "query-face-chip-selected" in block


def test_selected_query_faces_get_the_green_border():
    css = (STATIC / "style.css").read_text()
    start = css.index(".query-face-chip-selected")
    block = css[start : start + 200]
    assert "var(--success)" in block
    assert "border" in block


def test_query_face_functions_live_in_the_faces_part_file():
    js = (STATIC / "js" / "faces.js").read_text()
    for fn in ("loadQueryFaces", "toggleQueryFace", "queryFaceChipUrl"):
        assert fn in js
    assert "Object.assign" not in js
