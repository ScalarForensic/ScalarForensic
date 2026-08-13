import re
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


def test_hit_list_has_a_faces_badge_slot():
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="hit-scores"')
    block = html[start : start + 3000]
    assert "badge-faces" in block
    assert "'faces' in hit.scores" in block
    assert "hit.scores.faces.toFixed(" in block


def test_faces_filter_pill_exists_and_is_gated_on_availability():
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="hits-filters"')
    block = html[start : start + 2000]
    assert "hitsFilterFaces" in block
    assert "facesAvailable" in block


def test_matched_face_gets_a_green_border_and_a_score_beneath_it():
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="faces-grid"')
    block = html[start : start + 2000]
    assert "face-chip-matched" in block
    assert "faceMatchScore(face)" in block
    css = (STATIC / "style.css").read_text()
    matched = css[css.index(".face-chip-matched") :][:200]
    assert "var(--success)" in matched


def test_uncalibrated_banner_is_shown_with_face_results():
    html = (STATIC / "index.html").read_text()
    assert "faceCalibration?.banner" in html
    assert "faceCalibration?.model_reference_note" in html


def test_merged_hits_getter_is_in_computed_not_elsewhere():
    computed = (STATIC / "js" / "computed.js").read_text()
    assert "get mergedHits()" in computed
    assert "hitsFilterFaces" in computed
    for other in ("faces.js", "state.js", "helpers.js", "analysis.js"):
        assert "get mergedHits()" not in (STATIC / "js" / other).read_text()


def test_face_query_controls_sit_with_the_other_sliders():
    html = (STATIC / "index.html").read_text()
    # The whole .sliders block, up to the section that follows it.  Slicing to
    # the first </div> after "faceThreshold" instead would make the assertions
    # depend on the order of the rows, which is a layout decision, not a
    # requirement — and it traps whoever adds the next slider.
    start = html.index('<div class="sliders">')
    block = html[start : html.index('<div class="panel-header section-header">', start)]
    assert "faceLimit" in block
    assert "faceThreshold" in block
    assert "faceExactSearch" in block
    assert "0.363" not in block  # never a default in the UI controls
    assert "uncalibrated" in block.lower()


def test_header_legend_shows_the_face_modality():
    # Change-set 2026-08-13 item 1: the post-analysis header legend showed only
    # EXACT/ALTERED/SEMANTIC; the face modality must appear there too, gated on
    # availability like every other face surface.
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="analysis-pills"')
    block = html[start : html.index("</header>", start)]
    assert "analysis-pill-faces" in block
    assert "facesAvailable" in block
    css = (STATIC / "style.css").read_text()
    pill = css[css.index(".analysis-pill-faces") :].split("}")[0]
    assert "--success" in pill  # same green as badge-faces: one colour per modality


def test_query_controls_are_sectioned_per_search_function():
    # Change-set 2026-08-13 item 2: the flat slider list is divided into one
    # section per search function.  Sections gate their rows, so a control can
    # appear at most once and under its own function's heading.
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="sliders">')
    block = html[start : html.index('<div class="panel-header section-header">', start)]
    labels = re.findall(r'class="slider-section-label[^"]*">([^<]+)', block)
    assert labels == ["General", "SSCD", "DINOv2", "Face"]
    # Every face control sits under the Face heading, which is the last one.
    face_at = block.rindex('class="slider-section-label')
    for control in ("faceLimit", "faceThreshold", "faceExactSearch"):
        assert control not in block[:face_at]
        assert control in block[face_at:]


def test_basket_box_sits_in_the_left_panel_with_three_state_rows():
    # Change-set 2026-08-13 item 3b: scrollable selection basket below the
    # MATCHED / NOT MATCHED lists; checkmark toggles selected/unselected,
    # ctrl+click removes the row, Clear empties the basket.
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="face-basket"')
    assert start > html.index("Not Matched")
    block = html[start : start + 3000]
    assert "basketToggleRow(" in block
    assert "basketRemoveRow(" in block
    assert "row.selected" in block
    head = html[html.index("Face Selection") : start]
    assert "basketClear()" in head
    css = (STATIC / "style.css").read_text()
    basket = css[css.index(".face-basket {") :].split("}")[0]
    assert "overflow-y: auto" in basket  # scrollable


def test_ctrl_click_adds_faces_to_the_basket_on_both_sides():
    html = (STATIC / "index.html").read_text()
    qstart = html.index('<div class="query-faces"')
    qblock = html[qstart : qstart + 3000]
    assert "ctrlKey" in qblock and "toggleQueryFace(" in qblock
    mstart = html.index('<div class="faces-grid"')
    mblock = html[mstart : mstart + 3000]
    assert "ctrlKey" in mblock and "toggleHitFace(" in mblock


def test_plain_click_opens_the_hq_crop_on_both_sides():
    # Item 3c: uniform behaviour — the match side already links to the review
    # crop; the query side must be an equivalent link, not a selection toggle.
    html = (STATIC / "index.html").read_text()
    qstart = html.index('<div class="query-faces"')
    qblock = html[qstart : qstart + 3000]
    assert ':href="queryFaceChipUrl(face.index)"' in qblock
    assert 'target="_blank"' in qblock


def test_cross_highlight_is_wired_on_both_sides_with_an_operator_threshold():
    # Item 3d: pairs above the operator-set floor get the marker on both
    # sides.  The floor defaults to 0.0 elsewhere (state.js); here we hold the
    # marker classes and the slider in place — and 0.363 must not appear.
    html = (STATIC / "index.html").read_text()
    qstart = html.index('<div class="query-faces"')
    assert "face-cross-matched" in html[qstart : qstart + 3000]
    mstart = html.index('<div class="faces-grid"')
    assert "face-cross-matched" in html[mstart : mstart + 3000]
    sliders = html[
        html.index('<div class="sliders">') : html.index(
            '<div class="panel-header section-header">'
        )
    ]
    assert "faceCrossThreshold" in sliders
    assert "0.363" not in sliders
    css = (STATIC / "style.css").read_text()
    marker = css[css.index(".face-cross-matched") :].split("}")[0]
    assert "--accent" in marker


def test_compare_runs_for_the_already_selected_hit_and_on_change():
    # Same trap as loadFacesForHit: $watch alone never fires for the hit that
    # is already selected when the panel mounts.
    html = (STATIC / "index.html").read_text()
    init = re.search(r'<div class="faces-panel".*?x-init="([^"]*)"', html, re.S).group(1)
    assert init.count("runFaceCompare(") == 2
    assert "$watch('selectedHit'" in init


def test_selection_getters_live_in_computed_and_not_in_state():
    computed = (STATIC / "js" / "computed.js").read_text()
    for getter in (
        "get selectedQueryFaceIndices()",
        "get selectedFacePointIds()",
        "get faceCrossHighlight()",
    ):
        assert getter in computed
    state = (STATIC / "js" / "state.js").read_text()
    assert "selectedQueryFaceIndices" not in state  # derived, not stored
    assert "faceBasket" in state
    assert "faceCrossThreshold: 0.0" in state  # no manufactured default


def test_basket_search_sends_both_probe_origins():
    js = (STATIC / "js" / "faces.js").read_text()
    assert "face_indices" in js
    assert "point_ids" in js
    for fn in ("toggleHitFace", "basketToggleRow", "basketRemoveRow", "basketClear"):
        assert fn in js
    assert "Object.assign" not in js


def test_review_only_faces_never_enter_the_basket():
    # Vectorless faces cannot be probes; both add-paths must refuse them.
    js = (STATIC / "js" / "faces.js").read_text()
    toggle_q = js[js.index("toggleQueryFace(") :].split("},")[0]
    assert "searchable" in toggle_q
    toggle_h = js[js.index("toggleHitFace(") :].split("},")[0]
    assert "faceIsReviewOnly" in toggle_h


def test_face_chips_render_native_resolution_on_both_sides():
    # Item 3e: both boxes share the same chip geometry.  contain + auto keeps
    # the whole crop visible at its true aspect on the match side too.
    css = (STATIC / "style.css").read_text()
    block = css[css.index(".face-chip img") :].split("}")[0]
    assert "object-fit: contain" in block
    assert "max-width: 72px" in block and "width: auto" in block


def test_faces_only_view_orders_hits_by_their_face_score():
    computed = (STATIC / "js" / "computed.js").read_text()
    filtered = computed[computed.index("get filteredHits()") :].split("},")[0]
    assert "scores.faces" in filtered


def test_query_image_buttons_name_their_model():
    html = (STATIC / "index.html").read_text()
    assert "DINO Dist Stats" in html
    assert "DINO Audit" in html
    assert "FACE Dist Stats" in html
    assert "FACE Audit" in html


def test_face_dist_stats_modal_scopes_the_reference_threshold():
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="face-stats-modal-backdrop"')
    block = html[start : start + 4000]
    assert "model_reference_note" in block
    assert "review-only" in block  # the population statement


def test_server_error_detail_is_flattened_before_display():
    # c4 caught a 422 whose `detail` is FastAPI's array of validation objects:
    # assigned straight to the error field it renders as "[object Object]", so
    # the examiner is told nothing about why the search failed.  Every error
    # surface must go through the flattener, not `body.detail || '...'`.
    js = (STATIC / "js" / "faces.js").read_text()
    assert "faceErrText(" in js
    assert "body.detail ||" not in js
    for surface in (
        "queryFacesError",
        "faceSearchError",
        "faceCompareError",
        "faceStatsError",
        "faceAuditError",
    ):
        # Each surface is assigned more than once (reset to '' on entry, then
        # the server's message); only the ones carrying `detail` are in scope.
        assigned = [a for a in re.findall(rf"this\.{surface} = ([^;]+);", js) if "detail" in a]
        assert assigned, surface
        for a in assigned:
            assert "faceErrText(body.detail" in a, f"{surface}: {a}"
