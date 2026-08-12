from pathlib import Path

STATIC = Path("src/scalar_forensic/web/static")


def test_faces_part_is_loaded_before_app_js():
    html = (STATIC / "index.html").read_text()
    assert html.index("/static/js/faces.js") < html.index("/static/app.js")


def test_faces_part_registers_via_sfn_parts_not_object_assign():
    js = (STATIC / "js" / "faces.js").read_text()
    assert "__sfnParts" in js
    assert "Object.assign" not in js  # would evaluate getters instead of copying
