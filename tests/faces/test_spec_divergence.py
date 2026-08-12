"""The recorded divergence from the spec's calibration gate.

Split out of test_static_wiring.py deliberately: this asserts on the *docs*, not
on the shipped static assets, and the two files are edited by different hands.
"""

from pathlib import Path


def test_spec_records_the_uncalibrated_search_divergence():
    spec = Path("docs/specs/face-pipeline.md").read_text()
    s10 = spec[spec.index("## 10. Calibration") : spec.index("## 11.")]
    assert "DEPLOYMENT DIVERGENCE" in s10
    assert "2026-08-12" in s10
    assert "the record wins" in s10
