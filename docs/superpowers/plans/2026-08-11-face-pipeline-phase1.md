# Face Pipeline Phase 1 Implementation Plan

> **Execution status (2026-08-12):** Tasks 1–16 implemented on branch
> `feat/face-pipeline-phase1`, one commit per task, suite green (398 passed, 0 skipped) and
> `ruff check`/`format --check` clean. Deviations and open items:
> - Task 9 followed its **Interfaces** contract (3 chip artefacts, `write_thumbnail`, q=95,
>   `SFN_FACE_THUMB_SIZE`); the inline 2-tuple/q=92 code block beside it was a stale pre-amendment
>   draft. Task 10's `_cfg` test helper was missing four required `PipelineConfig` fields and was
>   corrected.
> - The YuNet landmark map **was** derived empirically (identity map confirmed on 10 real faces
>   from `data/sample_images`); the real-model test runs against a committed sample image, so no
>   `real_face.jpg` fixture is needed. It skips only when `models/` has no YuNet ONNX.
> - `scripts/download_models.py --yunet` uses the **git-lfs media host**, not
>   `raw.githubusercontent.com`, which serves a ~130-byte LFS pointer for this repo.
> - **Still open — the human half of the UX ground rule (Tasks 14, 16):** the face browser panel
>   and the pipeline explainer have had no hands-on maintainer testing. Both were landed as first
>   working versions, per the rule, and expect layout/interaction/copy iteration from feedback.
>   The explainer has no bbox/landmark canvas overlay yet.
> - Not exercised end-to-end: a real `--faces` index run (needs a running Qdrant and an
>   operator-supplied recognition ONNX, neither available here).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 1 of `docs/specs/face-pipeline.md` — an optional, disabled-by-default face modality that detects, quality-gates, aligns, embeds and stores face observations from indexed media, with a browse-only web UI. **No cross-file face search in this plan** (that is Phase 1b, gated on a calibration record; separate plan).

**Architecture:** Isolated package `src/scalar_forensic/faces/` (spec §4). YuNet ONNX detection at a capped input size with coordinates scaled back to oriented source pixels; two-stage quality gate; Umeyama 5-point alignment to a fixed 112×112 crop; operator-supplied ONNX embedder validated against a JSON manifest; observations stored as points in a case-scoped Qdrant sidecar collection `{SFN_COLLECTION}_faces` with deterministic IDs, processed markers, split safeguards, PNG/JPEG chip store, purge path and an append-only audit log.

**Tech Stack:** Python 3.11+, `opencv-python-headless>=4.10`, `onnxruntime>=1.18` (new PEP 735 dependency group `faces`), numpy, Pillow, qdrant-client, FastAPI, typer, pytest.

## Global Constraints

- Offline: no network I/O at runtime anywhere in `faces/`; models load from local paths only. Never depend on `insightface` or `ultralytics` (spec §3.1).
- No model weights are bundled or downloaded by the code (spec §3.2).
- Tests are hermetic: no network, no Qdrant server, no unshipped model files. Tests needing the real YuNet model use `pytest.mark.skipif` on the model path. Full suite: `uv run pytest -q`.
- Lint gate: `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts` (exactly what CI runs).
- `unittest.mock.patch` targets are per-module: patch where the name is *used* (CLAUDE.md gotcha).
- The faces package must never use `cv2` video I/O — only image ops (`FaceDetectorYN`, `warpAffine`, `Laplacian`, `cvtColor`) (spec §4.1).
- Canonical landmark order everywhere: **[left eye, right eye, nose tip, left mouth corner, right mouth corner]**, "left" = image-left (spec §6.1).
- Aligned crop size is fixed 112×112, `alignment_version = "arcface-112-v1"`; not configurable (spec §6.4).
- All new settings parse eagerly in `Settings.__init__` and raise `ValueError` with actionable messages (spec §13).
- New user-visible strings follow existing CLI/error style (see `config.py:offline_model_error` for tone).
- The code blocks in this plan are contracts, not formatting gospel: run `uv run ruff format` on every new file and take its output before the lint step — the plan's inline style (compound `;` statements in tests, packed literals) is not what ruff format emits, and CI enforces `ruff format --check`. Keep lines ≤ 100 chars (E501 is enabled).
- Commit after every green test cycle; conventional-commit messages; never push (local-only workflow).

## File Structure

```
src/scalar_forensic/faces/
  __init__.py        public API re-exports (mirror web/pipeline/__init__.py style)
  types.py           FaceDetection dataclass, LANDMARK_ORDER doc constant
  decode.py          load_for_detection() — EXIF-oriented full-res RGB decode
  detect.py          FaceDetector protocol, YuNetDetector adapter
  quality.py         pre_align_gate(), post_align_gate(), GateResult
  align.py           ARCFACE_DST, umeyama(), align_face()
  embed.py           EmbedderManifest, load_manifest(), OnnxFaceEmbedder
  provenance.py      PipelineConfig dataclass + config_hash()
  chips.py           chip_hash(), write_chips(), write_thumbnail(), chip_paths() (sharded store)
  store.py           FaceStore — collection mgmt, points, markers, meta, purge
  audit.py           AuditLog — append-only JSONL event log
  indexing.py        process_image() orchestration + FacePipeline loader
tests/faces/         one test module per source module (+ fixtures/)
tests/fixtures/faces/  golden alignment fixture (json + png), tiny ONNX embedder
src/scalar_forensic/config.py         add SFN_FACE_* (modify)
src/scalar_forensic/cli.py            add --faces flag + faces-purge command (modify)
src/scalar_forensic/web/routes/faces.py   browse endpoints (create)
src/scalar_forensic/web/app.py        register router (modify)
src/scalar_forensic/web/static/js/faces.js + index.html + app.js merge list (modify)
pyproject.toml, THIRD_PARTY_LICENSES.md, INSTALL.md, docs/face-matching-math.md,
docs/deployment.md  (modify/create)
```

---

### Task 1: Dependency group, package skeleton, detection types

**Files:**
- Modify: `pyproject.toml` (add `faces` to `[dependency-groups]`)
- Create: `src/scalar_forensic/faces/__init__.py`, `src/scalar_forensic/faces/types.py`
- Test: `tests/faces/test_types.py`, `tests/faces/__init__.py`

**Interfaces:**
- Produces: `FaceDetection(bbox: tuple[float, float, float, float], landmarks: np.ndarray  # (5,2) float32, confidence: float, detect_scale: float)`; `LANDMARK_ORDER: tuple[str, ...]`; `assert_canonical_landmarks(lm: np.ndarray) -> None`.

- [ ] **Step 1: Add the dependency group** (PEP 735 group, *not* an extra — matches existing `heif` group):

```toml
# in [dependency-groups], after heif = [...]
faces = [
    "opencv-python-headless>=4.10",
    "onnxruntime>=1.18",
]
```

**Also add `{ include-group = "faces" }` to the `dev` group** (alongside the existing
`{ include-group = "web" }`): CI runs `uv sync --dev` + `uv run pytest`, and the new test modules
import `cv2`/`onnxruntime` at collection time — without this, CI fails with collection errors,
and gating with `importorskip` would leave the alignment/detector logic untested in CI, which is
worse.

Run: `uv sync --dev` — expect a clean resolve.

- [ ] **Step 2: Write the failing test**

```python
# tests/faces/test_types.py
import numpy as np
import pytest

from scalar_forensic.faces.types import LANDMARK_ORDER, FaceDetection, assert_canonical_landmarks


def test_landmark_order_is_the_documented_contract():
    assert LANDMARK_ORDER == (
        "left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth",
    )


def test_face_detection_holds_canonical_shapes():
    lm = np.array([[10, 20], [30, 20], [20, 30], [12, 40], [28, 40]], dtype=np.float32)
    det = FaceDetection(bbox=(5.0, 10.0, 30.0, 35.0), landmarks=lm, confidence=0.93, detect_scale=0.5)
    assert det.landmarks.shape == (5, 2)
    assert det.confidence == pytest.approx(0.93)


def test_assert_canonical_rejects_swapped_eyes():
    # Right eye left of left eye ⇒ mirrored order ⇒ must raise.
    lm = np.array([[30, 20], [10, 20], [20, 30], [12, 40], [28, 40]], dtype=np.float32)
    with pytest.raises(ValueError, match="landmark order"):
        assert_canonical_landmarks(lm)


def test_assert_canonical_rejects_bad_shape():
    with pytest.raises(ValueError, match="5x2"):
        assert_canonical_landmarks(np.zeros((4, 2), dtype=np.float32))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/faces/test_types.py -q`
Expected: FAIL with `ModuleNotFoundError: scalar_forensic.faces`

- [ ] **Step 4: Implement**

```python
# src/scalar_forensic/faces/types.py
"""Shared detection types and the canonical landmark contract.

Canonical landmark order (spec §6.1): left eye, right eye, nose tip,
left mouth corner, right mouth corner — "left" meaning image-left
(the subject's right side).  Every detector adapter MUST reorder its
native output into this order; a swapped eye pair produces a mirrored
alignment that silently degrades matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LANDMARK_ORDER: tuple[str, ...] = (
    "left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth",
)


# eq=False: the default dataclass __eq__/__hash__ choke on the ndarray field
# ("truth value of an array is ambiguous"); identity comparison is all we need.
@dataclass(frozen=True, eq=False)
class FaceDetection:
    """One detected face in oriented source-pixel coordinates."""

    bbox: tuple[float, float, float, float]  # x, y, w, h
    landmarks: np.ndarray  # (5, 2) float32, canonical order
    confidence: float
    detect_scale: float  # source px * detect_scale = detector-input px


def assert_canonical_landmarks(lm: np.ndarray) -> None:
    """Sanity-check the canonical ordering invariants that survive pose.

    Checks are deliberately loose (faces rotate) but catch the two silent
    killers: swapped eye pair and swapped mouth pair.
    """
    if lm.shape != (5, 2):
        raise ValueError(f"landmarks must be 5x2, got {lm.shape}")
    if lm[0, 0] >= lm[1, 0]:
        raise ValueError("bad landmark order: left eye is not left of right eye")
    if lm[3, 0] >= lm[4, 0]:
        raise ValueError("bad landmark order: left mouth is not left of right mouth")
```

```python
# src/scalar_forensic/faces/__init__.py
"""Optional face modality (spec: docs/specs/face-pipeline.md).

Core modules never import this package at module level; availability is
probed via scalar_forensic.faces availability helpers.
"""

from scalar_forensic.faces.types import LANDMARK_ORDER, FaceDetection, assert_canonical_landmarks

__all__ = ["LANDMARK_ORDER", "FaceDetection", "assert_canonical_landmarks"]
```

Create empty `tests/faces/__init__.py`.

- [ ] **Step 5: Run tests, lint, commit**

Run: `uv run pytest tests/faces/test_types.py -q` → PASS; `uv run ruff check src tests` → clean.

```bash
git add pyproject.toml uv.lock src/scalar_forensic/faces tests/faces
git commit -m "feat(faces): package skeleton, dependency group, canonical landmark contract"
```

---

### Task 2: Settings — `SFN_FACE_*`

**Files:**
- Modify: `src/scalar_forensic/config.py` (append in `Settings.__init__` after the web-server block; reuse `_parse_bool/_parse_int/_parse_float/_parse_optional_path`)
- Test: `tests/faces/test_config.py`

**Interfaces:**
- Produces (on `Settings`): `faces_enabled: bool`, `face_detector: str`, `face_detector_model: Path | None`, `face_embedder_model: Path | None`, `face_collection: str` (derived default `f"{self.collection}_faces"`), `face_store_dir: Path | None` (default `data/faces`), `face_detect_max_size: int` (1600), `face_min_conf: float` (0.8), `face_min_size: int` (64), `face_min_sharpness: float` (25.0), `face_max_clipped: float` (0.6), `face_max_pose: float` (0.35), `face_crop_dilation: float` (0.15), `examiner_id: str | None`; method `face_startup_error() -> str | None`. **Every gate threshold lives here** — none may be a module constant, or provenance can't record it and Phase 1b's calibration record has nothing to supersede. (`SFN_FACE_TOPK_PER_GROUP` from spec §13 is deliberately absent: grouping is Phase 2.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/faces/test_config.py
import pytest

from scalar_forensic.config import Settings


def test_faces_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    s = Settings()
    assert s.faces_enabled is False
    assert s.face_collection == f"{s.collection}_faces"
    assert s.face_detect_max_size == 1600
    assert s.face_crop_dilation == pytest.approx(0.15)


def test_face_collection_follows_case_collection(monkeypatch):
    monkeypatch.setenv("SFN_COLLECTION", "case42")
    assert Settings().face_collection == "case42_faces"


def test_crop_dilation_bounds(monkeypatch):
    monkeypatch.setenv("SFN_FACE_CROP_DILATION", "0.9")
    with pytest.raises(ValueError, match="SFN_FACE_CROP_DILATION"):
        Settings()


def test_startup_error_when_enabled_without_models(monkeypatch, tmp_path):
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_EXAMINER_ID", "ex1")
    monkeypatch.delenv("SFN_FACE_DETECTOR_MODEL", raising=False)
    err = Settings().face_startup_error()
    assert err is not None and "SFN_FACE_DETECTOR_MODEL" in err


def test_startup_error_requires_examiner_id(monkeypatch, tmp_path):
    det = tmp_path / "yunet.onnx"; det.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"; emb.write_bytes(b"x")
    (tmp_path / "emb.onnx.manifest.json").write_text("{}")
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_FACE_DETECTOR_MODEL", str(det))
    monkeypatch.setenv("SFN_FACE_EMBEDDER_MODEL", str(emb))
    monkeypatch.delenv("SFN_EXAMINER_ID", raising=False)
    err = Settings().face_startup_error()
    assert err is not None and "SFN_EXAMINER_ID" in err


def test_no_startup_error_when_fully_configured(monkeypatch, tmp_path):
    det = tmp_path / "yunet.onnx"; det.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"; emb.write_bytes(b"x")
    (tmp_path / "emb.onnx.manifest.json").write_text("{}")
    monkeypatch.setenv("SFN_FACES_ENABLED", "true")
    monkeypatch.setenv("SFN_FACE_DETECTOR_MODEL", str(det))
    monkeypatch.setenv("SFN_FACE_EMBEDDER_MODEL", str(emb))
    monkeypatch.setenv("SFN_EXAMINER_ID", "ex1")
    assert Settings().face_startup_error() is None
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/faces/test_config.py -q` → FAIL (`AttributeError: faces_enabled`).

- [ ] **Step 3: Implement.** Append to `Settings.__init__` (after the web-server block):

```python
        # --- Face modality (optional; spec docs/specs/face-pipeline.md) ---
        # Disabled by default.  Enabling requires a detector model, an
        # embedder model + manifest, and SFN_EXAMINER_ID; validated by
        # face_startup_error() so entry points fail fast with guidance.
        self.faces_enabled: bool = self._parse_bool("SFN_FACES_ENABLED", default=False)
        self.face_detector: str = os.environ.get("SFN_FACE_DETECTOR", "yunet")
        if self.face_detector != "yunet":
            raise ValueError(
                f"SFN_FACE_DETECTOR={self.face_detector!r} is invalid. Supported: yunet"
            )
        self.face_detector_model: Path | None = self._parse_optional_path("SFN_FACE_DETECTOR_MODEL")
        self.face_embedder_model: Path | None = self._parse_optional_path("SFN_FACE_EMBEDDER_MODEL")
        self.face_collection: str = (
            os.environ.get("SFN_FACE_COLLECTION") or f"{self.collection}_faces"
        )
        self.face_store_dir: Path | None = self._parse_optional_path(
            "SFN_FACE_STORE_DIR", "data/faces"
        )
        self.face_detect_max_size: int = self._parse_int("SFN_FACE_DETECT_MAX_SIZE", 1600)
        if self.face_detect_max_size < 64:
            raise ValueError("SFN_FACE_DETECT_MAX_SIZE must be >= 64")
        self.face_min_conf: float = self._parse_float("SFN_FACE_MIN_CONF", 0.8)
        if not (0.0 < self.face_min_conf <= 1.0):
            raise ValueError("SFN_FACE_MIN_CONF must be in (0, 1]")
        self.face_min_size: int = self._parse_int("SFN_FACE_MIN_SIZE", 64)
        if self.face_min_size < 1:
            raise ValueError("SFN_FACE_MIN_SIZE must be >= 1")
        self.face_min_sharpness: float = self._parse_float("SFN_FACE_MIN_SHARPNESS", 25.0)
        self.face_max_clipped: float = self._parse_float("SFN_FACE_MAX_CLIPPED", 0.6)
        if not (0.0 < self.face_max_clipped <= 1.0):
            raise ValueError("SFN_FACE_MAX_CLIPPED must be in (0, 1]")
        self.face_max_pose: float = self._parse_float("SFN_FACE_MAX_POSE", 0.35)
        self.face_crop_dilation: float = self._parse_float("SFN_FACE_CROP_DILATION", 0.15)
        if not (0.0 < self.face_crop_dilation <= 0.5):
            raise ValueError("SFN_FACE_CROP_DILATION must be in (0, 0.5]")
        self.examiner_id: str | None = os.environ.get("SFN_EXAMINER_ID") or None
```

And the method (near `offline_model_error`):

```python
    def face_startup_error(self) -> str | None:
        """Actionable error if the face modality is enabled but unusable, else None.

        Checked at entry points (CLI --faces, sfn-web lifespan) so
        misconfiguration fails at startup, not at first detection.
        """
        if not self.faces_enabled:
            return None
        problems: list[str] = []
        if self.face_detector_model is None or not self.face_detector_model.exists():
            problems.append(
                f"  - SFN_FACE_DETECTOR_MODEL={str(self.face_detector_model)!r} not found.\n"
                "    Fetch the YuNet ONNX (MIT) once and point this at the local file."
            )
        if self.face_embedder_model is None or not self.face_embedder_model.exists():
            problems.append(
                f"  - SFN_FACE_EMBEDDER_MODEL={str(self.face_embedder_model)!r} not found.\n"
                "    ScalarForensic ships no recognition weights (see INSTALL.md, licensing);\n"
                "    supply an ONNX model plus its .manifest.json."
            )
        elif not Path(str(self.face_embedder_model) + ".manifest.json").exists():
            problems.append(
                f"  - Manifest not found: {self.face_embedder_model}.manifest.json\n"
                "    Every embedder model needs a manifest (see docs/specs/face-pipeline.md §6.3)."
            )
        if not self.examiner_id:
            problems.append(
                "  - SFN_EXAMINER_ID is required while faces are enabled (self-asserted\n"
                "    examiner identity, stamped on adjudications and audit-log entries)."
            )
        if not problems:
            return None
        return "Face modality is enabled (SFN_FACES_ENABLED=true) but not usable:\n" + "\n".join(
            problems
        )
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/faces/test_config.py -q` → PASS. Also `uv run pytest -q` (whole suite still green — Settings is constructed everywhere).

- [ ] **Step 5: Commit** — `git add src/scalar_forensic/config.py tests/faces/test_config.py && git commit -m "feat(faces): SFN_FACE_* settings with eager validation and startup check"`

---

### Task 3: Alignment — Umeyama to the ArcFace 112×112 template

**Files:**
- Create: `src/scalar_forensic/faces/align.py`, `scripts/gen_face_align_fixture.py`, `tests/fixtures/faces/golden_landmarks.json`, `tests/fixtures/faces/golden_aligned.png`
- Test: `tests/faces/test_align.py`

**Interfaces:**
- Produces: `ARCFACE_DST: np.ndarray  # (5,2) float32`; `ALIGNMENT_VERSION = "arcface-112-v1"`; `ALIGNED_SIZE = 112`; `umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray  # 2x3 float64`; `align_face(img_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray  # (112,112,3) uint8 RGB`.
- Consumes: canonical landmarks from Task 1.

- [ ] **Step 1: Write the failing tests**

```python
# tests/faces/test_align.py
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scalar_forensic.faces.align import (
    ALIGNED_SIZE, ALIGNMENT_VERSION, ARCFACE_DST, align_face, umeyama,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "faces"


def test_reference_points_match_insightface_arcface_dst():
    expected = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    np.testing.assert_allclose(ARCFACE_DST, expected, atol=1e-4)
    assert ALIGNMENT_VERSION == "arcface-112-v1"
    assert ALIGNED_SIZE == 112


def test_umeyama_recovers_a_known_similarity_transform():
    rng = np.random.default_rng(7)
    src = rng.uniform(0, 100, size=(5, 2))
    theta, scale, tx, ty = 0.3, 1.7, 12.0, -5.0
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = scale * src @ rot.T + [tx, ty]
    m = umeyama(src, dst)  # float64 in — float32 downcast costs ~6e-6 and fails tighter tols
    src_h = np.hstack([src, np.ones((5, 1))])
    np.testing.assert_allclose(src_h @ m.T, dst, atol=1e-4)


def test_umeyama_never_reflects():
    # Mirrored destination must still produce det(R) > 0 (reflection corrected).
    src = ARCFACE_DST.astype(np.float64)
    dst = src.copy(); dst[:, 0] = 112 - dst[:, 0]
    m = umeyama(src, dst)
    assert np.linalg.det(m[:, :2]) > 0


def test_identity_landmarks_produce_identity_crop():
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255
    out = align_face(img, ARCFACE_DST.copy())
    assert out.shape == (112, 112, 3)
    diff = np.abs(out.astype(int) - img.astype(int))
    assert diff.max() <= 1  # warp with ~identity matrix; allow 1 ULP of interpolation


def test_umeyama_matrix_matches_independent_reference():
    # Reference derived via a genuinely different formulation: the optimal
    # non-reflective 2-D similarity is w = a*z + b over complex numbers,
    # solved with plain lstsq (no SVD, no Umeyama).  Both minimise the same
    # objective, so the optima coincide; literals computed once, offline.
    src = np.array([[260.0, 210.0], [340.0, 205.0], [300.0, 260.0],
                    [270.0, 310.0], [335.0, 305.0]])
    m = umeyama(src, ARCFACE_DST.astype(np.float64))
    reference = np.array([
        [0.4187296417, -0.0148068730, -66.1912889251],
        [0.0148068730, 0.4187296417, -40.5883363192],
    ])
    np.testing.assert_allclose(m, reference, atol=1e-6)


def test_golden_fixture():
    meta = json.loads((FIXTURES / "golden_landmarks.json").read_text())
    src_img = np.array(Image.open(FIXTURES / meta["source_png"]))
    lm = np.array(meta["landmarks"], dtype=np.float32)
    expected = np.array(Image.open(FIXTURES / "golden_aligned.png"))
    out = align_face(src_img, lm)
    assert np.abs(out.astype(int) - expected.astype(int)).max() <= 1
```

Note on the golden fixture's limits: it is generated by the code under test, so it is a
**regression** fixture only — the matrix test above (independent reference values) is what
catches an initially wrong transform. Both are required.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/faces/test_align.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/scalar_forensic/faces/align.py
"""Umeyama 5-point similarity alignment to the ArcFace 112x112 template.

Implementation reference: Umeyama (1991), "Least-squares estimation of
transformation parameters between two point patterns" — NOT the research
corpus (its equations were stripped in export; spec §6.4).  Reference
points verified against insightface.utils.face_align.arcface_dst.
warpAffine parameters are pinned: bilinear, BORDER_CONSTANT black.
"""

from __future__ import annotations

import cv2
import numpy as np

ALIGNMENT_VERSION = "arcface-112-v1"
ALIGNED_SIZE = 112

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation+scale+translation), as 2x3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1  # reflection correction — a face alignment must never mirror
    rot = u @ s @ vt
    var_src = (src_c**2).sum() / n
    scale = float(np.trace(np.diag(d) @ s) / var_src)
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = mu_dst - scale * rot @ mu_src
    return m


def align_face(img_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp *img_rgb* so *landmarks* (canonical order) land on ARCFACE_DST."""
    m = umeyama(landmarks, ARCFACE_DST)
    return cv2.warpAffine(
        img_rgb, m, (ALIGNED_SIZE, ALIGNED_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
```

- [ ] **Step 4: Generate the golden fixture once, offline**

```python
# scripts/gen_face_align_fixture.py
"""One-shot generator for the alignment golden fixture (run once, commit output).

Uses a deterministic synthetic 'face': gradient background + asymmetric
markers, warped from hand-picked plausible landmark positions.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # noqa: E402 — scripts bootstrap
from scalar_forensic.faces.align import align_face  # noqa: E402

out = Path("tests/fixtures/faces")
out.mkdir(parents=True, exist_ok=True)
# Smooth gradient base, not per-pixel noise: bilinear resampling of
# high-frequency noise is the worst case for the ≤1-uint8 cross-build tolerance.
yy, xx = np.mgrid[0:480, 0:640]
img = np.stack([(xx * 255 // 640), (yy * 255 // 480), ((xx + yy) * 255 // 1120)], -1).astype(np.uint8)
img[100:110, 200:210] = [255, 0, 0]  # asymmetric hard-edged marker: catches mirroring
lm = [[260.0, 210.0], [340.0, 205.0], [300.0, 260.0], [270.0, 310.0], [335.0, 305.0]]
Image.fromarray(img).save(out / "golden_source.png")
aligned = align_face(img, np.array(lm, dtype=np.float32))
Image.fromarray(aligned).save(out / "golden_aligned.png")
(out / "golden_landmarks.json").write_text(
    json.dumps({"source_png": "golden_source.png", "landmarks": lm}, indent=2)
)
print("fixture written")
```

Run: `uv run python scripts/gen_face_align_fixture.py`

- [ ] **Step 5: Run tests, lint, commit**

Run: `uv run pytest tests/faces/test_align.py -q` → PASS.

```bash
git add src/scalar_forensic/faces/align.py scripts/gen_face_align_fixture.py tests/fixtures/faces tests/faces/test_align.py
git commit -m "feat(faces): Umeyama alignment to pinned arcface-112-v1 template with golden fixture"
```

---

### Task 4: Detection decode path

**Files:**
- Create: `src/scalar_forensic/faces/decode.py`
- Test: `tests/faces/test_decode.py`

**Interfaces:**
- Produces: `load_for_detection(data: bytes) -> np.ndarray  # oriented full-res RGB uint8 HxWx3`.
- Explicitly does **not** reuse `_open_rgb` (JPEG `draft()` downscales; spec §6.1).

- [ ] **Step 1: Write the failing tests**

```python
# tests/faces/test_decode.py
import io

import numpy as np
from PIL import Image

from scalar_forensic.faces.decode import load_for_detection


def _jpeg_bytes(img: Image.Image, exif: Image.Exif | None = None) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, exif=exif or Image.Exif())
    return buf.getvalue()


def test_full_resolution_no_draft_downscale():
    # 2000px wide: _open_rgb's draft() path would decode this at reduced scale.
    img = Image.new("RGB", (2000, 1400), (10, 20, 30))
    out = load_for_detection(_jpeg_bytes(img))
    assert out.shape == (1400, 2000, 3)
    assert out.dtype == np.uint8


def test_exif_orientation_applied():
    img = Image.new("RGB", (100, 60), (0, 0, 0))
    px = img.load()
    px[0, 0] = (255, 0, 0)  # top-left marker
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 270° CW to display upright
    out = load_for_detection(_jpeg_bytes(img, exif))
    assert out.shape[:2] == (100, 60)  # oriented: dims swap


def test_non_rgb_modes_converted():
    img = Image.new("L", (50, 50), 128)
    out = load_for_detection(_jpeg_bytes(img))
    assert out.shape == (50, 50, 3)
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/scalar_forensic/faces/decode.py
"""Full-resolution decode for face detection.

Deliberately NOT scalar_forensic.embedder._open_rgb: that path calls
PIL's JPEG draft() against the 331 px embedding cap, which destroys
small faces.  Faces are detected at full oriented resolution and the
detector applies its own input cap (spec §6.1).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps


def load_for_detection(data: bytes) -> np.ndarray:
    """Decode *data* to oriented full-resolution RGB uint8 (H, W, 3).

    Pillow's MAX_IMAGE_PIXELS decompression-bomb guard stays active here
    (deliberate: this path decodes at full resolution with no 331 px cap
    to protect it; SFN_MAX_IMAGE_PIXELS overrides for trusted ingestion,
    same as everywhere else).
    """
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit** — `git commit -m "feat(faces): full-resolution EXIF-oriented decode path for detection"` (after `git add`).

---

### Task 5: YuNet detector adapter

**Files:**
- Create: `src/scalar_forensic/faces/detect.py`
- Test: `tests/faces/test_detect.py`

**Interfaces:**
- Produces: `FaceDetector` protocol (`detect(img_rgb: np.ndarray) -> list[FaceDetection]`, properties `detector_id: str`, `model_hash: str`); `YuNetDetector(model_path: Path, max_size: int, score_threshold: float = 0.5)`; helper `_scaled_size(w: int, h: int, max_size: int) -> tuple[int, int, float]`.
- Consumes: `FaceDetection`, `assert_canonical_landmarks` (Task 1); `hash_file` from `scalar_forensic.embedder`.
- **The landmark map must be derived empirically, not assumed.** OpenCV labels YuNet's row as `[x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]` where "right eye" means the **subject's** right eye — which sits at *smaller x* in image coordinates, i.e. it may already be the canonical image-left point, making the correct map the identity `[0, 1, 2, 3, 4]`. Getting this wrong mirrors every alignment silently. At implementation time: run the real YuNet model on 2–3 real face photos, print the raw rows, and set `_YUNET_TO_CANONICAL` from observation. The synthetic-row unit test below can only pin *consistency between test and code* (it constructs rows from the same assumption), so the runtime enforcement is what actually protects us: `YuNetDetector.detect` calls `assert_canonical_landmarks()` on every emitted face — a wrong map raises on the first real frontal face. The real-model smoke test must assert the ordering invariant on actual detections, not an empty result.
- `mock.patch` cannot patch attributes on OpenCV's C-extension types — patch the module-level seam `_create_yunet` instead.

- [ ] **Step 1: Write the failing tests** (fake backend — hermetic; real-model test skips when absent)

```python
# tests/faces/test_detect.py
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scalar_forensic.faces.detect import YuNetDetector, _scaled_size

REAL_MODEL = os.environ.get("SFN_FACE_DETECTOR_MODEL", "")


def test_scaled_size_caps_long_side_and_reports_scale():
    w, h, scale = _scaled_size(3200, 2400, max_size=1600)
    assert (w, h) == (1600, 1200)
    assert scale == pytest.approx(0.5)
    # Small images are not upscaled.
    assert _scaled_size(640, 480, max_size=1600) == (640, 480, 1.0)


def _fake_backend(rows: np.ndarray):
    """Patch the _create_yunet seam (cv2's C-extension types reject setattr)."""
    stub = MagicMock()
    stub.detect.return_value = (1, rows)
    return patch("scalar_forensic.faces.detect._create_yunet", return_value=stub), stub


def test_detect_scales_coords_back_and_emits_canonical_order(tmp_path):
    model = tmp_path / "yunet.onnx"; model.write_bytes(b"weights")
    # One face at detector scale 0.5, bbox (10,20,30,40).  Landmark columns are
    # written in whatever _YUNET_TO_CANONICAL was empirically derived to expect;
    # this test pins scaling and internal consistency.  The *correctness* of the
    # map against reality is enforced by assert_canonical_landmarks inside
    # detect() plus the real-model test below — not by this synthetic row.
    # Row below assumes the identity map ([0..4]); adjust if derivation differs.
    rows = np.array([[10, 20, 30, 40, 40, 10, 60, 10, 50, 20, 42, 30, 58, 30, 0.94]], np.float32)
    ctx, stub = _fake_backend(rows)
    with ctx:
        det = YuNetDetector(model, max_size=1600)
        img = np.zeros((2400, 3200, 3), dtype=np.uint8)  # → scale 0.5
        faces = det.detect(img)
    assert len(faces) == 1
    f = faces[0]
    assert f.bbox == pytest.approx((20, 40, 60, 80))          # /0.5 back to source px
    assert f.detect_scale == pytest.approx(0.5)
    np.testing.assert_allclose(                                # canonical: le, re, nose, lm, rm
        f.landmarks,
        np.array([[80, 20], [120, 20], [100, 40], [84, 60], [116, 60]], np.float32),
    )
    stub.setInputSize.assert_called_once_with((1600, 1200))


def test_detect_drops_noncanonical_output_and_counts_it(tmp_path):
    # A row whose eye pair comes out swapped after the map is dropped (not
    # emitted, not raised — a rotated real face must not crash a run) and
    # counted, so a wholesale-wrong map is loudly visible in the stats.
    model = tmp_path / "yunet.onnx"; model.write_bytes(b"weights")
    rows = np.array([[10, 20, 30, 40, 60, 10, 40, 10, 50, 20, 42, 30, 58, 30, 0.94]], np.float32)
    ctx, _ = _fake_backend(rows)
    with ctx:
        det = YuNetDetector(model, max_size=1600)
        assert det.detect(np.zeros((100, 100, 3), np.uint8)) == []
        assert det.n_dropped_noncanonical == 1


def test_detect_converts_rgb_to_bgr_for_yunet(tmp_path):
    model = tmp_path / "yunet.onnx"; model.write_bytes(b"weights")
    ctx, stub = _fake_backend(np.empty((0, 15), np.float32))
    with ctx:
        det = YuNetDetector(model, max_size=1600)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[..., 0] = 200  # red channel in RGB
        det.detect(img)
    passed = stub.detect.call_args[0][0]
    assert passed[0, 0, 2] == 200 and passed[0, 0, 0] == 0  # red now in BGR channel 2


def test_no_faces_returns_empty(tmp_path):
    model = tmp_path / "yunet.onnx"; model.write_bytes(b"weights")
    stub = MagicMock(); stub.detect.return_value = (1, None)  # YuNet returns None for no faces
    with patch("scalar_forensic.faces.detect._create_yunet", return_value=stub):
        assert YuNetDetector(model, max_size=1600).detect(np.zeros((10, 10, 3), np.uint8)) == []


@pytest.mark.skipif(not Path(REAL_MODEL or "/nonexistent").exists(), reason="YuNet model not present")
def test_real_model_landmark_order_on_real_face():
    # Committed fixture: any permissively-licensed photo with one clear frontal
    # face (tests/fixtures/faces/real_face.jpg — implementer supplies, e.g. a
    # self-taken photo).  Asserts the ordering invariant on REAL detector output,
    # which is the only thing that can catch a wrong _YUNET_TO_CANONICAL map.
    from PIL import Image as _Image
    img = np.array(_Image.open(Path("tests/fixtures/faces/real_face.jpg")).convert("RGB"))
    det = YuNetDetector(Path(REAL_MODEL), max_size=1600)
    faces = det.detect(img)
    assert faces, "fixture must contain a detectable face"
    for f in faces:
        assert f.landmarks[0, 0] < f.landmarks[1, 0]  # left eye left of right eye
        assert f.landmarks[3, 0] < f.landmarks[4, 0]  # left mouth left of right mouth
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/scalar_forensic/faces/detect.py
"""Face detectors.  Default: YuNet via cv2.FaceDetectorYN (local ONNX, no network).

Adapter contract (spec §6.1): output in oriented source pixels, landmarks
reordered into the canonical order, RGB in / BGR handled internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from scalar_forensic.embedder import hash_file
from scalar_forensic.faces.types import FaceDetection, assert_canonical_landmarks

# Map from YuNet's native landmark column order to the canonical order.
# DERIVED EMPIRICALLY at implementation time (run the real model on real
# faces and read the raw rows — see the task notes; OpenCV's "right eye"
# is the subject's right eye, i.e. plausibly already image-left).  The
# identity map is the working hypothesis; assert_canonical_landmarks in
# detect() raises on the first real face if this is wrong.
_YUNET_TO_CANONICAL = [0, 1, 2, 3, 4]


def _create_yunet(model_path: str, score_threshold: float):
    """Seam for tests — cv2's extension types cannot be mock-patched directly."""
    return cv2.FaceDetectorYN.create(model_path, "", (0, 0), score_threshold=score_threshold)


class FaceDetector(Protocol):
    def detect(self, img_rgb: np.ndarray) -> list[FaceDetection]: ...
    @property
    def detector_id(self) -> str: ...
    @property
    def model_hash(self) -> str: ...


def _scaled_size(w: int, h: int, max_size: int) -> tuple[int, int, float]:
    long_side = max(w, h)
    if long_side <= max_size:
        return w, h, 1.0
    scale = max_size / long_side
    return round(w * scale), round(h * scale), scale


class YuNetDetector:
    def __init__(self, model_path: Path, max_size: int, score_threshold: float = 0.5) -> None:
        self._model_path = model_path
        self._max_size = max_size
        self._model_hash = hash_file(model_path)
        self._score_threshold = score_threshold  # recorded in provenance (PipelineConfig)
        self.n_dropped_noncanonical = 0
        self._net = _create_yunet(str(model_path), score_threshold)

    @property
    def detector_id(self) -> str:
        return "yunet"

    @property
    def model_hash(self) -> str:
        return self._model_hash

    def detect(self, img_rgb: np.ndarray) -> list[FaceDetection]:
        h, w = img_rgb.shape[:2]
        dw, dh, scale = _scaled_size(w, h, self._max_size)
        small = img_rgb if scale == 1.0 else cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        self._net.setInputSize((dw, dh))
        _, rows = self._net.detect(bgr)
        if rows is None or len(rows) == 0:
            return []
        out: list[FaceDetection] = []
        for row in rows:
            x, y, bw, bh = (float(v) / scale for v in row[0:4])
            native = row[4:14].reshape(5, 2) / scale
            lm = native[_YUNET_TO_CANONICAL].astype(np.float32)
            # Runtime guard against a wrong _YUNET_TO_CANONICAL map: drop
            # non-canonical output rather than raise — one legitimately rotated
            # face must not crash a whole indexing run.  A wholesale-wrong map
            # shows up as ~100% "landmark_order" rejections in the marker stats
            # and CLI summary, which is loud enough to catch immediately.
            try:
                assert_canonical_landmarks(lm)
            except ValueError:
                self.n_dropped_noncanonical += 1
                continue
            out.append(
                FaceDetection(
                    bbox=(x, y, bw, bh), landmarks=lm,
                    confidence=float(row[14]), detect_scale=scale,
                )
            )
        return out
```

- [ ] **Step 4: Run tests** → PASS (skip on the real-model test is expected). **Step 5: Commit** — `git commit -m "feat(faces): YuNet detector adapter with input cap, coord scale-back, canonical reorder"`.

---

### Task 6: Two-stage quality gate

**Files:**
- Create: `src/scalar_forensic/faces/quality.py`
- Test: `tests/faces/test_quality.py`

**Interfaces:**
- Produces: `GateResult(passed: bool, reason: str | None, subscores: dict[str, float])`; `pre_align_gate(det: FaceDetection, *, min_conf: float, min_size: int, max_pose: float) -> GateResult`; `post_align_gate(source_crop_gray: np.ndarray, *, min_sharpness: float, max_clipped_frac: float) -> GateResult`; `pose_ratio(landmarks) -> float`; rejection reason strings: `"confidence" | "size" | "pose" | "sharpness" | "exposure"` (plus `"landmark_order"` counted by the detector, Task 5).
- Sharpness is measured on the **native-resolution source crop**, not the 112×112 resample (spec §6.2). Every threshold — including the pose cutoff — is a keyword argument sourced from Settings (Task 2) and recorded in `PipelineConfig` (Task 8); **no module-level threshold constants**, or provenance can't record them and calibration can't supersede them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/faces/test_quality.py
import numpy as np
import pytest

from scalar_forensic.faces.quality import GateResult, pose_ratio, pre_align_gate, post_align_gate
from scalar_forensic.faces.types import FaceDetection

FRONTAL = np.array([[30, 30], [70, 30], [50, 55], [35, 75], [65, 75]], np.float32)


def _det(conf=0.95, size=100.0, lm=FRONTAL):
    return FaceDetection(bbox=(0, 0, size, size), landmarks=lm, confidence=conf, detect_scale=1.0)


def test_pre_gate_passes_good_face():
    r = pre_align_gate(_det(), min_conf=0.8, min_size=64, max_pose=0.35)
    assert r.passed and r.reason is None
    assert set(r.subscores) == {"confidence", "size", "pose"}


def test_pre_gate_rejects_low_confidence():
    r = pre_align_gate(_det(conf=0.5), min_conf=0.8, min_size=64, max_pose=0.35)
    assert not r.passed and r.reason == "confidence"


def test_pre_gate_rejects_small_face():
    # min side in detector-input px: bbox 40px at detect_scale 1.0
    r = pre_align_gate(_det(size=40.0), min_conf=0.8, min_size=64, max_pose=0.35)
    assert not r.passed and r.reason == "size"


def test_pre_gate_rejects_strong_profile():
    # Nose far outside the eye span → strong yaw.
    profile = np.array([[30, 30], [45, 30], [70, 50], [33, 75], [48, 75]], np.float32)
    r = pre_align_gate(_det(lm=profile), min_conf=0.8, min_size=64, max_pose=0.35)
    assert not r.passed and r.reason == "pose"


def test_pose_ratio_zero_for_symmetric_face():
    assert pose_ratio(FRONTAL) == pytest.approx(0.0, abs=0.05)


def test_post_gate_rejects_flat_crop_as_blurry():
    flat = np.full((80, 80), 128, dtype=np.uint8)
    r = post_align_gate(flat, min_sharpness=25.0, max_clipped_frac=0.6)
    assert not r.passed and r.reason == "sharpness"


def test_post_gate_rejects_clipped_exposure():
    sharp_but_clipped = np.zeros((80, 80), dtype=np.uint8)
    sharp_but_clipped[::2] = 255  # high Laplacian variance, but 100% clipped pixels
    r = post_align_gate(sharp_but_clipped, min_sharpness=25.0, max_clipped_frac=0.6)
    assert not r.passed and r.reason == "exposure"


def test_post_gate_passes_textured_crop():
    rng = np.random.default_rng(3)
    textured = rng.integers(60, 200, size=(80, 80), dtype=np.uint8)
    r = post_align_gate(textured, min_sharpness=25.0, max_clipped_frac=0.6)
    assert r.passed and set(r.subscores) == {"sharpness", "exposure"}
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/scalar_forensic/faces/quality.py
"""Two-stage quality gate (spec §6.2) — the primary false-positive lever.

Pre-alignment: detector confidence, size, pose-from-landmarks (cheap).
Post-alignment: sharpness and exposure on the native-resolution source
crop — NOT the 112x112 resample, whose Laplacian variance mostly
re-encodes the resize factor.

All thresholds are bootstrap values passed by the caller; the Phase 1b
face-calibration record supersedes them (spec §10.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from scalar_forensic.faces.types import FaceDetection


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None = None
    subscores: dict[str, float] = field(default_factory=dict)


def pose_ratio(landmarks: np.ndarray) -> float:
    """Yaw proxy: horizontal nose offset from the eye midpoint over eye span.

    0.0 = frontal; grows toward ±1 in profile.  Coarse by design — its job
    is rejecting strong profiles, not estimating angles.
    """
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_span = float(right_eye[0] - left_eye[0])
    if eye_span <= 1.0:
        return 1.0
    mid_x = (left_eye[0] + right_eye[0]) / 2.0
    return float(abs(nose[0] - mid_x) / eye_span)


def pre_align_gate(
    det: FaceDetection, *, min_conf: float, min_size: int, max_pose: float
) -> GateResult:
    min_side_input_px = min(det.bbox[2], det.bbox[3]) * det.detect_scale
    pose = pose_ratio(det.landmarks)
    subs = {"confidence": det.confidence, "size": min_side_input_px, "pose": pose}
    if det.confidence < min_conf:
        return GateResult(False, "confidence", subs)
    if min_side_input_px < min_size:
        return GateResult(False, "size", subs)
    if pose > max_pose:
        return GateResult(False, "pose", subs)
    return GateResult(True, None, subs)


def post_align_gate(
    source_crop_gray: np.ndarray, *, min_sharpness: float, max_clipped_frac: float
) -> GateResult:
    sharpness = float(cv2.Laplacian(source_crop_gray, cv2.CV_64F).var())
    clipped = float(np.mean((source_crop_gray <= 2) | (source_crop_gray >= 253)))
    subs = {"sharpness": sharpness, "exposure": clipped}
    if clipped > max_clipped_frac:
        return GateResult(False, "exposure", subs)
    if sharpness < min_sharpness:
        return GateResult(False, "sharpness", subs)
    return GateResult(True, None, subs)
```

Note the check order: exposure before sharpness, because a clipped strobe frame can have huge
Laplacian variance — the tests encode this.

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit** — `git commit -m "feat(faces): two-stage quality gate with per-check subscores and reasons"`.

---

### Task 7: Embedder manifest + ONNX embedder

**Files:**
- Create: `src/scalar_forensic/faces/embed.py`, `tests/faces/conftest.py` (tiny-ONNX fixture)
- Test: `tests/faces/test_embed.py`

**Interfaces:**
- Produces: `EmbedderManifest(input_name, layout: Literal["NCHW","NHWC"], channel_order: Literal["RGB","BGR"], dtype: Literal["float32"], input_size: int, mean: float, scale: float, output_name, embedding_dim: int)`; `load_manifest(model_path: Path) -> EmbedderManifest` (reads `<model>.manifest.json`); `OnnxFaceEmbedder(model_path: Path)` with `embed(crops: list[np.ndarray]) -> np.ndarray  # (N, dim) float32 L2-normalised`, `embedding_norms: np.ndarray  # (N,) pre-normalisation norms of the last batch`, `model_hash: str`, `manifest_hash: str`, `normalization_id: str  # f"affine-{mean}-{scale}"`.
- Consumes: 112×112 RGB uint8 crops from Task 3.
- Manifest validation errors and session-mismatch errors are `ValueError` with the offending field named.

- [ ] **Step 1: Generate and commit the tiny ONNX fixture — one-shot script, not a runtime export.**
  Runtime `torch.onnx.export` fails with the installed torch 2.11 (`dynamo=True` needs
  `onnxscript`; `dynamo=False` needs `onnx` — neither is a dependency), and a dense
  112×112×3→512 Linear is a ~77 MB file. Instead: a one-shot generator run **once** with the
  export deps supplied ad hoc, output committed to `tests/fixtures/faces/` (a conv+global-pool
  head is a few KB).

```python
# scripts/gen_face_embedder_fixture.py
"""One-shot generator for the tiny ONNX embedder fixture (run once, commit output).

Run with the export-only deps supplied ad hoc (they are NOT project deps):
    uv run --with onnx --with onnxscript python scripts/gen_face_embedder_fixture.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # noqa: E402 — scripts bootstrap

OUT = Path("tests/fixtures/faces")


class _TinyEmbedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 512, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))  # global average pool → (N, 512)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tiny_face.onnx"
    torch.onnx.export(
        _TinyEmbedder(), (torch.zeros(1, 3, 112, 112),), str(path),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    manifest = {
        "input_name": "input", "layout": "NCHW", "channel_order": "RGB",
        "dtype": "float32", "input_size": 112, "mean": 127.5, "scale": 128.0,
        "output_name": "embedding", "embedding_dim": 512,
    }
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"fixture written to {path}")


if __name__ == "__main__":
    main()
```

Then the conftest fixture just points at the committed files (copying to tmp when a test needs
to mutate the manifest):

```python
# tests/faces/conftest.py
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "faces" / "tiny_face.onnx"


@pytest.fixture(scope="session")
def tiny_onnx_model() -> Path:
    return _FIXTURE
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/faces/test_embed.py
import json
from pathlib import Path

import numpy as np
import pytest

from scalar_forensic.faces.embed import OnnxFaceEmbedder, load_manifest


def test_manifest_loads_and_validates(tiny_onnx_model):
    m = load_manifest(tiny_onnx_model)
    assert m.embedding_dim == 512 and m.layout == "NCHW" and m.channel_order == "RGB"


def test_manifest_missing_field_is_actionable(tiny_onnx_model, tmp_path):
    bad = tmp_path / "m.onnx"
    bad.write_bytes(tiny_onnx_model.read_bytes())
    manifest = json.loads(Path(str(tiny_onnx_model) + ".manifest.json").read_text())
    del manifest["embedding_dim"]
    Path(str(bad) + ".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="embedding_dim"):
        load_manifest(bad)


def test_embed_returns_l2_normalised_and_records_norms(tiny_onnx_model):
    emb = OnnxFaceEmbedder(tiny_onnx_model)
    crops = [np.random.default_rng(i).integers(0, 255, (112, 112, 3), np.uint8) for i in range(3)]
    out = emb.embed(crops)
    assert out.shape == (3, 512) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert emb.embedding_norms.shape == (3,) and (emb.embedding_norms > 0).all()


def test_embed_is_deterministic(tiny_onnx_model):
    emb = OnnxFaceEmbedder(tiny_onnx_model)
    crop = [np.full((112, 112, 3), 90, np.uint8)]
    np.testing.assert_array_equal(emb.embed(crop), emb.embed(crop))


def test_session_validation_catches_dim_mismatch(tiny_onnx_model, tmp_path):
    lying = tmp_path / "lying.onnx"
    lying.write_bytes(tiny_onnx_model.read_bytes())
    manifest = json.loads(Path(str(tiny_onnx_model) + ".manifest.json").read_text())
    manifest["embedding_dim"] = 256  # model actually emits 512
    Path(str(lying) + ".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="embedding_dim"):
        OnnxFaceEmbedder(lying)


def test_identity_metadata(tiny_onnx_model):
    emb = OnnxFaceEmbedder(tiny_onnx_model)
    assert len(emb.model_hash) == 64 and len(emb.manifest_hash) == 64
    assert emb.normalization_id == "affine-127.5-128.0"
```

- [ ] **Step 3: Run to verify failure** — FAIL (module missing).

- [ ] **Step 4: Implement**

```python
# src/scalar_forensic/faces/embed.py
"""Operator-supplied ONNX face embedder, described by a JSON manifest (spec §6.3).

CPU execution provider only in Phase 1; SFN_DEVICE does not apply here.
Threads are pinned to avoid OpenMP oversubscription against torch in the
same process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

from scalar_forensic.embedder import hash_file

_REQUIRED_FIELDS = (
    "input_name", "layout", "channel_order", "dtype", "input_size",
    "mean", "scale", "output_name", "embedding_dim",
)


@dataclass(frozen=True)
class EmbedderManifest:
    input_name: str
    layout: Literal["NCHW", "NHWC"]
    channel_order: Literal["RGB", "BGR"]
    dtype: Literal["float32"]
    input_size: int
    mean: float
    scale: float
    output_name: str
    embedding_dim: int


def load_manifest(model_path: Path) -> EmbedderManifest:
    manifest_path = Path(str(model_path) + ".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"Embedder manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text())
    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"Embedder manifest {manifest_path} missing fields: {', '.join(missing)}")
    # Coerce numerics: JSON "128" vs "128.0" must not change normalization_id,
    # which is a hard comparability field — two identical manifests would
    # otherwise refuse each other.
    raw["mean"] = float(raw["mean"])
    raw["scale"] = float(raw["scale"])
    raw["input_size"] = int(raw["input_size"])
    raw["embedding_dim"] = int(raw["embedding_dim"])
    m = EmbedderManifest(**{f: raw[f] for f in _REQUIRED_FIELDS})
    if m.layout not in ("NCHW", "NHWC"):
        raise ValueError(f"manifest layout must be NCHW or NHWC, got {m.layout!r}")
    if m.channel_order not in ("RGB", "BGR"):
        raise ValueError(f"manifest channel_order must be RGB or BGR, got {m.channel_order!r}")
    if m.input_size != 112:
        raise ValueError(f"manifest input_size must be 112 (arcface-112-v1), got {m.input_size}")
    return m


class OnnxFaceEmbedder:
    def __init__(self, model_path: Path) -> None:
        self.manifest = load_manifest(model_path)
        self.model_hash = hash_file(model_path)
        self.manifest_hash = hash_file(Path(str(model_path) + ".manifest.json"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._validate_session()
        self.embedding_norms: np.ndarray = np.empty(0, dtype=np.float32)

    @property
    def normalization_id(self) -> str:
        return f"affine-{self.manifest.mean}-{self.manifest.scale}"

    def _validate_session(self) -> None:
        m = self.manifest
        inputs = {i.name: i for i in self._session.get_inputs()}
        outputs = {o.name: o for o in self._session.get_outputs()}
        if m.input_name not in inputs:
            raise ValueError(f"model has no input {m.input_name!r}; found {list(inputs)}")
        if m.output_name not in outputs:
            raise ValueError(f"model has no output {m.output_name!r}; found {list(outputs)}")
        out_shape = outputs[m.output_name].shape
        # ONNX dynamic dims come back as str symbols — only validate concrete ints.
        if isinstance(out_shape[-1], int) and out_shape[-1] != m.embedding_dim:
            raise ValueError(
                f"manifest embedding_dim={m.embedding_dim} but model emits {out_shape[-1]}"
            )

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        m = self.manifest
        batch = np.stack(crops).astype(np.float32)
        if m.channel_order == "BGR":
            batch = batch[..., ::-1]
        batch = (batch - m.mean) / m.scale
        if m.layout == "NCHW":
            batch = batch.transpose(0, 3, 1, 2)
        (raw,) = self._session.run([m.output_name], {m.input_name: np.ascontiguousarray(batch)})
        norms = np.linalg.norm(raw, axis=1)
        self.embedding_norms = norms.astype(np.float32)
        return (raw / np.clip(norms[:, None], 1e-12, None)).astype(np.float32)
```

- [ ] **Step 5: Run tests, lint, commit** — `uv run pytest tests/faces/test_embed.py -q` → PASS; commit `feat(faces): manifest-validated ONNX embedder, CPU EP, L2-normalised with norm annotation`.

---

### Task 8: Provenance object

**Files:**
- Create: `src/scalar_forensic/faces/provenance.py`
- Test: `tests/faces/test_provenance.py`

**Interfaces:**
- Produces: `PipelineConfig` dataclass with fields `detector_id, detector_model_hash, detector_score_threshold, detect_max_size, embedder_model_name, embedder_model_hash, manifest_hash, embedder_dim, alignment_version, normalization_id, min_conf, min_size, min_sharpness, max_clipped, max_pose, crop_dilation, sfn_version, cv2_version, ort_version`; methods `to_payload() -> dict` and property `config_hash: str` (sha256 of the canonical-JSON of all fields **except** the version-info fields — library upgrades alone must not orphan collections; hard comparability is enforced by safeguards, Task 10, not by this hash). **Every gate threshold from Task 2 is a field here** — spec §7.1 requires "gate thresholds in force" in provenance, and the Phase 1b calibration record supersedes them by name.
- Consumes: nothing new; `sfn_version` via `importlib.metadata.version("scalar-forensic")` (confirm distribution name from `pyproject.toml` during implementation).

- [ ] **Step 1: Write the failing tests**

```python
# tests/faces/test_provenance.py
from scalar_forensic.faces.provenance import PipelineConfig


def _cfg(**over):
    base = dict(
        detector_id="yunet", detector_model_hash="d" * 64, detector_score_threshold=0.5,
        detect_max_size=1600, embedder_model_name="emb.onnx", embedder_model_hash="e" * 64,
        manifest_hash="m" * 64, embedder_dim=512, alignment_version="arcface-112-v1",
        normalization_id="affine-127.5-128.0", min_conf=0.8, min_size=64,
        min_sharpness=25.0, max_clipped=0.6, max_pose=0.35,
        crop_dilation=0.15, sfn_version="1.0", cv2_version="4.10", ort_version="1.18",
    )
    base.update(over)
    return PipelineConfig(**base)


def test_hash_is_stable_and_hex():
    assert _cfg().config_hash == _cfg().config_hash
    assert len(_cfg().config_hash) == 64


def test_hash_changes_with_comparability_fields():
    assert _cfg().config_hash != _cfg(embedder_model_hash="f" * 64).config_hash
    assert _cfg().config_hash != _cfg(min_conf=0.9).config_hash


def test_hash_ignores_library_versions():
    assert _cfg().config_hash == _cfg(cv2_version="9.9", ort_version="9.9", sfn_version="2.0").config_hash


def test_payload_round_trip_contains_everything():
    p = _cfg().to_payload()
    assert p["pipeline_config_hash"] == _cfg().config_hash
    for key in ("detector_id", "embedder_model_hash", "alignment_version", "cv2_version"):
        assert key in p
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: Implement**:

```python
# src/scalar_forensic/faces/provenance.py
"""Pipeline provenance recorded on every face point (spec §7.1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

_VERSION_INFO_FIELDS = {"sfn_version", "cv2_version", "ort_version"}


@dataclass(frozen=True)
class PipelineConfig:
    detector_id: str
    detector_model_hash: str
    detector_score_threshold: float
    detect_max_size: int
    embedder_model_name: str
    embedder_model_hash: str
    manifest_hash: str
    embedder_dim: int
    alignment_version: str
    normalization_id: str
    min_conf: float
    min_size: int
    min_sharpness: float
    max_clipped: float
    max_pose: float
    crop_dilation: float
    sfn_version: str
    cv2_version: str
    ort_version: str

    @property
    def config_hash(self) -> str:
        # Version-info fields are recorded but excluded from the hash:
        # a library upgrade alone must not orphan a collection.  Hard
        # comparability is enforced field-by-field in safeguards.
        hashed = {k: v for k, v in asdict(self).items() if k not in _VERSION_INFO_FIELDS}
        canon = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    def to_payload(self) -> dict:
        return {**asdict(self), "pipeline_config_hash": self.config_hash}
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit** — `feat(faces): hashed pipeline provenance object`.

---

### Task 9: Chip store (aligned PNG + review JPEG + browse thumbnail)

**Files:**
- Create: `src/scalar_forensic/faces/chips.py`
- Test: `tests/faces/test_chips.py`

**Interfaces:**
- Produces: `chip_hash(aligned_rgb: np.ndarray) -> str` (sha256 over `f"{h}x{w}:".encode() + raw bytes`, dimension-prefixed per `_frame_pixel_hash` convention); `chip_paths(store_dir: Path, chash: str) -> tuple[Path, Path, Path]` (sharded `store_dir/ab/<hash>.png`, `.review.jpg`, `.thumb.jpg`); `write_chips(store_dir: Path, aligned_rgb: np.ndarray, source_rgb: np.ndarray, bbox, dilation: float, thumb_size: int) -> str` (returns chip hash; idempotent — existing files are kept, matching the frame-store reuse pattern); `write_thumbnail(review_path: Path, thumb_path: Path, thumb_size: int) -> None` (downscale long side to `thumb_size`, never upscale — callable standalone for lazy regeneration by the chip endpoint); `dilated_clamped_bbox(bbox, dilation, w, h) -> tuple[int, int, int, int]`.
- Artefact roles per spec §7.3: aligned PNG = reproducibility (hashed, evidentiary), review JPEG q=95 = source-resolution examiner artefact (evidentiary), thumbnail = **derived, non-evidentiary, regenerable** — excluded from `chip_hash`, never referenced in audit events or purge accounting beyond file deletion. Review JPEG is written with `quality=95`.
- Settings addition (extends Task 2's contract): `SFN_FACE_THUMB_SIZE` (int > 0, default 256), parsed eagerly like the other `SFN_FACE_*` ints; add the corresponding valid/invalid cases to `tests/faces/test_config.py` when implementing this task.
- Tests additionally cover: thumbnail written alongside the other two artefacts; long side ≤ `thumb_size`; aspect ratio preserved; a review chip smaller than `thumb_size` is copied 1:1 (no upscale); `write_thumbnail` regenerates a deleted thumbnail; `chip_hash` unchanged by thumbnail presence/absence.

- [ ] **Step 1: Write the failing tests**

```python
# tests/faces/test_chips.py
import numpy as np

from scalar_forensic.faces.chips import chip_hash, chip_paths, dilated_clamped_bbox, write_chips


def test_chip_hash_is_dimension_prefixed():
    a = np.zeros((112, 112, 3), np.uint8)
    b = np.zeros((112 * 112 * 3,), np.uint8).reshape(56, 224, 3)
    assert chip_hash(a) != chip_hash(b)  # same bytes, different dims


def test_paths_are_sharded(tmp_path):
    png, jpg, thumb = chip_paths(tmp_path, "abcd" + "0" * 60)
    assert png.parent.name == "ab" and png.suffix == ".png" and jpg.name.endswith(".review.jpg")
    assert thumb.name.endswith(".thumb.jpg")


def test_dilation_clamps_to_image_bounds():
    # bbox at the corner: dilation must clamp, not go negative.
    x, y, w, h = dilated_clamped_bbox((0.0, 0.0, 100.0, 100.0), 0.15, img_w=640, img_h=480)
    assert (x, y) == (0, 0) and w <= 640 and h <= 480 and w > 100


def test_write_chips_round_trip_lossless_png(tmp_path):
    rng = np.random.default_rng(1)
    aligned = rng.integers(0, 255, (112, 112, 3), np.uint8)
    source = rng.integers(0, 255, (480, 640, 3), np.uint8)
    chash = write_chips(tmp_path, aligned, source, bbox=(100, 100, 80, 80), dilation=0.15)
    png, jpg = chip_paths(tmp_path, chash)
    assert png.exists() and jpg.exists()
    from PIL import Image
    np.testing.assert_array_equal(np.array(Image.open(png)), aligned)  # PNG is lossless
```

- [ ] **Step 2: verify failure. Step 3: Implement**

```python
# src/scalar_forensic/faces/chips.py
"""Face store: lossless aligned crop + human review chip (spec §7.3).

The PNG holds the exact 112x112 RGB tensor fed to the embedder
(pre-normalisation) and chip_hash() covers those exact bytes, so the
stored file authenticates the model input.  The review JPEG is the
unwarped, dilated source crop the examiner actually looks at.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def chip_hash(aligned_rgb: np.ndarray) -> str:
    h, w = aligned_rgb.shape[:2]
    hasher = hashlib.sha256(f"{h}x{w}:".encode())
    hasher.update(np.ascontiguousarray(aligned_rgb).tobytes())
    return hasher.hexdigest()


def chip_paths(store_dir: Path, chash: str) -> tuple[Path, Path, Path]:
    shard = store_dir / chash[:2]
    return (
        shard / f"{chash}.png",
        shard / f"{chash}.review.jpg",
        shard / f"{chash}.thumb.jpg",
    )


def dilated_clamped_bbox(
    bbox: tuple[float, float, float, float], dilation: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    dx, dy = w * dilation, h * dilation
    x0, y0 = max(0, int(x - dx)), max(0, int(y - dy))
    x1, y1 = min(img_w, int(x + w + dx)), min(img_h, int(y + h + dy))
    return x0, y0, x1 - x0, y1 - y0


def write_chips(
    store_dir: Path,
    aligned_rgb: np.ndarray,
    source_rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    dilation: float,
) -> str:
    chash = chip_hash(aligned_rgb)
    png, jpg = chip_paths(store_dir, chash)
    png.parent.mkdir(parents=True, exist_ok=True)
    if not png.exists():
        Image.fromarray(aligned_rgb).save(png, format="PNG")
    if not jpg.exists():
        x, y, w, h = dilated_clamped_bbox(bbox, dilation, source_rgb.shape[1], source_rgb.shape[0])
        if w > 0 and h > 0:  # bbox fully off-image after clamping: skip review chip
            Image.fromarray(source_rgb[y : y + h, x : x + w]).save(jpg, format="JPEG", quality=92)
    return chash
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit** — `feat(faces): sharded chip store — lossless aligned PNG + dilated review JPEG`.

---

### Task 10: FaceStore — collection, points, markers, meta, safeguards, purge

**Files:**
- Create: `src/scalar_forensic/faces/store.py`
- Test: `tests/faces/test_store.py`

**Interfaces:**
- Produces: `FaceStore(client: QdrantClient, collection, case_collection, embedder_dim)` — **injected client, matching `TagStore`'s constructor shape** (no per-request client construction; route tests patch the client, not the store) — with:
  - `ensure_collection(cfg: PipelineConfig, examiner_id: str, authorization_ref: str | None)` — creates the collection (vector name `"face"`, cosine, scalar quantization, `on_disk_payload=True`), payload indexes (`image_hash`, `image_path`, `video_hash`, `video_path`, `is_face`, `group_id` keyword; `quality` float; `frame_timecode_ms` integer — mirrors `indexer.py`'s hash+path convention so filename and time-range lookups need no hash round-trip, spec §7.1), and a **meta point** (payload-only, deterministic ID) recording `case_collection`, enablement record (`examiner_id`, `enabled_at`, `authorization_ref`), the hard-comparability tuple, and — informational only, per spec §7.3 — `face_store_dir` as configured plus `socket.gethostname()` of the app host at enablement (case-handover context; **not** part of the comparability tuple, later changes are audit-logged updates, not errors).
  - `check_compat(cfg: PipelineConfig) -> list[str]` — hard fields (`embedder_model_hash`, `manifest_hash`, `embedder_dim`, `alignment_version`, `normalization_id`) mismatch ⇒ raise `ValueError`; soft fields (`detector_id`, `detector_model_hash`, `detect_max_size`, gate values) mismatch ⇒ returned as warning strings; **absent meta fields are "unknown", not mismatch** (follows `safeguards.py` stance). Wrong `case_collection` ⇒ raise.
  - `face_point_id(image_hash, frame_timecode_ms, bbox) -> str` — `uuid5(NAMESPACE_URL, f"face:{image_hash}:{frame_timecode_ms or ''}:{x}:{y}:{w}:{h}:{ALIGNMENT_VERSION}")` with bbox rounded to ints. **`alignment_version` is part of the ID** (spec §7.1): a future `arcface-112-v2` re-index coexists rather than overwriting v1 points in place (the hard safeguard refuses mixed collections anyway, but the ID must not lie about identity). `observation_key(...)` — the model-independent coordinates only: `f"{image_hash}:{ts}:{x}:{y}:{w}:{h}"` (no alignment version — labels must survive re-alignment).
  - `upsert_faces(points: list[PointStruct])`, `marker_point(image_hash, video_hash, cfg_hash, n_detected, n_kept, rejected: dict[str, int]) -> PointStruct` — **marker ID is keyed on the processed unit's `image_hash`** (`uuid5(NAMESPACE_URL, f"face-marker:{image_hash}")`), i.e. one marker per image and per video *frame*; a per-frame marker must never overwrite a per-video one or `processed_hashes()` skips the rest of the video after frame 1. An optional video-level rollup (`f"face-marker-video:{video_hash}"`, aggregated counts) is written once by the CLI after a video's frames complete.
  - `collection_is_new() -> bool` (collection absent from `get_collections()` — lets the CLI prompt for the authorization reference only on first activation), `list_faces(image_hash) -> list[dict]`, `processed_hashes(cfg_hash) -> set[str]` (returns processed `image_hash`es), `purge_media(image_hash) -> PurgeResult(n_points: int, chip_hashes: list[str])` — the caller unlinks `chip_paths()` for each returned hash (spec §7.5 requires chip files to go too), `purge_all() -> PurgeResult` — deletes face/marker points **by filter, preserving the meta point**: the enablement record is an auditable act and must survive routine purges; the purge itself is audit-logged.
- Consumes: `PipelineConfig` (Task 8), `qdrant_scroll_all` (`indexer.py`), qdrant models. Follows `tags.py` sidecar patterns (deterministic namespace UUID, payload-only meta/marker points, dummy-vector-free — here the collection has a real `face` vector; marker/meta points simply omit the vector).
- All tests mock `QdrantClient` at `scalar_forensic.faces.store.QdrantClient` (patch where used).

- [ ] **Step 1: Write the failing tests**

```python
# tests/faces/test_store.py
from unittest.mock import MagicMock, patch

import pytest

from scalar_forensic.faces.store import FaceStore
from scalar_forensic.faces.provenance import PipelineConfig


def _cfg(**over):
    base = dict(
        detector_id="yunet", detector_model_hash="d" * 64, detect_max_size=1600,
        embedder_model_name="emb.onnx", embedder_model_hash="e" * 64,
        manifest_hash="m" * 64, embedder_dim=512, alignment_version="arcface-112-v1",
        normalization_id="affine-127.5-128.0", min_conf=0.8, min_size=64,
        crop_dilation=0.15, sfn_version="1.0", cv2_version="4.10", ort_version="1.18",
    )
    base.update(over)
    return PipelineConfig(**base)


@pytest.fixture()
def store():
    client = MagicMock()  # injected, TagStore-style — no class patching needed
    client.get_collections.return_value.collections = []
    yield FaceStore(client, "case1_faces", "case1", 512), client


def test_point_id_is_deterministic_and_bbox_rounded(store):
    s, _ = store
    a = s.face_point_id("h1", 1234, (10.4, 20.6, 30.0, 40.0))
    b = s.face_point_id("h1", 1234, (10.0, 21.0, 30.0, 40.0))
    assert a == b  # rounds to same ints
    assert a != s.face_point_id("h1", 5678, (10, 21, 30, 40))
    assert a == s.face_point_id("h1", 1234, (10.4, 20.6, 30.0, 40.0))  # stable


def test_observation_key_matches_point_id_inputs(store):
    s, _ = store
    assert s.observation_key("h1", None, (1.0, 2.0, 3.0, 4.0)) == "h1::1:2:3:4"


def test_ensure_collection_creates_with_quantization_and_indexes(store):
    s, client = store
    s.ensure_collection(_cfg(), examiner_id="ex1", authorization_ref="case order 7")
    create = client.create_collection.call_args
    assert "face" in create.kwargs["vectors_config"]
    assert create.kwargs["quantization_config"] is not None
    assert create.kwargs["on_disk_payload"] is True
    indexed = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
    assert {
        "image_hash",
        "image_path",
        "video_hash",
        "video_path",
        "is_face",
        "group_id",
        "quality",
        "frame_timecode_ms",
    } <= indexed
    # Meta point recorded the enablement + case binding.
    upserted = client.upsert.call_args.kwargs["points"]
    meta = upserted[0].payload
    assert meta["case_collection"] == "case1"
    assert meta["enablement"]["examiner_id"] == "ex1"


def _existing_meta(client, payload):
    rec = MagicMock(); rec.payload = payload
    client.retrieve.return_value = [rec]


def test_check_compat_hard_mismatch_raises(store):
    s, client = store
    _existing_meta(client, {"case_collection": "case1", "embedder_model_hash": "x" * 64,
                            "manifest_hash": "m" * 64, "embedder_dim": 512,
                            "alignment_version": "arcface-112-v1",
                            "normalization_id": "affine-127.5-128.0"})
    with pytest.raises(ValueError, match="embedder_model_hash"):
        s.check_compat(_cfg())


def test_check_compat_soft_mismatch_warns(store):
    s, client = store
    _existing_meta(client, {"case_collection": "case1", "embedder_model_hash": "e" * 64,
                            "manifest_hash": "m" * 64, "embedder_dim": 512,
                            "alignment_version": "arcface-112-v1",
                            "normalization_id": "affine-127.5-128.0",
                            "detector_model_hash": "old" * 21 + "x"})
    warnings = s.check_compat(_cfg())
    assert any("detector_model_hash" in w for w in warnings)


def test_check_compat_absent_fields_are_unknown_not_mismatch(store):
    s, client = store
    _existing_meta(client, {"case_collection": "case1"})
    assert s.check_compat(_cfg()) == []


def test_check_compat_wrong_case_collection_raises(store):
    s, client = store
    _existing_meta(client, {"case_collection": "other_case"})
    with pytest.raises(ValueError, match="case"):
        s.check_compat(_cfg())


def test_purge_media_deletes_points_and_returns_chip_hashes(store):
    s, client = store
    rec1 = MagicMock(); rec1.id = "p1"; rec1.payload = {"chip_hash": "c" * 64}
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", return_value=iter([rec1])):
        result = s.purge_media("h1")
    assert result.n_points == 1
    assert result.chip_hashes == ["c" * 64]  # caller unlinks chip files (spec §7.5)
    client.delete.assert_called_once()


def test_purge_all_preserves_meta_point(store):
    s, client = store
    with patch("scalar_forensic.faces.store.qdrant_scroll_all", return_value=iter([])):
        s.purge_all()
    # Points deleted by filter (is_face / is_face_marker), never delete_collection:
    # the meta point carries the enablement record and must survive routine purges.
    client.delete_collection.assert_not_called()
    client.delete.assert_called()
```

- [ ] **Step 2: verify failure. Step 3: Implement** `store.py` (~180 lines). Key excerpts the tests pin down:

```python
# src/scalar_forensic/faces/store.py
"""Face observation store: case-scoped Qdrant sidecar collection (spec §7)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct,
    QuantizationSearchParams, ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    VectorParams,
)

from scalar_forensic.faces.align import ALIGNMENT_VERSION
from scalar_forensic.faces.provenance import PipelineConfig
from scalar_forensic.indexer import qdrant_scroll_all

FACE_VECTOR_NAME = "face"
_META_POINT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "sfn:face-collection-meta"))
_HARD_FIELDS = (
    "embedder_model_hash", "manifest_hash", "embedder_dim",
    "alignment_version", "normalization_id",
)
_SOFT_FIELDS = ("detector_id", "detector_model_hash", "detect_max_size", "min_conf", "min_size")


class FaceStore:
    def __init__(self, client: QdrantClient, collection, case_collection, embedder_dim):
        # Injected client, matching TagStore — callers own client lifetime.
        self.client = client
        self.collection = collection
        self.case_collection = case_collection
        self.embedder_dim = embedder_dim

    def face_point_id(self, image_hash, frame_timecode_ms, bbox) -> str:
        key = self.observation_key(image_hash, frame_timecode_ms, bbox)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"face:{key}:{ALIGNMENT_VERSION}"))

    def observation_key(self, image_hash, frame_timecode_ms, bbox) -> str:
        # Model-independent: no alignment version — labels reference this and
        # must survive re-alignment/re-detection under new versions.
        x, y, w, h = (round(v) for v in bbox)
        ts = "" if frame_timecode_ms is None else str(frame_timecode_ms)
        return f"{image_hash}:{ts}:{x}:{y}:{w}:{h}"

    def ensure_collection(self, cfg: PipelineConfig, examiner_id: str,
                          authorization_ref: str | None) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={FACE_VECTOR_NAME: VectorParams(
                    size=self.embedder_dim, distance=Distance.COSINE)},
                quantization_config=ScalarQuantization(scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, always_ram=True)),
                on_disk_payload=True,
            )
            for field, schema in (
                ("image_hash", PayloadSchemaType.KEYWORD),
                ("image_path", PayloadSchemaType.KEYWORD),
                ("video_hash", PayloadSchemaType.KEYWORD),
                ("video_path", PayloadSchemaType.KEYWORD),
                ("is_face", PayloadSchemaType.KEYWORD),
                ("group_id", PayloadSchemaType.KEYWORD),
                ("quality", PayloadSchemaType.FLOAT),
                ("frame_timecode_ms", PayloadSchemaType.INTEGER),
            ):
                self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema)
            meta_payload = {
                "is_face_meta": True,
                "case_collection": self.case_collection,
                "enablement": {
                    "examiner_id": examiner_id,
                    "enabled_at": datetime.now(UTC).isoformat(),
                    "authorization_ref": authorization_ref,
                },
                **{f: getattr(cfg, f) for f in _HARD_FIELDS + _SOFT_FIELDS},
            }
            self.client.upsert(collection_name=self.collection, points=[
                PointStruct(id=_META_POINT_ID, vector={}, payload=meta_payload)])
        # else: leave meta as recorded at creation; check_compat() judges it.
    ...
```

`check_compat` reads the meta point via `client.retrieve([_META_POINT_ID])`, raises on
`case_collection` mismatch, raises listing every mismatching hard field, returns warning strings
for soft fields, and skips fields absent from the stored payload ("unknown, not mismatch").
`purge_media(image_hash)` scrolls points filtered on `image_hash` (via `qdrant_scroll_all`),
deletes them by ID, and returns `PurgeResult(n_points, chip_hashes)` — the caller unlinks the
chip files. `purge_all()` deletes face and marker points **by filter** (`is_face` /
`is_face_marker`), never `delete_collection` — the meta point (case binding + enablement record)
is an auditable act and survives routine purges; the CLI confirms and audit-logs the purge.
`processed_hashes(cfg_hash)` scrolls marker points (`is_face_marker=True`, matching
`faces_pipeline_config_hash=cfg_hash`) and returns their `image_hash` set. `marker_point(...)`
builds a payload-only `PointStruct` with deterministic ID
`uuid5(NAMESPACE_URL, f"face-marker:{image_hash}")` — **per processed unit** (per image, per
video *frame*), never per video, or frame N's marker overwrites the video's and
`processed_hashes()` skips the remaining frames — with payload `{"is_face_marker": True,
"image_hash": ..., "video_hash": ..., "faces_processed_at": ..., "faces_pipeline_config_hash":
cfg_hash, "n_detected": ..., "n_kept": ..., "n_rejected": rejected}`.

- [ ] **Step 4: Run tests** → PASS; full suite still green. **Step 5: Commit** — `feat(faces): case-scoped face collection with split safeguards, markers, meta, purge`.

---

### Task 11: Audit log

**Files:**
- Create: `src/scalar_forensic/faces/audit.py`
- Test: `tests/faces/test_audit.py`

**Interfaces:**
- Produces: `AuditLog(path: Path)` with `append(event_type: str, examiner_id: str, **fields) -> None` (JSONL line: `{"ts": iso-utc, "event": event_type, "examiner_id": ..., **fields}`; parent dir created; file opened append-only per call — no held handle) and `iter_events() -> Iterator[dict]`. Event types used in Phase 1: `"enablement"`, `"index_run"`, `"purge"`. Default path: `data/face_audit.log` next to the store dir's parent (constructed by callers from `settings.face_store_dir.parent / "face_audit.log"`; callers handle `face_store_dir=None` by placing it in `data/`).

- [ ] **Step 1: Failing tests**

```python
# tests/faces/test_audit.py
import json

from scalar_forensic.faces.audit import AuditLog


def test_append_writes_jsonl_with_timestamp(tmp_path):
    log = AuditLog(tmp_path / "sub" / "face_audit.log")
    log.append("purge", examiner_id="ex1", image_hash="h1", n_deleted=3)
    lines = (tmp_path / "sub" / "face_audit.log").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "purge" and rec["examiner_id"] == "ex1" and rec["n_deleted"] == 3
    assert "ts" in rec


def test_append_only_accumulates(tmp_path):
    log = AuditLog(tmp_path / "a.log")
    log.append("enablement", examiner_id="ex1")
    log.append("index_run", examiner_id="ex1", n_kept=5)
    assert [e["event"] for e in log.iter_events()] == ["enablement", "index_run"]
```

- [ ] **Step 2: verify failure. Step 3: Implement** (~30 lines: `Path.mkdir(parents=True, exist_ok=True)`, `open(path, "a")` per append, `json.dumps` with `datetime.now(UTC).isoformat()`).
- [ ] **Step 4: tests PASS. Step 5: Commit** — `feat(faces): append-only JSONL audit log`.

---

### Task 12: Orchestration + CLI (`--faces`, `sfn faces-purge`)

**Files:**
- Create: `src/scalar_forensic/faces/indexing.py`
- Modify: `src/scalar_forensic/cli.py` (add `--faces` option to `index`; add `faces_purge` command registered on the existing typer app — check how `main()` wires commands and follow it)
- Test: `tests/faces/test_indexing.py`, `tests/faces/test_cli_faces.py`

**Interfaces:**
- Produces: `FacePipeline.from_settings(settings) -> FacePipeline` (loads detector + embedder + store + audit; raises `ValueError(settings.face_startup_error())` when unusable); `FacePipeline.process_image(data: bytes, image_hash: str, image_path: str, video_hash: str | None = None, video_path: str | None = None, frame_timecode_ms: int | None = None) -> FaceIndexResult(n_detected: int, n_kept: int, rejected: dict[str, int], points: list[PointStruct])` — pure orchestration: decode → detect → pre-gate → align → post-gate → embed (one batch per image) → chips → build points + marker. Caller upserts (keeps Qdrant I/O at the edge, testable core).
- CLI, three load-bearing corrections verified against `cli.py`:
  1. **`main()` is `typer.run(index)` (cli.py:1545) — a single-command app.** Adding a named
     command would force `sfn index <dir>` syntax and break every documented invocation. So:
     `sfn` gains only the `--faces` option; purge ships as a **second console script**
     `sfn-faces = "scalar_forensic.cli:faces_main"` (pyproject `[project.scripts]`, same split
     as `sfn`/`sfn-web`), where `faces_main()` is its own `typer.Typer()` with a `purge`
     command: `sfn-faces purge --media <hash> | --all` (confirmation prompt on `--all`).
  2. **The `--dino/--sscd` guard (cli.py:428) must accept `--faces` alone**: relax to "at least
     one of --dino, --sscd, --faces", and audit the paths that assume ≥1 embedding spec
     (`_dedup_by_hash`, `_check_collection_compat_cli`, `_print_summary`, the batch loop) so a
     faces-only run skips them cleanly rather than crashing on an empty spec list.
  3. **Face processing gets its own pass, not a hook inside the embedding batch loop** — that
     loop iterates only *not-yet-indexed* media (`_dedup_by_hash`), so hooking it means a
     `--faces` run over an already-indexed case silently produces zero faces. The face pass
     iterates **all discovered images + all stored frame paths**, filtered by
     `store.processed_hashes(cfg.config_hash)` (the markers are the face-side idempotency
     mechanism), reading bytes itself. It runs after the embedding phase completes (frames must
     exist on disk first).
  - Summary line: `faces: 132 kept / 517 detected (385 rejected: 200 size, …)`; per-video
    rollup markers written after each video's frames complete; one `index_run` audit event with
    totals at the end. First-time `ensure_collection` prompts for a free-text
    `authorization_ref` (typer.prompt, empty allowed with a warning) and appends an
    `"enablement"` audit event (spec §11).
- Consumes: everything from Tasks 2–11. Payload assembly per spec §7.1 (`image_hash`, `image_path`, `is_face: True`, `video_hash`, `video_path`, `frame_timecode_ms`, `observation_key`, `bbox` ints, `landmarks`, `det_conf`, `detect_scale`, `quality_*` subscores, **`quality`** composite, `embedding_norm`, `chip_hash`, provenance via `cfg.to_payload()`, `indexed_at`).
- **Composite `quality` float** (the field the Task 10 payload index serves; bootstrap formula,
  recorded in the method doc): `quality = min(det_conf, 1 - pose/max_pose, min(1.0, sharpness / (2 * min_sharpness)), 1 - clipped/max_clipped)`, clamped to [0, 1] — each subscore normalised
  against its own gate threshold so 0 = at the gate boundary. It is a browse/ranking aid, not
  evidence; the raw subscores stay alongside it.

- [ ] **Step 1: Failing orchestration test** (fake detector/embedder — no models needed):

```python
# tests/faces/test_indexing.py
import io
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from scalar_forensic.faces.indexing import FacePipeline
from scalar_forensic.faces.types import FaceDetection

FRONTAL = np.array([[130, 130], [170, 130], [150, 155], [135, 175], [165, 175]], np.float32)


def _img_bytes() -> bytes:
    rng = np.random.default_rng(5)
    img = rng.integers(30, 220, (300, 300, 3), np.uint8)
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _pipeline(detections):
    detector = MagicMock()
    detector.detect.return_value = detections
    detector.detector_id = "yunet"; detector.model_hash = "d" * 64
    embedder = MagicMock()
    embedder.embed.side_effect = lambda crops: np.eye(512, dtype=np.float32)[: len(crops)]
    embedder.embedding_norms = np.full(len(detections), 21.7, np.float32)
    store = MagicMock()
    store.face_point_id.side_effect = lambda h, t, b: f"id-{h}-{t}"
    store.observation_key.side_effect = lambda h, t, b: f"{h}:{t or ''}:obs"
    return FacePipeline(
        detector=detector, embedder=embedder, store=store, audit=MagicMock(),
        cfg=MagicMock(config_hash="c" * 64, to_payload=lambda: {"pipeline_config_hash": "c" * 64}),
        min_conf=0.8, min_size=64, min_sharpness=25.0, max_clipped=0.6, max_pose=0.35,
        crop_dilation=0.15, store_dir=None,
    )


def test_process_image_keeps_good_face_and_builds_payload():
    det = FaceDetection(bbox=(100, 100, 100, 100), landmarks=FRONTAL, confidence=0.95, detect_scale=1.0)
    result = _pipeline([det]).process_image(_img_bytes(), image_hash="h1", image_path="/x.png")
    assert result.n_detected == 1 and result.n_kept == 1
    payload = result.points[0].payload
    assert payload["is_face"] is True and payload["image_hash"] == "h1"
    assert payload["observation_key"] == "h1::obs"
    assert "quality_sharpness" in payload and payload["embedding_norm"] > 0
    assert 0.0 <= payload["quality"] <= 1.0  # composite the payload index serves


def test_process_image_counts_rejections_by_reason():
    weak = FaceDetection(bbox=(100, 100, 100, 100), landmarks=FRONTAL, confidence=0.3, detect_scale=1.0)
    result = _pipeline([weak]).process_image(_img_bytes(), image_hash="h1", image_path="/x.png")
    assert result.n_kept == 0 and result.rejected == {"confidence": 1}
    assert result.points == []  # rejected faces are never persisted


def test_zero_faces_is_a_valid_result():
    result = _pipeline([]).process_image(_img_bytes(), image_hash="h1", image_path="/x.png")
    assert result.n_detected == 0 and result.n_kept == 0
```

- [ ] **Step 2: verify failure. Step 3: Implement** `indexing.py`: `FacePipeline` dataclass holding the five collaborators + gate params; `process_image` follows the spec §5 order exactly, converts the aligned crop to gray via the **source crop** for `post_align_gate` (crop the source at the un-dilated bbox, `cv2.cvtColor(..., cv2.COLOR_RGB2GRAY)`), batches all surviving crops into one `embedder.embed` call, writes chips only when `store_dir` is not None, and assembles `PointStruct(id=..., vector={"face": emb}, payload=...)`. `from_settings` wires `YuNetDetector(settings.face_detector_model, settings.face_detect_max_size)`, `OnnxFaceEmbedder(settings.face_embedder_model)`, `FaceStore(QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key), settings.face_collection, settings.collection, embedder.manifest.embedding_dim)` (client constructed here, injected — TagStore pattern), `AuditLog(...)`, and builds `PipelineConfig` (library versions via `cv2.__version__`, `ort.__version__`, `importlib.metadata`; all gate thresholds from Settings).

- [ ] **Step 4: CLI wiring.** In `cli.py` `index(...)`: add `faces: bool = typer.Option(False, "--faces", help="Also detect, embed and store faces (requires SFN_FACES_ENABLED=true)")`. Guard at startup:

```python
    if faces:
        err = settings.face_startup_error() if settings.faces_enabled else (
            "SFN_FACES_ENABLED must be 'true' to use --faces (see docs/specs/face-pipeline.md)."
        )
        if err:
            typer.echo(f"[ERROR] {err}", err=True)
            raise typer.Exit(1)
        from scalar_forensic.faces.indexing import FacePipeline  # deferred: optional deps
        face_pipeline = FacePipeline.from_settings(settings)
        if face_pipeline.store.collection_is_new():
            auth_ref = typer.prompt(
                "First face-collection activation. Authorization reference (free text,"
                " recorded in the enablement record; empty allowed)", default="",
            )
            face_pipeline.store.ensure_collection(face_pipeline.cfg, settings.examiner_id, auth_ref or None)
            face_pipeline.audit.append("enablement", examiner_id=settings.examiner_id,
                                       authorization_ref=auth_ref or None)
        face_pipeline.store.check_compat(face_pipeline.cfg)  # raises on hard mismatch
        _faces_done = face_pipeline.store.processed_hashes(face_pipeline.cfg.config_hash)
```

Then add the **dedicated face pass** after the embedding phase completes (a new top-level block
near the end of `index()`, before the summary): build the face work list as all discovered image
paths plus all frame paths under the frame store for this run's videos, map each to its
`image_hash` (reusing `_file_hashes` / the hash cache), subtract
`store.processed_hashes(cfg.config_hash)`, then for each remaining item read bytes, call
`face_pipeline.process_image(...)`, and `store.upsert_faces(result.points + [marker])`.
Write a per-video rollup marker after each video's frames finish; accumulate and print the
summary; append one `index_run` audit event with totals. Relax the cli.py:428 guard to accept
`--faces` alone and short-circuit the embedding phase when no embedding specs were requested.

Add the purge entry point in `cli.py`:

```python
faces_app = typer.Typer(help="Face-modality maintenance commands.")


@faces_app.command()
def purge(
    media: str = typer.Option(None, "--media", help="Purge faces for one media sha256"),
    all_: bool = typer.Option(False, "--all", help="Purge ALL face observations"),
) -> None:
    ...  # Settings() → face_startup_error check → FaceStore → purge_media/purge_all
    # → unlink chip files for returned chip_hashes → AuditLog.append("purge", ...)
    # `--all` requires typer.confirm("Delete ALL face observations …?").


def faces_main() -> None:
    faces_app()
```

and in `pyproject.toml` `[project.scripts]`: `sfn-faces = "scalar_forensic.cli:faces_main"` —
`sfn` itself stays a single-command `typer.run(index)` app, unbroken.

CLI test (`tests/faces/test_cli_faces.py`): use `typer.testing.CliRunner` — `--faces` without `SFN_FACES_ENABLED` exits 1 with the actionable message; `sfn-faces purge --media h1` (invoke `faces_app` directly) with `FaceStore` patched where it is *used* (`scalar_forensic.cli.FaceStore` if imported at module level, else the deferred-import site) appends a `purge` audit event, unlinks the chip files, and reports the count.

- [ ] **Step 5: Run the full suite** — `uv run pytest -q` → all green (existing CLI tests must still pass unchanged). **Step 6: Commit** — `feat(faces): indexing orchestration, --faces CLI flag, faces-purge command`.

---

### Task 13: Web — browse endpoints + availability flag

**Files:**
- Create: `src/scalar_forensic/web/routes/faces.py`
- Modify: `src/scalar_forensic/web/app.py` (one `include_router` line)
- Test: `tests/faces/test_routes_faces.py`

**Interfaces:**
- Produces (all JSON; router pattern copied from `routes/tags.py`):
  - `GET /api/faces/availability` → `{"faces_available": bool, "reason": str | null}` — false with reason when disabled/misconfigured/collection absent. This is the `faces_available` capability flag; it is **not** an entry in `get_available_modes` (spec §4.3).
  - `GET /api/faces/by-image/{image_hash}` → `{"faces": [{point_id, observation_key, bbox, landmarks, det_conf, quality_*, chip_hash, frame_timecode_ms}]}` via `FaceStore.list_faces`.
  - `GET /api/faces/chip/{chip_hash}`, `GET /api/faces/chip/{chip_hash}/review` and `GET /api/faces/chip/{chip_hash}/thumb` → PNG / JPEG / JPEG from the chip store (404 when absent; degraded-evidence mode note in the availability payload when the store dir is unset). The `/thumb` route serves the browse thumbnail and, when the thumbnail file is missing but the review chip exists, **lazily regenerates it** via `write_thumbnail` (Task 9) before serving — thumbnails are derived artefacts (spec §7.3) and self-heal. **`chip_hash` is validated against `^[0-9a-f]{64}$` before any filesystem access** — it is a path component (follow the hash-param validation precedent in `routes/files.py`/`video.py`).
- Consumes: `FaceStore` (Task 10), Settings (Task 2). Patch targets in tests: `scalar_forensic.web.routes.faces.Settings` and `scalar_forensic.web.routes.faces.FaceStore` (patch where used — CLAUDE.md gotcha).
- Also modify `app.py`'s `lifespan` to log a startup warning with `settings.face_startup_error()` when faces are enabled but unusable (Task 2's docstring promises this check runs at web startup; the availability endpoint then carries the same reason to the UI).
- No search endpoint in this plan (Phase 1b).

- [ ] **Step 1: Failing tests** — FastAPI `TestClient` against the app; three cases: availability false by default (no env), availability true with mocked Settings+FaceStore, `by-image` returns the store's payload list, chip 404 when file missing, chip 200 with correct content-type when present (write a fixture chip into `tmp_path` and point the mocked settings at it).

```python
# tests/faces/test_routes_faces.py — core case shown; write all five described above
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from scalar_forensic.web.app import app


def test_availability_false_when_disabled(monkeypatch):
    monkeypatch.delenv("SFN_FACES_ENABLED", raising=False)
    resp = TestClient(app).get("/api/faces/availability")
    assert resp.status_code == 200
    body = resp.json()
    assert body["faces_available"] is False and "SFN_FACES_ENABLED" in body["reason"]
```

- [ ] **Step 2: verify failure. Step 3: Implement** `routes/faces.py` (~90 lines, `router = APIRouter()`; each endpoint constructs `Settings()` per request like `routes/files.py` does; availability endpoint returns early on each failure condition with a human-readable reason, including `"degraded-evidence mode: face store disabled"` as a non-blocking `note` field when `face_store_dir` is None). Register in `app.py`: `app.include_router(faces_routes.router)`.

- [ ] **Step 4: tests PASS; full suite green. Step 5: Commit** — `feat(faces): browse/availability/chip web endpoints`.

---

### Task 14: Frontend — face browser panel

**Files:**
- Create: `src/scalar_forensic/web/static/js/faces.js`
- Modify: `src/scalar_forensic/web/static/index.html` (add `<script src="/static/js/faces.js"></script>` after the `reset.js` tag and **before** `/static/app.js` — note the leading-slash URL form the existing tags use), `src/scalar_forensic/web/static/app.js` (update its load-order comment only — **there is no merge list**: parts self-register by pushing onto `window.__sfnParts`, and `sfn()` iterates that array), `src/scalar_forensic/web/static/js/state.js` (add `facesAvailable: false, facesForHit: [], facesLoading: false`), `src/scalar_forensic/web/static/js/lifecycle.js` (one `checkFacesAvailability()` call in init)
- Test: `tests/faces/test_static_wiring.py`

**Interfaces:**
- Produces (Alpine part file: an object pushed via `(window.__sfnParts = window.__sfnParts || []).push({...})`, copying the registration idiom from an existing part — **never `Object.assign` merging**, per CLAUDE.md): `checkFacesAvailability()`, `loadFacesForHit(imageHash)`, `faceChipUrl(chipHash)`, `faceReviewUrl(chipHash)`, `faceThumbUrl(chipHash)`. UI: a "Faces" section on the selected-result detail view, rendered only when `facesAvailable`; the grid renders **thumbnails** (`/thumb`), with the full-resolution review chip opened on click; each face shows det_conf and quality subscores and the observation key — copy uses "similar faces"/"face observations" language, never "identified person" (spec §8). No search UI in this plan.
- **Iterative UX ground rule (spec §12):** land the endpoints, wiring test, and a first working panel autonomously, then **stop and hand over for hands-on testing** — the maintainer drives the running UI and gives feedback; layout/interaction polish iterates from that feedback rather than being finalised from this plan. Applies to this task and Task 16.
- Computed getters (if any are needed) go in `computed.js` per CLAUDE.md.

- [ ] **Step 1: Failing wiring test** (there are no existing static-wiring tests to mirror — this is the first; keep it to the one assertion that matters, load order):

```python
# tests/faces/test_static_wiring.py
from pathlib import Path

STATIC = Path("src/scalar_forensic/web/static")


def test_faces_part_is_loaded_before_app_js():
    html = (STATIC / "index.html").read_text()
    assert html.index("/static/js/faces.js") < html.index("/static/app.js")
```

- [ ] **Step 2: verify failure. Step 3: Implement** `faces.js` following the structure of an existing part (read `js/evidence.js` first and copy its registration + fetch idioms exactly): fetch wrappers around the three endpoints, defensive `facesAvailable` gating, and a small render block in `index.html`'s detail panel guarded by `x-show="facesAvailable && facesForHit.length"`.

- [ ] **Step 4: tests PASS; full suite + ruff green. Step 5: Manual smoke** — `./run.sh sfn-web`, confirm the app boots with faces disabled and `/api/faces/availability` returns a reason (drop-zone screensaver gotcha applies if driving via browser automation). **Step 6: Commit** — `feat(faces): face browser panel behind faces_available flag`.

---

### Task 15: Docs and licensing

**Files:**
- Modify: `INSTALL.md` (new "Face modality (optional)" section), `THIRD_PARTY_LICENSES.md` (opencv-python-headless, onnxruntime entries), `docs/deployment.md` (legal-assumptions note per spec §11), `CLAUDE.md` (one line in Commands for `--faces`, one gotcha line: faces tests skip without models)
- Create: `docs/face-matching-math.md` (method-chain skeleton)

**Interfaces:** none — documentation task, but its content is specified:

- [ ] **Step 0: Extend `scripts/download_models.py` with `--yunet`** (spec §3.2 names this script as the fetch mechanism; it already has `--dino`/`--sscd` — follow its existing structure: download the pinned opencv_zoo YuNet ONNX to `models/`, print the SFN_FACE_DETECTOR_MODEL line to add to `.env`). Add a test only if the script has existing test coverage; otherwise match its current (untested-script) status.
- [ ] **Step 1: INSTALL.md section** must state, in this order: how to enable (`SFN_FACES_ENABLED`, `SFN_EXAMINER_ID`, model paths); storage location (`SFN_FACE_STORE_DIR` accepts any absolute path — other drive = other path, other server = NFS/SMB mount; the store is relocatable because the DB references chips by hash only, per spec §7.3); how to fetch YuNet (`uv run python scripts/download_models.py --yunet`, MIT, one-time online, offline afterwards); the **recognition-weights licensing reality** verbatim from spec §6.3 (no permissive recognition model known; InsightFace-family weights are research-only; "non-commercial" is *not* automatically satisfied by government use; the choice and its legal review are the operator's, in writing); the manifest format with a filled-in InsightFace-convention example.
- [ ] **Step 2: docs/face-matching-math.md skeleton** — sections: Detection (YuNet, input cap, confidence), Quality gate (two stages, all subscores, current thresholds and their *bootstrap* status), Alignment (Umeyama, the five reference points, `arcface-112-v1`), Embedding (manifest, L2, cosine), What is stored (payload walk-through), What is NOT claimed (no identification, twins/relatives limitation, demographic-differential statement, no accuracy claim — per-deployment method annex arrives with the Phase 1b calibration record).
- [ ] **Step 3: docs/deployment.md** — append the spec §11 paragraph: assumptions-not-conclusions, deployer duties (DPIA, AI-Act role assessment, authorisation regimes), controls support-but-do-not-establish compliance.
- [ ] **Step 4:** `uv run ruff check src tests scripts` + full suite one last time.
- [ ] **Step 5: Commit** — `docs(faces): install/licensing, method-chain doc, deployment legal notes`.

---

### Task 16: Pipeline explainer view (spec §6.6)

**Files:**
- Create: none new — extends `web/routes/faces.py` (Task 13) and `web/static/js/faces.js` (Task 14)
- Modify: `src/scalar_forensic/web/static/index.html` (explainer panel markup)
- Test: extend `tests/faces/test_routes_faces.py`

**Interfaces:**
- `GET /api/faces/explain/{point_id}` → one JSON bundle assembled **entirely from persisted data** (no pipeline step re-runs): source media reference (`image_hash`, `video_hash`, `frame_timecode_ms`), bbox + landmarks in source coordinates, per-step records (detection, pre-align gate, post-align gate) each as `{scores, thresholds_in_force, passed}` where thresholds come from the observation's stored `pipeline_config` — **not** current env (the view describes what happened at index time), chip URLs (aligned / review / thumb), embedder metadata (model name + hash, dim, normalisation), and the source file's rejection counts by reason from the processed marker.
- Frontend: an "How was this face processed?" action on each face in the browser panel, opening a step-by-step view per spec §6.6: source image with bbox + 5 landmarks drawn as a client-side canvas/SVG overlay from payload coordinates → detection score vs threshold → pre-align subscores → dilated review crop → aligned 112×112 crop with the five ArcFace reference points overlaid → post-align subscores → embedding metadata. Each step has one fixed plain-language sentence (court-facing copy rules: "similar faces", never "identified person"; scores as raw values, never percentages). Rejection counts of the source file are shown at the end ("this file: 14 detections, 11 rejected — 6 size, 4 pose, 1 exposure").
- Tests: explain endpoint returns the bundle for a mocked stored observation; thresholds in the response come from stored `pipeline_config`, not env (set a conflicting env value in the test and assert the stored one wins); 404 on unknown point id; hash/id validated before any store access.

- [ ] **Step 1: Failing tests** for the explain endpoint (mocked `FaceStore`, same patch targets as Task 13). **Step 2: verify failure. Step 3: Implement** the endpoint (~40 lines: fetch point, reshape payload into the step bundle) and the frontend panel following the Task 14 idioms. **Step 4: tests + suite + ruff green. Step 5: Manual walkthrough** with the maintainer per the iterative UX ground rule (Task 14) — the explainer is explicitly an educational artefact; its legibility is judged by a human, not by tests. **Step 6: Commit** — `feat(faces): pipeline explainer view`.

---

## Amendments (2026-08-11, evening review with maintainer)

Applied above; listed here so task-local code blocks written before the amendment are read correctly:

- **Thumbnail tier added** (spec §7.3): Task 9 writes three artefacts (`chip_paths` returns three paths, `write_chips` takes `thumb_size`), Task 13 serves `/thumb` with lazy regeneration, Task 14 grids render thumbnails. Any earlier code block in this plan showing `write_chips(...)` without `thumb_size` or a two-tuple from `chip_paths` is superseded by the Task 9 interface — including the Task 12 orchestration calls, which must pass `settings.face_thumb_size`.
- **`SFN_FACE_THUMB_SIZE`** joins the Task 2 settings contract (int > 0, default 256); its config tests are added when Task 9 lands.
- **Task 16 (explainer view)** added per spec §6.6; Phase 1 scope now ends at Task 16.
- **Filename/time lookups** (spec §7.1): the faces collection additionally stores `video_path` and indexes `image_path`, `video_path` (keyword) and `frame_timecode_ms` (integer), mirroring `indexer.py`'s hash+path convention; `FacePipeline.process_image` gained a `video_path` parameter and the Task 12 payload assembly includes it.
- **Iterative frontend UX** ground rule added to Task 14 (applies to 14 and 16).
- **Store relocatability** (spec §7.3): Task 10's meta point additionally records `face_store_dir` + app-host hostname at enablement (informational, not comparability); Task 15's INSTALL.md documents drive/server placement via path/mount.

## Self-Review (performed; then externally reviewed and revised)

This plan was reviewed by an independent agent that executed candidate test code against the
installed environment (torch 2.11, qdrant-client 1.18, typer 0.27, Pillow 12.3) and read the
real `cli.py`/`app.js`. All confirmed findings are integrated above; the load-bearing ones:

- **CI/deps:** `faces` group is included in `dev` (CI runs `uv sync --dev`); the ONNX test
  fixture is committed, generated by a one-shot script with ad-hoc export deps (runtime
  `torch.onnx.export` fails on installed torch without `onnx`/`onnxscript`).
- **Landmark order:** the YuNet→canonical map is *empirically derived at implementation time*
  (identity map is the hypothesis, not `[1,0,2,4,3]`); non-canonical output is dropped and
  counted at runtime, and the real-model test asserts ordering on a real face — the synthetic
  test alone cannot catch a wrong map (it shares the map's assumption).
- **CLI:** `sfn` remains a single-command `typer.run` app; purge lives in a new `sfn-faces`
  console script. `--faces` alone is valid (guard relaxed). Face processing is its own pass over
  all media filtered by face markers — hooking the embedding batch loop would skip everything
  already image-indexed.
- **Markers:** keyed per processed unit (per frame), with a per-video rollup — one marker per
  video would self-overwrite and skip frames 2..n.
- **Purge:** returns chip hashes for file deletion; `purge_all` deletes by filter and preserves
  the meta/enablement point.
- **Provenance:** all five gate thresholds + detector score threshold are Settings + PipelineConfig
  fields (no module constants), and the composite `quality` the payload index serves is defined.
- **Frontend:** parts self-register on `window.__sfnParts` (verified in `app.js:11`); there is no
  merge list to edit; script tag uses the `/static/...` form after `reset.js` (`index.html:1797`).
- **Spec coverage (Phase 1 scope):** §4 → Tasks 1, 2, 10, 13; §5 → Task 12; §6.1 → Tasks 4, 5;
  §6.2 → Task 6; §6.3 → Task 7; §6.4 → Task 3; §7.1 → Tasks 8, 10, 12; §7.2 → Task 10; §7.3 →
  Task 9; §7.4 → Tasks 10, 12; §7.5 → Tasks 10, 12; §11 audit + enablement → Tasks 10–12; §12
  Phase 1 UI → Tasks 13, 14; §13 → Task 2; §3.2 model fetch → Task 15 Step 0; docs → Task 15.
  **Deliberately out of plan (Phase 1b+):** search endpoints, calibration precedence, duplicate
  collapse, adjudications sidecar `{}_face_labels` (ships with search, when there is something
  to adjudicate), within-file grouping and `SFN_FACE_TOPK_PER_GROUP` (Phase 2).
- **Remaining implementer-verification points** (each has a loud failure mode wired in): the
  empirical landmark map (runtime drop-counter + real-model test), the skimage-vs-complex-lstsq
  matrix literals in Task 3 (recompute if `ARCFACE_DST` or the fixture landmarks change), and
  ruff format's opinion of every code block (run `ruff format`, take its output).
