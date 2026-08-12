# Face Query Flow (Phase 1b) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an uploaded query image's faces selectable probes for a cross-file face search,
render FACES as a first-class hit mode alongside EXACT / ALTERED / SEMANTIC with the raw
per-face score displayed, and name the per-model explainer surfaces (DINO pair + an equivalent
face pair).

**Architecture:** Query-side faces are produced by a **session-scoped** detect→gate→align→embed
pass that writes nothing to Qdrant and nothing to the face chip store; the vectors live only in
the in-process `Session` object and are never sent to the browser (the client passes face
*indices*). A new `POST /api/faces/search` runs an exact kNN against the existing
`{SFN_COLLECTION}_faces` named vector `face`, which structurally cannot return review-only
observations because they carry no vector. The frontend merges the returned face hits into the
existing hit list as `hit.scores.faces`, reusing the fixed-slot badge pattern already used for
EXACT/ALTER/SEMAN.

**Tech Stack:** FastAPI (sync `def` handlers → Starlette threadpool), Qdrant
(`query_points`, `using="face"`, `SearchParams(exact=True)`), onnxruntime CPU EP (SFace 128-d),
OpenCV YuNet, Alpine.js part files merged by property descriptors, pytest.

**Specs:**
- `docs/specs/face-query-ux.md` — the request **and the binding RULING appended at its end**.
- `docs/specs/face-pipeline.md` §6 (components), §7 (storage), §8 (query semantics),
  §10 (calibration), §11 (legal/logging/enablement), §12 (phasing), §13 (config).
- Measurement trail: `docs/fleet/runbook.md`, entries of 2026-08-12.

---

## Global Constraints

Every task's requirements implicitly include this section.

1. **The score is displayed, uncalibrated.** It is labelled as a **raw model output with no
   confidence interval and no calibrated threshold**. This is the maintainer's ruling of
   2026-08-12 (`face-query-ux.md`, RULING §1) and is not to be re-litigated.
2. **SFace's 0.363 may appear only as "the model authors' reference figure", never as this
   deployment's threshold.** It must never be a default value of any slider, any env var, or any
   server-side filter. It may appear as an annotated marker on a distribution plot and as a
   sentence in an explainer.
3. **The "uncalibrated — not evidential" banner and the opt-in enablement gate remain**
   (RULING §2). Face search is reachable only when `/api/faces/availability` returns
   `faces_available: true`.
4. **Review-only observations are vectorless and must stay unreachable by search.** Never attach
   a vector to one; never substitute a payload filter for that guarantee (spec §6.2, §7.1). This
   applies to *query-side* faces too: a query face that fails the embedding gate gets
   `vector = None` in the session and is refused as a probe with HTTP 400.
5. **UI language is fixed:** "similar faces", "face observations", never "identified person",
   never "confidence", never a percentage. Raw cosine under a named model (spec §8).
6. **Nothing from the query path is ever persisted** to Qdrant or to `SFN_FACE_STORE_DIR`
   (spec §8: "processed identically in session scope and never persisted"; the web analyze path
   is ephemeral by design — indexing is CLI-only).
7. **Search is exact by default**: `SearchParams(exact=True)`. ANN is an explicit opt-in and the
   mode used is recorded in the query log (spec §8).
8. **Every face query is logged** to `data/faces/face_audit.log` via the existing
   `faces/audit.py` `AuditLog` (spec §11, Phase 1 deliverable). New event type: `"query"`.
9. **Comparability is hard-gated.** Before any query-time embedding is compared against the
   collection, `FaceStore.check_compat(cfg)` must be consulted; a **hard**-field mismatch
   (`embedder_model_hash`, `manifest_hash`, `embedder_dim`, `alignment_version`,
   `normalization_id`) refuses the search with HTTP 409. Soft mismatches are returned as
   warnings and displayed (spec §7.2).
10. **Patch targets are per-module** (CLAUDE.md): patch where the name is *used*, e.g.
    `scalar_forensic.web.routes.faces.Settings`, not the package that re-exports it.
11. **Frontend conventions** (CLAUDE.md): new component code goes in the matching part file;
    computed getters go in `computed.js`; parts are merged by property descriptors — **never**
    `Object.assign`; part `<script>` tags must precede `app.js`.
12. **Testing UI changes requires a cache-bust**: `/?cachebust=N`, and re-set the `style.css`
    href separately — the app sends no cache headers on `/`.
13. Suite bar: `uv run pytest -q` (478 passed / 5 skipped at 2026-08-12) stays green and
    hermetic — no network, no unshipped model. `uv run ruff check src tests scripts` and
    `uv run ruff format --check src tests scripts` must be clean.

### Deployment divergence from the spec, recorded deliberately

Two divergences are introduced by this plan. Both are **recorded in the spec** in Stage 6; do not
ship the code without that edit.

- **§10 / §12 gate face search on an active face-calibration record.** No record exists. The
  maintainer ruled search ships uncalibrated with the score visible. §10.4's precedence rule
  still stands for the day a record exists: the record wins.
- **§8 places uploaded probes in Phase 2** ("Probe = a stored face observation… Phase 2 adds
  uploaded probe images"). The maintainer's request is an uploaded-probe flow, so that part of
  Phase 2 is pulled into 1b. The session-scope-and-never-persist rule §8 attaches to it is kept
  in full.

### Known spec/reality mismatches an implementer will trip over

- Spec §7.1 says the `face` vector is 512-d and §6.6 step 7 says "512-d"; the adopted model
  (SFace) is **128-d**. The dimension is read from the manifest at runtime
  (`EmbedderManifest.embedding_dim`, hard-gated by `check_compat`). Never hardcode either number.
- The frontend badge literals are `EXACT` / `ALTER` / `SEMAN` (`index.html:1242-1264`), while the
  filter pills read `ALTERED` / `SEMANTIC` (`index.html:1179-1195`). Follow the local convention
  in each place: badge `FACES` (5 chars fits the existing slot), pill `FACES`.

### Stage 0 — operational prerequisite (no code, not a shippable stage)

`ALTERED` cannot be exercised against `danny_validation`: it was indexed `--dino` only, so the
collection has no `sscd` vector and `get_available_modes` will not offer `altered`
(`web/pipeline/modes.py:176-177`). Face search does not depend on this; only the four-mode UI
demo does. Before the maintainer's acceptance pass, re-index with both:

```bash
./run.sh sfn data/images/danny_validation --dino --sscd
```

This is an index-time change to the case collection only; it does not touch
`faces_danny_validation`. If it is not run, Stage 4's mode filter must be demonstrated with
EXACT / SEMANTIC / FACES and the ALTERED pill will be absent, which is correct behaviour, not a
defect.

---

## File structure

**New files**

| Path | Responsibility |
|---|---|
| `src/scalar_forensic/web/pipeline/faces_query.py` | Session-scoped query-face detection + embedding. Orchestration only; every primitive is imported from `scalar_forensic.faces.*` so there is one implementation of each pipeline step. Writes nothing anywhere. |
| `src/scalar_forensic/web/pipeline/faces_search.py` | Cross-file face kNN, collapse, hit assembly, compat gate. |
| `src/scalar_forensic/web/pipeline/faces_stats.py` | Face score-distribution statistics (the face twin of `stats.py`). |
| `tests/faces/test_query_faces.py` | Stage 1 |
| `tests/faces/test_face_search.py` | Stage 2 |
| `tests/faces/test_face_audit_stats.py` | Stage 7 backend |

**Modified files**

| Path | Change |
|---|---|
| `src/scalar_forensic/web/session.py` | `QueryFace` dataclass; `FileEntry.query_faces` |
| `src/scalar_forensic/web/routes/faces.py` | 5 new endpoints + audit-log helper |
| `src/scalar_forensic/web/pipeline/__init__.py` | re-export the new public names |
| `src/scalar_forensic/config.py` | `SFN_FACE_QUERY_MAX_FACES` |
| `src/scalar_forensic/faces/store.py` | `search_faces()` read method |
| `src/scalar_forensic/web/static/js/state.js` | new state properties |
| `src/scalar_forensic/web/static/js/faces.js` | query-face + search + match-score functions |
| `src/scalar_forensic/web/static/js/computed.js` | `mergedHits`, `filteredHits` change |
| `src/scalar_forensic/web/static/index.html` | query-face strip, FACES badge slot + pill, face sliders, banner, renamed buttons, face audit/stats modals |
| `src/scalar_forensic/web/static/style.css` | green-border and score classes, face modals |
| `tests/faces/test_static_wiring.py` | frontend assertions per stage |
| `tests/faces/test_config.py` | new env var |
| `docs/specs/face-pipeline.md` | §10 divergence note (Stage 6) |
| `docs/face-matching-math.md` | §6 "what is NOT claimed" addition (Stage 6) |
| `CLAUDE.md` | one gotcha line (Stage 2) |

`docs/specs/face-pipeline.md`, `docs/face-matching-math.md` and `CLAUDE.md` are **not owned by
the implementing session** — claim them with `cx o <path>` before editing, and prefer `Edit`
over `Write`.

---

## Endpoint contracts (written out once; tasks reference them)

### 1. `POST /api/faces/query-faces` — detect faces in the uploaded query image

Request: `multipart/form-data`
```
session_id : str   (required)
file_id    : str   (required)
timecode_ms: int   (optional; for a video upload, the frame to detect on)
```

Response `200`:
```json
{
  "faces": [
    {
      "index": 0,
      "bbox": [412, 118, 148, 191],
      "landmarks": [[451,175],[518,172],[486,215],[457,247],[513,245]],
      "det_conf": 0.945,
      "detect_scale": 1.0,
      "searchable": true,
      "embedding_status": "embedded",
      "embedding_exclusion_reason": null,
      "quality": {"confidence": 0.945, "size": 147.8, "pose": 0.08,
                  "sharpness": 61.2, "exposure": 0.01},
      "chip_url": "/api/faces/query-chip/<session_id>/<file_id>/0"
    }
  ],
  "n_detected": 6,
  "n_searchable": 2,
  "n_review_only": 3,
  "rejected": {"confidence": 1},
  "pipeline_config_hash": "9f2c…",
  "embedder": {"embedder_model_name": "face_recognition_sface_2021dec.onnx",
               "embedder_model_hash": "0ba9fbfa…", "manifest_hash": "…",
               "embedder_dim": 128, "normalization_id": "affine-0.0-1.0",
               "alignment_version": "arcface-112-v1"},
  "truncated": false
}
```
`quality` subscores that were never measured are `null`, never `0.0` (spec §6.2).
`searchable` is `true` iff a vector was produced. **No vector is ever present in the response.**

Errors: `404 {"detail": "session not found"}` / `{"detail": "file not found in session"}`;
`400 {"detail": "not an image"}`; `503 {"detail": "<availability reason>"}` when the face
modality is unavailable; `413 {"detail": "too many faces detected (max N)"}` is **not** used —
detection is truncated to `SFN_FACE_QUERY_MAX_FACES` with `"truncated": true` instead.

### 2. `GET /api/faces/query-chip/{session_id}/{file_id}/{face_index}` — session review crop

Returns `image/jpeg` (the dilated, source-resolution review crop held in memory for that
session), `Cache-Control: no-store`. `404` when the session, file or index is unknown. The chip
is **never** written to `SFN_FACE_STORE_DIR`.

### 3. `POST /api/faces/search` — cross-file face search

Request: `multipart/form-data`
```
session_id   : str   (required)
file_id      : str   (required)
face_indices : str   (required; comma-separated indices into the query-faces response)
limit        : int   (default 10, ge=1, le=50)      — face TOP K, per probe
threshold    : float (default 0.0, ge=0.0, le=1.0)  — display floor on raw cosine
exact        : bool  (default true)                 — SearchParams(exact=…)
collapse     : bool  (default true)                 — one hit per distinct image_hash
```

Response `200`:
```json
{
  "hits": [
    {
      "image_hash": "ca4bed…",
      "image_path": "/evidence/danny2.jpeg",
      "score": 0.6097,
      "query_face_index": 0,
      "face": {
        "point_id": "8f1e…", "observation_key": "ca4bed…::412:118:148:191",
        "bbox": [412,118,148,191], "det_conf": 0.945, "quality": 0.71,
        "review_url": "/api/faces/chip/<review_chip_hash>/review",
        "thumb_url": "/api/faces/chip/<review_chip_hash>/thumb"
      },
      "is_video_frame": false, "video_hash": null, "video_path": null,
      "frame_timecode_ms": null,
      "n_collapsed": 1
    }
  ],
  "n_probes": 2,
  "search_mode": "exact",
  "threshold": 0.0,
  "limit": 10,
  "calibration": {
    "status": "uncalibrated",
    "active_record": null,
    "banner": "Uncalibrated — not evidential. The number below is the raw cosine similarity this model produced. There is no confidence interval and no calibrated threshold for this deployment.",
    "model_reference_threshold": 0.363,
    "model_reference_note": "0.363 is the SFace authors' published same/different reference figure for this model. It is not a threshold calibrated on this deployment's material and must not be read as one."
  },
  "embedder": { … same block as endpoint 1 … },
  "compat": {"ok": true, "warnings": []}
}
```

Errors:
- `400 {"detail": "face 3 is review-only and has no vector; it cannot be searched"}` — a
  non-searchable index was passed. This is the guarantee of Global Constraint 4 surfacing.
- `400 {"detail": "no face indices given"}`, `400 {"detail": "unknown face index 9"}`.
- `404` session/file unknown; `409 {"detail": "<compat message>"}` on a hard-field mismatch;
  `503` when the face modality or collection is unavailable.

### 4. `GET /api/faces/audit?image_hash=<sha256>` — face-model audit for a hit

Response `200` (all fields read from persisted payload; **no pipeline step is re-run**):
```json
{
  "image_hash": "ca4bed…",
  "n_observations": 4,
  "detector": {"detector_id": "yunet", "detector_model_hash": "…",
               "detector_score_threshold": 0.5, "detect_max_size": 1600},
  "embedder": { … same block as endpoint 1 … },
  "gates_in_force_at_index_time": {"min_conf": 0.8, "min_size": 64, "max_pose": 0.35,
                                   "min_sharpness": 25.0, "max_clipped": 0.6,
                                   "review_min_conf": 0.6, "review_min_size": 36,
                                   "crop_dilation": 0.25},
  "pipeline_config_hash": "9f2c…",
  "library_versions": {"cv2": "…", "onnxruntime": "…", "sfn_version": "…"},
  "file_totals": {"n_detected": 6, "n_kept": 2, "n_review_only": 3,
                  "review_only_reasons": {"size": 3}, "n_rejected": 1,
                  "rejected": {"confidence": 1}},
  "enablement": {"examiner_id": "m4v1", "enabled_at": "…", "authorization_ref": "…"},
  "compat": {"ok": true, "warnings": []},
  "caveat": "This describes how the machine produced these face observations. It is an investigative lead, not an identification."
}
```
`400 "Invalid hash"` on a non-`[0-9a-f]{64}` argument; `404` when no face marker exists for that
image; `503` when the collection is unreachable.

### 5. `POST /api/faces/dist-stats` — face score distribution for one query face

Request: `multipart/form-data` — `session_id`, `file_id`, `face_index: int`,
`sample_size: int = 10000 (ge=1, le=50000)`.

Response `200` — deliberately the **same field set** as `SemanticStats`
(`web/pipeline/stats.py:25-39`) so the two modalities read alike, plus three face-specific keys:
```json
{
  "sample_size": 10000, "count": 5,
  "min_score": -0.0142, "p10": 0.01, "p25": 0.09, "median": 0.11,
  "p75": 0.42, "p90": 0.60, "max_score": 0.6097,
  "mean": 0.24, "stdev": 0.29,
  "histogram": [0,0,1,0,…],
  "population": "embedded face observations in <collection>_faces; review-only observations carry no vector and are structurally absent from this distribution",
  "model_reference_threshold": 0.363,
  "model_reference_note": "…as in endpoint 3…"
}
```
Numeric stats stay on the **raw cosine** scale; the histogram is on the `[-1,1] → [0,1]`
normalised scale with 20 buckets — identical to `stats.py:95-101`, and the UI must label both
scales the way the DINO modal already does.

Errors: `404` session/file/index; `400 "face <i> is review-only and has no vector"`;
`503` collection unreachable.

---

## Stage 1: Session-scoped query-face detection

Independently shippable: after this stage the examiner can see the faces the tool found in their
uploaded image. No search exists yet.

**Files:**
- Create: `src/scalar_forensic/web/pipeline/faces_query.py`
- Create: `tests/faces/test_query_faces.py`
- Modify: `src/scalar_forensic/web/session.py` (add `QueryFace`, `FileEntry.query_faces`)
- Modify: `src/scalar_forensic/web/routes/faces.py` (endpoints 1 and 2)
- Modify: `src/scalar_forensic/web/pipeline/__init__.py` (re-exports)
- Modify: `src/scalar_forensic/config.py:164-230` region (add `SFN_FACE_QUERY_MAX_FACES`)
- Modify: `tests/faces/test_config.py`

**Interfaces:**
- Consumes: `scalar_forensic.faces.decode.load_for_detection(data: bytes) -> np.ndarray`;
  `faces.detect.YuNetDetector(model_path, max_size, score_threshold)`;
  `faces.quality.review_gate(det, *, min_conf, min_size) -> GateResult`;
  `faces.quality.pre_align_gate(det, *, min_conf, min_size, max_pose) -> GateResult`;
  `faces.quality.post_align_gate(source_crop_gray, *, min_sharpness, max_clipped) -> GateResult`;
  `faces.align.align_face(img_rgb, landmarks) -> np.ndarray`;
  `faces.embed.OnnxFaceEmbedder(model_path).embed(crops) -> np.ndarray`;
  `faces.chips.dilated_clamped_bbox(bbox, dilation, img_w, img_h)`;
  `faces.provenance.PipelineConfig` / `config_hash` / `to_payload`;
  `web.session.get_session(session_id) -> Session | None`.
- Produces:
  ```python
  # web/pipeline/faces_query.py
  def detect_query_faces(
      data: bytes, settings: Settings, *, max_faces: int
  ) -> QueryFaceResult: ...

  @dataclass
  class QueryFaceResult:
      faces: list[QueryFace]          # web.session.QueryFace
      n_detected: int
      n_searchable: int
      n_review_only: int
      rejected: dict[str, int]
      truncated: bool
      cfg: PipelineConfig

  def query_embedder_block(cfg: PipelineConfig) -> dict[str, object]: ...
  ```
  ```python
  # web/session.py
  @dataclass
  class QueryFace:
      index: int
      bbox: tuple[float, float, float, float]
      landmarks: list[list[float]]
      det_conf: float
      detect_scale: float
      quality: dict[str, float | None]
      embedding_status: str            # "embedded" | "review_only"
      embedding_exclusion_reason: str | None
      vector: list[float] | None       # None for review_only — never populated
      review_jpeg: bytes | None        # in-memory only, never written to disk
  ```

**Why a separate orchestration module rather than reusing `FacePipeline.process_image`:**
`process_image` (`faces/indexing.py:142`) builds `PointStruct`s and writes chip files. Reusing it
here would put a persistence-capable code path one argument away from the query flow. The
primitives are all imported from `scalar_forensic.faces.*`, so no pipeline *step* is duplicated —
only the ~60-line orchestration is, and it is the part that must provably not persist.

- [x] **Step 1: Write the failing tests**

Create `tests/faces/test_query_faces.py`:

```python
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scalar_forensic.faces.types import FaceDetection
from scalar_forensic.web.app import app
from scalar_forensic.web.pipeline.faces_query import detect_query_faces

client = TestClient(app)


def _det(size: float, conf: float) -> FaceDetection:
    return FaceDetection(
        bbox=(10.0, 10.0, size, size),
        landmarks=np.array(
            [[20.0, 25.0], [40.0, 25.0], [30.0, 35.0], [22.0, 45.0], [38.0, 45.0]]
        ),
        confidence=conf,
        detect_scale=1.0,
    )


def _settings(tmp_path, **over):
    s = MagicMock()
    s.faces_enabled = True
    s.face_startup_error.return_value = None
    s.face_collection = "case1_faces"
    s.collection = "case1"
    s.face_store_dir = tmp_path
    s.face_detect_max_size = 1600
    s.face_min_conf = 0.8
    s.face_min_size = 64
    s.face_max_pose = 0.35
    s.face_min_sharpness = 25.0
    s.face_max_clipped = 0.6
    s.face_review_min_conf = 0.6
    s.face_review_min_size = 36
    s.face_crop_dilation = 0.25
    s.face_query_max_faces = 25
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_review_only_query_face_gets_no_vector(tmp_path):
    """The Phase 1 exclusion guarantee, carried into the query path."""
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[1.0] + [0.0] * 127], dtype=np.float32)
    detector = MagicMock()
    detector.detect.return_value = [_det(120.0, 0.95), _det(40.0, 0.9)]
    detector.detector_id = "yunet"
    detector.model_hash = "abc"

    with (
        patch("scalar_forensic.web.pipeline.faces_query.load_for_detection", return_value=img),
        patch("scalar_forensic.web.pipeline.faces_query._detector", return_value=detector),
        patch("scalar_forensic.web.pipeline.faces_query._embedder", return_value=embedder),
    ):
        result = detect_query_faces(b"jpegbytes", _settings(tmp_path), max_faces=25)

    assert result.n_detected == 2
    embedded = [f for f in result.faces if f.embedding_status == "embedded"]
    review = [f for f in result.faces if f.embedding_status == "review_only"]
    assert len(embedded) == 1 and len(review) == 1
    assert embedded[0].vector is not None
    assert review[0].vector is None
    assert review[0].embedding_exclusion_reason == "size"
    # Unmeasured subscores are null, never 0.0 (spec §6.2)
    assert review[0].quality["sharpness"] is None


def test_query_faces_are_never_persisted(tmp_path):
    """No Qdrant client, no file under the chip store."""
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[1.0] + [0.0] * 127], dtype=np.float32)
    detector = MagicMock()
    detector.detect.return_value = [_det(120.0, 0.95)]
    detector.detector_id = "yunet"
    detector.model_hash = "abc"

    with (
        patch("scalar_forensic.web.pipeline.faces_query.load_for_detection", return_value=img),
        patch("scalar_forensic.web.pipeline.faces_query._detector", return_value=detector),
        patch("scalar_forensic.web.pipeline.faces_query._embedder", return_value=embedder),
    ):
        result = detect_query_faces(b"jpegbytes", _settings(tmp_path), max_faces=25)

    assert result.faces[0].review_jpeg is not None       # held in memory
    assert list(tmp_path.rglob("*")) == []               # nothing written
    assert "QdrantClient" not in open(
        "src/scalar_forensic/web/pipeline/faces_query.py", encoding="utf-8"
    ).read()


def test_detection_is_truncated_at_the_configured_cap(tmp_path):
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    embedder = MagicMock()
    embedder.embed.return_value = np.array(
        [[1.0] + [0.0] * 127] * 2, dtype=np.float32
    )
    detector = MagicMock()
    detector.detect.return_value = [_det(120.0, 0.95)] * 5
    detector.detector_id = "yunet"
    detector.model_hash = "abc"

    with (
        patch("scalar_forensic.web.pipeline.faces_query.load_for_detection", return_value=img),
        patch("scalar_forensic.web.pipeline.faces_query._detector", return_value=detector),
        patch("scalar_forensic.web.pipeline.faces_query._embedder", return_value=embedder),
    ):
        result = detect_query_faces(b"j", _settings(tmp_path), max_faces=2)

    assert result.truncated is True
    assert len(result.faces) == 2


def test_query_faces_endpoint_404s_on_unknown_session():
    with patch("scalar_forensic.web.routes.faces.get_session", return_value=None):
        resp = client.post(
            "/api/faces/query-faces", data={"session_id": "nope", "file_id": "x"}
        )
    assert resp.status_code == 404


def test_query_faces_endpoint_never_returns_a_vector(tmp_path, monkeypatch):
    from scalar_forensic.web.session import QueryFace

    face = QueryFace(
        index=0,
        bbox=(1.0, 2.0, 3.0, 4.0),
        landmarks=[[1.0, 2.0]] * 5,
        det_conf=0.9,
        detect_scale=1.0,
        quality={"confidence": 0.9, "size": 120.0, "pose": 0.1,
                 "sharpness": 40.0, "exposure": 0.0},
        embedding_status="embedded",
        embedding_exclusion_reason=None,
        vector=[1.0] + [0.0] * 127,
        review_jpeg=b"\xff\xd8",
    )
    entry = MagicMock(is_video=False, temp_path=tmp_path / "q.jpg", query_faces=[face])
    (tmp_path / "q.jpg").write_bytes(b"jpegbytes")
    session = MagicMock(files=[entry])

    result = MagicMock(
        faces=[face], n_detected=1, n_searchable=1, n_review_only=0,
        rejected={}, truncated=False, cfg=MagicMock(),
    )
    with (
        patch("scalar_forensic.web.routes.faces.Settings", return_value=_settings(tmp_path)),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=session),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.detect_query_faces", return_value=result),
        patch("scalar_forensic.web.routes.faces.query_embedder_block", return_value={}),
    ):
        body = client.post(
            "/api/faces/query-faces", data={"session_id": "s", "file_id": "f"}
        ).json()

    assert body["faces"][0]["searchable"] is True
    assert "vector" not in body["faces"][0]
    assert body["faces"][0]["chip_url"].endswith("/query-chip/s/f/0")
```

Add to `tests/faces/test_config.py`:

```python
def test_face_query_max_faces_defaults_and_parses(monkeypatch):
    monkeypatch.delenv("SFN_FACE_QUERY_MAX_FACES", raising=False)
    assert Settings().face_query_max_faces == 25
    monkeypatch.setenv("SFN_FACE_QUERY_MAX_FACES", "4")
    assert Settings().face_query_max_faces == 4
    monkeypatch.setenv("SFN_FACE_QUERY_MAX_FACES", "0")
    with pytest.raises(ValueError):
        Settings()
```

- [x] **Step 2: Run them and confirm they fail**

```
uv run pytest tests/faces/test_query_faces.py tests/faces/test_config.py -q
```
Expected: `ModuleNotFoundError: No module named 'scalar_forensic.web.pipeline.faces_query'`
and `AttributeError: … face_query_max_faces`.

- [x] **Step 3: Add the config knob**

In `src/scalar_forensic/config.py`, beside the other `SFN_FACE_*` parses (the block ending
around line 229 with `face_thumb_size`):

```python
        self.face_query_max_faces = self._parse_int(
            "SFN_FACE_QUERY_MAX_FACES", default=25, min_value=1
        )
```
Match the exact helper name and signature used by the neighbouring `_parse_int` calls — read
lines 178-229 before writing this line.

- [x] **Step 4: Add the session types**

In `src/scalar_forensic/web/session.py`, above `FileEntry`:

```python
@dataclass
class QueryFace:
    """A face detected in an *uploaded* image.

    Session-scoped: it is never written to Qdrant and never written to the face
    chip store (spec §8 — the web analyze path is ephemeral by design).  A face
    that fails the embedding gate carries ``vector=None``; the review-only
    exclusion guarantee (spec §6.2) is structural on the query side too.
    """

    index: int
    bbox: tuple[float, float, float, float]
    landmarks: list[list[float]]
    det_conf: float
    detect_scale: float
    quality: dict[str, float | None]
    embedding_status: str
    embedding_exclusion_reason: str | None
    vector: list[float] | None
    review_jpeg: bytes | None
```

and add to `FileEntry`:

```python
    query_faces: list[QueryFace] | None = None
```

- [x] **Step 5: Implement `faces_query.py`**

```python
"""Session-scoped face detection for *uploaded* query images.

Deliberately separate from ``faces/indexing.py``: that module builds PointStructs
and writes chip files, and the query path must not be one argument away from
persisting anything.  Every pipeline *step* here is imported from
``scalar_forensic.faces.*`` — only the orchestration differs.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from scalar_forensic.config import Settings
from scalar_forensic.faces.align import align_face
from scalar_forensic.faces.chips import dilated_clamped_bbox
from scalar_forensic.faces.decode import load_for_detection
from scalar_forensic.faces.detect import YuNetDetector
from scalar_forensic.faces.embed import OnnxFaceEmbedder
from scalar_forensic.faces.provenance import PipelineConfig
from scalar_forensic.faces.quality import post_align_gate, pre_align_gate, review_gate
from scalar_forensic.web.session import QueryFace

_REVIEW_QUALITY = 95

_detector_cache: dict[str, YuNetDetector] = {}
_embedder_cache: dict[str, OnnxFaceEmbedder] = {}


def _detector(settings: Settings) -> YuNetDetector:
    key = f"{settings.face_detector_model}:{settings.face_detect_max_size}"
    if key not in _detector_cache:
        _detector_cache[key] = YuNetDetector(
            settings.face_detector_model, settings.face_detect_max_size
        )
    return _detector_cache[key]


def _embedder(settings: Settings) -> OnnxFaceEmbedder:
    key = str(settings.face_embedder_model)
    if key not in _embedder_cache:
        _embedder_cache[key] = OnnxFaceEmbedder(settings.face_embedder_model)
    return _embedder_cache[key]


@dataclass
class QueryFaceResult:
    faces: list[QueryFace]
    n_detected: int
    n_searchable: int
    n_review_only: int
    rejected: dict[str, int]
    truncated: bool
    cfg: PipelineConfig


def query_embedder_block(cfg: PipelineConfig) -> dict[str, object]:
    payload = cfg.to_payload()
    keys = (
        "embedder_model_name", "embedder_model_hash", "manifest_hash",
        "embedder_dim", "normalization_id", "alignment_version",
    )
    return {k: payload.get(k) for k in keys}


def detect_query_faces(
    data: bytes, settings: Settings, *, max_faces: int
) -> QueryFaceResult:
    img = load_for_detection(data)
    detector = _detector(settings)
    embedder = _embedder(settings)
    cfg = PipelineConfig.from_settings(settings, detector, embedder)

    dets = detector.detect(img)
    n_detected = len(dets)
    truncated = n_detected > max_faces
    dets = dets[:max_faces]

    faces: list[QueryFace] = []
    rejected: dict[str, int] = {}
    to_embed: list[tuple[int, np.ndarray]] = []
    h, w = img.shape[:2]

    for det in dets:
        rev = review_gate(
            det,
            min_conf=settings.face_review_min_conf,
            min_size=settings.face_review_min_size,
        )
        if not rev.passed:
            reason = rev.reason or "unknown"
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        pre = pre_align_gate(
            det,
            min_conf=settings.face_min_conf,
            min_size=settings.face_min_size,
            max_pose=settings.face_max_pose,
        )
        quality: dict[str, float | None] = {
            "confidence": pre.subscores.get("confidence"),
            "size": pre.subscores.get("size"),
            "pose": pre.subscores.get("pose"),
            "sharpness": None,
            "exposure": None,
        }

        x, y, cw, ch = dilated_clamped_bbox(
            det.bbox, settings.face_crop_dilation, w, h
        )
        crop = img[y : y + ch, x : x + cw]
        buf = io.BytesIO()
        Image.fromarray(crop).save(buf, format="JPEG", quality=_REVIEW_QUALITY)

        face = QueryFace(
            index=len(faces),
            bbox=tuple(float(v) for v in det.bbox),
            landmarks=[[float(a), float(b)] for a, b in det.landmarks],
            det_conf=float(det.confidence),
            detect_scale=float(det.detect_scale),
            quality=quality,
            embedding_status="review_only",
            embedding_exclusion_reason=pre.reason,
            vector=None,
            review_jpeg=buf.getvalue(),
        )
        faces.append(face)
        if not pre.passed:
            continue

        aligned = align_face(img, det.landmarks)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        post = post_align_gate(
            gray,
            min_sharpness=settings.face_min_sharpness,
            max_clipped=settings.face_max_clipped,
        )
        face.quality["sharpness"] = post.subscores.get("sharpness")
        face.quality["exposure"] = post.subscores.get("exposure")
        if not post.passed:
            face.embedding_exclusion_reason = post.reason
            continue
        to_embed.append((face.index, aligned))

    if to_embed:
        vecs = embedder.embed([a for _, a in to_embed])
        for (idx, _), vec in zip(to_embed, vecs, strict=True):
            faces[idx].vector = [float(v) for v in vec]
            faces[idx].embedding_status = "embedded"
            faces[idx].embedding_exclusion_reason = None

    return QueryFaceResult(
        faces=faces,
        n_detected=n_detected,
        n_searchable=sum(1 for f in faces if f.vector is not None),
        n_review_only=sum(1 for f in faces if f.vector is None),
        rejected=rejected,
        truncated=truncated,
        cfg=cfg,
    )
```

`PipelineConfig.from_settings(...)` may not exist under that name — read
`src/scalar_forensic/faces/provenance.py` and `faces/indexing.py:68-140` and construct
`PipelineConfig` the way `FacePipeline.from_settings` does. Do not invent a constructor.

- [x] **Step 6: Add endpoints 1 and 2 to `routes/faces.py`**

Handlers must be **sync `def`** (not `async def`) so Starlette runs the CPU-bound ONNX work in
its threadpool instead of blocking the event loop. Add near the existing helpers:

```python
def _entry_for(session, file_id: str):
    for e in session.files:
        if e.file_id == file_id:
            return e
    return None


def _require_faces(settings: Settings) -> None:
    if not settings.faces_enabled:
        raise HTTPException(503, "Face modality is disabled (set SFN_FACES_ENABLED=true).")
    err = settings.face_startup_error()
    if err:
        raise HTTPException(503, err)
```

Endpoint 1 caches its result on the entry (`entry.query_faces = result.faces`) so endpoint 2 and
Stage 2 can address faces by index, and re-detects when called again (the examiner may change
the selected video frame). Serialise exactly the field set in the contract — build the dict
explicitly; do **not** `dataclasses.asdict` the `QueryFace`, or the vector and JPEG bytes leak
into the response.

- [x] **Step 7: Re-export from the pipeline package**

In `src/scalar_forensic/web/pipeline/__init__.py`, import `detect_query_faces`,
`QueryFaceResult`, `query_embedder_block` from `.faces_query` and add all three to `__all__`
(keep the list alphabetised as it already is).

- [x] **Step 8: Run the tests**

```
uv run pytest tests/faces/ -q
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
```
Expected: all pass, ruff rc=0.

- [x] **Step 9: Commit**

```bash
git add src/scalar_forensic/web/pipeline/faces_query.py \
        src/scalar_forensic/web/pipeline/__init__.py \
        src/scalar_forensic/web/session.py \
        src/scalar_forensic/web/routes/faces.py \
        src/scalar_forensic/config.py \
        tests/faces/test_query_faces.py tests/faces/test_config.py
git commit -m "feat(faces): session-scoped face detection on the uploaded query image"
```

---

## Stage 2: Cross-file face search

Independently shippable: after this stage the search exists and is exercisable with `curl`,
before any UI consumes it.

**Files:**
- Create: `src/scalar_forensic/web/pipeline/faces_search.py`
- Create: `tests/faces/test_face_search.py`
- Modify: `src/scalar_forensic/faces/store.py` (add `search_faces`)
- Modify: `src/scalar_forensic/web/routes/faces.py` (endpoint 3 + `_audit_log`)
- Modify: `src/scalar_forensic/web/pipeline/__init__.py`
- Modify: `tests/faces/test_store.py` (the store method)
- Modify: `CLAUDE.md` (one gotcha line)

**Interfaces:**
- Consumes: `QueryFace.vector` from Stage 1; `FaceStore.check_compat(cfg) -> list[str]`;
  `faces.store.FACE_VECTOR_NAME`; `faces.audit.AuditLog.append(event_type, examiner_id, **fields)`.
- Produces:
  ```python
  # faces/store.py
  def search_faces(
      self, vector: list[float], *, limit: int, threshold: float, exact: bool
  ) -> list[dict]: ...        # [{"point_id": str, "score": float, **payload}]

  # web/pipeline/faces_search.py
  UNCALIBRATED_BANNER: str
  MODEL_REFERENCE_THRESHOLD: float = 0.363
  MODEL_REFERENCE_NOTE: str

  def search_query_faces(
      store, probes: list[tuple[int, list[float]]], *,
      limit: int, threshold: float, exact: bool, collapse: bool,
  ) -> list[dict]: ...        # hits in the endpoint-3 shape, score-desc
  ```

- [x] **Step 1: Write the failing tests**

Create `tests/faces/test_face_search.py`:

```python
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scalar_forensic.web.app import app
from scalar_forensic.web.pipeline.faces_search import (
    MODEL_REFERENCE_THRESHOLD,
    search_query_faces,
)

client = TestClient(app)


def _payload(image_hash, score, review="rev1", **over):
    d = {
        "point_id": f"pid-{image_hash}",
        "score": score,
        "image_hash": image_hash,
        "image_path": f"/evidence/{image_hash}.jpg",
        "observation_key": f"{image_hash}::1:2:3:4",
        "bbox": [1, 2, 3, 4],
        "det_conf": 0.9,
        "quality": 0.7,
        "review_chip_hash": review,
        "is_video_frame": False,
        "video_hash": None,
        "video_path": None,
        "frame_timecode_ms": None,
    }
    d.update(over)
    return d


def test_collapse_keeps_one_hit_per_image_hash_with_the_best_score():
    store = MagicMock()
    store.search_faces.return_value = [
        _payload("aaa", 0.91), _payload("aaa", 0.44), _payload("bbb", 0.60)
    ]
    hits = search_query_faces(
        store, [(0, [1.0, 0.0])], limit=10, threshold=0.0, exact=True, collapse=True
    )
    assert [h["image_hash"] for h in hits] == ["aaa", "bbb"]
    assert hits[0]["score"] == pytest.approx(0.91)
    assert hits[0]["n_collapsed"] == 2


def test_hits_carry_the_query_face_index_that_produced_them():
    store = MagicMock()
    store.search_faces.side_effect = [[_payload("aaa", 0.5)], [_payload("bbb", 0.8)]]
    hits = search_query_faces(
        store, [(0, [1.0, 0.0]), (3, [0.0, 1.0])],
        limit=10, threshold=0.0, exact=True, collapse=True,
    )
    by_hash = {h["image_hash"]: h for h in hits}
    assert by_hash["aaa"]["query_face_index"] == 0
    assert by_hash["bbb"]["query_face_index"] == 3


def test_search_is_exact_by_default_and_records_the_mode():
    store = MagicMock()
    store.search_faces.return_value = []
    search_query_faces(
        store, [(0, [1.0])], limit=5, threshold=0.2, exact=True, collapse=True
    )
    kwargs = store.search_faces.call_args.kwargs
    assert kwargs["exact"] is True
    assert kwargs["threshold"] == 0.2
    assert kwargs["limit"] == 5


def test_review_only_probe_is_refused_with_400(tmp_path):
    from scalar_forensic.web.session import QueryFace

    face = QueryFace(
        index=0, bbox=(1.0, 2.0, 3.0, 4.0), landmarks=[[1.0, 2.0]] * 5,
        det_conf=0.7, detect_scale=1.0,
        quality={"confidence": 0.7, "size": 40.0, "pose": None,
                 "sharpness": None, "exposure": None},
        embedding_status="review_only", embedding_exclusion_reason="size",
        vector=None, review_jpeg=b"\xff\xd8",
    )
    entry = MagicMock(query_faces=[face])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
    ):
        resp = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    assert resp.status_code == 400
    assert "review-only" in resp.json()["detail"]


def test_hard_compat_mismatch_refuses_the_search():
    store = MagicMock()
    store.check_compat.return_value = ["embedder_model_hash mismatch: abc != def"]
    entry = MagicMock(query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")])
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        resp = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    assert resp.status_code == 409


def test_response_labels_the_score_as_uncalibrated_and_scopes_0363():
    """The ruling: show the number, label it honestly, never claim 0.363 as ours."""
    store = MagicMock()
    store.check_compat.return_value = []
    store.search_faces.return_value = []
    entry = MagicMock(
        query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")]
    )
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        patch("scalar_forensic.web.routes.faces._audit_log"),
    ):
        body = client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        ).json()

    cal = body["calibration"]
    assert cal["status"] == "uncalibrated"
    assert cal["active_record"] is None
    assert "not evidential" in cal["banner"]
    assert cal["model_reference_threshold"] == MODEL_REFERENCE_THRESHOLD
    assert "not a threshold calibrated on this deployment" in cal["model_reference_note"]
    # 0.363 must never be the applied threshold
    assert body["threshold"] == 0.0


def test_every_search_is_written_to_the_face_audit_log():
    store = MagicMock()
    store.check_compat.return_value = []
    store.search_faces.return_value = []
    entry = MagicMock(
        file_hash="deadbeef",
        query_faces=[MagicMock(vector=[1.0], embedding_status="embedded")],
    )
    log = MagicMock()
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.get_session", return_value=MagicMock()),
        patch("scalar_forensic.web.routes.faces._entry_for", return_value=entry),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
        patch("scalar_forensic.web.routes.faces._audit_log", return_value=log),
    ):
        client.post(
            "/api/faces/search",
            data={"session_id": "s", "file_id": "f", "face_indices": "0"},
        )
    log.append.assert_called_once()
    assert log.append.call_args.args[0] == "query"
    fields = log.append.call_args.kwargs
    assert fields["search_mode"] == "exact"
    assert fields["face_calibration_id"] is None
    assert "n_results" in fields and "probe_hash" in fields
```

Add to `tests/faces/test_store.py` (uses the existing `store` fixture at
`tests/faces/test_store.py:37-41`):

```python
def test_search_faces_uses_the_named_face_vector_and_exact_mode(store):
    st, client = store
    client.query_points.return_value.points = []
    st.search_faces([1.0, 0.0], limit=7, threshold=0.3, exact=True)
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "case1_faces"
    assert kwargs["using"] == "face"
    assert kwargs["limit"] == 7
    assert kwargs["score_threshold"] == 0.3
    assert kwargs["search_params"].exact is True
```

- [x] **Step 2: Run them and confirm they fail**

```
uv run pytest tests/faces/test_face_search.py tests/faces/test_store.py -q
```
Expected: `ModuleNotFoundError … faces_search`, `AttributeError: 'FaceStore' object has no
attribute 'search_faces'`, and 404s from the missing route.

- [x] **Step 3: Add `FaceStore.search_faces`**

In `src/scalar_forensic/faces/store.py`, in the **reads** section (after `list_faces`, around
line 320):

```python
    def search_faces(
        self, vector: list[float], *, limit: int, threshold: float, exact: bool
    ) -> list[dict]:
        """kNN over the named ``face`` vector.

        Review-only observations carry no vector at all (see ``clear_face_vector``
        and ``indexing.py``), so they are structurally unreachable here.  This is
        the guarantee — do **not** add a payload filter to "make sure"; a filter
        that a later query forgets is exactly the failure mode the vectorless
        design avoids.
        """
        res = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=FACE_VECTOR_NAME,
            limit=limit,
            score_threshold=threshold,
            search_params=SearchParams(exact=exact),
            with_payload=True,
        )
        return [
            {"point_id": str(p.id), "score": float(p.score), **(p.payload or {})}
            for p in res.points
        ]
```
Import `SearchParams` from `qdrant_client.models` alongside the existing model imports at the
top of `store.py`.

- [x] **Step 4: Implement `faces_search.py`**

```python
"""Cross-file face search (spec §8, Phase 1b).

The score this returns is a **raw cosine under a named model**.  There is no
face-calibration record in this deployment (spec §10) and the maintainer ruled on
2026-08-12 that the number is shown anyway, labelled for what it is.  The
constants below are the only place that copy lives.
"""

from __future__ import annotations

UNCALIBRATED_BANNER = (
    "Uncalibrated — not evidential. The number below is the raw cosine similarity "
    "this model produced. There is no confidence interval and no calibrated "
    "threshold for this deployment."
)

MODEL_REFERENCE_THRESHOLD = 0.363

MODEL_REFERENCE_NOTE = (
    "0.363 is the SFace authors' published same/different reference figure for this "
    "model. It is not a threshold calibrated on this deployment's material and must "
    "not be read as one."
)


def calibration_block() -> dict[str, object]:
    return {
        "status": "uncalibrated",
        "active_record": None,
        "banner": UNCALIBRATED_BANNER,
        "model_reference_threshold": MODEL_REFERENCE_THRESHOLD,
        "model_reference_note": MODEL_REFERENCE_NOTE,
    }


def search_query_faces(
    store,
    probes: list[tuple[int, list[float]]],
    *,
    limit: int,
    threshold: float,
    exact: bool,
    collapse: bool,
) -> list[dict]:
    """One kNN per probe, merged.  No aggregation across observations (spec §8)."""
    raw: list[dict] = []
    for face_index, vector in probes:
        for row in store.search_faces(
            vector, limit=limit, threshold=threshold, exact=exact
        ):
            raw.append(_hit(row, face_index))

    raw.sort(key=lambda h: -h["score"])
    if not collapse:
        return raw

    best: dict[str, dict] = {}
    for hit in raw:
        seen = best.get(hit["image_hash"])
        if seen is None:
            best[hit["image_hash"]] = hit
        else:
            seen["n_collapsed"] += 1
    return sorted(best.values(), key=lambda h: -h["score"])


def _hit(row: dict, face_index: int) -> dict:
    review = row.get("review_chip_hash")
    return {
        "image_hash": row.get("image_hash"),
        "image_path": row.get("image_path"),
        "score": row["score"],
        "query_face_index": face_index,
        "face": {
            "point_id": row["point_id"],
            "observation_key": row.get("observation_key"),
            "bbox": row.get("bbox"),
            "det_conf": row.get("det_conf"),
            "quality": row.get("quality"),
            "review_url": f"/api/faces/chip/{review}/review" if review else None,
            "thumb_url": f"/api/faces/chip/{review}/thumb" if review else None,
        },
        "is_video_frame": row.get("frame_timecode_ms") is not None,
        "video_hash": row.get("video_hash"),
        "video_path": row.get("video_path"),
        "frame_timecode_ms": row.get("frame_timecode_ms"),
        "n_collapsed": 1,
    }
```

- [x] **Step 5: Implement endpoint 3 and `_audit_log`**

In `routes/faces.py`:

```python
def _audit_log(settings: Settings) -> AuditLog:
    audit_dir = (
        settings.face_store_dir.parent if settings.face_store_dir else Path("data")
    )
    return AuditLog(audit_dir / "face_audit.log")
```
(This mirrors `faces/indexing.py:103-109` and `cli.py:1940-1942`. It is the **first** web route
that writes the audit log — spec §11 requires every face query to be logged.)

The handler, sync `def`, in order:
1. `settings = Settings()`; `_require_faces(settings)`.
2. Resolve session/entry → 404s.
3. Parse `face_indices` → refuse unknown indices (400) and refuse any face whose `vector is
   None` with the message `f"face {i} is review-only and has no vector; it cannot be searched"`.
4. Build the store (`QdrantClient` + `FaceStore` as `_store()` already does); read the
   `PipelineConfig` used for the query faces from `entry.query_faces`' producing run — recompute
   it via `detect_query_faces`' `cfg` if it was cached on the entry, otherwise rebuild it from
   `Settings`. **`check_compat(cfg)`**: if any returned message names a `_HARD_FIELDS` key,
   raise `HTTPException(409, "; ".join(msgs))`; otherwise carry them as
   `{"ok": True, "warnings": msgs}`.
5. `search_query_faces(...)`.
6. `_audit_log(settings).append("query", settings.examiner_id, probe_hash=entry.file_hash,
   face_indices=idxs, collection=settings.face_collection,
   pipeline_config_hash=cfg.config_hash(), face_calibration_id=None,
   search_mode="exact" if exact else "ann", threshold=threshold, limit=limit,
   n_results=len(hits), top_scores=[h["score"] for h in hits[:5]])`.
   Read `faces/provenance.py` for the actual `config_hash` accessor name before writing this.
7. Return the endpoint-3 JSON, with `calibration_block()` and the embedder block from Stage 1.

- [x] **Step 6: Run the tests**

```
uv run pytest tests/faces/ -q
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
```

- [x] **Step 7: Run the integration test against a throwaway Qdrant**

CLAUDE.md: `tests/faces/test_store_integration.py` is the only test that can observe the
exclusion guarantee against a real store, and this stage adds the first production search path.

```
SFN_TEST_QDRANT_URL=http://<qdrant>:6333 uv run pytest tests/faces/test_store_integration.py -q
```
Expected: passes, 0 skipped. If it skips, say so in the commit message rather than claiming the
guarantee was verified.

- [x] **Step 8: Record the new gotcha in CLAUDE.md**

Claim it first (`cx o CLAUDE.md`), then add one line under **Gotchas**:

```
- Face search is uncalibrated by deliberate ruling (2026-08-12): the raw cosine is displayed and
  labelled as such. SFace's 0.363 is the model authors' reference figure only — never a default,
  a filter or a deployment threshold. See `docs/specs/face-pipeline.md` §10.
```

- [x] **Step 9: Commit**

```bash
git add src/scalar_forensic/web/pipeline/faces_search.py \
        src/scalar_forensic/web/pipeline/__init__.py \
        src/scalar_forensic/faces/store.py \
        src/scalar_forensic/web/routes/faces.py \
        tests/faces/test_face_search.py tests/faces/test_store.py CLAUDE.md
git commit -m "feat(faces): cross-file face search endpoint with query logging"
```

---

## Stage 3: Query-side face strip with selection and green borders

Independently shippable: the examiner sees and selects faces on the left; nothing is searched yet.

**Files:**
- Modify: `src/scalar_forensic/web/static/js/state.js`
- Modify: `src/scalar_forensic/web/static/js/faces.js`
- Modify: `src/scalar_forensic/web/static/index.html` (Panel 2, query image, after line 787)
- Modify: `src/scalar_forensic/web/static/style.css`
- Modify: `tests/faces/test_static_wiring.py`

**Interfaces:**
- Consumes: `POST /api/faces/query-faces`, `GET /api/faces/query-chip/…` (Stage 1).
- Produces (all in `faces.js`, merged into `sfn()`): `loadQueryFaces()`,
  `toggleQueryFace(face)`, `queryFaceSelected(index)`, `queryFaceChipUrl(index)`,
  `queryFaceStatusLabel(face)`, `selectAllQueryFaces()`, `clearQueryFaceSelection()`.
  State: `queryFaces`, `queryFacesLoading`, `queryFacesError`, `selectedQueryFaceIndices`,
  `queryFacesTruncated`.

Face UI code goes in **`faces.js`** (same modality, already loaded before `app.js` at
`index.html:1872` — no new script tag, no ordering risk).

- [ ] **Step 1: Write the failing tests**

Append to `tests/faces/test_static_wiring.py`:

```python
def test_query_face_strip_selects_only_searchable_faces():
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="query-faces"')
    block = html[start : start + 2500]
    assert "toggleQueryFace(" in block
    assert "face.searchable" in block          # review-only chips are not selectable
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
```

- [ ] **Step 2: Run and confirm failure**

```
uv run pytest tests/faces/test_static_wiring.py -q
```
Expected: `ValueError: substring not found`.

- [ ] **Step 3: Add state**

In `state.js`, inside the existing `// ── Face modality …` block (lines 19-27):

```js
    // Query-side faces (session-scoped; never indexed)
    queryFaces: [],
    queryFacesLoading: false,
    queryFacesError: '',
    queryFacesTruncated: false,
    selectedQueryFaceIndices: [],
```

- [ ] **Step 4: Add the functions to `faces.js`**

```js
    // ── Query-side faces (session-scoped, never persisted) ────────────────
    async loadQueryFaces() {
      if (!this.facesAvailable || !this.sessionId || !this.selectedFileId) return;
      this.queryFacesLoading = true;
      this.queryFacesError = '';
      this.queryFaces = [];
      this.selectedQueryFaceIndices = [];
      try {
        const fd = new FormData();
        fd.append('session_id', this.sessionId);
        fd.append('file_id', this.selectedFileId);
        const resp = await fetch('/api/faces/query-faces', { method: 'POST', body: fd });
        const body = await resp.json();
        if (!resp.ok) { this.queryFacesError = body.detail || 'face detection failed'; return; }
        this.queryFaces = body.faces || [];
        this.queryFacesTruncated = body.truncated === true;
        // Pre-select every searchable face: the examiner de-selects, rather than
        // starting from an empty selection that looks like "no faces found".
        this.selectedQueryFaceIndices = this.queryFaces
          .filter(f => f.searchable).map(f => f.index);
      } catch (e) {
        this.queryFacesError = String(e);
      } finally {
        this.queryFacesLoading = false;
      }
    },

    queryFaceChipUrl(index) {
      return `/api/faces/query-chip/${this.sessionId}/${this.selectedFileId}/${index}`;
    },

    queryFaceSelected(index) {
      return this.selectedQueryFaceIndices.includes(index);
    },

    toggleQueryFace(face) {
      if (!face.searchable) return;   // review-only faces have no vector
      const i = this.selectedQueryFaceIndices.indexOf(face.index);
      if (i === -1) this.selectedQueryFaceIndices.push(face.index);
      else this.selectedQueryFaceIndices.splice(i, 1);
    },

    selectAllQueryFaces() {
      this.selectedQueryFaceIndices = this.queryFaces
        .filter(f => f.searchable).map(f => f.index);
    },

    clearQueryFaceSelection() {
      this.selectedQueryFaceIndices = [];
    },

    queryFaceStatusLabel(face) {
      if (face.searchable) return 'searchable';
      const why = face.embedding_exclusion_reason;
      return why ? `not searchable — ${why} below threshold` : 'not searchable';
    },
```

- [ ] **Step 5: Add the markup**

In `index.html`, immediately after the query `image-box` (closes at line 787) and before the
video frame navigation at 789:

```html
          <!-- Query-side faces.  Session-scoped: detected on upload, never indexed. -->
          <div class="query-faces" x-show="facesAvailable && queryFaces.length"
               x-init="$watch('selectedFileId', () => loadQueryFaces())">
            <div class="query-faces-head">
              <span x-text="`${queryFaces.length} face observations in this image`"></span>
              <button class="btn-tiny" @click="selectAllQueryFaces()">All</button>
              <button class="btn-tiny" @click="clearQueryFaceSelection()">None</button>
            </div>
            <div class="query-faces-grid">
              <template x-for="face in queryFaces" :key="`qf-${face.index}`">
                <div class="query-face-chip"
                     :class="{
                       'query-face-chip-selected': queryFaceSelected(face.index),
                       'query-face-chip-unsearchable': !face.searchable,
                     }"
                     @click="toggleQueryFace(face)">
                  <img :src="queryFaceChipUrl(face.index)" alt="">
                  <span class="query-face-status" x-text="queryFaceStatusLabel(face)"></span>
                </div>
              </template>
            </div>
            <p class="faces-caveat" x-show="queryFacesTruncated">
              Detection was capped; not every face in this image is shown.
            </p>
          </div>
```

Note the `x-init` runs `loadQueryFaces()` only via the watcher. **Also call it once**, the way
`index.html:936-939` had to be fixed on 2026-08-12 (runbook: "the watcher fires only on
*change*"): write it as
`x-init="loadQueryFaces(); $watch('selectedFileId', () => loadQueryFaces())"`.

- [ ] **Step 6: Add the CSS**

Next to the existing face-chip block (`style.css:2248-2310`):

```css
.query-faces { margin-top: 6px; }
.query-faces-head { display: flex; gap: 6px; align-items: center;
                    font-size: 11px; color: var(--muted); margin-bottom: 4px; }
.query-faces-grid { display: flex; gap: 6px; flex-wrap: wrap; }
.query-face-chip { border: 2px solid transparent; border-radius: 3px;
                   padding: 2px; cursor: pointer; text-align: center; }
.query-face-chip img { display: block; max-width: 72px; width: auto; object-fit: contain; }
.query-face-chip-selected { border-color: var(--success); }
.query-face-chip-unsearchable { cursor: not-allowed; opacity: 0.65; }
.query-face-status { display: block; font-size: 10px; color: var(--muted); }
```
`--success` is `#2ecc71` (`style.css:53`) — the same green already used for `.hit-card.mark-pos`.
Read the surrounding block and match its spacing/variable conventions rather than pasting blind.

- [ ] **Step 7: Run tests, then look at the page**

```
uv run pytest tests/faces/ -q
uv run ruff check src tests scripts
node --check src/scalar_forensic/web/static/js/faces.js
```
Then `./run.sh sfn-web` and open **`http://localhost:8080/?cachebust=1`** — plain `/` will serve
a stale page. Re-set the `style.css` href too; the query string does not bust it.

- [ ] **Step 8: Commit**

```bash
git add src/scalar_forensic/web/static/js/state.js \
        src/scalar_forensic/web/static/js/faces.js \
        src/scalar_forensic/web/static/index.html \
        src/scalar_forensic/web/static/style.css \
        tests/faces/test_static_wiring.py
git commit -m "feat(web): selectable query-side faces with a green selection border"
```

---

## Stage 4: FACES hits in the hit list, mode filter, score under the matched face

**Files:**
- Modify: `src/scalar_forensic/web/static/js/faces.js` (`runFaceSearch`, match-score map)
- Modify: `src/scalar_forensic/web/static/js/state.js`
- Modify: `src/scalar_forensic/web/static/js/computed.js` (`mergedHits`; `filteredHits` at 45-58)
- Modify: `src/scalar_forensic/web/static/index.html` (badge slot 4, FACES pill, banner,
  matched-face border + score in the Best Match face grid at 944-961)
- Modify: `src/scalar_forensic/web/static/style.css`
- Modify: `tests/faces/test_static_wiring.py`

**Interfaces:**
- Consumes: `POST /api/faces/search` (Stage 2); `selectedQueryFaceIndices` (Stage 3);
  the existing `facesForHit` array (`faces.js:21`).
- Produces: `runFaceSearch()`; state `faceHits`, `faceSearchLoading`, `faceSearchError`,
  `faceCalibration`, `faceMatchScores` (`{point_id: score}`), `hitsFilterFaces`;
  computed getter `mergedHits`; helpers `faceMatchScore(face)`, `faceIsMatched(face)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/faces/test_static_wiring.py`:

```python
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
```

- [ ] **Step 2: Run and confirm failure**

```
uv run pytest tests/faces/test_static_wiring.py -q
```
Expected: `ValueError: substring not found` / `AssertionError`.

- [ ] **Step 3: State**

Add to `state.js` in the face block:

```js
    faceHits: [],
    faceSearchLoading: false,
    faceSearchError: '',
    faceCalibration: null,
    faceMatchScores: {},        // {point_id: raw cosine}
    hitsFilterFaces: true,
    faceLimit: 10,
    faceThreshold: 0.0,
    faceExactSearch: true,
```

- [ ] **Step 4: `runFaceSearch()` in `faces.js`**

```js
    async runFaceSearch() {
      if (!this.facesAvailable || !this.selectedQueryFaceIndices.length) {
        this.faceHits = []; this.faceMatchScores = {}; return;
      }
      this.faceSearchLoading = true;
      this.faceSearchError = '';
      try {
        const fd = new FormData();
        fd.append('session_id', this.sessionId);
        fd.append('file_id', this.selectedFileId);
        fd.append('face_indices', this.selectedQueryFaceIndices.join(','));
        fd.append('limit', this.faceLimit);
        fd.append('threshold', this.faceThreshold);
        fd.append('exact', this.faceExactSearch ? 'true' : 'false');
        const resp = await fetch('/api/faces/search', { method: 'POST', body: fd });
        const body = await resp.json();
        if (!resp.ok) { this.faceSearchError = body.detail || 'face search failed'; return; }
        this.faceHits = body.hits || [];
        this.faceCalibration = body.calibration || null;
        const scores = {};
        for (const h of this.faceHits) scores[h.face.point_id] = h.score;
        this.faceMatchScores = scores;
      } catch (e) {
        this.faceSearchError = String(e);
      } finally {
        this.faceSearchLoading = false;
      }
    },

    faceIsMatched(face) {
      return Object.prototype.hasOwnProperty.call(this.faceMatchScores, face.id);
    },

    faceMatchScore(face) {
      const s = this.faceMatchScores[face.id];
      return typeof s === 'number' ? s.toFixed(4) : '';
    },
```
`face.id` is what `/api/faces/by-image/` returns (`routes/faces.py:86-97` sets `{"id": rec.id,
…}`), and `hit.face.point_id` is the same id stringified — compare as strings.

- [ ] **Step 5: `mergedHits` in `computed.js`, and the `filteredHits` change**

Add above `filteredHits` (currently `computed.js:45-58`):

```js
    get mergedHits() {
      // Face hits arrive from a second endpoint; merge them into the image-mode
      // hit list by image_hash so one row is one medium across all four modes.
      const base = this.selectedFile ? this.selectedFile.hits : [];
      if (!this.faceHits.length) return base;
      const byHash = new Map();
      for (const fh of this.faceHits) byHash.set(fh.image_hash, fh);
      const merged = base.map(h => {
        const fh = byHash.get(h.image_hash);
        if (!fh) return h;
        byHash.delete(h.image_hash);
        return { ...h, scores: { ...h.scores, faces: fh.score }, face_match: fh.face };
      });
      // Media found only by face, appended score-desc.
      const faceOnly = [...byHash.values()]
        .sort((a, b) => b.score - a.score)
        .map(fh => ({
          path: fh.image_path, image_hash: fh.image_hash,
          scores: { faces: fh.score }, face_match: fh.face,
          is_video_frame: fh.is_video_frame, video_hash: fh.video_hash,
          video_path: fh.video_path, frame_timecode_ms: fh.frame_timecode_ms,
          exif: null, exif_geo_data: null, model_provenance: null,
          matched_frames: null, query_timecodes: null,
          best_query_timecode_ms: null, is_reference: false,
        }));
      return merged.concat(faceOnly);
    },
```

Then change `filteredHits` — it currently reads `this.selectedFile.hits` at line 47 and has three
`return true` lines at 49-51. Two edits, nothing else:

```js
      let hits = this.mergedHits.filter(h => {
```
and add a fourth clause after the `semantic` one:
```js
        if ('faces' in h.scores && this.hitsFilterFaces) return true;
```

Face-only rows have no `path`-derived tag/mark state; verify `hitKey(hit)` (`helpers.js`) does
not blow up on them before shipping — read `helpers.js:hitKey` and, if it keys on a field that
face-only rows lack, populate that field in the `faceOnly` mapping instead of changing `hitKey`.

- [ ] **Step 6: Markup**

Badge slot 4, after the SEMAN group (`index.html:1253-1263`), matching the fixed-slot pattern
exactly:

```html
                <!-- Fixed slot 4: reserved when the face modality is available -->
                <div class="pill-group"
                     x-show="facesAvailable"
                     :style="'faces' in hit.scores && hitsFilterFaces ? '' : 'visibility:hidden'">
                  <span class="badge badge-faces">FACES</span>
                  <span class="pill-score pill-score-faces"
                        x-text="'faces' in hit.scores ? hit.scores.faces.toFixed(4) : ''"></span>
                </div>
```

Filter pill, after the SEMANTIC pill (`index.html:1188-1193`):

```html
        <template x-if="facesAvailable">
          <button class="hit-filter-pill hit-filter-pill-faces"
                  :class="{off: !hitsFilterFaces}"
                  @click="hitsFilterFaces = !hitsFilterFaces; selectedHitKey = null; matchSrc = null">FACES</button>
        </template>
```
Also relax the `x-show` on the `hits-filters` container (`index.html:1180-1181`) to include
`|| facesAvailable`, or the pill row stays hidden in a dino-only collection.

Matched face + score, inside the Best Match face grid (`index.html:944-961`) — add to the
existing `:class` on `.face-chip` and add one span:

```html
                  'face-chip-matched': faceIsMatched(face),
```
```html
                    <span class="face-chip-score" x-show="faceIsMatched(face)"
                          x-text="`cosine ${faceMatchScore(face)}`"></span>
```

Banner, immediately above the face-hit-bearing hit list (put it in the hits panel header area,
`index.html:1171-1176`):

```html
      <div class="faces-uncal-banner" x-show="faceCalibration">
        <span x-text="faceCalibration?.banner"></span>
        <span class="faces-uncal-note" x-text="faceCalibration?.model_reference_note"></span>
      </div>
```

- [ ] **Step 7: CSS**

```css
.badge-faces { background: rgba(46, 204, 113, 0.12); color: var(--success); }
.pill-score-faces { color: var(--success); }
.hit-filter-pill-faces { border-color: var(--success); }
.face-chip-matched { border: 2px solid var(--success); border-radius: 3px; }
.face-chip-score { display: block; font-size: 10px; color: var(--success); }
.faces-uncal-banner { font-size: 11px; padding: 4px 6px; margin-bottom: 6px;
                      border-left: 3px solid var(--danger); color: var(--muted); }
.faces-uncal-note { display: block; margin-top: 2px; }
```
Match the surrounding declaration style at `style.css:902-915` (badges) and `2233-2364` (faces).

- [ ] **Step 8: Run and inspect**

```
uv run pytest tests/faces/ -q
node --check src/scalar_forensic/web/static/js/computed.js
node --check src/scalar_forensic/web/static/js/faces.js
uv run ruff check src tests scripts
```
Then the live check at `/?cachebust=2`: select a query face, confirm a FACES row appears, the
badge shows the raw cosine to 4 dp, the banner is visible, and the matched face in the Best
Match panel carries a green border with the score beneath it.

- [ ] **Step 9: Commit**

```bash
git add src/scalar_forensic/web/static/js/ src/scalar_forensic/web/static/index.html \
        src/scalar_forensic/web/static/style.css tests/faces/test_static_wiring.py
git commit -m "feat(web): FACES hit mode, filter pill and matched-face score"
```

---

## Stage 5: Face query controls beside TOP K and the DINOv2 threshold

**Files:**
- Modify: `src/scalar_forensic/web/static/index.html` (`.sliders`, lines 603-625)
- Modify: `src/scalar_forensic/web/static/js/faces.js` (`debouncedFaceQuery`)
- Modify: `tests/faces/test_static_wiring.py`

**Interfaces:** consumes `faceLimit`, `faceThreshold`, `faceExactSearch` (Stage 4 state) and
`runFaceSearch()`.

- [ ] **Step 1: Write the failing test**

```python
def test_face_query_controls_sit_with_the_other_sliders():
    html = (STATIC / "index.html").read_text()
    start = html.index('<div class="sliders">')
    block = html[start : html.index("</div>", html.index("faceThreshold")) + 6]
    assert "faceLimit" in block
    assert "faceThreshold" in block
    assert "faceExactSearch" in block
    assert "0.363" not in block          # never a default in the UI controls
    assert "uncalibrated" in block.lower()
```

- [ ] **Step 2: Run and confirm failure**

```
uv run pytest tests/faces/test_static_wiring.py::test_face_query_controls_sit_with_the_other_sliders -q
```

- [ ] **Step 3: Add the controls**

Inside `.sliders`, after the DINOv2 threshold row (`index.html:617-624`):

```html
        <div class="slider-row" x-show="facesAvailable">
          <div class="slider-label">
            <span title="Maximum face observations returned per selected query face">Face Top K</span>
            <span class="slider-value" x-text="faceLimit"></span>
          </div>
          <input type="range" min="1" max="50" step="1" x-model.number="faceLimit"
                 @input="debouncedFaceQuery()">
        </div>
        <div class="slider-row" x-show="facesAvailable">
          <div class="slider-label">
            <span title="Display floor on the raw cosine. Uncalibrated: this is not a decision threshold.">Face score floor (raw cosine, uncalibrated)</span>
            <span class="slider-value" x-text="faceThreshold.toFixed(2)"></span>
          </div>
          <input type="range" min="0" max="1" step="0.01" x-model.number="faceThreshold"
                 @input="debouncedFaceQuery()">
        </div>
        <div class="slider-row" x-show="facesAvailable">
          <label class="slider-label">
            <input type="checkbox" x-model="faceExactSearch" @change="runFaceSearch()">
            <span title="Exact kNN. Turning this off uses the approximate index; the mode is recorded in the query log.">Exact face search</span>
          </label>
        </div>
```
`faceThreshold` default stays **0.0**. Do not seed it with 0.363 (Global Constraint 2).

- [ ] **Step 4: Debounce in `faces.js`**

Mirror `analysis.js:97-99`:

```js
    debouncedFaceQuery() {
      clearTimeout(this._faceQueryTimer);
      this._faceQueryTimer = setTimeout(() => this.runFaceSearch(), 300);
    },
```
and add `_faceQueryTimer: null,` to the `state.js` face block.

- [ ] **Step 5: Run and inspect**

```
uv run pytest tests/faces/ -q && node --check src/scalar_forensic/web/static/js/faces.js
```
Live at `/?cachebust=3`: dragging Face Top K re-runs the face search only; the DINO sliders are
unaffected.

- [ ] **Step 6: Commit**

```bash
git add src/scalar_forensic/web/static/index.html \
        src/scalar_forensic/web/static/js/faces.js \
        src/scalar_forensic/web/static/js/state.js \
        tests/faces/test_static_wiring.py
git commit -m "feat(web): face query controls beside Top K and the DINOv2 threshold"
```

---

## Stage 6: Record the divergence in the spec

Docs only. Ship it in the same release as Stage 2 — the ruling says the code must not silently
contradict the spec.

**Files:**
- Modify: `docs/specs/face-pipeline.md` §10 (claim with `cx o` first)
- Modify: `docs/face-matching-math.md` §6
- Modify: `tests/faces/test_static_wiring.py` (a docs assertion; see below)

- [ ] **Step 1: Write the failing test**

A test is warranted here because the divergence note is the artefact a reviewer will look for.

```python
def test_spec_records_the_uncalibrated_search_divergence():
    spec = Path("docs/specs/face-pipeline.md").read_text()
    s10 = spec[spec.index("## 10. Calibration") : spec.index("## 11.")]
    assert "DEPLOYMENT DIVERGENCE" in s10
    assert "2026-08-12" in s10
    assert "the record wins" in s10
```

- [ ] **Step 2: Run and confirm failure**

```
uv run pytest tests/faces/test_static_wiring.py::test_spec_records_the_uncalibrated_search_divergence -q
```

- [ ] **Step 3: Append to `docs/specs/face-pipeline.md` §10**

Insert immediately before the `---` that closes §10 (currently line 670):

```markdown
**DEPLOYMENT DIVERGENCE — 2026-08-12, by the maintainer's ruling.** This deployment ships
Phase 1b face search **without** an active face-calibration record, and **displays the raw
pairwise cosine**. The gate above was not met and was deliberately waived; the ruling is
recorded verbatim in `docs/specs/face-query-ux.md`. What holds instead:

- The score is presented as a raw model output, labelled as having **no confidence interval and
  no calibrated threshold**. No FMR/FPIR tier is claimed anywhere in the UI or the docs.
- The model authors' published reference figure (SFace, 0.363) may be shown, always attributed
  to the model authors and never as this deployment's threshold. It is not a default of any
  control and is applied nowhere in the search path.
- The "uncalibrated — not evidential" banner and the §11 opt-in enablement gate remain in force.
- §10.4's precedence rule is unchanged: when a record exists, **the record wins**, and the UI
  copy above is replaced by the record's T, CI and provenance.
- Nothing in §6.2's exclusion guarantee is relaxed: review-only observations remain vectorless
  and unreachable by search, on the index side and on the query side alike.

**Also diverging from §8/§12:** uploaded probes, listed as Phase 2, are pulled into Phase 1b
because the maintainer's query flow is upload-driven. The §8 constraint they carry is kept in
full — query-side faces are session-scoped, written to no collection and to no chip store.
```

- [ ] **Step 4: Append to `docs/face-matching-math.md` §6 ("What is NOT claimed")**

```markdown
- **No calibrated threshold, and no error rate, is claimed for face search in this deployment.**
  The cosine shown beneath a matched face is the raw model output. The number that would tell an
  examiner what it means — a face-calibration record with stated pair counts and confidence
  intervals (spec §10) — does not exist here. SFace's published 0.363 same/different figure is
  the model authors' measurement on their material, not a measurement on this operator's.
```

- [ ] **Step 5: Run the test and commit**

```bash
uv run pytest tests/faces/test_static_wiring.py -q
git add docs/specs/face-pipeline.md docs/face-matching-math.md tests/faces/test_static_wiring.py
git commit -m "docs(spec): record the deliberate divergence from the calibration gate"
```

---

## Stage 7: Per-model explainer buttons — DINO pair renamed, face pair added

Independent of the ruling: this surface reports what the pipeline *did*.

**Files:**
- Create: `src/scalar_forensic/web/pipeline/faces_stats.py`
- Create: `tests/faces/test_face_audit_stats.py`
- Modify: `src/scalar_forensic/web/routes/faces.py` (endpoints 4 and 5)
- Modify: `src/scalar_forensic/web/pipeline/__init__.py`
- Modify: `src/scalar_forensic/web/static/index.html` (buttons 754-763; two new modals)
- Modify: `src/scalar_forensic/web/static/js/faces.js`, `state.js`, `style.css`
- Modify: `tests/faces/test_static_wiring.py`

**Interfaces:**
- Consumes: `FaceStore.search_faces` (Stage 2); `FaceStore.get_marker`,
  `FaceStore.list_faces`, `FaceStore.check_compat`; `QueryFace.vector` (Stage 1).
- Produces:
  ```python
  # web/pipeline/faces_stats.py
  @dataclass
  class FaceScoreStats:      # same field set as web/pipeline/stats.py SemanticStats
      sample_size: int; count: int
      min_score: float; p10: float; p25: float; median: float
      p75: float; p90: float; max_score: float
      mean: float; stdev: float; histogram: list[int]

  def face_score_stats(store, vector: list[float], *, sample_size: int) -> FaceScoreStats: ...
  def face_audit(store, image_hash: str, cfg) -> dict: ...
  ```

**Naming decision, and the honesty problem in it.** The maintainer asked for `DINO Dist Stats` /
`DINO Audit`. `Dist Stats` is DINOv2-only and the rename is exactly right. **The Audit modal is
not**: `/api/hit-provenance` returns both `semantic` (DINOv2) and `altered` (SSCD) provenance
(`web/pipeline/provenance.py:353-356, 374-409`), so a bare "DINO Audit" under-describes it. Ship
the requested label and add a one-line subtitle in the modal naming both models — this satisfies
the instruction without printing a false scope. Flagged to the maintainer as a DECIDE; if they
prefer `Image Audit (DINOv2 + SSCD)`, it is a two-string change.

- [ ] **Step 1: Write the failing tests**

Create `tests/faces/test_face_audit_stats.py`:

```python
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scalar_forensic.web.app import app
from scalar_forensic.web.pipeline.faces_stats import face_score_stats

client = TestClient(app)


def test_face_score_stats_has_the_same_field_set_as_the_dino_stats():
    from scalar_forensic.web.pipeline.stats import SemanticStats

    store = MagicMock()
    store.search_faces.return_value = [
        {"point_id": f"p{i}", "score": s} for i, s in enumerate([0.1, 0.2, 0.6, 0.9])
    ]
    stats = face_score_stats(store, [1.0, 0.0], sample_size=100)
    assert set(vars(stats)) == set(SemanticStats.__dataclass_fields__)
    assert len(stats.histogram) == 20
    assert stats.max_score == pytest.approx(0.9)
    assert stats.count == 4


def test_face_score_stats_are_computed_with_exact_search():
    store = MagicMock()
    store.search_faces.return_value = [{"point_id": "p", "score": 0.5}]
    face_score_stats(store, [1.0], sample_size=50)
    assert store.search_faces.call_args.kwargs["exact"] is True
    assert store.search_faces.call_args.kwargs["threshold"] == 0.0


def test_face_audit_reports_index_time_thresholds_not_current_env():
    store = MagicMock()
    store.get_marker.return_value = {
        "n_detected": 6, "n_kept": 2, "n_review_only": 3,
        "review_only_reasons": {"size": 3}, "rejected": {"confidence": 1},
    }
    store.list_faces.return_value = [
        {"id": "p1", "min_conf": 0.8, "min_size": 64, "review_min_size": 36,
         "detector_id": "yunet", "detector_model_hash": "d1",
         "embedder_model_hash": "e1", "manifest_hash": "m1", "embedder_dim": 128,
         "alignment_version": "arcface-112-v1", "normalization_id": "affine-0.0-1.0",
         "pipeline_config_hash": "c1", "cv2_version": "4.10", "ort_version": "1.19",
         "sfn_version": "0.1", "crop_dilation": 0.25, "max_pose": 0.35,
         "min_sharpness": 25.0, "max_clipped": 0.6, "review_min_conf": 0.6,
         "detector_score_threshold": 0.5, "detect_max_size": 1600,
         "embedder_model_name": "sface"}
    ]
    store.check_compat.return_value = []
    with (
        patch("scalar_forensic.web.routes.faces.Settings"),
        patch("scalar_forensic.web.routes.faces._require_faces"),
        patch("scalar_forensic.web.routes.faces.FaceStore", return_value=store),
        patch("scalar_forensic.web.routes.faces.QdrantClient", MagicMock()),
    ):
        body = client.get("/api/faces/audit?image_hash=" + "a" * 64).json()

    assert body["gates_in_force_at_index_time"]["review_min_size"] == 36
    assert body["file_totals"]["n_review_only"] == 3
    assert body["detector"]["detector_id"] == "yunet"
    assert "caveat" in body


def test_face_audit_rejects_a_malformed_hash():
    with patch("scalar_forensic.web.routes.faces.Settings"):
        assert client.get("/api/faces/audit?image_hash=zz").status_code == 400
```

Append to `tests/faces/test_static_wiring.py`:

```python
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
    assert "review-only" in block          # the population statement
```

- [ ] **Step 2: Run and confirm failure**

```
uv run pytest tests/faces/test_face_audit_stats.py tests/faces/test_static_wiring.py -q
```

- [ ] **Step 3: Implement `faces_stats.py`**

Mirror `web/pipeline/stats.py:78-101` exactly for the statistics — mean, `statistics.stdev` when
n≥2 else 0.0, `statistics.quantiles(scores, n=100, method="inclusive")` indices 9/24/49/74/89,
and the 20-bucket histogram on the `[-1,1] → [0,1]` normalised scale while the numeric fields
stay on the raw cosine scale. Read that file and match it line for line; the point of this
feature is that the two modalities read alike.

`face_score_stats` calls `store.search_faces(vector, limit=sample_size, threshold=0.0,
exact=True)`. `face_audit` assembles the endpoint-4 dict from `get_marker` + the first row of
`list_faces` (payload fields are identical across a file's observations) + `check_compat`.

- [ ] **Step 4: Add endpoints 4 and 5 to `routes/faces.py`**, sync `def`, reusing
`_require_hash`, `_require_faces`, `_store`, `_entry_for`.

- [ ] **Step 5: Rename the two DINO buttons**

`index.html:757` — `x-text="semanticStatsLoading ? '…' : 'DINO Dist Stats'"`
`index.html:762` — `x-text="auditLoading ? '…' : 'DINO Audit'"`

In the Audit modal header (`index.html:105`), add beneath the title:

```html
          <p class="audit-modal-sub">Image embedding models: DINOv2 (semantic) and SSCD (altered).</p>
```

In the Dist Stats modal header (`index.html:34`), the title already reads "DINOv2 Score
Distribution" — leave it.

- [ ] **Step 6: Add the two FACE buttons and their modals**

Beside the DINO pair (`index.html:754-763`):

```html
        <template x-if="facesAvailable && queryFaces.some(f => f.searchable)">
          <button class="btn-tiny" :disabled="faceStatsLoading"
                  @click="openFaceStats()"
                  x-text="faceStatsLoading ? '…' : 'FACE Dist Stats'"></button>
        </template>
        <template x-if="facesAvailable && selectedHit">
          <button class="btn-tiny" :disabled="faceAuditLoading"
                  @click="openFaceAudit()"
                  x-text="faceAuditLoading ? '…' : 'FACE Audit'"></button>
        </template>
```

Two new modals, structurally cloned from the existing pair so the explanations have the same
shape: `.face-stats-modal-backdrop` (title "Face model — score distribution", the same histogram
+ percentile table, plus the population statement and the attributed 0.363 marker rendered in a
visibly different style from any deployment threshold) and `.face-audit-modal-backdrop` (title
"Face model — forensic audit", sections Detector / Gate thresholds at index time / Alignment /
Embedder / File totals / Enablement / Compatibility, and the caveat line). Copy the DINO modals'
markup conventions rather than inventing new ones.

State to add: `faceStats`, `faceStatsLoading`, `faceStatsError`, `showFaceStats`,
`faceAudit`, `faceAuditLoading`, `faceAuditError`, `showFaceAudit`.
Functions in `faces.js`: `openFaceStats()` (posts `session_id`, `file_id`, and the **first
selected** query face index; if none is selected, show "select a query face first"),
`openFaceAudit()` (GET with `selectedHit.image_hash`), plus their close handlers.

- [ ] **Step 7: Run everything**

```
uv run pytest -q
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
node --check src/scalar_forensic/web/static/js/faces.js
```
Expected: the whole suite green (previous bar 478 passed / 5 skipped, plus the new tests; the 5
skips remain unless `SFN_TEST_QDRANT_URL` is set). Then the live pass at `/?cachebust=4`.

- [ ] **Step 8: Commit**

```bash
git add src/scalar_forensic/web/pipeline/faces_stats.py \
        src/scalar_forensic/web/pipeline/__init__.py \
        src/scalar_forensic/web/routes/faces.py \
        src/scalar_forensic/web/static/ \
        tests/faces/test_face_audit_stats.py tests/faces/test_static_wiring.py
git commit -m "feat(web): per-model explainer buttons — DINO pair renamed, face pair added"
```

---

## Deferred, with reasons (do not silently implement)

- **SSCD near-duplicate set flagging on face hits** (spec §8: "media that are SSCD-near-duplicates
  of each other are flagged as a duplicate set"). Collapse *by `image_hash`* ships in Stage 2;
  the SSCD grouping needs an SSCD index pass (Stage 0) plus a second lookup per hit against the
  case collection. It is a self-contained follow-up stage and is not required for any part of the
  maintainer's request.
- **Shadow-mode adjudication workflow** (spec §10.5, §9). Phase 1b scope in the spec, but not in
  the maintainer's request; it needs the `{}_face_labels` sidecar collection, which does not
  exist.
- **Face-calibration tooling and record schema** (spec §10.4). Explicitly waived by the ruling
  for now; §10.4's precedence rule means adding it later changes the display path only.
- **Aggregation across observations of the same identity.** Spec §8 forbids it in 1b — one hit per
  observation, no aggregation, until an identity grouping exists.

## Things this plan is not sure about — verify before or during implementation

1. **Whether the web process can carry the face models at all.** Nothing in the web tier has ever
   loaded YuNet or the ONNX embedder; Phase 1 was CLI-index / web-read. Unknowns: whether
   `SFN_FACE_DETECTOR_MODEL` / `SFN_FACE_EMBEDDER_MODEL` are reliably set in the environment
   `./run.sh sfn-web` starts, first-call model-load latency, and ORT-vs-torch thread interaction
   in one process (the embedder already pins `intra_op_num_threads = 1`,
   `faces/embed.py:366-367`, which is the mitigation, and sync `def` handlers keep the event loop
   free). **Measure the first `POST /api/faces/query-faces` on danny1.jpeg before building
   Stage 3**; if it is slow enough to need a warm-up at lifespan or a progress state, that is a
   Stage 1 addendum, not a redesign — the endpoint contract is unaffected.
2. **`PipelineConfig` construction outside `FacePipeline`.** The exact constructor/classmethod
   `faces_query.py` needs is assumed, not verified. Read `faces/provenance.py` and
   `faces/indexing.py:68-140` and follow whatever exists.
3. **`hitKey()` on face-only hit rows.** Stage 4 synthesises hit objects that never came from
   `/api/query`. If `hitKey` (in `helpers.js`) keys on a field those rows lack, they will collide
   in the `x-for` `:key` and Alpine will mis-render. Read `hitKey` first and populate the field.
4. **Where face-only hits belong in the ordering.** This plan appends them after the image-mode
   hits, score-desc. Interleaving by score across modalities is equally defensible and mixes
   incomparable scales. Spec §12 says Phase 1 frontend UX is developed iteratively with the
   maintainer — expect this to be revisited in a hands-on round.
5. **`FACE Audit` when a hit has no face observations.** The plan 404s. An empty-state panel
   ("no faces were detected in this medium") may read better than an error; decide with the
   maintainer during the acceptance pass.
6. **The `DINO Audit` label under-describes a modal that also reports SSCD.** Shipping the
   requested label plus a subtitle; see the Stage 7 naming note.

## Acceptance pass (maintainer, in the running UI)

Prerequisites the 2026-08-12 runbook proved are easy to miss: the case collection must actually
be indexed with image vectors (`--faces` alone writes none), `SFN_INPUT_DIR` must be set or
`/api/metadata` returns 403, and **`/?cachebust=N`** is mandatory or a stale page will be
debugged.

1. Upload `danny1.jpeg`. The query panel shows face chips; the three sub-36 px faces are labelled
   not-searchable and cannot be selected.
2. Upload `danny2.jpeg`. Its face is selectable, green-bordered when selected.
3. Run the search: a FACES row appears for danny2's own medium at cosine ≈ 1.0, and the
   uncalibrated banner is visible.
4. Toggle the FACES pill: the face-only rows disappear and return.
5. In the Best Match panel, the matched face carries a green border with `cosine 0.xxxx` beneath.
6. `FACE Audit` names YuNet, SFace, `arcface-112-v1`, `affine-0.0-1.0`, and reports
   `review_min_size = 36` — the value in force at index time, not today's env.
7. `FACE Dist Stats` renders the distribution; 0.363 appears attributed to the model authors and
   nowhere as this deployment's threshold.
8. `tail data/faces/face_audit.log` shows one `"query"` event per search, with `search_mode`,
   `n_results` and `face_calibration_id: null`.
