# Face Review/Embedding Gate Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the single face quality gate into a low-bar review path (persist native-resolution crops for hand identification) and the unchanged high-bar embedding path, so small faces yield examinable crops without ever entering the similarity population.

**Architecture:** `pre_align_gate` stops being the sole admission decision. A new `review_gate` admits detections for retention; the existing checks then decide embedding. Detections partition three ways — rejected, review-only, embeddable. Review-only observations are Qdrant points carrying **no named vector**, so a vector search structurally cannot return them; the payload flag `embedding_status` is annotation only. Chip hashes are domain-separated so aligned and review artefacts can never alias.

**Tech Stack:** Python 3.11+, opencv-python-headless, onnxruntime, numpy, Pillow, qdrant-client 1.18.0, Qdrant server v1.17.1, FastAPI, typer, pytest, Alpine.js.

**Spec:** `docs/superpowers/specs/2026-08-12-face-review-embedding-split-design.md`
**Parent spec:** `docs/specs/face-pipeline.md`

## Global Constraints

- Offline: no network I/O at runtime anywhere in `faces/`; models load from local paths only.
- Tests hermetic by default: no network, no Qdrant server, no unshipped model files. The one live-Qdrant test (Task 11) is marked and skipped by default, like the real-YuNet test.
- Lint gate, exactly what CI runs: `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts`. Run `uv run ruff format` on every file you touch before linting. Lines ≤ 100 chars (E501 enabled).
- Full suite: `uv run pytest -q`. Baseline before this plan: **398 passed, 0 skipped**.
- `unittest.mock.patch` targets are per-module: patch where the name is *used*, not where it is defined.
- Canonical landmark order everywhere: `[left eye, right eye, nose tip, left mouth, right mouth]`, "left" = image-left.
- Aligned crop is fixed 112×112, `alignment_version = "arcface-112-v1"`.
- New settings parse eagerly in `Settings.__init__` and never raise for a *default* value conflicting with an explicit one — clamp and report via `face_startup_error()`.
- UI copy rules: face **observations**, never identifications. Review-only observations are additionally labelled **not comparable**. Never "similar people"; "similar faces" only.
- Commit after every green test cycle; conventional-commit messages; **never push** (local-only workflow).
- Payload field names, fixed across all tasks: `embedding_status` (`"embedded"` | `"review_only"`), `embedding_exclusion_reason` (`str | None`), `aligned_chip_hash` (`str | None`), `review_chip_hash` (`str | None`). There is no `chip_kind` and no bare `chip_hash` on new points.

---

### Task 0: Preflight — confirm the migration window is open

**Files:** none (verification only)

**Interfaces:**
- Consumes: nothing
- Produces: a go/no-go for the whole plan

The spec assumes no medium has ever been indexed with faces, which makes every schema change free. That is an operational claim, not a code fact. Verify before writing code.

- [x] **Step 1: Check whether a face collection exists and is populated**

Qdrant is not published to the host by default (`docker-compose.yml:44`). If `curl` fails, that is expected; use the container.

```bash
docker exec scalarforensic-qdrant-1 sh -c \
  'wget -qO- http://localhost:6333/collections' || \
  curl -s http://localhost:6333/collections
```

Expected: no collection whose name ends in `_faces`. If one exists, check its point count:

```bash
docker exec scalarforensic-qdrant-1 sh -c \
  'wget -qO- http://localhost:6333/collections/<name>_faces'
```

- [x] **Step 2: Decide** — 2026-08-12: Qdrant reports `{"collections":[]}` (queried at
container IP 172.20.0.2:6333; the container has neither `wget` nor `curl`). No `_faces`
collection has ever existed. Migration window is open; the plan's free-schema-change
assumption holds.

If no `_faces` collection exists, or it exists with `points_count: 0`, the window is open — proceed to Task 1.

**If it is populated, STOP and report to the maintainer.** Do not proceed. The spec's "Migration and preflight" section must be rewritten with a migration path first: existing points would have a bare `chip_hash` and no `embedding_status`, and every task below assumes neither exists.

---

### Task 1: Review thresholds in Settings, with clamping

**Files:**
- Modify: `src/scalar_forensic/config.py:179` (after `face_min_size`), and `face_startup_error()` at `config.py:303`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Settings.face_review_min_conf: float`, `Settings.face_review_min_size: int` (both already clamped to be no stricter than their embedding counterparts), and `Settings.face_threshold_notes() -> list[str]` returning human-readable clamp/floor notices.

Why clamping rather than raising: `Settings.__init__` parses the face block unconditionally (`faces_enabled` is only consulted at `config.py:309`), and `Settings()` is constructed per request (`routes/faces.py:53, 89, 111, 245, 278`). Raising would 500 every route — including non-face routes — for an operator who set `SFN_FACE_MIN_CONF=0.5` and never touched the review variable, because the *default* review value 0.6 exceeds their explicit 0.5. A default must never invalidate a user's explicit value.

- [x] **Step 1: Write the failing tests**

```python
def test_review_thresholds_default(monkeypatch):
    monkeypatch.delenv("SFN_FACE_REVIEW_MIN_CONF", raising=False)
    monkeypatch.delenv("SFN_FACE_REVIEW_MIN_SIZE", raising=False)
    s = Settings()
    assert s.face_review_min_conf == 0.6
    assert s.face_review_min_size == 48


def test_review_thresholds_clamped_to_embedding_gate(monkeypatch):
    # An explicit embedding threshold below the review DEFAULT must not raise.
    monkeypatch.setenv("SFN_FACE_MIN_CONF", "0.5")
    monkeypatch.setenv("SFN_FACE_MIN_SIZE", "32")
    s = Settings()
    assert s.face_review_min_conf == 0.5
    assert s.face_review_min_size == 32
    notes = s.face_threshold_notes()
    assert any("SFN_FACE_REVIEW_MIN_CONF" in n for n in notes)
    assert any("SFN_FACE_REVIEW_MIN_SIZE" in n for n in notes)


def test_review_conf_below_detector_floor_is_noted(monkeypatch):
    monkeypatch.setenv("SFN_FACE_REVIEW_MIN_CONF", "0.2")
    s = Settings()
    assert any("0.5" in n for n in s.face_threshold_notes())


def test_review_thresholds_reject_nonsense(monkeypatch):
    monkeypatch.setenv("SFN_FACE_REVIEW_MIN_SIZE", "0")
    with pytest.raises(ValueError, match="SFN_FACE_REVIEW_MIN_SIZE"):
        Settings()
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_config.py -k review -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'face_review_min_conf'`.

- [x] **Step 3: Implement**

In `config.py`, immediately after the `face_min_size` block (`config.py:179-181`):

```python
        # Review path (spec: 2026-08-12 gate-split design).  Admits faces for
        # hand examination only — never for embedding.  Clamped, never raising:
        # this block parses even when faces are disabled and Settings() is
        # built per request, so a default must not invalidate an explicit value.
        self._face_threshold_notes: list[str] = []
        review_conf = self._parse_float("SFN_FACE_REVIEW_MIN_CONF", 0.6)
        if not 0 < review_conf <= 1:
            raise ValueError("SFN_FACE_REVIEW_MIN_CONF must be in (0, 1]")
        review_size = self._parse_int("SFN_FACE_REVIEW_MIN_SIZE", 48)
        if review_size < 1:
            raise ValueError("SFN_FACE_REVIEW_MIN_SIZE must be >= 1")
        if review_conf > self.face_min_conf:
            self._face_threshold_notes.append(
                f"SFN_FACE_REVIEW_MIN_CONF ({review_conf}) exceeds SFN_FACE_MIN_CONF "
                f"({self.face_min_conf}); clamped to {self.face_min_conf}. The review "
                "gate can never be stricter than the embedding gate."
            )
            review_conf = self.face_min_conf
        if review_size > self.face_min_size:
            self._face_threshold_notes.append(
                f"SFN_FACE_REVIEW_MIN_SIZE ({review_size}) exceeds SFN_FACE_MIN_SIZE "
                f"({self.face_min_size}); clamped to {self.face_min_size}."
            )
            review_size = self.face_min_size
        if review_conf < _DETECTOR_SCORE_FLOOR:
            self._face_threshold_notes.append(
                f"SFN_FACE_REVIEW_MIN_CONF ({review_conf}) is below the detector's own "
                f"score threshold ({_DETECTOR_SCORE_FLOOR}); no face below that ever "
                "reaches the gate, so the lower value has no effect."
            )
        self.face_review_min_conf: float = review_conf
        self.face_review_min_size: int = review_size
```

Add near the other module constants in `config.py`:

```python
# Mirrors YuNetDetector's default score_threshold (faces/detect.py); faces
# below it never reach any gate.  Duplicated rather than imported: config.py
# must not import cv2 transitively.
_DETECTOR_SCORE_FLOOR = 0.5
```

And a public accessor next to `face_startup_error()`:

```python
    def face_threshold_notes(self) -> list[str]:
        """Non-fatal notices about face threshold clamping (spec §Config)."""
        return list(self._face_threshold_notes)
```

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_config.py -k review -v`
Expected: PASS (4 tests).

- [x] **Step 5: Surface the notes in `face_startup_error()`**

`face_startup_error()` returns `None` when configuration is usable. Clamping is not an error, so do not change its return contract. Instead have the CLI and web lifespan print the notes. Add to `cli.py` where `face_startup_error()` is already consulted, and to the lifespan in `web/app.py` beside the existing faces warning:

```python
        for _note in settings.face_threshold_notes():
            typer.secho(f"note: {_note}", fg=typer.colors.YELLOW)
```

```python
        for note in settings.face_threshold_notes():
            log.warning("face threshold: %s", note)
```

- [x] **Step 6: Full suite and lint**

Run: `uv run pytest -q && uv run ruff format src tests && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: 402 passed.

- [x] **Step 7: Commit**

```bash
git add src/scalar_forensic/config.py src/scalar_forensic/cli.py src/scalar_forensic/web/app.py tests/test_config.py
git commit -m "feat(faces): review-path thresholds, clamped not raised"
```

---

### Task 2: The review gate

**Files:**
- Modify: `src/scalar_forensic/faces/quality.py` (after `pre_align_gate`, `quality.py:43-55`)
- Test: `tests/faces/test_quality.py`

**Interfaces:**
- Consumes: `Settings.face_review_min_conf`, `Settings.face_review_min_size` (Task 1)
- Produces: `review_gate(det: FaceDetection, *, min_conf: float, min_size: int) -> GateResult`

`GateResult` already exists in `quality.py` with fields `passed: bool`, `reason: str | None`, `subscores: dict[str, float]`. Reuse it unchanged. Size is measured in detector-input pixels exactly as `pre_align_gate` does (`quality.py:46`), so the two gates are directly comparable.

- [x] **Step 1: Write the failing tests**

```python
def _det(conf=0.9, w=100.0, h=100.0, scale=1.0):
    return FaceDetection(
        bbox=(0.0, 0.0, w, h),
        landmarks=np.array(
            [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32
        ),
        confidence=conf,
        detect_scale=scale,
    )


def test_review_gate_passes_a_small_but_confident_face():
    # 40px: below the 64px embedding floor, above a 24px review floor.
    r = review_gate(_det(conf=0.93, w=40.0, h=49.0), min_conf=0.6, min_size=24)
    assert r.passed is True
    assert r.reason is None
    assert r.subscores["size"] == 40.0


def test_review_gate_rejects_below_size():
    r = review_gate(_det(w=20.0, h=20.0), min_conf=0.6, min_size=24)
    assert r.passed is False
    assert r.reason == "size"


def test_review_gate_rejects_below_confidence():
    r = review_gate(_det(conf=0.55), min_conf=0.6, min_size=24)
    assert r.passed is False
    assert r.reason == "confidence"


def test_review_gate_measures_size_in_detector_input_px():
    # A large face in a downscaled image is small to the detector.
    r = review_gate(_det(w=100.0, h=100.0, scale=0.2), min_conf=0.6, min_size=24)
    assert r.passed is False
    assert r.subscores["size"] == 20.0


def test_review_gate_ignores_pose():
    # Pose is an embedding concern only: a profile face is still worth a look.
    det = _det()
    object.__setattr__(
        det,
        "landmarks",
        np.array([[10, 40], [14, 40], [12, 60], [10, 80], [14, 80]], dtype=np.float32),
    )
    assert review_gate(det, min_conf=0.6, min_size=24).passed is True
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/faces/test_quality.py -k review_gate -v`
Expected: FAIL — `ImportError: cannot import name 'review_gate'`.

- [x] **Step 3: Implement**

```python
def review_gate(det: FaceDetection, *, min_conf: float, min_size: int) -> GateResult:
    """Admit a detection for hand review (spec: gate-split design).

    Deliberately weaker than pre_align_gate: no pose check.  Pose degrades an
    *alignment*, and this path never aligns — the examiner looks at the
    unwarped source crop, where a profile face is perfectly examinable.
    """
    min_side_input_px = min(det.bbox[2], det.bbox[3]) * det.detect_scale
    subs = {"confidence": det.confidence, "size": min_side_input_px}
    if det.confidence < min_conf:
        return GateResult(False, "confidence", subs)
    if min_side_input_px < min_size:
        return GateResult(False, "size", subs)
    return GateResult(True, None, subs)
```

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/faces/test_quality.py -k review_gate -v`
Expected: PASS (5 tests).

- [x] **Step 5: Commit**

```bash
git add src/scalar_forensic/faces/quality.py tests/faces/test_quality.py
git commit -m "feat(faces): review_gate — confidence and size, no pose"
```

---

### Task 3: Provenance and soft-compatibility fields

**Files:**
- Modify: `src/scalar_forensic/faces/provenance.py:12-32` (`PipelineConfig`)
- Modify: `src/scalar_forensic/faces/store.py:43` (`_SOFT_FIELDS`)
- Test: `tests/faces/test_provenance.py`, `tests/faces/test_store.py`

**Interfaces:**
- Consumes: Task 1's settings
- Produces: `PipelineConfig.review_min_conf: float`, `PipelineConfig.review_min_size: int`, both inside `config_hash`

The review thresholds go **inside** `config_hash` deliberately. `processed_hashes` (`store.py:242`) matches markers on exact hash equality, so a threshold outside the hash would leave already-indexed media matching their old marker — lowering the review floor would silently pick up no new faces, with no signal that it did nothing. They also join `_SOFT_FIELDS`, which is the existing mechanism for "changes which faces get in, not what the vectors mean" — without that, `check_compat` reports a stale account of admission criteria.

- [x] **Step 1: Write the failing tests**

```python
def _cfg(**over):
    base = dict(
        detector_id="yunet",
        detector_model_hash="d" * 64,
        detector_score_threshold=0.5,
        detect_max_size=1600,
        embedder_model_name="test.onnx",
        embedder_model_hash="e" * 64,
        manifest_hash="m" * 64,
        embedder_dim=512,
        alignment_version="arcface-112-v1",
        normalization_id="affine-127.5-128.0",
        min_conf=0.8,
        min_size=64,
        min_sharpness=25.0,
        max_clipped=0.6,
        max_pose=0.35,
        crop_dilation=0.15,
        review_min_conf=0.6,
        review_min_size=48,
        sfn_version="0.0.0",
        cv2_version="5.0.0",
        ort_version="1.18.0",
    )
    base.update(over)
    return PipelineConfig(**base)


def test_review_thresholds_change_config_hash():
    # Must change: processed_hashes() keys on this, and a retuned review floor
    # has to force reprocessing or it silently admits nothing new.
    assert _cfg().config_hash != _cfg(review_min_size=32).config_hash
    assert _cfg().config_hash != _cfg(review_min_conf=0.5).config_hash


def test_review_thresholds_are_in_the_payload():
    p = _cfg().to_payload()
    assert p["review_min_conf"] == 0.6
    assert p["review_min_size"] == 48
```

```python
def test_review_thresholds_are_soft_not_hard(fake_client):
    # Changing them must not raise — they cannot change what a vector means.
    store = _store(fake_client, meta=_cfg().to_payload())
    notes = store.check_compat(_cfg(review_min_size=24))
    assert any("review_min_size" in n for n in notes)
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/faces/test_provenance.py tests/faces/test_store.py -k review -v`
Expected: FAIL — `TypeError: PipelineConfig.__init__() got an unexpected keyword argument 'review_min_conf'`.

- [x] **Step 3: Implement**

In `provenance.py`, add to `PipelineConfig` immediately after `crop_dilation`:

```python
    review_min_conf: float
    review_min_size: int
```

In `store.py:43`, extend `_SOFT_FIELDS`:

```python
_SOFT_FIELDS = (
    "detector_id",
    "detector_model_hash",
    "detect_max_size",
    "min_conf",
    "min_size",
    "review_min_conf",
    "review_min_size",
)
```

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/faces/ -k review -v`
Expected: PASS.

- [x] **Step 5: Fix every other construction site**

`PipelineConfig` gains two required fields, so all existing constructions break. Find and fix them:

```bash
grep -rn "PipelineConfig(" src tests | grep -v "def "
```

In `indexing.py:62-67` (the `from_settings` path) pass `review_min_conf=settings.face_review_min_conf, review_min_size=settings.face_review_min_size`. In test helpers add the two fields to the base dict.

- [x] **Step 6: Full suite and lint**

Run: `uv run pytest -q && uv run ruff format src tests && uv run ruff check src tests`
Expected: all green.

- [x] **Step 7: Commit**

```bash
git add src/scalar_forensic/faces/provenance.py src/scalar_forensic/faces/store.py tests/
git commit -m "feat(faces): record review thresholds in provenance and soft compat"
```

---

### Task 4: Domain-separated chip hashes and review-only chip writing

**Files:**
- Modify: `src/scalar_forensic/faces/chips.py`
- Test: `tests/faces/test_chips.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `aligned_chip_hash(aligned_rgb: np.ndarray) -> str` (replaces `chip_hash`)
  - `review_chip_hash(crop_rgb: np.ndarray) -> str`
  - `chip_paths(store_dir: Path, chash: str) -> tuple[Path, Path, Path]` (unchanged: png, review, thumb). Kept as-is deliberately, but note the wart: its 3-tuple now spans both hash domains, so `chip_paths(dir, aligned_hash)[1]` names a review JPEG that can never exist. Task 9 resolves the ambiguity where it matters, by choosing the path builder from `embedding_status` rather than letting an endpoint index blindly into the tuple.
  - `review_chip_paths(store_dir: Path, chash: str) -> tuple[Path, Path]` (review, thumb)
  - `write_aligned_chips(store_dir, aligned_rgb, source_rgb, bbox, dilation, thumb_size) -> tuple[str, str | None]` (renamed `write_chips`) returning `(aligned_hash, review_hash)`. The aligned PNG is stored under the aligned hash; **the review JPEG and thumbnail are stored under the review hash**, exactly as `write_review_chips` does. Both observation kinds therefore resolve review artefacts identically, and an embedded face's review chip is byte-shared with a review-only face that produced the same crop.
  - `write_review_chips(store_dir, source_rgb, bbox, dilation, thumb_size) -> str | None` — returns `None` when the dilated bbox clamps to zero area, so a caller never records a hash for files that were not written

Why domain separation: the same dimension-prefixed RGB array can legitimately arise once as an aligned crop and once as a native review crop. Paths are chosen by hash plus suffix alone, so without separation a review-only observation could be served another observation's aligned PNG.

- [x] **Step 1: Write the failing tests**

```python
def test_hash_domains_are_separated():
    arr = np.full((112, 112, 3), 7, dtype=np.uint8)
    assert aligned_chip_hash(arr) != review_chip_hash(arr)


def test_hashes_are_dimension_sensitive():
    a = np.zeros((112, 112, 3), dtype=np.uint8)
    b = np.zeros((56, 224, 3), dtype=np.uint8)
    assert review_chip_hash(a) != review_chip_hash(b)


def test_write_review_chips_writes_two_files_and_no_png(tmp_path):
    src = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    chash = write_review_chips(
        tmp_path, src, bbox=(100.0, 100.0, 80.0, 80.0), dilation=0.15, thumb_size=256
    )
    assert chash is not None
    review, thumb = review_chip_paths(tmp_path, chash)
    assert review.exists() and thumb.exists()
    png, _, _ = chip_paths(tmp_path, chash)
    assert not png.exists()


def test_write_review_chips_returns_none_for_offimage_bbox(tmp_path):
    src = np.zeros((100, 100, 3), dtype=np.uint8)
    assert (
        write_review_chips(
            tmp_path, src, bbox=(500.0, 500.0, 40.0, 40.0), dilation=0.15, thumb_size=256
        )
        is None
    )
    assert not any(tmp_path.rglob("*.jpg"))


def test_review_thumbnail_never_upscales(tmp_path):
    # Load-bearing: the review path's honesty rests on native resolution.
    src = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    chash = write_review_chips(
        tmp_path, src, bbox=(80.0, 80.0, 40.0, 40.0), dilation=0.15, thumb_size=256
    )
    review, thumb = review_chip_paths(tmp_path, chash)
    with Image.open(review) as r, Image.open(thumb) as t:
        assert t.size == r.size
        assert max(t.size) < 256
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/faces/test_chips.py -k "review or domain" -v`
Expected: FAIL — `ImportError: cannot import name 'review_chip_hash'`.

- [x] **Step 3: Implement**

Replace `chip_hash` in `chips.py` and add the review functions:

```python
_ALIGNED_DOMAIN = b"aligned-rgb-v1\0"
_REVIEW_DOMAIN = b"review-source-rgb-v1\0"


def _domain_hash(domain: bytes, arr: np.ndarray) -> str:
    h, w = arr.shape[:2]
    hasher = hashlib.sha256(domain + f"{h}x{w}:".encode())
    hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


def aligned_chip_hash(aligned_rgb: np.ndarray) -> str:
    """Identity of the exact 112x112 tensor fed to the embedder."""
    return _domain_hash(_ALIGNED_DOMAIN, aligned_rgb)


def review_chip_hash(crop_rgb: np.ndarray) -> str:
    """Identity of the native-resolution source crop an examiner reviews.

    Domain-separated from aligned_chip_hash: the same pixel array can arise
    in both roles, and the chip endpoints resolve files by hash + suffix.
    """
    return _domain_hash(_REVIEW_DOMAIN, crop_rgb)


def review_chip_paths(store_dir: Path, chash: str) -> tuple[Path, Path]:
    shard = store_dir / chash[:2]
    return shard / f"{chash}.review.jpg", shard / f"{chash}.thumb.jpg"


def write_review_chips(
    store_dir: Path,
    source_rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    dilation: float,
    thumb_size: int,
) -> str | None:
    """Write the review JPEG and thumbnail for a review-only observation.

    Returns None when the dilated bbox clamps to zero area: a review-only
    observation whose crop does not exist is useless, so the caller rejects
    it rather than storing a hash for files that were never written.
    """
    x, y, w, h = dilated_clamped_bbox(bbox, dilation, source_rgb.shape[1], source_rgb.shape[0])
    if w <= 0 or h <= 0:
        return None
    crop = source_rgb[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    chash = review_chip_hash(crop)
    jpg, thumb = review_chip_paths(store_dir, chash)
    jpg.parent.mkdir(parents=True, exist_ok=True)
    if not jpg.exists():
        Image.fromarray(crop).save(jpg, format="JPEG", quality=_REVIEW_QUALITY)
    if not thumb.exists():
        write_thumbnail(jpg, thumb, thumb_size)
    return chash
```

Rename `write_chips` to `write_aligned_chips`, change its internal `chip_hash(aligned_rgb)` call to `aligned_chip_hash(aligned_rgb)` for the PNG only, and have it delegate the review JPEG and thumbnail to `write_review_chips` so those land under the review-domain hash:

```python
def write_aligned_chips(
    store_dir: Path,
    aligned_rgb: np.ndarray,
    source_rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    dilation: float,
    thumb_size: int,
) -> tuple[str, str | None]:
    """Write the aligned PNG plus the review artefacts.

    Returns (aligned_hash, review_hash).  The PNG is content-addressed in the
    aligned domain because it authenticates the exact model input; the review
    JPEG lives in the review domain so both observation kinds resolve review
    artefacts by the same rule.
    """
    ahash = aligned_chip_hash(aligned_rgb)
    png, _, _ = chip_paths(store_dir, ahash)
    png.parent.mkdir(parents=True, exist_ok=True)
    if not png.exists():
        Image.fromarray(aligned_rgb).save(png, format="PNG")
    rhash = write_review_chips(store_dir, source_rgb, bbox, dilation, thumb_size)
    return ahash, rhash
```

Add a test that both hashes come back and that the review JPEG sits under the review hash, not the aligned one:

```python
def test_write_aligned_chips_stores_review_under_review_hash(tmp_path):
    aligned = np.full((112, 112, 3), 3, dtype=np.uint8)
    src = np.random.default_rng(2).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    ahash, rhash = write_aligned_chips(
        tmp_path, aligned, src, bbox=(100.0, 100.0, 80.0, 80.0),
        dilation=0.15, thumb_size=256,
    )
    assert ahash != rhash
    png, aligned_review, _ = chip_paths(tmp_path, ahash)
    review, thumb = review_chip_paths(tmp_path, rhash)
    assert png.exists()
    assert review.exists() and thumb.exists()
    assert not aligned_review.exists()
```

Update its import sites:

```bash
grep -rn "write_chips\|chip_hash(" src tests
```

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/faces/test_chips.py -v`
Expected: PASS.

- [x] **Step 5: Full suite and lint**

Run: `uv run pytest -q && uv run ruff format src tests && uv run ruff check src tests`

- [x] **Step 6: Commit**

```bash
git add src/scalar_forensic/faces/chips.py src/scalar_forensic/faces/ tests/faces/
git commit -m "feat(faces): domain-separated chip hashes; review-only chip writer"
```

---

### Task 5: Store — review-only points, vector demotion, purge reference safety

**Files:**
- Modify: `src/scalar_forensic/faces/store.py`
- Test: `tests/faces/test_store.py`

**Interfaces:**
- Consumes: Task 4's hashes
- Produces:
  - `FaceStore.clear_face_vector(point_ids: list[str]) -> None` — calls `delete_vectors` for `FACE_VECTOR_NAME`
  - `FaceStore.unreferenced_chip_hashes(hashes: list[str]) -> list[str]` — of the given hashes, those no surviving point references
  - `purge_all()` also matches review-only points (they carry `is_face: True`, so no filter change is needed — assert it)

Demotion is the case the first design draft missed. Point IDs (`store.py:75`) exclude `config_hash`, but `processed_hashes` (`store.py:242`) keys idempotency **on** it — so a threshold change reprocesses every medium at the same point IDs. A face demoted from embedded to review-only is upserted with `vector={}` over a point that currently holds a vector. Whether `upsert` clears it is an implementation detail of Qdrant; a forensic guarantee does not run on an implication. Clear it explicitly. The call is idempotent, so no read-before-write is needed.

- [x] **Step 1: Write the failing tests**

```python
def test_clear_face_vector_calls_delete_vectors(fake_client):
    store = _store(fake_client)
    store.clear_face_vector(["id-a", "id-b"])
    assert fake_client.deleted_vectors == [("id-a", "id-b")] or fake_client.deleted_vectors
    call = fake_client.delete_vectors_calls[0]
    assert call["vectors"] == [FACE_VECTOR_NAME]
    assert set(call["points"]) == {"id-a", "id-b"}


def test_clear_face_vector_noop_on_empty(fake_client):
    _store(fake_client).clear_face_vector([])
    assert fake_client.delete_vectors_calls == []


def test_unreferenced_chip_hashes_keeps_shared_chips(fake_client):
    # A chip still referenced by a surviving observation must not be unlinked.
    fake_client.points = [
        _rec("p1", {"is_face": True, "review_chip_hash": "shared"}),
    ]
    store = _store(fake_client)
    assert store.unreferenced_chip_hashes(["shared", "orphan"]) == ["orphan"]


def test_review_only_points_are_purged_by_purge_all(fake_client):
    fake_client.points = [
        _rec("p1", {"is_face": True, "embedding_status": "review_only",
                    "review_chip_hash": "r1"}),
    ]
    store = _store(fake_client)
    result = store.purge_all()
    assert result.n_points == 1
    assert "r1" in result.chip_hashes
```

Extend the existing fake Qdrant client in `tests/faces/conftest.py` with:

```python
    def __init__(self):
        ...
        self.delete_vectors_calls: list[dict] = []

    def delete_vectors(self, collection_name, vectors, points):
        self.delete_vectors_calls.append(
            {"collection": collection_name, "vectors": list(vectors), "points": list(points)}
        )
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/faces/test_store.py -k "clear_face_vector or unreferenced or review_only" -v`
Expected: FAIL — `AttributeError: 'FaceStore' object has no attribute 'clear_face_vector'`.

- [x] **Step 3: Implement**

```python
    def clear_face_vector(self, point_ids: list[str]) -> None:
        """Remove the named face vector from points, keeping their payloads.

        Called when an observation is demoted to review-only on re-index.
        Point IDs are stable across runs (face_point_id excludes config_hash)
        while idempotency keys on config_hash, so a threshold change rewrites
        existing points in place.  An upsert with no vector must not be
        trusted to clear a previously stored one: a review-only point that
        kept its vector would still be returned by similarity search.
        """
        if not point_ids:
            return
        self.client.delete_vectors(
            collection_name=self.collection,
            vectors=[FACE_VECTOR_NAME],
            points=list(point_ids),
        )

    def unreferenced_chip_hashes(self, hashes: list[str]) -> list[str]:
        """Of *hashes*, those no surviving point still references.

        Chip files are content-addressed and therefore shared between
        observations with byte-identical crops — common for review chips
        across exact-duplicate media.  Unlinking a shared chip would break a
        surviving observation's evidence.
        """
        if not hashes:
            return []
        wanted = set(hashes)
        still_referenced: set[str] = set()
        for rec in qdrant_scroll_all(
            self.client,
            self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="is_face", match=MatchValue(value=True))]
            ),
            limit=_SCROLL_LIMIT,
            with_payload=["aligned_chip_hash", "review_chip_hash"],
        ):
            payload = rec.payload or {}
            for key in ("aligned_chip_hash", "review_chip_hash"):
                value = payload.get(key)
                if value in wanted:
                    still_referenced.add(value)
        return [h for h in hashes if h not in still_referenced]
```

Update `_purge_by_filter` (`store.py:300`) to collect both hash fields:

```python
            payload = rec.payload or {}
            for key in ("aligned_chip_hash", "review_chip_hash"):
                chash = payload.get(key)
                if chash:
                    chip_hashes.append(chash)
```

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/faces/test_store.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/scalar_forensic/faces/store.py tests/faces/
git commit -m "feat(faces): explicit vector demotion and chip reference safety"
```

---

### Task 6: Three-way partition in the pipeline

**Files:**
- Modify: `src/scalar_forensic/faces/indexing.py:30-37` (`FaceIndexResult`), `:128-219` (`process_image`)
- Test: `tests/faces/test_indexing.py`

**Interfaces:**
- Consumes: Tasks 2, 4, 5
- Produces: `FaceIndexResult` gains `n_review_only: int`, `review_only_reasons: dict[str, int]`, `review_only_point_ids: list[str]`

This is the highest-risk task in the plan. Embeddings are index-aligned with the embeddable list (`indexing.py:171-215`); pairing them against any combined list misassigns vectors — attaching one person's vector to another's observation, silently. The sentinel test below exists specifically to catch that.

**Do not run `./run.sh sfn <dir> --faces` or `uv run sfn-faces purge` between Tasks 4 and 8.** Task 4 moved review artefacts to the review-domain hash and Task 5 made `_purge_by_filter` read `aligned_chip_hash`/`review_chip_hash`, but until this task rewrites the payload the pipeline still writes a bare `chip_hash`. In that window purge deletes the points, finds no hashes, unlinks nothing, and records `n_chip_files=0` — it would report having removed biometric crops that are still on disk. Chips written in the window are also unreachable in the UI, which still derives its URLs from `chip_hash`.

- [x] **Step 1: Write the failing tests**

```python
def test_interleaved_outcomes_pair_each_embedding_with_its_own_face(fake_pipeline):
    # SENTINEL: embeddings array has one row per EMBEDDABLE face only.
    # Alternating outcomes is what breaks a naive enumerate() over a combined list.
    dets = [
        _det(w=200.0),   # embeddable
        _det(w=50.0),    # review-only (below 64px)
        _det(w=200.0),   # embeddable
        _det(w=50.0),    # review-only
    ]
    fake_pipeline.detector.faces = dets
    fake_pipeline.embedder.vectors = [[1.0] + [0.0] * 511, [0.0, 1.0] + [0.0] * 510]
    result = fake_pipeline.process_image(b"x", "hash", "path.jpg")

    assert result.n_kept == 2
    assert result.n_review_only == 2
    embedded = [p for p in result.points if p.payload["embedding_status"] == "embedded"]
    assert embedded[0].vector[FACE_VECTOR_NAME][0] == 1.0
    assert embedded[1].vector[FACE_VECTOR_NAME][1] == 1.0
    assert len(fake_pipeline.embedder.received) == 2


def test_all_review_only_image_still_yields_observations(fake_pipeline):
    # Guards the early-return: this is exactly what danny1.jpeg produces.
    fake_pipeline.detector.faces = [_det(w=50.0), _det(w=50.0), _det(w=50.0)]
    result = fake_pipeline.process_image(b"x", "hash", "path.jpg")
    assert result.n_review_only == 3
    assert len(result.points) == 3
    assert fake_pipeline.embedder.received == []


def test_review_only_points_carry_no_vector(fake_pipeline):
    fake_pipeline.detector.faces = [_det(w=50.0)]
    point = fake_pipeline.process_image(b"x", "hash", "path.jpg").points[0]
    assert point.vector == {}
    assert point.payload["embedding_status"] == "review_only"
    assert point.payload["embedding_exclusion_reason"] == "size"
    assert point.payload["is_face"] is True
    assert point.payload["aligned_chip_hash"] is None


def test_review_only_failures_are_not_counted_as_rejections(fake_pipeline):
    fake_pipeline.detector.faces = [_det(w=50.0), _det(w=10.0)]
    r = fake_pipeline.process_image(b"x", "hash", "path.jpg")
    assert r.n_review_only == 1
    assert r.review_only_reasons == {"size": 1}
    assert r.rejected == {"size": 1}
    assert r.n_detected == r.n_kept + r.n_review_only + sum(r.rejected.values())


def test_degenerate_crop_is_rejected_not_retained(fake_pipeline):
    # A review-only observation whose crop does not exist is useless.
    fake_pipeline.detector.faces = [_det(x=5000.0, y=5000.0, w=50.0)]
    r = fake_pipeline.process_image(b"x", "hash", "path.jpg")
    assert r.n_review_only == 0
    assert r.rejected == {"size": 1}
    assert r.points == []
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/faces/test_indexing.py -k "interleaved or review_only or degenerate" -v`
Expected: FAIL — `AttributeError: 'FaceIndexResult' object has no attribute 'n_review_only'`.

- [x] **Step 3: Extend the result dataclass**

```python
@dataclass
class FaceIndexResult:
    n_detected: int = 0
    n_kept: int = 0
    n_review_only: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    review_only_reasons: dict[str, int] = field(default_factory=dict)
    points: list[PointStruct] = field(default_factory=list)
    review_only_point_ids: list[str] = field(default_factory=list)
```

- [x] **Step 4: Rewrite the partition loop**

Replace `process_image`'s body from the `kept` loop (`indexing.py:143`) through `if not kept: return result` (`:169`):

```python
        embeddable: list[tuple] = []   # (det, aligned, pre_subs, post_subs)
        review_only: list[tuple] = []  # (det, review_subs, exclusion_reason)

        def _count(bucket: dict[str, int], reason: str) -> None:
            bucket[reason] = bucket.get(reason, 0) + 1

        for det in detections:
            rev = review_gate(
                det, min_conf=self.review_min_conf, min_size=self.review_min_size
            )
            if not rev.passed:
                _count(result.rejected, rev.reason)
                continue
            # The review crop must exist, or the observation has no reason to be.
            x0, y0, cw, ch = dilated_clamped_bbox(
                det.bbox, self.crop_dilation, img.shape[1], img.shape[0]
            )
            if cw <= 0 or ch <= 0:
                _count(result.rejected, "size")
                continue

            pre = pre_align_gate(
                det, min_conf=self.min_conf, min_size=self.min_size, max_pose=self.max_pose
            )
            if not pre.passed:
                review_only.append((det, rev.subscores, pre.reason))
                continue
            aligned = align_face(img, det.landmarks)
            x, y, w, h = (int(v) for v in det.bbox)
            crop = img[max(0, y) : y + h, max(0, x) : x + w]
            if crop.size == 0:
                review_only.append((det, rev.subscores, "size"))
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            post = post_align_gate(
                gray, min_sharpness=self.min_sharpness, max_clipped_frac=self.max_clipped
            )
            if not post.passed:
                review_only.append((det, rev.subscores, post.reason))
                continue
            embeddable.append((det, aligned, pre.subscores, post.subscores))

        for _, _, reason in review_only:
            _count(result.review_only_reasons, reason)
        result.n_review_only = len(review_only)

        if not embeddable and not review_only:
            return result

        embeddings = (
            self.embedder.embed([e[1] for e in embeddable])
            if embeddable
            else np.empty((0, 0), dtype=np.float32)
        )
        norms = np.asarray(self.embedder.embedding_norms) if embeddable else np.empty(0)
        provenance = self.cfg.to_payload()
        indexed_at = datetime.now(UTC).isoformat()
```

Note `dilated_clamped_bbox` must be added to the imports from `scalar_forensic.faces.chips`.

- [x] **Step 5: Build both point kinds**

Keep the existing embedded-point loop (it already enumerates `embeddable`, which is now the only list paired with `embeddings`), changing only its chip call and payload keys:

```python
            aligned_hash = None
            review_hash = None
            if self.store_dir is not None:
                aligned_hash, review_hash = write_aligned_chips(
                    self.store_dir, aligned, img,
                    bbox=det.bbox, dilation=self.crop_dilation, thumb_size=self.thumb_size,
                )
```

where the payload replaces `"chip_hash": chip_hash` with:

```python
                "aligned_chip_hash": aligned_hash,
                "review_chip_hash": review_hash,
                "embedding_status": "embedded",
                "embedding_exclusion_reason": None,
```

Then append the review-only loop:

```python
        for det, rev_subs, reason in review_only:
            review_hash = None
            if self.store_dir is not None:
                review_hash = write_review_chips(
                    self.store_dir, img,
                    bbox=det.bbox, dilation=self.crop_dilation, thumb_size=self.thumb_size,
                )
            point_id = self.store.face_point_id(image_hash, frame_timecode_ms, det.bbox)
            result.review_only_point_ids.append(point_id)
            result.points.append(
                PointStruct(
                    id=point_id,
                    vector={},
                    payload={
                        "is_face": True,
                        "image_hash": image_hash,
                        "image_path": image_path,
                        "video_hash": video_hash,
                        "video_path": video_path,
                        "frame_timecode_ms": frame_timecode_ms,
                        "observation_key": self.store.observation_key(
                            image_hash, frame_timecode_ms, det.bbox
                        ),
                        "bbox": [int(round(v)) for v in det.bbox],
                        "landmarks": det.landmarks.tolist(),
                        "det_conf": det.confidence,
                        "detect_scale": det.detect_scale,
                        "quality_confidence": rev_subs["confidence"],
                        "quality_size": rev_subs["size"],
                        "quality_pose": None,
                        "quality_sharpness": None,
                        "quality_exposure": None,
                        "quality": None,
                        "embedding_norm": None,
                        "aligned_chip_hash": None,
                        "review_chip_hash": review_hash,
                        "embedding_status": "review_only",
                        "embedding_exclusion_reason": reason,
                        "indexed_at": indexed_at,
                        **provenance,
                    },
                )
            )
        result.n_kept = sum(
            1 for p in result.points if p.payload["embedding_status"] == "embedded"
        )
        return result
```

`FacePipeline` gains `review_min_conf: float` and `review_min_size: int` fields, populated in `from_settings` from Task 1's settings.

- [x] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/faces/test_indexing.py -v`
Expected: PASS.

- [x] **Step 7: Wire demotion at the call site**

In `cli.py`, where `upsert_faces` is called, clear vectors on demoted points immediately after:

```python
                face_pipeline.store.upsert_faces(_res.points)
                # Every review-only point, not only genuinely demoted ones:
                # delete_vectors is idempotent and ignores absent vectors, so
                # first-time review-only observations cost nothing.  The list
                # is named for what it holds, because Task 8 reports it --
                # calling first sightings "demoted" would misdescribe them.
                # Only ever pass review-only ids: clearing an embedded point's
                # vector destroys data recoverable only by a full re-index.
                face_pipeline.store.clear_face_vector(_res.review_only_point_ids)
```

- [x] **Step 8: Full suite and lint**

Run: `uv run pytest -q && uv run ruff format src tests && uv run ruff check src tests`

- [x] **Step 9: Commit**

```bash
git add src/scalar_forensic/faces/indexing.py src/scalar_forensic/cli.py tests/faces/
git commit -m "feat(faces): three-way partition — rejected, review-only, embeddable"
```

---

### Task 7: Marker counts, rollups, and the non-canonical drop counter

**Files:**
- Modify: `src/scalar_forensic/faces/store.py:182-238` (`marker_point`, `video_rollup_point`), `src/scalar_forensic/faces/detect.py`
- Test: `tests/faces/test_store.py`, `tests/faces/test_detect.py`

**Interfaces:**
- Consumes: Task 6's result fields
- Produces: `marker_point(..., n_review_only: int, review_only_reasons: dict[str, int], n_dropped_noncanonical: int)`, same three added to `video_rollup_point`

`n_kept` keeps meaning "embedded", so nothing already written is redefined. `n_dropped_noncanonical` currently increments on the detector (`detect.py:90`) and is persisted nowhere and read by nothing — a silent subtraction from `n_detected`. A design staking its case on the honest account of a medium has to record it.

- [x] **Step 1: Write the failing tests**

```python
def test_marker_records_review_only_counts(fake_client):
    store = _store(fake_client)
    p = store.marker_point(
        "img", None, "cfg", n_detected=6, n_kept=1,
        rejected={"confidence": 2},
        n_review_only=3, review_only_reasons={"size": 2, "pose": 1},
        n_dropped_noncanonical=0,
    )
    assert p.payload["n_review_only"] == 3
    assert p.payload["review_only_reasons"] == {"size": 2, "pose": 1}
    assert p.payload["n_dropped_noncanonical"] == 0
    total = (
        p.payload["n_kept"]
        + p.payload["n_review_only"]
        + sum(p.payload["n_rejected"].values())
    )
    assert total == p.payload["n_detected"]


def test_rollup_records_review_only_counts(fake_client):
    p = _store(fake_client).video_rollup_point(
        "vid", "cfg", n_detected=2, n_kept=1, rejected={},
        n_frames=1, n_review_only=1, review_only_reasons={"size": 1},
        n_dropped_noncanonical=0,
    )
    assert p.payload["n_review_only"] == 1
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/faces/test_store.py -k "review_only_counts" -v`
Expected: FAIL — `TypeError: marker_point() got an unexpected keyword argument 'n_review_only'`.

- [x] **Step 3: Implement**

Add the three parameters to both builders and to their payloads:

```python
                "n_review_only": n_review_only,
                "review_only_reasons": review_only_reasons,
                "n_dropped_noncanonical": n_dropped_noncanonical,
```

In `indexing.py`, carry the detector's counter onto the result so the caller can pass it: read `getattr(self.detector, "n_dropped_noncanonical", 0)` before detection and again after, and record the delta on `FaceIndexResult.n_dropped_noncanonical: int = 0`.

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/faces/ -v`
Expected: PASS. Update the existing marker/rollup call sites in `cli.py` to pass the new arguments from `FaceIndexResult`.

- [x] **Step 5: Full suite, lint, commit**

```bash
uv run pytest -q && uv run ruff format src tests && uv run ruff check src tests
git add src/scalar_forensic/faces/ src/scalar_forensic/cli.py tests/faces/
git commit -m "feat(faces): persist review-only and non-canonical-drop counts"
```

---

### Task 8: CLI summary and audit record

**Files:**
- Modify: `src/scalar_forensic/cli.py:1595-1655` (aggregation, summary line, audit record), `:1735-1750` (purge chip unlink)
- Test: `tests/test_cli_faces.py`

**Interfaces:**
- Consumes: Tasks 6, 7
- Produces: nothing downstream

The audit `index_run` record is spec §7.4's auditable account of the run. As written it reports `n_kept` and `n_rejected` only, so under the split it would **understate how many biometric crops were written to disk**. For a modality whose defensibility rests on its audit trail, that is not cosmetic.

- [x] **Step 1: Write the failing tests**

```python
def test_cli_summary_reconciles_counts(run_faces_cli, capsys):
    run_faces_cli(detected=6, kept=1, review_only=3, rejected={"confidence": 2})
    out = capsys.readouterr().out
    assert "1 comparable" in out
    assert "3 retained for review" in out
    assert "2 rejected" in out


def test_audit_index_run_records_review_only(run_faces_cli, audit_events):
    run_faces_cli(detected=6, kept=1, review_only=3, rejected={"confidence": 2})
    ev = [e for e in audit_events() if e["event"] == "index_run"][-1]
    assert ev["n_review_only"] == 3
    assert ev["review_only_reasons"] == {"size": 3}
    assert ev["n_kept"] + ev["n_review_only"] + sum(ev["n_rejected"].values()) == ev["n_detected"]


def test_purge_keeps_chips_still_referenced(tmp_path, fake_client):
    # Content-addressed chips are shared; purging one medium must not break another.
    ...  # assert a chip referenced by a surviving observation is not unlinked
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cli_faces.py -k "reconciles or index_run or still_referenced" -v`
Expected: FAIL on the missing summary text.

- [x] **Step 3: Implement the summary**

Replace the summary line (`cli.py:1634-1638`):

```python
        _rej_str = ", ".join(f"{_n} {_r}" for _r, _n in sorted(_face_rejected.items()))
        _rev_str = ", ".join(f"{_n} {_r}" for _r, _n in sorted(_face_review_reasons.items()))
        typer.echo(
            f"faces: {_face_detected:,} detected  │  {_face_kept:,} comparable"
            + f"  │  {_face_review_only:,} retained for review"
            + (f" ({_rev_str})" if _rev_str else "")
            + (f"  │  {sum(_face_rejected.values()):,} rejected: {_rej_str}" if _rej_str else "")
            + (f"  │  {_face_failed:,} failed" if _face_failed else "")
        )
```

Add `n_review_only=_face_review_only, review_only_reasons=_face_review_reasons` to the `audit.append("index_run", ...)` call, and accumulate both counters alongside the existing `_face_kept` / `_face_rejected` aggregation.

- [x] **Step 4: Implement purge reference safety**

At `cli.py:1741`, filter through the store before unlinking:

Three limits of this check must be stated in the audit record and the docs rather than engineered away, because each is a deployment property, not a code defect:

1. **Collection-scoped, chip store global.** `face_collection` is per case but `face_store_dir` defaults to one `data/faces` for every case, and content-addressing does not stop at a collection boundary — two cases holding the same image yield the same review hash and one file. `unreferenced_chip_hashes` scrolls only its own collection, so purging case A can unlink a chip case B still references. Task 12 must document that `SFN_FACE_STORE_DIR` is set per case, matching the cross-case rule `check_compat` already enforces for vectors.
2. **Check-then-unlink race.** A concurrent index run can write a referencing point between the check and the `unlink()`; because the file then already exists, `write_review_chips` will not recreate it, leaving a dangling reference. Purge assumes a single writer.
3. Both are mitigated but not cured by chips being re-derivable from the source media.

```python
    n_chip_files = 0
    if settings.face_store_dir is not None:
        unreferenced = store.unreferenced_chip_hashes(result.chip_hashes)
        for chash in unreferenced:
            # Both builders, deliberately: a hash may be an aligned hash (PNG)
            # or a review hash (JPEG + thumb), and the caller cannot tell which.
            # The two overlap on the review pair; `exists()` keeps the count honest.
            for path in (
                *chip_paths(Path(settings.face_store_dir), chash),
                *review_chip_paths(Path(settings.face_store_dir), chash),
            ):
                if path.exists():
                    path.unlink()
                    n_chip_files += 1
```

- [x] **Step 5: Run to verify they pass, then full suite, lint, commit**

```bash
uv run pytest tests/test_cli_faces.py -v && uv run pytest -q
uv run ruff format src tests && uv run ruff check src tests
git add src/scalar_forensic/cli.py tests/
git commit -m "feat(faces): reconcile CLI summary and audit record with the split"
```

---

### Task 9: Web endpoints — kinds, ordering, and a payload-driven explainer

**Files:**
- Modify: `src/scalar_forensic/web/routes/faces.py` (`/by-image`, `/chip/{hash}`, `/explain/{point_id}`), `src/scalar_forensic/faces/store.py:280` (`list_faces`)
- Test: `tests/web/test_routes_faces.py`

**Interfaces:**
- Consumes: Tasks 5, 6
- Produces: `/by-image` entries always carry `embedding_status` and `embedding_exclusion_reason`

Three concrete defects to fix, all found in review:

1. The explainer hardcodes `"passed": True` on all five steps (`routes/faces.py:128`) — only possible because today just kept faces are stored.
2. It emits a `chips.aligned` URL whenever a chip hash is set (`routes/faces.py:226`), which for a review-only face is a guaranteed 404.
3. The aligned-chip endpoint never resolves the observation, so it cannot tell the two chip kinds apart.

- [x] **Step 1: Write the failing tests**

```python
def test_by_image_orders_embedded_before_review_only(client, fake_store):
    fake_store.faces = [
        {"point_id": "b", "embedding_status": "review_only", "det_conf": 0.99},
        {"point_id": "a", "embedding_status": "embedded", "quality": 0.4},
    ]
    body = client.get("/api/faces/by-image/abc").json()
    assert [f["point_id"] for f in body["faces"]] == ["a", "b"]
    assert all("embedding_status" in f for f in body["faces"])


def test_aligned_chip_404s_for_review_only_observation(client, fake_store):
    r = client.get("/api/faces/chip/reviewhash")
    assert r.status_code == 404
    assert "review-only" in r.json()["detail"].lower()


def test_explain_marks_embedding_step_not_performed(client, fake_store):
    fake_store.face = {
        "embedding_status": "review_only",
        "embedding_exclusion_reason": "size",
        "quality_size": 40.0, "min_size": 64,
        "review_chip_hash": "r1", "aligned_chip_hash": None,
    }
    body = client.get("/api/faces/explain/<uuid>").json()
    embed_step = [s for s in body["steps"] if s["step"] == "embedding"][0]
    assert embed_step["passed"] is False
    assert "size" in embed_step["sentence"]
    assert body["chips"]["aligned"] is None
    assert body["embedding_status"] == "review_only"


def test_explain_still_marks_embedded_faces_passed(client, fake_store):
    fake_store.face = {"embedding_status": "embedded", "aligned_chip_hash": "a1"}
    body = client.get("/api/faces/explain/<uuid>").json()
    assert all(s["passed"] for s in body["steps"])
    assert body["chips"]["aligned"] is not None
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/web/test_routes_faces.py -k "review_only or orders or not_performed" -v`
Expected: FAIL — the explainer returns `passed: True` for every step.

- [x] **Step 3: Implement**

In the explain handler, derive each step's `passed` from the payload rather than hardcoding, and add the terminal step:

```python
    status = face.get("embedding_status", "embedded")
    excluded_at = face.get("embedding_exclusion_reason")
    _STEP_FOR_REASON = {
        "confidence": "pre_align_gate",
        "size": "pre_align_gate",
        "pose": "pre_align_gate",
        "sharpness": "post_align_gate",
        "exposure": "post_align_gate",
    }
    failing_step = _STEP_FOR_REASON.get(excluded_at) if status == "review_only" else None
```

Then for each step dict, replace `"passed": True` with `"passed": step_name != failing_step`, and append:

```python
        {
            "step": "embedding",
            "sentence": (
                "This face was measured and stored as a comparable numeric fingerprint."
                if status == "embedded"
                else (
                    f"This face was NOT embedded, because its {excluded_at} check did not "
                    "meet the threshold in force when the file was processed. It is kept "
                    "for examination by eye and is never compared with other faces."
                )
            ),
            "scores": {},
            "thresholds_in_force": {},
            "passed": status == "embedded",
        },
```

Gate the aligned chip URL on the aligned hash specifically:

```python
    aligned = face.get("aligned_chip_hash")
    review = face.get("review_chip_hash")
    ...
        "chips": {
            "aligned": f"/api/faces/chip/{aligned}" if aligned else None,
            "review": f"/api/faces/chip/{review}/review" if review else None,
            "thumb": f"/api/faces/chip/{review}/thumb" if review else None,
        },
        "embedding_status": status,
        "embedding_exclusion_reason": excluded_at,
```

For the aligned-chip endpoint, return 404 with a clear detail when the PNG is absent:

```python
    if not png_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No aligned chip for this observation: it is review-only and was never "
                "aligned or embedded."
            ),
        )
```

In `list_faces` (`store.py:280`), sort embedded observations first, each group by descending quality (embedded) or descending `det_conf` (review-only), since review-only faces have `quality: None`.

- [x] **Step 4: Run to verify they pass, then full suite, lint, commit**

```bash
uv run pytest tests/web/ -v && uv run pytest -q
uv run ruff format src tests && uv run ruff check src tests
git add src/scalar_forensic/web/ src/scalar_forensic/faces/store.py tests/web/
git commit -m "feat(faces): payload-driven explainer and chip-kind-aware endpoints"
```

---

### Task 10: Frontend — labelling and honest display resolution

**Files:**
- Modify: `src/scalar_forensic/web/static/js/faces.js`, `src/scalar_forensic/web/static/index.html` (face panel ~`:939`, explainer ~`:961`), `src/scalar_forensic/web/static/style.css:2243`
- Test: manual, in the running UI

**Interfaces:**
- Consumes: Task 9's payload fields
- Produces: nothing downstream

`style.css:2243` renders every chip at `width:72px; height:72px; object-fit: cover`. The stored review JPEG is native resolution, but the browser upscales *and* centre-crops it — worst for exactly the small faces this design adds, hiding part of the crop and fabricating display pixels. Shipping that while the spec claims "never upscales" is a gap a competent cross-examination finds.

- [x] **Step 1: Add the helper to `faces.js`**

New component code goes in the matching part file; computed getters belong in `computed.js`. This is a method, so it belongs here:

```javascript
    // Review-only observations are kept for examination by eye and are never
    // compared against anything.  The distinction must be visible, not a tooltip.
    faceIsReviewOnly(face) { return face?.embedding_status === 'review_only'; },

    faceStatusLabel(face) {
      return this.faceIsReviewOnly(face) ? 'review only — not comparable' : 'comparable';
    },
```

- [x] **Step 2: Mark review-only chips in the grid**

In `index.html`, on the face chip element add `:class="{ 'face-chip-review-only': faceIsReviewOnly(face) }"` and a visible caption bound to `faceStatusLabel(face)`.

- [x] **Step 3: Fix the display resolution**

In `style.css`, replace the blanket rule so small crops are neither upscaled nor cropped:

```css
.face-chip img {
  display: block;
  width: 72px;
  height: 72px;
  object-fit: cover;
}
/* Review-only chips are the small ones this modality exists to preserve.
   Never upscale and never crop them: the examiner must see the whole crop
   at its true resolution, or the "native resolution" claim is false. */
.face-chip-review-only img {
  width: auto;
  height: auto;
  max-width: 72px;
  max-height: 72px;
  object-fit: contain;
  image-rendering: pixelated;
}
```

- [x] **Step 4: Verify in the running UI**

Run `./run.sh sfn-web`, open a medium with both kinds, and confirm: review-only chips are visibly labelled, are not upscaled or cropped, and the explainer shows the embedding step as not performed with the failing check named. Remember the drop zone fades after 5 s idle and swallows pointer events — it wakes on `mousemove`.

- [x] **Step 5: Commit**

```bash
git add src/scalar_forensic/web/static/
git commit -m "feat(faces): label review-only observations and stop upscaling their chips"
```

---

### Task 11: Live-Qdrant integration test for the exclusion guarantee

**Files:**
- Create: `tests/faces/test_store_integration.py`
- Test: itself

**Interfaces:**
- Consumes: Tasks 5, 6
- Produces: nothing downstream

The exclusion guarantee is the design's load-bearing claim, and no hermetic test can observe it: asserting `PointStruct.vector == {}` inspects a constructor, not Qdrant's storage or search behaviour. This is the only test that can catch a demotion regression. Marked and skipped by default, like the real-YuNet test.

- [x] **Step 1: Write the test**

```python
"""Live-Qdrant check for the review-only exclusion guarantee.

Skipped unless SFN_TEST_QDRANT_URL is set.  Qdrant is not published to the
host by default (docker-compose.yml); add a local override with
  ports: ["127.0.0.1:6333:6333"]
then run:  SFN_TEST_QDRANT_URL=http://localhost:6333 uv run pytest -q
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from scalar_forensic.faces.store import FACE_VECTOR_NAME

_URL = os.environ.get("SFN_TEST_QDRANT_URL")
pytestmark = pytest.mark.skipif(not _URL, reason="SFN_TEST_QDRANT_URL not set")


def test_demoted_point_is_not_returned_by_vector_search():
    client = QdrantClient(url=_URL)
    name = f"sfn_test_faces_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        collection_name=name,
        vectors_config={FACE_VECTOR_NAME: VectorParams(size=4, distance=Distance.COSINE)},
    )
    try:
        pid = str(uuid.uuid4())
        client.upsert(
            collection_name=name,
            points=[PointStruct(id=pid, vector={FACE_VECTOR_NAME: [1.0, 0, 0, 0]},
                                payload={"embedding_status": "embedded"})],
            wait=True,
        )
        hits = client.query_points(
            collection_name=name, query=[1.0, 0, 0, 0], using=FACE_VECTOR_NAME, limit=10
        ).points
        assert [h.id for h in hits] == [pid], "precondition: the embedded point is findable"

        # Demote exactly as the pipeline does: upsert vectorless, then clear.
        client.upsert(
            collection_name=name,
            points=[PointStruct(id=pid, vector={}, payload={"embedding_status": "review_only"})],
            wait=True,
        )
        client.delete_vectors(
            collection_name=name, vectors=[FACE_VECTOR_NAME], points=[pid], wait=True
        )

        hits = client.query_points(
            collection_name=name, query=[1.0, 0, 0, 0], using=FACE_VECTOR_NAME, limit=10
        ).points
        assert hits == [], "a review-only point must be unreachable by vector search"

        got = client.retrieve(collection_name=name, ids=[pid], with_payload=True)
        assert got[0].payload["embedding_status"] == "review_only", "payload must survive"
    finally:
        client.delete_collection(collection_name=name)


def test_clear_face_vector_on_absent_points_is_a_noop():
    """The call site passes every review-only id, not only demoted ones.

    Most of those points have never held a vector, and some may not exist at
    all when the clear runs.  The whole demotion design assumes the server
    treats both as no-ops rather than erroring -- verify it against a real
    server instead of inferring it from the client signature.
    """
    client = QdrantClient(url=_URL)
    name = f"sfn_test_faces_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        collection_name=name,
        vectors_config={FACE_VECTOR_NAME: VectorParams(size=4, distance=Distance.COSINE)},
    )
    try:
        vectorless = str(uuid.uuid4())
        client.upsert(
            collection_name=name,
            points=[PointStruct(id=vectorless, vector={}, payload={"is_face": True})],
            wait=True,
        )
        # A point that exists but never held a vector, plus one that does not
        # exist at all -- the two cases the call site actually produces.
        client.delete_vectors(
            collection_name=name,
            vectors=[FACE_VECTOR_NAME],
            points=[vectorless, str(uuid.uuid4())],
            wait=True,
        )
        got = client.retrieve(collection_name=name, ids=[vectorless], with_payload=True)
        assert got[0].payload["is_face"] is True, "payload must survive a no-op clear"
    finally:
        client.delete_collection(collection_name=name)
```

- [x] **Step 2: Verify it skips by default**

Run: `uv run pytest tests/faces/test_store_integration.py -v`
Expected: 2 skipped.

- [x] **Step 3: Verify it passes against live Qdrant**

Create `docker-compose.override.yml` with `services: {qdrant: {ports: ["127.0.0.1:6333:6333"]}}`, `docker compose up -d qdrant`, then:

Run: `SFN_TEST_QDRANT_URL=http://localhost:6333 uv run pytest tests/faces/test_store_integration.py -v`
Expected: PASS.

**If the post-demotion search still returns the point, STOP.** That is the failure mode the design exists to prevent; report it rather than working around it.

- [x] **Step 4: Decide on the override file**

`docker-compose.override.yml` is not gitignored and is picked up automatically. Either add it to `.gitignore` or delete it after the run — do not commit it silently.

- [x] **Step 5: Commit**

```bash
git add tests/faces/test_store_integration.py
git commit -m "test(faces): live-Qdrant proof that demoted points leave the search space"
```

---

### Task 12: Documentation

**Files:**
- Modify: `docs/specs/face-pipeline.md` (§6.2, §7.3, §7.4, §13), `INSTALL.md` (face settings table), `CLAUDE.md`
- Create: nothing

**Interfaces:**
- Consumes: everything above
- Produces: nothing

- [x] **Step 1: Amend the parent spec**

In §6.2, replace "A face must pass all checks to be embedded. **Rejected faces are not persisted**" with the two-gate account: the review gate admits for retention, the existing checks decide embedding, and faces retained without embedding are review-only observations that carry no vector. In §7.3 document `aligned_chip_hash` / `review_chip_hash` and the domain separation. In §7.4 document `n_review_only`, `review_only_reasons` and `n_dropped_noncanonical`, and the invariant. In §13 add both new variables.

Add to §6.5 (Phase 2 grouping) an explicit open question: how review-only observations participate in within-file grouping and group counts. Do not resolve it.

- [x] **Step 2: Update INSTALL.md**

Add to the settings table:

```
| `SFN_FACE_REVIEW_MIN_CONF` | `0.6` | Confidence floor for keeping a face for hand review |
| `SFN_FACE_REVIEW_MIN_SIZE` | `48` | bbox min side (detector-input px) for hand review |
```

Add a short paragraph: faces clearing the review bar but not the embedding bar are kept as review-only observations — croppable and examinable, never compared. Both thresholds are bootstrap values pending calibration.

Also document the chip-store scoping rule, which the code cannot enforce: `SFN_FACE_COLLECTION` is per case but `SFN_FACE_STORE_DIR` defaults to a single `data/faces`, and chips are content-addressed, so two cases holding the same image share one chip file. Purge's reference check (Task 8) scrolls only its own collection and would unlink a chip another case still references. **Set `SFN_FACE_STORE_DIR` per case**, for the same reason `check_compat` refuses to mix biometric data across cases. Note also that purge assumes a single writer: it checks for references and then unlinks, so a concurrent index run can leave a dangling chip reference. Both are recoverable — chips are re-derivable from the source media — but neither should be discovered during an examination.

- [x] **Step 3: Update CLAUDE.md**

Add one line under the face bullets:

```markdown
- Faces have two gates: review (keep the native-resolution crop) and embedding (align + vector).
  Review-only observations are vectorless points — that is what keeps them out of search, not a
  payload filter. Never give them a vector.
```

- [x] **Step 4: Full suite, lint, commit**

```bash
uv run pytest -q && uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
git add docs/ INSTALL.md CLAUDE.md
git commit -m "docs(faces): record the review/embedding gate split"
```

---

## Validation run (after Task 12)

Not a code task — the first real exercise of the pipeline, which has never run end to end.

- [ ] Export a throwaway non-evidential embedder ONNX (112×112 → 512-d) plus manifest into `models/`, named so it can never be mistaken for a recognition model (e.g. `NOT_FOR_EVIDENCE_random_512.onnx`).
- [ ] `docker compose up -d qdrant` with the ports override from Task 11.
- [ ] `SFN_FACES_ENABLED=true SFN_EXAMINER_ID=<id> ./run.sh sfn analysis_test --faces` — the first activation prompts interactively for an authorization reference; the maintainer types it.
- [ ] Confirm: `danny2.jpeg`'s 148 px face is embedded; `danny1.jpeg`'s three ~40 px faces are review-only with `size` as the exclusion reason; the CLI summary reconciles; the audit record's counts match what is on disk.
- [ ] Confirm in the UI: review crops legible at native resolution without upscaling, both populations labelled distinctly, the explainer names the failing check.
- [ ] Then judge whether 48 is the right review floor. That judgement is the calibration evidence, and it cannot be obtained by picking a constant.

## Self-review notes

**Spec coverage.** Every section of the design maps to a task: clamping → 1; review gate → 2; `config_hash` and `_SOFT_FIELDS` → 3; chip identity → 4; demotion, purge safety, `is_face` → 5; partition, pairing hazard, degenerate crops, exclusion reason → 6; counts and the non-canonical counter → 7; CLI and audit → 8; endpoints and explainer → 9; UI labelling and display resolution → 10; the live guarantee → 11; docs → 12. The preflight the design requires is Task 0.

**Deliberately deferred.** Explainer visuals (spec §6.6 items 1, 4, 5) are the next workstream. The `purge_all` video-rollup omission (`store.py:325`) is a pre-existing bug, recorded in the design's "Out of scope" and fixed separately — Task 7 adds rollup fields without touching that filter, so it neither fixes nor worsens it.

**Resolved during review of this plan.** An earlier draft had embedded faces carrying a `review_chip_hash` whose files did not exist, because `write_aligned_chips` stored the review JPEG under the *aligned* hash. Task 4 now stores review artefacts in the review domain for both observation kinds, so `write_aligned_chips` returns both hashes and Task 6 needs no reconciling helper. The single rule is: the aligned PNG is addressed in the aligned domain, every review artefact in the review domain.

**One consequence worth knowing.** Because review artefacts are now content-addressed on the source crop, an embedded face and a review-only face with byte-identical crops share the same review JPEG on disk. That is intended — it is why Task 5's `unreferenced_chip_hashes` exists, and why purge must never unlink a chip another observation still references.
