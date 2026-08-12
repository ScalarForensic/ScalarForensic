# Setup — face validation run (`danny*`)

**For:** the validation run that closes `docs/superpowers/plans/2026-08-12-face-review-embedding-split.md`.
**Written:** 2026-08-12. **Branch:** `feat/face-pipeline-phase1` (local only).

This is the first end-to-end exercise of the face pipeline. Everything below runs on
your machine; nothing leaves it. Steps 1–5 are setup and can be done any time. Step 6
is the run itself — **do not run it until Task 8 has landed** (until then `sfn-faces
purge` unlinks chips unconditionally and can remove a chip another observation still
references).

Times are rough: setup is ~10 minutes plus the ONNX export if you use the throwaway
embedder.

---

## 1. Install the optional face dependencies

The face modality is behind a dependency group (`opencv-python-headless`, `onnxruntime`);
neither is installed by a plain `uv sync`.

```bash
cd /home/user01/Schreibtisch/gitea/ScalarForensic
uv sync --group faces
```

Verify:

```bash
uv run python -c "import cv2, onnxruntime; print(cv2.__version__, onnxruntime.__version__)"
```

## 2. Fetch the YuNet detector

MIT-licensed, downloaded once, hash-checked by the script:

```bash
uv run python scripts/download_models.py --yunet
```

Lands at `models/face_detection_yunet_2023mar.onnx` (`models/` is gitignored).

Side benefit: this also un-skips the one test that can catch a wrong YuNet landmark-column
map. Worth running the suite once afterwards to confirm it passes:

```bash
uv run pytest -q -k yunet
```

## 3. Provide an embedder

ScalarForensic ships no recognition weights (licensing — see INSTALL.md). You need an
ONNX model **plus a sidecar `<model>.onnx.manifest.json`**. Pick one of the two:

### Option A — your real embedder

Place the `.onnx` in `models/` and write the manifest next to it. The manifest fields the
loader requires (see `docs/specs/face-pipeline.md` §6.3):

```json
{
  "input_name": "input",
  "layout": "NCHW",
  "channel_order": "RGB",
  "dtype": "float32",
  "input_size": 112,
  "mean": 127.5,
  "scale": 128.0,
  "output_name": "embedding",
  "embedding_dim": 512
}
```

Adjust to match your model — `input_size`, `embedding_dim`, `mean`/`scale` and
`channel_order` in particular. A wrong `channel_order` produces plausible-looking but
silently wrong vectors.

### Option B — throwaway non-evidential embedder

For validating *plumbing* — counts, gates, chips, audit record, UI — random weights are
sufficient and cannot be mistaken for a recognition result. This exports a 112×112 → 512-d
model with the deliberately alarming name the plan calls for:

```bash
uv run --with onnx --with onnxscript python - <<'PY'
import json
from pathlib import Path
import torch

OUT = Path("models"); OUT.mkdir(exist_ok=True)
path = OUT / "NOT_FOR_EVIDENCE_random_512.onnx"


class _Rand(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 512, kernel_size=3, stride=2)

    def forward(self, x):
        return self.conv(x).mean(dim=(2, 3))


torch.onnx.export(
    _Rand().eval(),
    (torch.zeros(1, 3, 112, 112),),
    str(path),
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    external_data=False,  # keep weights inside the .onnx so hash_file() covers them
)
Path(str(path) + ".manifest.json").write_text(json.dumps({
    "input_name": "input", "layout": "NCHW", "channel_order": "RGB",
    "dtype": "float32", "input_size": 112, "mean": 127.5, "scale": 128.0,
    "output_name": "embedding", "embedding_dim": 512,
}, indent=2))
print("wrote", path)
PY
```

`onnx` and `onnxscript` are export-only and intentionally not project dependencies;
`--with` supplies them for this one command without touching the lockfile.

**Do not use Option B for anything but this validation.** Its similarity scores are noise.

## 4. Start Qdrant with a host-reachable port

The compose file deliberately does not publish Qdrant — `sfn-web` reaches it over the
internal network. The CLI here runs on the host, so it needs `localhost:6333`. Add a
local override — note this repo does **not** gitignore `docker-compose.override.yml`, so
either add it to `.gitignore` or delete it at teardown rather than committing it:

```bash
cat > docker-compose.override.yml <<'YML'
services:
  qdrant:
    ports: ["127.0.0.1:6333:6333"]
YML

docker compose up -d qdrant
curl -s localhost:6333/collections
```

Expect `{"result":{"collections":[]},...}` on a clean instance.

## 5. Configure the environment

Set these for the run. `SFN_FACE_STORE_DIR` is **per case** — this is the operator
discipline decision (Option A): chips are content-addressed and purge's reference check
only scrolls its own collection, so two cases sharing one store directory can unlink each
other's chips.

```bash
export SFN_FACES_ENABLED=true
export SFN_EXAMINER_ID="<your examiner id>"
export SFN_FACE_DETECTOR_MODEL=models/face_detection_yunet_2023mar.onnx
export SFN_FACE_EMBEDDER_MODEL=models/NOT_FOR_EVIDENCE_random_512.onnx   # or your real one
export SFN_FACE_COLLECTION=faces_danny_validation
export SFN_FACE_STORE_DIR=data/faces/danny_validation
export SFN_COLLECTION=danny_validation
```

Dry-run the configuration check without indexing anything — a misconfiguration fails at
startup with an actionable message rather than at first detection:

```bash
uv run python -c "
from scalar_forensic.config import Settings
s = Settings()
print(s.face_startup_error() or 'config OK')
print('threshold notes:', s.face_threshold_notes())
"
```

Thresholds in force for the run (defaults; both review values are bootstrap numbers
pending exactly this calibration):

| Variable | Default | Meaning |
|---|---|---|
| `SFN_FACE_MIN_CONF` | `0.8` | embedding confidence floor |
| `SFN_FACE_MIN_SIZE` | `64` | embedding bbox min side (detector-input px) |
| `SFN_FACE_REVIEW_MIN_CONF` | `0.6` | retention confidence floor |
| `SFN_FACE_REVIEW_MIN_SIZE` | `48` | retention bbox min side |

Review values are clamped to never exceed their embedding counterparts; the clamp is
reported through `face_threshold_notes()`, printed above and at the start of the run.

## 6. Run it — after Task 8 lands

```bash
./run.sh sfn analysis_test --faces
```

The first activation of a new face collection prompts interactively for an authorization
reference (free text, recorded in the enablement record; empty is allowed but warns).
Type yours — do not leave it blank for a run you intend to cite.

## 7. What to check

**On `danny2.jpeg`** — one ~148 px face, expected **embedded**.
**On `danny1.jpeg`** — three ~40 px faces, expected **review-only** with exclusion reason
`size` (40 px clears the 48 px review floor only if the detector reports it larger than my
estimate — if they are rejected outright rather than retained, that itself is the
calibration finding, not a bug).

**CLI summary** — one line, must reconcile:
`detected = comparable + retained for review + rejected (+ failed)`

**Audit record** — `data/face_audit.log` (JSONL; sits beside the store dir's parent):

```bash
uv run python -c "
import json
for line in open('data/face_audit.log'):
    e = json.loads(line)
    if e['event'] == 'index_run':
        print(json.dumps(e, indent=2))
" | tail -40
```

Check `n_detected == n_kept + n_review_only + sum(n_rejected.values())`, and that
`n_dropped_noncanonical` is present. A non-zero `n_dropped_noncanonical` on this data is
the signal for a wrong YuNet landmark-column map — treat it as a stop-and-investigate.

**On disk** — review chips are JPEG + thumb, aligned chips are PNG:

```bash
find data/faces/danny_validation -type f | sort
```

Counts must match: one aligned PNG per embedded face, one review JPEG+thumb per retained
face (embedded faces have review artefacts too — they share the review domain).

**In the UI** (`./run.sh sfn-web`, then the face panel for each image):

- review crops legible at native resolution, **not upscaled** to look like aligned chips;
- the two populations labelled distinctly — a reviewer must never mistake a review-only
  observation for a comparable one;
- the explainer names the failing check and marks the embedding step *not performed*
  rather than passed.

> The drop zone fades after 5 s idle and swallows pointer events; move the mouse to wake it.

**Then the actual judgement:** is 48 the right review floor? Look at the `danny1` crops
and decide whether a face that size is worth an examiner's time. That judgement is the
calibration evidence and is the reason this run exists — it cannot be obtained by picking
a constant.

## 8. Teardown

```bash
uv run sfn-faces purge --all          # or --media <sha256>
docker compose down                    # add -v to drop the Qdrant volume too
rm -f docker-compose.override.yml
rm -rf data/faces/danny_validation
```

Keep `data/face_audit.log` — the audit trail is append-only by design and there is no
reason to delete it.
