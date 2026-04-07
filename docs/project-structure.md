# Suggested Project Structure

```
ScalarForensic/
├── src/
│   └── scalar_forensic/
│       ├── __init__.py
│       ├── cli.py                  # entrypoint: starts FastAPI via uvicorn
│       ├── config.py               # settings loaded from env / config.toml
│       │
│       ├── api/
│       │   ├── app.py              # FastAPI app factory
│       │   └── routes/
│       │       ├── ingest.py       # POST /ingest  — upload + hash + enqueue
│       │       ├── query.py        # POST /query   — nearest-neighbour search
│       │       ├── files.py        # GET  /files   — list, detail, status
│       │       └── health.py       # GET  /health, GET /stats
│       │
│       ├── storage/
│       │   ├── files.py            # filesystem helpers (save, delete, path resolution)
│       │   ├── metadata.py         # SQLite: files table, pipeline status per file
│       │   └── vector_db.py        # Qdrant client wrapper (collections, upsert, search)
│       │
│       ├── pipelines/
│       │   ├── hashing.py          # SHA-256 exact dedup
│       │   ├── dedup.py            # SSCD semantic near-duplicate detection
│       │   ├── person_reid.py      # StrongSORT + OSNet via BoxMOT
│       │   ├── scene_matching.py   # MegaLoc / EigenPlaces VPR
│       │   └── audio_profiling.py  # audfprint + ECAPA-TDNN + Whisper
│       │
│       └── models/
│           ├── sscd.py             # SSCD ResNet-50 wrapper (load, embed)
│           ├── reid.py             # OSNet wrapper (load, embed)
│           ├── vpr.py              # MegaLoc / EigenPlaces wrapper
│           └── speaker.py          # ECAPA-TDNN wrapper (SpeechBrain)
│
├── frontend/
│   ├── index.html                  # upload form + file list
│   ├── query.html                  # search / results view
│   └── static/
│       ├── style.css
│       └── app.js
│
├── tests/
│   ├── __init__.py
│   ├── test_hashing.py
│   ├── test_dedup.py
│   └── test_api.py
│
├── docs/
│   ├── project-structure.md        # this file
│   ├── research/                   # model/library evaluation reports
│   └── advisory/                   # strategic notes
│
├── data/                           # local runtime data — gitignored
│   ├── uploads/                    # ingested files
│   ├── models/                     # downloaded model weights
│   └── scalar_forensic.db          # SQLite metadata store
│
├── pyproject.toml
├── README.md
├── ROADMAP.md
└── .gitignore
```

## Key decisions

**`src/` layout** — keeps the installable package separate from tests and tooling; already set up in `pyproject.toml`.

**`models/` vs `pipelines/`** — models are stateful singletons (load once, reuse); pipelines are stateless functions that call a model and write results to storage. This separation makes it easy to swap a model without touching the pipeline logic.

**`storage/metadata.py` (SQLite)** — lightweight, zero-dependency metadata store for file records and per-pipeline status. If the dataset grows large or multi-process writes become a bottleneck, swap to PostgreSQL with minimal interface changes.

**`storage/vector_db.py`** — thin wrapper around `qdrant-client`; keeps Qdrant-specific code out of the pipelines. One collection per embedding type (e.g. `dedup`, `reid`, `scene`, `speaker`).

**`frontend/`** — plain HTML/JS served as static files by FastAPI (`StaticFiles`). No build step needed for Phase 1. Replace with a proper framework later if the UI grows complex.

**`data/`** — never committed. Add to `.gitignore`:
```
data/
```
Model weights live in `data/models/` and are downloaded on first run or via a CLI command (`scalar-forensic download-models`).
