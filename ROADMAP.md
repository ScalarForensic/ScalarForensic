# Roadmap

## Phase 1 — Foundation

Get a working skeleton end-to-end before any ML.

- [ ] Basic FastAPI app wired into `cli.py` (`uv run scalar-forensic` starts the server)
- [ ] File ingestion endpoint: accept upload, write to local storage, return a file ID
- [ ] Exact-duplicate detection via SHA-256 hashing on ingest (reject or flag known files)
- [ ] SQLite metadata store: file ID, original filename, SHA-256, ingest timestamp, status
- [ ] Minimal frontend: upload form, file list with status badges
- [ ] Health endpoint (`GET /health`) and basic stats endpoint (`GET /stats`)

**Done when:** a file can be uploaded, stored, hashed, and appear in the frontend list.

---

## Phase 2 — Vector Storage

Introduce Qdrant and the first semantic pipeline.

- [ ] Qdrant running as a single Docker container (see `docs/research/vector-db-evaluation.md`)
- [ ] SSCD ResNet-50 embedding pipeline: generate a 512-dim copy-detection descriptor per image
- [ ] Store embeddings in Qdrant alongside file metadata payload
- [ ] Query endpoint: upload a file, return nearest neighbours with similarity scores
- [ ] Frontend: query results page showing matched files and scores
- [ ] Threshold-based dedup flag: mark files with cosine similarity ≥ 0.75 as near-duplicates

**Done when:** uploading an altered copy of an existing image surfaces the original in results.

---

## Phase 3 — Analysis Pipelines

Add the remaining analysis modules one at a time.

- [ ] **Person Re-ID** — StrongSORT tracker + OSNet embeddings via BoxMOT; store tracklet embeddings per video clip
- [ ] **Scene matching** — MegaLoc VPR (Tier 2/3) or EigenPlaces (Tier 1 CPU); store scene descriptors per frame sample
- [ ] **Audio profiling** — audfprint fingerprinting for exact audio match; ECAPA-TDNN speaker embeddings for near-match; Whisper turbo for transcription
- [ ] Pipeline status tracking: each file carries per-pipeline state (pending / running / done / failed)
- [ ] Frontend: per-file detail view showing pipeline results

**Done when:** a media file returns results from all four pipelines.

---

## Phase 4 — Hardening

- [ ] Config file (`config.toml` or env-based) covering storage paths, Qdrant host, model paths, similarity thresholds
- [ ] Background task queue (start with `asyncio`, swap to Celery/ARQ if needed)
- [ ] Docker Compose: app + Qdrant + optional GPU worker
- [ ] Basic access control (API key header)
- [ ] Export endpoint: download search results as CSV/JSON

---

## Out of Scope (for now)

- Horizontal scaling / sharding
- Cloth-changing Re-ID (not mature enough — revisit when TryHarder weights become available)
- SegFormer preprocessing for VPR (no benchmark evidence of improvement)
