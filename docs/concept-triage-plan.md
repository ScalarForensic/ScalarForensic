# Concept-Triage: Investigator-in-the-Loop Discovery for ScalarForensic

## Context

**Why this change is being made.** ScalarForensic today offers three pull-based similarity modes — `exact` (SHA-256), `altered` (SSCD cosine ≥ 0.75), and `semantic` (DINOv2 cosine ≥ 0.55) — each anchored to a **single** query image. This is a strong foundation but has three forensically important blind spots:

1. **No notion of a "concept" larger than one image.** An investigator who has labelled five items as incriminating and three look-alikes as benign cannot feed that knowledge back into retrieval. They must pick one reference and re-run from scratch.
2. **Cosine thresholds are hard to defend.** "0.55" is not legible to a reviewer, a supervisor, or a court. "8 of 10 labelled reference pairs place this candidate on the incriminating side of the decision boundary" is.
3. **No active learning.** Every case starts cold. Labels produced during review are discarded at session end.

Qdrant's **Discovery API** (context pairs + optional target) and **Recommendation API** (positive/negative example lists) map almost exactly onto the mental model investigators already use during triage: *"more like these, less like those."* The Discovery API's triplet-satisfaction score is an integer count of pair constraints met — a directly interpretable, explainable ranking signal.

**Intended outcome.** Add a new mode — **"Concept Triage"** — where the investigator builds a concept during review (two lists: positive references + negative references, optionally a target anchor), queries the corpus via Qdrant's native `discover` / context-search against **both** named vectors (`dino` + `sscd`), and gets results ranked by a defensible integer triplet score plus a per-model cosine margin. Concepts are persistable, reusable across cases, and composable — including a **reverse mode** that surfaces provably-benign material to *exclude* from review (reducing examiner fatigue and exposure to traumatic content).

This is the first feature in ScalarForensic that uses Qdrant as more than a distance index — it uses Qdrant as a *reasoning layer over labelled examples*.

---

## Architecture

### Qdrant APIs used

| API | Qdrant SDK call | When |
|-----|-----------------|------|
| Context search (no target) | `client.query_points(query=DiscoverQuery(discover=DiscoverInput(target=None, context=[ContextPair(positive=pid, negative=pid), ...])), using=<vec>)` | Default triage mode — "what in the corpus sits on the incriminating side of these boundaries?" |
| Discovery search (target + context) | same, with `target=<point_id_or_vector>` | When an anchor exists (suspect screenshot, seed image) — "similar to this AND on the bad side of the boundary" |
| Recommend (average_vector / best_score) | `client.query_points(query=RecommendQuery(recommend=RecommendInput(positive=[...], negative=[...], strategy=<s>)))` | Baseline/fallback when the investigator wants a single ranked cosine score rather than triplet satisfaction; also used when a concept has no negative examples yet |
| `lookup_from` | `lookup_from=LookupLocation(collection="reference_concepts", vector="dino")` | Phase 2: reference vectors live in a **separate** collection for chain-of-custody; the case collection only references them by ID |

### Feature shape

1. **Concept object** — a named, persisted set of positive/negative point IDs (+ optional target, + polarity), stored as a payload-only point in a sidecar Qdrant collection `sfn_concepts`. Mirrors the existing `is_video=True` payload-only-anchor pattern from `indexer.upsert_video_records`.

2. **Discovery engine** — `run_discovery` builds either a `DiscoverInput` (has pairs) or a `RecommendInput` (no negatives) and calls `client.query_points`. `run_discovery_dual` runs the query against both `dino` and `sscd` named vectors and fuses the two rankings: items present in **both** result sets rank higher than items in only one (semantic AND copy-detection agreeing is a strong signal). The fused score is `(triplet_dino + triplet_sscd, max(margin_dino, margin_sscd))`.

3. **Web API**:
   - `POST /api/concept` — create
   - `GET /api/concepts`, `GET /api/concept/{id}` — read
   - `POST /api/concept/{id}/mark` — append one `(point_id, "positive"|"negative")` — used by "mark" buttons on result cards for active learning during review
   - `DELETE /api/concept/{id}` — delete
   - `POST /api/triage` — run discovery against the case collection using a concept

4. **CLI** — `sfn-triage --concept <id> --limit N [--reverse] [--report out.jsonl]` for scripted / batch operation.

### Why this is innovative

- **Pair-based boundaries, not single-point thresholds.** 3 positives + 3 negatives defines 9 triplet constraints — far more expressive than any single reference + cosine cutoff, without training a classifier.
- **Active-learning without retraining.** The embedding model is frozen; the "model" that adapts is the concept. Every mark during review improves the next query.
- **Dual-vector fusion is forensically meaningful.** `dino` finds semantic cousins; `sscd` finds re-encoded / cropped / watermarked variants. Items appearing in both are the highest-confidence candidates.
- **Reverse (exculpatory) mode.** Flip polarity to surface material the concept places on the *benign* side — auto-hide from review queue to reduce exposure to traumatic content.
- **Chain-of-custody aware.** Externally-labelled known-bad material (NCMEC / Project VIC / CAID) can live in a separate `reference_concepts` collection referenced via `lookup_from` — the case collection never ingests those vectors.
- **Defensible scoring.** Integer triplet-satisfaction scores are legible outside the technical team.

---

## Files

**Existing (read before modifying):**
- `src/scalar_forensic/indexer.py:22-50` — `qdrant_scroll_all`
- `src/scalar_forensic/indexer.py:253-283` — `upsert_video_records` (payload-only-point pattern to mirror)
- `src/scalar_forensic/web/pipeline.py:860-909` — `_query_vector` (query_points call shape)
- `src/scalar_forensic/web/pipeline.py:937-969` — standalone `QdrantClient` construction
- `src/scalar_forensic/web/app.py:273-345` — `/api/query` envelope

**New:**
- `src/scalar_forensic/concepts.py` — Concept dataclass + persistence
- `src/scalar_forensic/discovery.py` — Discovery/Recommend engine + fusion
- `src/scalar_forensic/web/app.py` — five new routes (additions only)
- `src/scalar_forensic/cli.py` — `sfn-triage` entry point (new Typer app)
- `tests/test_concepts.py`, `tests/test_discovery.py`

**Modified:**
- `src/scalar_forensic/config.py` — add `concepts_collection` and `reference_collection`
- `pyproject.toml` — register `sfn-triage` console script

---

## Scope cut for v1

- **UI is deferred.** The REST endpoints land in v1 so operators can drive the feature via `curl`/CLI. The Alpine.js result-card "mark" buttons and Concept tab are a separate change.
- **`lookup_from` is stubbed but not wired to an ingestion path.** `reference_collection` is accepted from config and plumbed through `run_discovery`, but there is no CLI to populate it yet.
- **No authentication / audit logging** — the repo is currently airgapped-trust; separate change.
- **No case / evidence / device metadata** — separate change; concepts are standalone for now.

---

## Verification

**Unit tests:**
- `run_discovery` builds correct `DiscoverInput` (pairs present) vs `RecommendInput` (no negatives) vs reverse polarity.
- `run_discovery_dual` fusion: items in both rank above items in one.
- Concept persistence round-trip (create → get → mark → get → delete).

**Manual smoke:**
1. `docker compose up -d qdrant`, `sfn <fixture-dir>` with `--dino --sscd`.
2. `curl POST /api/concept` with 3 indexed point IDs as positive, 3 as negative.
3. `curl POST /api/triage` with the concept — verify `triplet_score`, `matched_modes: ["dino","sscd"]`.
4. `POST /api/concept/{id}/mark` with a new point ID → re-run triage → verify ranking changed.
5. `sfn-triage --concept <id> --reverse --limit 20 --report out.jsonl` — verify JSONL rows include per-vector triplet scores.
