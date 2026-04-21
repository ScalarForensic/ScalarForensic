# Concept Triage

Concept Triage is an investigator-in-the-loop retrieval mode that lets you encode accumulated review judgements ("these are incriminating, those are not") into a named, reusable **Concept** and use it to drive ranked Qdrant queries against the case collection.

It replaces single-image cosine queries with pair-based decision boundaries: every `(positive, negative)` pair you supply defines one triplet constraint. The resulting **triplet satisfaction score** — an integer count of how many constraints a candidate satisfies — is directly legible to reviewers and courts without needing to choose or defend a cosine threshold.

---

## Comparison with the existing query modes

| Mode | What it searches for | Score |
|------|---------------------|-------|
| `exact` | SHA-256 identical files | — |
| `altered` | Near-copies (SSCD cosine ≥ 0.75) | cosine |
| `semantic` | Visual semantic matches (DINOv2 cosine ≥ 0.55) | cosine |
| **concept** | Material satisfying investigator-labelled pair constraints | **integer triplet count** |

---

## Concepts

A **Concept** is a named object that persists in a sidecar Qdrant collection (`sfn_concepts` by default, configurable via `SFN_CONCEPTS_COLLECTION`).

```
Concept {
    concept_id   # UUIDv5 derived deterministically from the name
    name         # human-readable, stable ID key
    positive_ids # Qdrant point IDs of incriminating reference material
    negative_ids # Qdrant point IDs of known-benign material
    target_id    # optional single anchor point (activates Discovery mode)
    polarity     # "incriminating" | "exculpatory" — metadata only
    notes        # free-form text
}
```

The `concept_id` is derived from `name` via UUIDv5, so creating the same name twice **replaces** the existing concept in place. Use `mark` to amend incrementally.

Point IDs stored in a concept are **not** copies of the vectors — Qdrant resolves them to vectors at query time, so the concept automatically reflects any re-indexing of the underlying points.

---

## Query selection

The engine picks the Qdrant API automatically based on what the concept contains:

| Concept state | Qdrant API used | Score meaning |
|---|---|---|
| Has positive + negative IDs | **ContextQuery** (no target) | integer triplet count |
| Has positive + negative IDs + target | **DiscoverQuery** (target + context) | integer triplet count |
| Has only positive IDs (or target, no negatives) | **RecommendQuery** (best-score strategy) | cosine similarity |
| Has only negative IDs | error — add a positive or use `--reverse` | — |

### Context pairs and the cap

Pairs are built as the cartesian product of `positive_ids × negative_ids`, capped at 64 pairs (`_MAX_CONTEXT_PAIRS`). When the cap applies, the sampling is done round-robin to guarantee every positive and every negative appears in at least one pair rather than silently dropping references.

### Triplet satisfaction score

For each `(positive_ref, negative_ref, candidate)` triplet, Qdrant increments the score by 1 if the candidate is closer to `positive_ref` than to `negative_ref`. The final score is the count of satisfied triplets. Higher is stronger.

A concept with 5 positives × 5 negatives defines 25 constraints. A score of 23/25 is a very strong positive signal; a score of 2/25 suggests the candidate belongs on the benign side.

---

## Dual-vector fusion

Running in `mode=dual` (the default) issues the concept query against **both** the `dino` vector (DINOv2, 1024-dim, semantic) and the `sscd` vector (SSCD, 512-dim, copy-detection) independently, then fuses the results.

Fusion sort order:

1. **`matched_modes` count** — items ranked by both `dino` and `sscd` (length 2) sort above items ranked by only one.
2. **`fused_triplet_score`** — sum of per-vector triplet scores.
3. **`fused_cosine_margin`** — max of per-vector cosine scores.

Cross-space agreement is forensically meaningful: DINOv2 finds semantic cousins; SSCD finds re-encoded, cropped, or watermarked variants. An item that both spaces place on the incriminating side of the concept boundary is a stronger candidate than one that only one space agrees with.

---

## Reverse / exculpatory mode

Passing `reverse=True` swaps positive and negative lists at query time, surfacing material the concept places on the **benign** side of the boundary. This is used to auto-hide low-risk material from the review queue, reducing examiner exposure to traumatic content.

The concept's stored `polarity` field is metadata only — it is a label for human reference and audit. The actual swap is controlled entirely by the `reverse` flag at query time.

---

## Active learning

The `mark` operation appends a point ID to a concept's positive or negative list without recreating the concept. If the point is already in the other list it is moved (most-recent label wins). Re-running triage after a round of marking produces an updated ranking — the concept is the only thing that changes between runs; the embedding models stay frozen.

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `SFN_CONCEPTS_COLLECTION` | `sfn_concepts` | Qdrant sidecar collection for concept persistence |
| `SFN_REFERENCE_COLLECTION` | *(unset)* | Separate collection for externally-labelled reference material (NCMEC/CAID). When set, concept point IDs are resolved against this collection instead of the case collection. |

The reference collection pathway (`lookup_from`) supports chain-of-custody requirements: known-bad hashes from external databases never need to be ingested into the case collection; the case collection only ever references them by ID.

---

## REST API

All endpoints are served by the ScalarForensic web server.

### `GET /api/concepts`

List all concepts, ordered by most recently updated.

**Response**
```json
{
  "concepts": [
    {
      "concept_id": "...",
      "name": "...",
      "positive_ids": [...],
      "negative_ids": [...],
      "target_id": null,
      "polarity": "incriminating",
      "notes": "",
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

### `POST /api/concept`

Create or replace a concept.

**Form fields**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Human-readable name (also the stable ID key) |
| `positive_ids` | no | Comma-separated Qdrant point IDs |
| `negative_ids` | no | Comma-separated Qdrant point IDs |
| `target_id` | no | Single anchor point ID |
| `polarity` | no | `"incriminating"` (default) or `"exculpatory"` |
| `notes` | no | Free-form text |

### `GET /api/concept/{concept_id}`

Retrieve a single concept by its UUID.

### `POST /api/concept/{concept_id}/mark`

Append a point ID to the concept's positive or negative list.

**Form fields**: `point_id` (required), `role` (`"positive"` or `"negative"`, required).

### `POST /api/concept/{concept_id}/unmark`

Remove a point ID from the concept's positive and negative lists.

**Form fields**: `point_id` (required).

### `DELETE /api/concept/{concept_id}`

Delete a concept. Returns 404 if not found.

### `POST /api/triage`

Run a concept query against the case collection.

**Form fields**

| Field | Default | Description |
|---|---|---|
| `concept_id` | required | UUID of the concept to run |
| `mode` | `dual` | `"dual"`, `"dino"`, or `"sscd"` |
| `limit` | `50` | Maximum result count (1–1000) |
| `reverse` | `false` | Exculpatory mode — swap positive/negative roles |

**Response**
```json
{
  "hits": [
    {
      "point_id": "...",
      "matched_modes": ["dino", "sscd"],
      "triplet_score_dino": 9,
      "triplet_score_sscd": 8,
      "cosine_margin_dino": 4.0,
      "cosine_margin_sscd": 3.0,
      "fused_triplet_score": 17,
      "fused_cosine_margin": 4.0,
      "image_path": "...",
      "image_hash": "...",
      "is_video_frame": false,
      "video_path": null,
      "frame_timecode_ms": null
    }
  ],
  "concept_id": "...",
  "mode": "dual"
}
```

---

## CLI (`sfn-triage`)

The `sfn-triage` console script covers the same operations without the web UI, suitable for scripted / batch workflows.

### Create a concept

```sh
sfn-triage create "beach-grooming" \
  --positive "abc123,def456,ghi789" \
  --negative "jkl012,mno345" \
  --notes "beach location, adult-child proximity"
```

Returns JSON including the assigned `concept_id`.

### List concepts

```sh
sfn-triage list
```

### Inspect a concept

```sh
sfn-triage show <concept_id>
```

### Mark a reference

```sh
sfn-triage mark <concept_id> <point_id> --role positive
sfn-triage mark <concept_id> <point_id> --role negative
```

### Run triage

```sh
# Print ranked hits to stdout
sfn-triage run <concept_id> --mode dual --limit 100

# Write JSONL report
sfn-triage run <concept_id> --limit 100 --report hits.jsonl

# Exculpatory / reverse mode
sfn-triage run <concept_id> --reverse --limit 50 --report benign.jsonl
```

JSONL rows include `concept_id`, `point_id`, per-vector triplet and cosine scores, `matched_modes`, `path`, `image_hash`, `is_video_frame`, `video_path`, and `frame_timecode_ms`.

### Delete a concept

```sh
sfn-triage delete <concept_id>
```

Exits with code 0 if the concept existed, 1 if not found.

---

## Typical workflow

1. Index the case collection with `sfn --dino --sscd`.
2. Run an initial semantic query (`/api/query`) to surface a first batch of candidates.
3. Create a concept with a few clearly incriminating and clearly benign points from that batch.
4. Run `/api/triage` — the triplet-ranked results replace the cosine-ranked list.
5. For each new result card, click **mark positive** or **mark negative** to update the concept.
6. Re-run triage. The ranking adapts to accumulated labels without retraining any model.
7. Use `--reverse` at the end of the session to identify low-risk material that can be excluded from the formal review queue.
