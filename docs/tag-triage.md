# Tag Triage

Tag Triage is an investigator-in-the-loop retrieval mode that lets you encode accumulated review judgements ("these are hits, those are not") into a named, reusable **Tag** and use it to drive ranked Qdrant queries against the case collection.

It replaces single-image cosine queries with pair-based decision boundaries: every `(positive, negative)` pair you supply defines one triplet constraint. The resulting **triplet satisfaction score** — an integer count of how many constraints a candidate satisfies — is directly legible to reviewers and courts without needing to choose or defend a cosine threshold.

Only DINOv2 vectors are used for triage. SSCD is a copy-detector and not appropriate for category-based classification.

---

## Comparison with the existing query modes

| Mode | What it searches for | Score |
|------|---------------------|-------|
| `exact` | SHA-256 identical files | — |
| `altered` | Near-copies (SSCD cosine ≥ 0.75) | cosine |
| `semantic` | Visual semantic matches (DINOv2 cosine ≥ 0.55) | cosine |
| **tag** | Material satisfying investigator-labelled pair constraints | **integer triplet count** |

---

## Tags

A **Tag** is a named object that persists in a sidecar Qdrant collection (`sfn_tags` by default, configurable via `SFN_TAGS_COLLECTION`).

```
Tag {
    tag_id       # UUIDv5 derived deterministically from the name
    name         # human-readable, stable ID key
    positive_ids # Qdrant point IDs of reference hits
    negative_ids # Qdrant point IDs of known-benign material
    target_id    # optional explicit anchor (see below)
    notes        # free-form text
}
```

The `tag_id` is derived from `name` via UUIDv5, so creating the same name twice **replaces** the existing tag in place. Use `mark` to amend incrementally.

Point IDs stored in a tag are **not** copies of the vectors — Qdrant resolves them to vectors at query time, so the tag automatically reflects any re-indexing of the underlying points.

---

## Query selection

The engine picks the Qdrant API automatically based on what the tag contains:

| Tag state | Qdrant API used | Score meaning |
|---|---|---|
| Has positive + negative IDs | **DiscoverQuery** (anchor = `positives[0]` or explicit `target_id`) | integer triplet count |
| Has only positive IDs (no negatives) | **RecommendQuery** (best-score strategy) | cosine similarity |
| Has only negative IDs | error — add a positive or use `--reverse` | — |

---

## Triplet satisfaction score

For each `(positive_ref, negative_ref, candidate)` triplet, Qdrant increments the score by 1 if the candidate is closer to `positive_ref` than to `negative_ref`. The final score is the count of satisfied triplets.

Pairs are built as the cartesian product of `positive_ids × negative_ids`, capped at 64 (`_MAX_CONTEXT_PAIRS`). When the cap applies, sampling is round-robin so every positive and every negative appears in at least one pair.

**How to read the score:** the UI shows it as `N / M` where `N` is the satisfied count and `M` is the total pairs for that tag. A tag with 5 positives × 5 negatives defines 25 pairs. A score of 23/25 is a very strong signal; 2/25 suggests the candidate is on the benign side.

---

## The anchor

The **anchor** (called `target_id` in the API) is the reference image the search starts from. Qdrant Discovery works in two stages:

1. Find candidates that are similar to the anchor.
2. Rank them by how many of the tag's positive/negative triplet constraints they satisfy.

The anchor determines *which region* of the vector space the search explores; the triplet constraints determine *which side of the boundary* within that region ranks highest.

**Implicit anchor (default):** when no explicit `target_id` is set, the engine automatically uses `positives[0]` as the anchor. This is sufficient for most cases — the first positive example already points the search at the right region.

**Explicit anchor:** use "⚓ Set as anchor" on any triage hit to override the auto-anchor with a specific image. This is useful when you have a particular suspect image you want to use as the starting point while still applying the tag's accumulated positive/negative labels as the ranking boundary. To clear it and return to the auto-anchor, click the button again or use "Clear anchor" in the tag editor.

---

## Recommend mode (no negatives)

When a tag has positive examples but no negatives, Qdrant's Recommendation API is used instead of Discovery. The returned score is a cosine similarity to the closest positive, not a triplet count. The UI displays this as "X.XXX cosine similarity" and notes "recommendation mode — add negatives to enable triplet scoring."

Add at least one negative example (a clearly benign image) to activate Discovery and get the more interpretable triplet scores.

---

## Reverse mode

Passing `reverse=True` swaps positive and negative lists at query time, surfacing material the tag places on the **benign** side of the boundary. This is used to auto-hide low-risk material from the review queue, reducing examiner exposure to traumatic content.

---

## Active learning

The `mark` operation appends a point ID to a tag's positive or negative list without recreating the tag. If the point is already in the other list it is moved (most-recent label wins). Re-running triage after a round of marking produces an updated ranking — the tag is the only thing that changes between runs; the embedding models stay frozen.

---

## Query-image evaluation

In addition to querying against the indexed case collection, the web UI exposes a **Query images** source toggle that evaluates a tag against images uploaded in the current session — without ingesting them into the case collection.

This is useful for rapid triage of suspected material before deciding whether to formally ingest it:

1. Upload one or more query images via the standard upload flow.
2. In the Tag Triage panel, switch the source toggle to **Query images**.
3. Click Run — the tag's reference vectors are fetched from Qdrant and the triplet scoring is computed locally in NumPy against the session's query image embeddings.

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `SFN_TAGS_COLLECTION` | `sfn_tags` | Qdrant sidecar collection for tag persistence |
| `SFN_REFERENCE_COLLECTION` | *(unset)* | Separate collection for externally-labelled reference material (NCMEC/CAID). When set, tag point IDs are resolved against this collection instead of the case collection — a chain-of-custody boundary so known-bad hashes never enter the case collection. |

---

## REST API

All endpoints are served by the ScalarForensic web server.

### `GET /api/tags`

List all tags, ordered by most recently updated.

**Response**
```json
{
  "tags": [
    {
      "tag_id": "...",
      "name": "...",
      "positive_ids": [...],
      "negative_ids": [...],
      "target_id": null,
      "notes": "",
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

### `POST /api/tag`

Create or replace a tag by *name*.

**Form fields**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Human-readable name (also the stable ID key) |
| `positive_ids` | no | Comma-separated Qdrant point IDs |
| `negative_ids` | no | Comma-separated Qdrant point IDs |
| `target_id` | no | Single anchor point ID (overrides auto-anchor) |
| `notes` | no | Free-form text |

### `GET /api/tag/{tag_id}`

Retrieve a single tag by its UUID.

### `POST /api/tag/{tag_id}/mark`

Append a point ID to the tag's positive or negative list.

**Form fields**: `point_id` (required), `role` (`"positive"` or `"negative"`, required).

### `POST /api/tag/{tag_id}/unmark`

Remove a point ID from both lists.

**Form fields**: `point_id` (required).

### `POST /api/tag/{tag_id}/set-target`

Set or clear the explicit anchor for a tag.

**Form fields**: `target_id` (optional — omit or send empty to clear and return to auto-anchor).

### `DELETE /api/tag/{tag_id}`

Delete a tag. Returns 404 if not found.

### `POST /api/triage`

Run a tag query against the indexed case collection using DINOv2.

**Form fields**

| Field | Default | Description |
|---|---|---|
| `tag_id` | required | UUID of the tag to run |
| `limit` | `50` | Maximum result count (1–500) |
| `reverse` | `false` | Swap positive/negative roles (exculpatory mode) |

**Response**
```json
{
  "tag": { "tag_id": "...", "name": "...", ... },
  "reverse": false,
  "limit": 50,
  "hits": [
    {
      "point_id": "...",
      "triplet_score": 9,
      "cosine_margin": 0.812,
      "path": "...",
      "image_hash": "...",
      "is_video_frame": false,
      "video_path": null,
      "frame_timecode_ms": null
    }
  ]
}
```

`triplet_score` is `null` when the tag has no negatives (Recommend mode); in that case `cosine_margin` carries the cosine similarity.

### `POST /api/tags/classify`

Bulk-evaluate a list of image hashes against all tags using in-memory triplet scoring.

For each hash, returns the names of tags whose triplet threshold the image satisfies (≥ half the defined pairs).

**JSON body**
```json
{
  "image_hashes": ["sha256hex1", "sha256hex2"]
}
```

**Response**
```json
{
  "by_hash": {
    "sha256hex1": ["weapons", "csam"],
    "sha256hex2": []
  }
}
```

### `POST /api/tags/classify-session`

Evaluate tag membership for all images in an active session using their in-memory embeddings.

**JSON body**: `{"session_id": "..."}`.  
**Response**: same `by_hash` shape as `/api/tags/classify`.

### `POST /api/triage/query-images`

Evaluate a tag against uploaded query images in the current session, without running a Qdrant search against the indexed collection.

**Form fields**

| Field | Default | Description |
|---|---|---|
| `tag_id` | required | UUID of the tag to evaluate |
| `session_id` | required | Session ID from the upload flow |
| `limit` | `50` | Maximum result count |

**Response** — same hit shape as `/api/triage`, with `file_id` replacing `point_id` and `filename` instead of `path`.

---

## CLI (`sfn-triage`)

### Create a tag

```sh
sfn-triage create "beach-grooming" \
  --positive "abc123,def456,ghi789" \
  --negative "jkl012,mno345" \
  --notes "beach location, adult-child proximity"
```

Use `--target <point_id>` to set an explicit anchor.

### List tags

```sh
sfn-triage list
```

### Inspect a tag

```sh
sfn-triage show <tag_id>
```

### Mark a reference

```sh
sfn-triage mark <tag_id> <point_id> --role positive
sfn-triage mark <tag_id> <point_id> --role negative
```

### Run triage

```sh
# Print ranked hits to stdout
sfn-triage run <tag_id> --limit 100

# Write JSONL report
sfn-triage run <tag_id> --limit 100 --report hits.jsonl

# Reverse mode (exculpatory)
sfn-triage run <tag_id> --reverse --limit 50 --report benign.jsonl
```

JSONL rows include `tag_id`, `point_id`, `triplet_score`, `cosine_margin`, `path`, `image_hash`, `is_video_frame`, `video_path`, and `frame_timecode_ms`.

### Delete a tag

```sh
sfn-triage delete <tag_id>
```

Exits with code 0 if the tag existed, 1 if not found.

### Migrate concepts to tags

If you have an existing deployment using the old `sfn_concepts` collection:

```sh
sfn-triage migrate-concepts-to-tags
```

---

## Typical workflow

1. Index the case collection with `sfn --dino`.
2. Run an initial semantic query (`/api/query`) to surface a first batch of candidates.
3. Create a tag with a few clearly relevant (positive) and clearly benign (negative) points from that batch.
4. Run `/api/triage` — the triplet-ranked results replace the cosine-ranked list. The first positive is used as the Discovery anchor automatically; the UI shows scores as `N / M pairs satisfied`.
5. For each new result card, click **+** or **−** to mark the point as positive or negative for the active tag.
6. Re-run triage. The ranking adapts to accumulated labels without retraining any model.
7. If the default anchor (first positive) is not giving good results, use "⚓ Set as anchor" on any hit to pin a more representative image.
8. Use reverse mode at the end of the session to identify low-risk material that can be excluded from the formal review queue.
9. Optionally use `/api/tags/classify` or the **Query images** source to evaluate new material before ingestion.
