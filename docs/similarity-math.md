# Similarity Mathematics in ScalarForensic

This document gives a complete, self-contained account of every similarity
formula used by the application — from raw pixel data to the scores shown in
the UI.  It is the authoritative reference for forensic experts who need to
understand or justify a numerical result.

---

## 1. Embedding: from pixels to vectors

Before any similarity can be computed, each image is converted to a fixed-length
numerical vector by a deep neural network.  The application uses two models.

### 1.1 DINOv2 (semantic similarity)

**Architecture:** Vision Transformer ViT-L/14, trained by Meta AI with
self-supervised DINO v2 objectives on diverse web imagery.

**What is extracted:** the `[CLS]` token from the final transformer layer.
This token aggregates a global, holistic summary of image content.  The raw
activation has dimension 1 024 (ViT-L hidden size).

**Normalization:** immediately after extraction the raw vector is L2-normalized:

```
v_dino = cls_token / ‖cls_token‖₂
```

The result is a unit vector (`‖v_dino‖₂ = 1`) on a 1 024-dimensional unit
hypersphere.

**Input resolution:** configurable via `SFN_NORMALIZE_SIZE` (default 224 px).
The preprocessor resizes the short side to `normalize_size`, then center-crops
to `normalize_size × normalize_size`, then applies ImageNet mean/std
normalization.  See `docs/normalization.md` for the full preprocessing pipeline
and the resolution vs. compute trade-off.

### 1.2 SSCD (copy detection)

**Architecture:** ResNet-50 with a gem-pooling head, trained by Meta AI with
a self-supervised copy-detection objective on image augmentation pairs.

**What is extracted:** the global gem-pool descriptor.  Dimension: 512.

**Normalization:** depends on `SFN_SSCD_N_CROPS`:

- **n_crops = 1 (default):** the center-crop embedding is returned as-is; the
  model's own output layer produces an L2-normalized vector (`‖v_sscd‖₂ = 1`).

- **n_crops = 5:** five crops (center + four corners) are embedded in one
  forward pass, each individually L2-normalized, then averaged and
  L2-normalized again:
  ```
  v_sscd = normalize( (1/5) Σᵢ normalize(eᵢ) )
  ```
  The double-normalization ensures the final vector is unit-norm regardless
  of how similar or dissimilar the crops are.

**Input resolution:** 331 px short side → five 288×288 crops.

---

## 2. Cosine similarity

All stored vectors are unit-norm.  The similarity between two images is their
**cosine similarity**, which equals the dot product when both vectors are
unit-norm:

```
sim(a, b)  =  a · b / (‖a‖₂ · ‖b‖₂)
           =  a · b          (because ‖a‖₂ = ‖b‖₂ = 1)
           =  Σᵢ aᵢ bᵢ
```

**Range:** [−1, 1].  Visual embeddings from the same model and domain are
almost always non-negative, so in practice the working range is [0, 1].

- **1.0** — vectors are identical (or indistinguishable by the model).
- **0.0** — vectors are orthogonal; no semantic relationship captured.
- **−1.0** — antipodal; not seen in practice for visual embeddings.

**Storage in Qdrant:** all collections are created with
`Distance.COSINE`.  Because vectors are already unit-norm when stored,
Qdrant's cosine computation (`1 − angular_distance`) is numerically equivalent
to a plain dot product.

**Local computation (NumPy, `query_eval.py`):** the same formula is implemented
explicitly with a small epsilon to guard against the zero-vector edge case:

```python
sims = (refs @ query) / (‖refs‖₂ · ‖query‖₂ + ε)   # ε = 1e-12
```

---

## 3. Query modes and their scoring functions

### 3.1 Exact mode

**Score:** none (binary match / no-match).

Two images match if and only if their SHA-256 hash over the raw file bytes is
identical.  No vector arithmetic is involved.

### 3.2 Altered mode (near-copy detection)

**Model:** SSCD  
**Score:** cosine similarity `sim(v_query_sscd, v_candidate_sscd)`  
**Default threshold:** 0.75 (configurable per query via the UI slider)

A candidate passes if `sim ≥ threshold`.  The score surfaced in the UI is the
raw cosine similarity.

SSCD was trained specifically to be invariant to common image manipulations
(colour adjustments, cropping, compression, watermarking) while remaining
sensitive to content differences.  It is appropriate for finding copies and
derivatives of a specific query image, not for category-level classification.

### 3.3 Semantic mode (category-level search)

**Model:** DINOv2  
**Score:** cosine similarity `sim(v_query_dino, v_candidate_dino)`  
**Default threshold:** 0.55 (configurable per query via the UI slider)

A candidate passes if `sim ≥ threshold`.  The score shown in the UI is the raw
cosine similarity.

DINOv2 captures high-level visual semantics (scene type, object category,
composition) and is appropriate for finding images that depict the same kind
of content as the query even when the specific content differs.

---

## 4. Tag Triage scoring

Tag Triage replaces single-query cosine thresholds with a richer, investigator-
defined decision boundary expressed as accumulated positive/negative reference
examples.

### 4.1 Triplet constraint and satisfaction score (Discovery mode)

Used when the tag has **at least one positive and at least one negative** reference.

**Context pairs:** the cartesian product of `positive_ids × negative_ids`,
capped at `MAX_CONTEXT_PAIRS = 64`.  When the product exceeds the cap, pairs
are sampled diagonal-first (round-robin) so that every positive and every
negative appears in at least one pair even under the cap.

```
pairs = { (pᵢ, nⱼ) | pᵢ ∈ positives, nⱼ ∈ negatives },  |pairs| ≤ 64
```

**Triplet satisfaction:** for a candidate c, one constraint `(pᵢ, nⱼ)` is
satisfied if c is cosine-closer to the positive reference than to the negative:

```
sat(c, pᵢ, nⱼ) = 𝟙[ sim(c, pᵢ) > sim(c, nⱼ) ]
              = 𝟙[ v_c · v_pᵢ  >  v_c · v_nⱼ ]    (unit-norm vectors)
```

**Triplet score** (the integer N displayed in "N / M"):

```
score(c) = Σ_{(pᵢ,nⱼ) ∈ pairs} sat(c, pᵢ, nⱼ)
```

- **Maximum possible score** M = `|pairs|` (≤ 64).
- **Score 0** — c is on the negative side of every constraint; strongly benign.
- **Score M** — c satisfies every constraint; very strong positive signal.

The integer nature of the score is intentional: it is directly legible to
reviewers and courts without requiring expertise in floating-point thresholds.

**Anchor (target):** Qdrant's Discovery API requires an anchor point that
defines the region of vector space to search.  The anchor determines *which
candidates are retrieved*; the triplet score then ranks them.  If no explicit
`target_id` is set on the tag, the first positive ID is used automatically.
To override, use "⚓ Set as anchor" in the UI.

### 4.2 Recommend mode (no negatives)

Used when `/api/triage` is called on a tag that has **positives but no
negatives** — a degenerate state that should not arise in normal use.  The
intended workflow (see `docs/tag-triage.md`) creates tags with at least one
positive and one negative from the outset so that Discovery mode is active
from the very first triage run.

**Score:** maximum cosine similarity to any positive reference:

```
score(c) = max_i sim(c, pᵢ)   =   max_i (v_c · v_pᵢ)
```

Strategy used: `BEST_SCORE` (Qdrant's recommendation strategy that picks the
single closest positive rather than averaging across positives).  This is
appropriate when the positive set is visually diverse — averaging would pull
the reference centroid toward an empty region of the space.

The UI notes "recommendation mode — add negatives to enable triplet scoring."

**Default threshold:** 0.5 (configurable per triage run via the "Cosine
threshold" slider, which is only shown when the active tag has no negatives).

### 4.3 Explore mode (label-bootstrapping)

Explore uses a `ContextQuery` (no anchor) rather than a `DiscoverQuery`
(anchored) when both positives and negatives are present:

```
score(c) = Σ_{(pᵢ,nⱼ) ∈ pairs} sat(c, pᵢ, nⱼ)     (same as §4.1)
```

The difference from triage: without an anchor, Qdrant evaluates the full
collection and returns candidates near the *decision boundary* (score ≈ M/2),
which are the most informative for the next round of labelling.

When either list is empty (cold start), a `SampleQuery(RANDOM)` is used:
candidates are drawn uniformly at random with no vector scoring.

### 4.4 Classify threshold (bulk classification)

`POST /api/tags/classify` and `POST /api/tags/classify-session` test many
images against all tags at once.  A tag is assigned to an image if the
triplet score clears a 75% threshold:

```
threshold(tag) = max(1,  ⌊|pairs| × 0.75⌋)
```

For a tag with 5 positives × 4 negatives = 20 pairs, the threshold is
`max(1, 15) = 15`.  An image must satisfy at least 15 of the 20 constraints
to be classified under that tag.

Implementation note: the code uses `n_pairs * 3 // 4`, which is identical to
`⌊n_pairs × 0.75⌋` for all non-negative integers (`n × 3 // 4 = ⌊3n/4⌋ = ⌊n × 0.75⌋`).

In Recommend mode (no negatives) the classify threshold is replaced by the
configurable `cosine_threshold` parameter (default 0.5).

---

## 5. Local NumPy scoring (query-image triage)

`POST /api/triage/query-images` evaluates uploaded session images against a
tag's reference vectors without storing anything in Qdrant.  The same triplet
logic is replicated in NumPy:

```python
# query_eval.py — _cosine_sims
sims(q, refs) = (refs @ q) / (‖refs‖_row_wise · ‖q‖ + ε)

# score_query_vector
pos_sims = sims(q, pos_vecs)
neg_sims = sims(q, neg_vecs)
triplet_score = Σ_{(i,j)} 𝟙[ pos_sims[i] > neg_sims[j] ]
cosine_margin = max(pos_sims)   # best positive similarity
```

This is mathematically identical to §4.1 and §4.2 for the on-disk case.
Only DINOv2 vectors are used.

---

## 6. Summary table

| Mode | Model | Score type | Formula |
|------|-------|------------|---------|
| `exact` | — | binary | SHA-256 hash equality |
| `altered` | SSCD 512-d | cosine similarity | `v_q · v_c` |
| `semantic` | DINOv2 1024-d | cosine similarity | `v_q · v_c` |
| `tag` (Discovery) | DINOv2 | integer triplet count | `Σ 𝟙[v_c·v_p > v_c·v_n]` |
| `tag` (Recommend) | DINOv2 | cosine similarity | `max_i (v_c · v_pᵢ)` |
| `explore` (context) | DINOv2 | integer triplet count | `Σ 𝟙[v_c·v_p > v_c·v_n]` |
| `explore` (random) | — | none | uniform random sample |

All cosine computations assume unit-norm vectors; the dot product and cosine
similarity are numerically identical under this assumption.

---

## 7. Why unit-norm vectors?

Storing unit-norm vectors has two practical consequences:

1. **Cosine = dot product.** The division by `‖a‖ · ‖b‖` in the cosine formula
   is always `1.0 / 1.0 = 1`, so the comparison reduces to a dot product.
   Dot products are a single matrix-multiply away from GPU hardware (cuBLAS,
   rocBLAS), making large-scale retrieval fast.

2. **Scale invariance.** The magnitude of a raw activation vector carries
   no semantic meaning for retrieval — only the direction matters.
   Normalizing removes magnitude as a confounding factor.

Both DINOv2 (`F.normalize(cls, p=2, dim=1)`) and SSCD (model output +
optional crop average + `F.normalize`) apply L2-normalization before any
vector is stored in Qdrant or used for scoring.
