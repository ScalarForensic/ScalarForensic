"""Concept-Triage: Qdrant Discovery and Recommendation query engine.

Builds :class:`~qdrant_client.models.DiscoverInput` or
:class:`~qdrant_client.models.RecommendInput` queries from a
:class:`~scalar_forensic.tags.Tag` and runs them through the
unified ``client.query_points`` endpoint against the DINOv2 (``dino``)
named vector.  Only DINOv2 is used for semantic triage — SSCD is a
copy-detector and is not appropriate for category-based tagging.

Scoring semantics
-----------------

* When the tag has at least one ``(positive, negative)`` pair,
  Qdrant Discovery is used and the returned ``score`` is an *integer*
  triplet-satisfaction count — for each pair the candidate is closer
  to the positive than the negative, the score increments by one.
* When the tag has only positive examples (or only a target, no
  negatives), Qdrant Recommendation is used instead and the returned
  ``score`` is a cosine similarity.  The ``triplet_score`` field on
  the returned :class:`DiscoveryHit` is left as ``None`` in that case
  and the ``cosine_margin`` field carries the cosine score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    ContextPair,
    ContextQuery,
    DiscoverInput,
    DiscoverQuery,
    Filter,
    HasIdCondition,
    LookupLocation,
    RecommendInput,
    RecommendQuery,
    RecommendStrategy,
    Sample,
    SampleQuery,
)

from scalar_forensic.tags import Tag


@runtime_checkable
class ConceptLike(Protocol):
    """Structural interface shared by Concept and Tag."""
    positive_ids: list[str | int]
    negative_ids: list[str | int]
    target_id: str | int | None

logger = logging.getLogger(__name__)

# The "with_payload" field list that every triage hit needs — a superset
# of what the existing vector-search results carry, so the web response
# envelope for /api/triage can reuse the /api/query hit-card renderer.
_TRIAGE_PAYLOAD_FIELDS: list[str] = [
    "image_path",
    "image_hash",
    "exif",
    "exif_geo_data",
    "is_video_frame",
    "video_path",
    "video_hash",
    "frame_timecode_ms",
    "dino_model_name",
    "dino_model_hash",
    "sscd_model_name",
    "sscd_model_hash",
]

# Hard cap on the number of context pairs built from a concept's
# positive × negative cartesian product.  More pairs tighten the
# decision boundary but also slow the server-side triplet evaluation.
# 64 is well below any Qdrant operational limit while still giving
# the investigator a lot of expressive headroom (e.g. 8 positives ×
# 8 negatives = 64 pairs).
_MAX_CONTEXT_PAIRS = 64

# Recommendation strategy used when the concept has no negatives.
# BEST_SCORE matches the intent "most similar to any of my positives"
# better than AVERAGE_VECTOR when the positive set is visually diverse.
_DEFAULT_RECOMMEND_STRATEGY = RecommendStrategy.BEST_SCORE


@dataclass
class DiscoveryHit:
    """One ranked result from a concept query against a single named vector."""

    point_id: str | int
    vector_name: str  # "dino" or "sscd"
    # None when the concept has no negatives (Recommend fallback path).
    triplet_score: int | None
    # Raw Qdrant score.  For Discovery queries this is the same integer
    # value reported as triplet_score; for Recommend queries it is the
    # cosine similarity to the best positive example.
    cosine_margin: float
    payload: dict = field(default_factory=dict)



def _build_context_pairs(
    positives: list[str | int], negatives: list[str | int]
) -> list[ContextPair]:
    """Return the cartesian product of (positive, negative) pairs, capped.

    The cap is applied by round-robin sampling rather than truncation so
    every positive and every negative is represented at least once when
    the cap kicks in.
    """
    pairs: list[ContextPair] = []
    if not positives or not negatives:
        return pairs
    # Round-robin interleaving: for each pair index k, pick positive
    # k % len(pos) and negative k % len(neg).  This guarantees coverage
    # until the cap is hit.
    total = len(positives) * len(negatives)
    limit = min(total, _MAX_CONTEXT_PAIRS)
    full: list[tuple[str | int, str | int]] = [
        (p, n) for p in positives for n in negatives
    ]
    # Re-order so that the first min(len(pos), len(neg)) entries cover a
    # distinct positive *and* a distinct negative, guaranteeing each
    # reference contributes to the boundary even when the cap is low.
    reordered: list[tuple[str | int, str | int]] = []
    short = min(len(positives), len(negatives))
    for i in range(short):
        reordered.append((positives[i], negatives[i]))
    for pair in full:
        if pair not in reordered:
            reordered.append(pair)
    for pos, neg in reordered[:limit]:
        pairs.append(ContextPair(positive=pos, negative=neg))
    return pairs


def _exclude_filter(point_ids: list[str | int]) -> Filter | None:
    """Return a Qdrant ``Filter`` that excludes the given point IDs, or None."""
    if not point_ids:
        return None
    return Filter(must_not=[HasIdCondition(has_id=point_ids)])


def _merge_filter(user_filter: Filter | None, exclude: Filter | None) -> Filter | None:
    """Conjoin a caller-supplied filter with the must_not-exclude filter."""
    if exclude is None:
        return user_filter
    if user_filter is None:
        return exclude
    # Qdrant Filter is dataclass-like; build a fresh one combining both.
    combined_must_not = list(user_filter.must_not or []) + list(exclude.must_not or [])
    return Filter(
        must=user_filter.must,
        should=user_filter.should,
        must_not=combined_must_not,
    )


def _resolve_polarity(
    tag: Tag, reverse: bool
) -> tuple[list[str | int], list[str | int]]:
    """Apply reverse-polarity by swapping positive and negative lists.

    Reverse triage turns "more like positives, less like negatives"
    into "more like negatives, less like positives" — used to surface
    provably-benign material that can be excluded from review.  Callers
    opt in via the *reverse* flag.
    """
    if reverse:
        return list(tag.negative_ids), list(tag.positive_ids)
    return list(tag.positive_ids), list(tag.negative_ids)


def run_discovery(
    client: QdrantClient,
    collection: str,
    tag: Tag,
    *,
    vector_name: str,
    limit: int = 50,
    filter_: Filter | None = None,
    reverse: bool = False,
    reference_collection: str | None = None,
    exclude_references: bool = True,
) -> list[DiscoveryHit]:
    """Run a tag query against a single named vector.

    Chooses the query kind automatically:

    * Tag has negatives → Discovery (target+context or context-only).
    * Tag has only positives (and optionally a target) → Recommend.
    * Tag has neither → raises :class:`ValueError`.

    When at least one positive is present and no explicit target is set,
    the first positive is used as an implicit anchor so DiscoverQuery fires
    immediately without the user needing to call "Set from hit".

    When *exclude_references* is True (the default), the tag's own
    reference points are filtered out of the result set so the top hits
    are discovered material rather than the examples the investigator
    already labelled.
    """
    positives, negatives = _resolve_polarity(tag, reverse)
    target = tag.target_id
    # Auto-anchor: use the first positive as implicit target when no
    # explicit target is set — always fires DiscoverQuery when possible.
    if target is None and positives:
        target = positives[0]

    if not positives and not negatives and target is None:
        raise ValueError(
            f"Tag {tag.tag_id!r} has no positive, negative, or target "
            "references; cannot build a Discovery or Recommend query."
        )

    ref_ids: list[str | int] = []
    if exclude_references:
        ref_ids.extend(positives)
        ref_ids.extend(negatives)
        if target is not None and target not in ref_ids:
            ref_ids.append(target)
    query_filter = _merge_filter(filter_, _exclude_filter(ref_ids))

    lookup_from = (
        LookupLocation(collection=reference_collection, vector=vector_name)
        if reference_collection
        else None
    )

    pairs = _build_context_pairs(positives, negatives)

    if pairs:
        # Qdrant distinguishes two context-based query kinds:
        #   • ``DiscoverQuery`` requires a non-null ``target`` (an anchor).  It
        #     returns points similar to the target AND satisfying the triplet
        #     constraints encoded by the context pairs.
        #   • ``ContextQuery`` is context-only (no target).  It returns points
        #     ranked purely by triplet-satisfaction over the pairs.
        # The auto-target logic above ensures target is set whenever a positive
        # exists, so ContextQuery is only reached on negative-only (reversed) tags.
        if target is not None:
            query = DiscoverQuery(
                discover=DiscoverInput(target=target, context=pairs)
            )
        else:
            query = ContextQuery(context=pairs)
    else:
        if not positives and target is None:
            raise ValueError(
                f"Tag {tag.tag_id!r} has only negative references; "
                "cannot build a Recommend query (Qdrant requires ≥1 positive "
                "or a target).  Add a positive example or a target_id, or use "
                "reverse=True if this tag is exculpatory."
            )
        query = RecommendQuery(
            recommend=RecommendInput(
                positive=[target, *positives] if target is not None else positives,
                negative=None,
                strategy=_DEFAULT_RECOMMEND_STRATEGY,
            )
        )

    try:
        result = client.query_points(
            collection_name=collection,
            query=query,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=_TRIAGE_PAYLOAD_FIELDS,
            lookup_from=lookup_from,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Discovery query failed on %s/%s: %s", collection, vector_name, exc
        )
        return []

    is_discover = bool(pairs)
    hits: list[DiscoveryHit] = []
    for r in result.points:
        triplet: int | None = int(round(r.score)) if is_discover else None
        hits.append(
            DiscoveryHit(
                point_id=r.id,
                vector_name=vector_name,
                triplet_score=triplet,
                cosine_margin=float(r.score),
                payload=r.payload or {},
            )
        )
    return hits


def run_explore(
    client: QdrantClient,
    collection: str,
    positive_ids: list[str | int],
    negative_ids: list[str | int],
    *,
    vector_name: str = "dino",
    limit: int = 50,
    filter_: Filter | None = None,
) -> tuple[list[DiscoveryHit], str]:
    """Surface candidates for tag bootstrapping and iterative diversity injection.

    Two strategies, chosen automatically:

    * **context** — both *positive_ids* and *negative_ids* non-empty:
      ``ContextQuery`` ranks points by how many (positive, negative) triplet
      pairs they satisfy.  Points near the decision boundary score at roughly
      half the pair count, making them the most informative for labelling.
    * **random** — one or both lists empty:
      ``SampleQuery(Sample.Random)`` returns uniformly random points with no
      vector scoring, giving unbiased cold-start coverage.

    Already-labelled points (both lists combined) are excluded so successive
    explore runs surface fresh candidates.

    Returns ``(hits, strategy)`` where *strategy* is ``"context"`` or
    ``"random"``.
    """
    exclude_ids: list[str | int] = list(positive_ids) + list(negative_ids)
    query_filter = _merge_filter(filter_, _exclude_filter(exclude_ids))

    use_context = bool(positive_ids) and bool(negative_ids)

    if use_context:
        pairs = _build_context_pairs(positive_ids, negative_ids)
        query: ContextQuery | SampleQuery = ContextQuery(context=pairs)
        using: str | None = vector_name
    else:
        query = SampleQuery(sample=Sample.RANDOM)
        using = None  # random sampling is vector-agnostic

    try:
        result = client.query_points(
            collection_name=collection,
            query=query,
            using=using,
            limit=limit,
            query_filter=query_filter,
            with_payload=_TRIAGE_PAYLOAD_FIELDS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Explore query failed on %s: %s", collection, exc)
        return [], "random"

    strategy = "context" if use_context else "random"
    hits: list[DiscoveryHit] = []
    for r in result.points:
        triplet = int(round(r.score)) if use_context else None
        hits.append(
            DiscoveryHit(
                point_id=r.id,
                vector_name=vector_name if use_context else "random",
                triplet_score=triplet,
                cosine_margin=float(r.score) if use_context else 0.0,
                payload=r.payload or {},
            )
        )
    return hits, strategy


def run_triage(
    client: QdrantClient,
    collection: str,
    tag: Tag,
    *,
    limit: int = 50,
    filter_: Filter | None = None,
    reverse: bool = False,
    reference_collection: str | None = None,
    exclude_references: bool = True,
) -> list[DiscoveryHit]:
    """Run a tag triage query using DINOv2 semantic embeddings only."""
    return run_discovery(
        client,
        collection,
        tag,
        vector_name="dino",
        limit=limit,
        filter_=filter_,
        reverse=reverse,
        reference_collection=reference_collection,
        exclude_references=exclude_references,
    )
