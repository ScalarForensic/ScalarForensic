"""Cross-file face search (spec §8, Phase 1b).

The score this returns is a **raw cosine under a named model**.  There is no
face-calibration record in this deployment (spec §10) and the maintainer ruled on
2026-08-12 that the number is shown anyway, labelled for what it is.  The
constants below are the only place that copy lives.

One kNN per probe, merged — never an aggregate across observations.  Averaging
two faces' scores would produce a number that describes no comparison anyone
made, and an examiner cannot go back to the evidence behind it.
"""

from __future__ import annotations

UNCALIBRATED_BANNER = (
    "Uncalibrated — not evidential. The number below is the raw cosine similarity "
    "this model produced. There is no confidence interval and no calibrated "
    "threshold for this deployment."
)

# The SFace authors' published figure.  It is quoted, never applied: it is not a
# default for any slider, env var or server-side filter, and the score floor
# defaults to 0.0.  See the RULING in docs/specs/face-query-ux.md.
MODEL_REFERENCE_THRESHOLD = 0.363

MODEL_REFERENCE_NOTE = (
    "0.363 is the SFace authors' published same/different reference figure for this "
    "model. It is not a threshold calibrated on this deployment's material and must "
    "not be read as one."
)


def calibration_block() -> dict[str, object]:
    """The calibration status every face-search response carries."""
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
    """One kNN per probe, merged.  No aggregation across observations (spec §8).

    *probes* is ``(query_face_index, vector)``; the index travels onto every hit
    so the UI can say which of the examiner's selected faces produced it.
    """
    raw: list[dict] = []
    for face_index, vector in probes:
        for row in store.search_faces(vector, limit=limit, threshold=threshold, exact=exact):
            raw.append(_hit(row, face_index))

    raw.sort(key=lambda h: -h["score"])
    if not collapse:
        return raw

    # Collapse to one hit per medium, keeping the *best-scoring* observation —
    # raw is already score-descending, so the first sighting of a hash is it.
    # n_collapsed counts how many observations that hit stands for, so a
    # collapsed row never silently hides that a file matched more than once.
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
            # Only the review artefacts: the aligned PNG is addressed in its own
            # hash domain and a review-only observation has none, so emitting a
            # URL for it would be a guaranteed 404 that reads as a lost file.
            "review_url": f"/api/faces/chip/{review}/review" if review else None,
            "thumb_url": f"/api/faces/chip/{review}/thumb" if review else None,
        },
        # Derived from the timecode rather than the stored flag: a frame always
        # has one, and the flag is absent on observations written before it.
        "is_video_frame": row.get("frame_timecode_ms") is not None,
        "video_hash": row.get("video_hash"),
        "video_path": row.get("video_path"),
        "frame_timecode_ms": row.get("frame_timecode_ms"),
        "n_collapsed": 1,
    }
