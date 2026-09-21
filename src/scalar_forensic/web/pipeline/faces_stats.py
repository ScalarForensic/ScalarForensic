"""Face-modality audit and score-distribution statistics (spec §8, Phase 1b).

Two read-only reports over the face collection:

* ``face_score_stats`` — the distribution of raw cosines one query face draws out
  of the collection.  Its arithmetic mirrors ``web/pipeline/stats.py`` field for
  field and line for line, because the point of the feature is that the semantic
  and the face modality read alike in the UI: the same percentiles, the same
  20-bucket histogram, the same two scales.  The numeric fields stay on the
  **raw cosine** scale; only the histogram is normalised to ``[0, 1]``.
* ``face_audit`` — how the machine produced the face observations of one indexed
  medium.  Assembled *entirely from persisted payload*: no pipeline step is
  re-run and the current environment is never consulted, so the report describes
  the gates that were in force at index time rather than the ones in force today.

Nothing here applies a threshold.  This deployment has no calibration record
(spec §10 DEPLOYMENT DIVERGENCE); the SFace authors' 0.363 travels as an
annotation on a plot and never as a filter, so the distribution is drawn with
``threshold=0.0``.
"""

from __future__ import annotations

import statistics as _statistics
from dataclasses import dataclass
from types import SimpleNamespace

from scalar_forensic.faces.provenance import EMBEDDER_IDENTITY_FIELDS, LEGACY_QUALITY_GATE_VERSION
from scalar_forensic.faces.store import _HARD_FIELDS, _SOFT_FIELDS

_HIST_BUCKETS = 20  # 0.05-wide buckets covering normalised [0.0, 1.0] (cosine [-1,1] → [0,1])


@dataclass
class FaceScoreStats:
    """The same statistics as ``SemanticStats``, with explicit scale metadata."""

    sample_size: int  # how many points were requested
    count: int  # how many were actually returned
    min_score: float
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    max_score: float
    mean: float
    stdev: float
    # _HIST_BUCKETS counts on normalised [0,1] scale; bucket i covers [i*0.05, (i+1)*0.05)
    histogram: list[int]
    score_scale: str = "raw_cosine"
    # Histogram positions use (clamp(score, -1, 1) + 1) / 2, not raw cosine.
    histogram_scale: str = "normalized_cosine"


def face_score_stats(store, vector: list[float], *, sample_size: int) -> FaceScoreStats:
    """Score distribution of one probe vector against the face collection.

    Exact top-N neighbors with no score floor. This is a selected nearest-neighbor
    sample, not an estimate of the full collection distribution. The eligible
    observations are *embedded* — review-only observations carry
    no vector and are structurally absent from it (spec §6.2).
    """
    rows = store.search_faces(vector, limit=sample_size, threshold=None, exact=True)
    scores = [float(r["score"]) for r in rows]

    n = len(scores)
    if n == 0:
        # An empty collection is a fact about the population, not an error: the
        # caller still gets a well-formed, obviously-empty distribution.
        return FaceScoreStats(
            sample_size=sample_size,
            count=0,
            min_score=0.0,
            p10=0.0,
            p25=0.0,
            median=0.0,
            p75=0.0,
            p90=0.0,
            max_score=0.0,
            mean=0.0,
            stdev=0.0,
            histogram=[0] * _HIST_BUCKETS,
        )

    mean = sum(scores) / n
    stdev = _statistics.stdev(scores) if n >= 2 else 0.0

    if n >= 2:
        cuts = _statistics.quantiles(scores, n=100, method="inclusive")
        p10 = cuts[9]
        p25 = cuts[24]
        median = cuts[49]
        p75 = cuts[74]
        p90 = cuts[89]
    else:
        p10 = p25 = median = p75 = p90 = scores[0]

    histogram = [0] * _HIST_BUCKETS
    for s in scores:
        # Normalise cosine score from [-1, 1] to [0, 1] before bucketing.
        # Note: the numeric stats (min/max/percentiles) are on the raw cosine scale;
        # the histogram intentionally uses the normalised scale so the UI threshold
        # marker (also normalised before bucket lookup) aligns correctly.
        normalised = (max(-1.0, min(s, 1.0)) + 1.0) / 2.0
        idx = min(max(int(normalised * _HIST_BUCKETS), 0), _HIST_BUCKETS - 1)
        histogram[idx] += 1

    return FaceScoreStats(
        sample_size=sample_size,
        count=n,
        min_score=min(scores),
        p10=p10,
        p25=p25,
        median=median,
        p75=p75,
        p90=p90,
        max_score=max(scores),
        mean=mean,
        stdev=stdev,
        histogram=histogram,
    )


_GATE_FIELDS = (
    "quality_gate_version",
    "min_conf",
    "min_size",
    "max_pose",
    "min_sharpness",
    "max_clipped",
    "review_min_conf",
    "review_min_size",
    "crop_dilation",
)

_DETECTOR_FIELDS = (
    "detector_id",
    "detector_model_hash",
    "detector_score_threshold",
    "detect_max_size",
)

_AUDIT_CAVEAT = (
    "This describes how the machine produced these face observations. It is an "
    "investigative lead, not an identification."
)


def _stored_config(row: dict) -> SimpleNamespace:
    """The observation's own recorded configuration, as a comparison target.

    ``check_compat`` reads the hard and soft fields off an attribute bag; feeding
    it the *persisted* values rather than a config built from today's env is what
    keeps this report a statement about index time.  A field the payload never
    carried comes through as ``None``; absent hard provenance cannot establish
    comparability with a collection carrying that identity.
    """
    return SimpleNamespace(**{f: row.get(f) for f in (*_HARD_FIELDS, *_SOFT_FIELDS)})


def _enablement(store) -> dict | None:
    """Who turned the modality on, from the collection's meta point (spec §11).

    Contextual, not load-bearing: an older collection may carry no meta at all,
    and the audit must still describe the observations it does have.
    """
    try:
        meta = store._meta_payload()
    except Exception:  # noqa: BLE001 - the collection may be unreachable mid-report
        return None
    if not isinstance(meta, dict):
        return None
    enablement = meta.get("enablement")
    return enablement if isinstance(enablement, dict) else None


def face_audit(
    store,
    image_hash: str,
    *,
    video_hash: str | None = None,
    frame_timecode_ms: int | None = None,
) -> dict | None:
    """Endpoint-4 body for one indexed medium, or ``None`` when it has no faces.

    The collection is checked against the observation's own persisted provenance,
    so that nothing in this report depends on the environment the web process
    happens to be running under. Marker-only records report compatibility as
    unknown because they carry no observation provenance to compare.
    """
    scope = (
        {}
        if video_hash is None
        else {
            "video_hash": video_hash,
            "frame_timecode_ms": frame_timecode_ms,
        }
    )
    marker = store.get_marker(image_hash, **scope)
    faces = store.list_faces(image_hash, **scope)
    if marker is None and not faces:
        return None

    # Every observation of one medium was produced by one pass, so the provenance
    # fields are identical across its rows — the first row carries them all.
    row = {"quality_gate_version": LEGACY_QUALITY_GATE_VERSION, **faces[0]} if faces else {}

    if not faces:
        # Markers record totals, not the provenance needed for a comparison.
        # An absent observation cannot demonstrate an embedding mismatch.
        ok = None
        warnings = [
            "Compatibility is unknown: no surviving face observations carry index-time provenance."
        ]
    else:
        try:
            warnings = list(store.check_compat(_stored_config(row)))
            ok = True
        except ValueError as exc:
            # Preserve incomparable observations so the examiner can see why.
            warnings = [str(exc)]
            ok = False

    marker = marker or {}
    return {
        "image_hash": image_hash,
        "n_observations": len(faces),
        "detector": {f: row.get(f) for f in _DETECTOR_FIELDS},
        "embedder": {f: row.get(f) for f in EMBEDDER_IDENTITY_FIELDS},
        "gates_in_force_at_index_time": {f: row.get(f) for f in _GATE_FIELDS},
        "pipeline_config_hash": row.get("pipeline_config_hash"),
        "library_versions": {
            "cv2": row.get("cv2_version"),
            "onnxruntime": row.get("ort_version"),
            "sfn_version": row.get("sfn_version"),
        },
        "file_totals": {
            "n_detected": marker.get("n_detected"),
            "n_kept": marker.get("n_kept"),
            "n_review_only": marker.get("n_review_only"),
            "review_only_reasons": marker.get("review_only_reasons") or {},
            "n_rejected": marker.get("n_rejected"),
            "rejected": marker.get("n_rejected") or {},
        },
        "enablement": _enablement(store),
        "compat": {"ok": ok, "warnings": warnings},
        "caveat": _AUDIT_CAVEAT,
    }
