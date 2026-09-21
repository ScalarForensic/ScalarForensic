# Face-query acceptance measurements — 2026-08-12

Historical validation evidence, not instructions to operate on today's case
database. The later configuration change did not itself establish small-face
searchability. Current corpus/storage state was not rechecked during cleanup.

## Initial acceptance boundary

At the recorded 64 px embedding floor, the acceptance query was `danny2.jpeg`.
It covered selection, highlighting, raw cosine, face query controls, audit and
distribution statistics, and the refusal of review-only faces as probes.

| Image | Face | Detection size | Result at embedding floor 64 |
| --- | --- | --- | --- |
| danny2.jpeg | 0 | 147.8 px | Comparable |
| danny1.jpeg | 0 | 40.1 px | Excluded for size |
| danny1.jpeg | 1 | 46.9 px | Excluded for size |
| danny1.jpeg | 2 | 40.8 px | Excluded for size |

**The acceptance covered the large query face, not searchability of the 40–47 px
cohort.** Rendering review-only crops is not coverage of their embeddings and
does not justify lowering an embedding threshold.

## Measured model/preprocessing checks

Recorded during the same validation work:

- SFace: 112×112 input, 128-dimensional output; RGB, mean 0.0, scale 1.0.
  Comparison with OpenCV `FaceRecognizerSF.feature` yielded cosine 1.000000.
- Same face after JPEG q=60 re-encoding: cosine 0.9898.
- Different people in the validation pair: cosine 0.1042.
- The 46.9 px face versus its 147.8 px counterpart: cosine 0.6097.
  The other two small faces scored approximately −0.0142 against that query.
- Review crop dilation 0.25 produced a crop 1.50× the detection box.

These are sample measurements, not calibrated operating thresholds or a grant
of model redistribution rights.

## Same-day threshold change and unverified follow-up

The recorded change `b91fd61` lowered embedding size 64→40 and review size
48→24. Earlier notes claiming a shipped review default of 36 were incorrect.

The historical `faces_danny_validation` collection had the same counts before
and after the attempted reindex:

| Snapshot | Observations | With vector | Vectorless |
| --- | --- | --- | --- |
| Before | 17 | 2 | 15 |
| After | 17 | 2 | 15 |

Already-indexed media had been skipped; the configuration change had not
retroactively created vectors. A separately authorized rebuild/re-measurement
would have been needed to test the admitted cohort. The old expected 5/12 split
was a prediction, not an observed result.

The historical collection name differed from the derived default. This is a
general validation warning: confirm exact target collections and isolate test
storage before any rebuild/purge. No destructive command or old server address
is retained as a runnable recipe. The standalone workflow remains a
[deferred proposal](../specs/stale-observation-purge.md).
