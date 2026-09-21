# Face-driven query interface

Implemented query workflow, originating in the maintainer's 2026-08-12 request.
This is an interface/interpretation contract, not an unstarted implementation plan.

## Workflow

1. Upload query media into session scope; query processing does not persist it
   to indexed collections.
2. Show the query image and its detected faces. Search only selected comparable
   faces; review-only crops remain viewable but cannot be selected as probes.
3. Show matched media on the right, identifying EXACT, SEMANTIC, ALTERED and
   FACES contributions. Expose mode filters and face search parameters.
4. Highlight selected query faces and compared hit faces, with raw model scores.
   Selection identity includes the source video frame and detection token.
5. Offer per-model audit and distribution-statistics views, so the examiner
   can distinguish what each model did and which population its statistics describe.

The face routes provide query detection, stored-observation browsing, search,
comparison, explanation, audit and distribution statistics. The API identity
contract is in [face pipeline §8](face-pipeline.md#8-query-semantics-phase-1b).

## Interpretation ruling

The maintainer explicitly authorized showing raw face similarity without a
deployment calibration record on 2026-08-12. This does not authorize presenting
a model output as identification or calibrated confidence.

- Keep the “Uncalibrated — not evidential” disclosure and opt-in enablement.
- Show that no confidence interval or calibrated deployment threshold exists.
- Any model-author reference threshold is labeled as such, never substituted
  for measurement on the deployment's material.
- Review-only observations remain vectorless and excluded from similarity.
- If calibration is implemented later, its provenance/precedence must follow
  [face pipeline §10](face-pipeline.md#10-calibration-and-validation-release-gate-for-search).

## Examiner-visible evidence

Show comparable/review-only counts, collapsed face-match counts, compatibility
warnings and the embedder identity returned by search/compare. A failed metadata
or comparison request is not a finding of “no match.” Marker-only face audit
compatibility is unknown when there are no surviving observation provenance fields.

Query review links serve the full session review crop; thumbnail display must
not replace the review source. Both source selection and delayed responses must
remain bound to the current case.

## Validation boundary

The [historical face-query measurements](../qa/face-query-acceptance-2026-08-12.md)
retain the model/preprocessing checks and small-face cohort caveats. They are
not a statement about the operator's current database. Synthetic browser checks
in `tests/browser/` cover rendering and interaction; live-Qdrant/model and
deployment-specific calibration remain separate evidence.
