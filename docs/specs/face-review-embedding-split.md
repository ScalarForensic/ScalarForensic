# Face quality gates: review versus embedding

Implemented design originating on 2026-08-12. Supplements
[face pipeline](face-pipeline.md) §§6–7. Historical execution plans are not
current requirements; this document retains the storage and interpretation contract.

## Two purposes, two gates

Native-resolution review crops support human inspection; aligned 112×112 crops
support model comparison. The review path must not invent pixels to pass an
embedding gate. Detections failing embedding quality may still have useful
review evidence.

- Review checks confidence and size. Its effective thresholds are clamped to
  be no stricter than embedding thresholds; defaults must not invalidate a
  stricter user configuration elsewhere in Settings construction.
- Embedding additionally checks pose, sharpness and exposure. Persist the
  exclusion reason and the thresholds actually used.
- Pose measures nose displacement along the eye-to-eye axis, normalized by
  eye separation. In-plane rotation alone must not turn a frontal face into
  a rejected profile. Landmark validation preserves semantic eye/mouth order
  under rotation and rejects mirrored, degenerate or nonfinite geometry.
- Detector-input resizing and thumbnail downsampling are distinct from review
  crop creation. The source crop stays at native resolution; display must
  neither upscale nor silently centre-crop away evidence.
- Degenerate crops are rejected, not retained as observations with missing files.

Threshold values belong to current configuration and recorded provenance,
not a historical validation constant. The
[acceptance measurements](../qa/face-query-acceptance-2026-08-12.md) record why
changing a floor alone did not validate the small-face cohort.

## Storage and identity

Review-only observations carry `embedding_status: review_only` and no named
face vector. Search exclusion is structural, not dependent on every caller
remembering a payload filter. They remain ordinary face observations for browse,
counts and purge.

Reindexing an embedded observation as review-only must explicitly remove its
old vector. An upsert of an empty vector alone is not the demotion guarantee.
Promotion supplies the new embedding. Integration tests must query the actual
store after demotion, rather than only inspect an outgoing PointStruct.

Chips are domain-separated by purpose:

- aligned hash prefix: `aligned-rgb-v1\0`;
- review hash prefix: `review-source-rgb-v1\0`.

Each hashes dimensions and RGB content within its domain. This prevents equal
pixel arrays in different roles from aliasing filenames. Embedded observations
carry aligned and review hashes; review-only ones carry only a review hash.
Aligned-chip requests for review-only observations return 404 rather than an
unrelated aligned image.

Create new thumbnails directly from the native crop in memory, before review
JPEG compression, with one JPEG generation and no upscaling. Existing review
JPEGs and thumbnails are reused without rewriting shared artifacts. The browse
endpoint can regenerate a missing thumbnail from a stored review JPEG; when no
resize is needed it copies the JPEG bytes. Resized regeneration can differ from
the original thumbnail because only the compressed review crop is available.
Thumbnails remain derived, non-evidentiary artifacts outside chip identity.

Chips may be shared by multiple observations. Purge and reindex cleanup remove
an artifact only when no surviving observation references it. Orphan cleanup
must not erase a sibling source/frame's shared chip. Current observation IDs
include source-media/timecode scope; legacy identity changes require an explicit
rebuild, not the old assumption that all collections are empty.

## Partitioning and accounting

1. Rejected detections increment the rejection breakdown.
2. Retained detections partition into embeddable and review-only populations.
3. Embeddings are paired only with the embeddable subset, even when input
   outcomes alternate; an interleaved sentinel regression protects this.
4. An all-review-only image still produces observations and review crops.

Keep the invariant
`n_detected == n_kept + n_review_only + sum(rejected.values())`.
Review-only exclusions are retentions, not rejections. CLI, markers, video
rollups and audit must account for both populations. Record detector rows
dropped for noncanonical landmarks separately rather than hiding their count.
Purge includes face observations, markers and video rollups.

Both review and embedding thresholds belong in config identity: otherwise
lowering a review floor can silently leave old processed markers matching.
Admission thresholds may be soft comparability fields, but must still trigger
appropriate reprocessing. Do not confuse “which faces were admitted” with
“what vector space these embeddings occupy.”

`quality_gate_version` records the selection algorithm as well as numerical
thresholds. `eye-axis-v2` uses roll-invariant pose and landmark checks. Older
observations and collection metadata without this field mean `image-x-v1`;
stored probes retain their historical config hash. A version difference is
soft compatibility: existing embeddings remain comparable, with a warning.
New indexing records the version in its config hash so old processed markers
do not skip reevaluation on the next face-indexing run. Reading a stored face
does not reprocess or relabel it.

## Presentation and later work

Explain each retained observation from persisted index-time provenance.
Review-only observations have a persistent not-comparable label, no aligned-chip
link, and an explanation of why embedding was not performed.

Raw cross-file face similarity is implemented under the explicit uncalibrated
display ruling in [face query UX](face-query-ux.md). Grouping/clustering and a
deployment-calibration record are separate future work, not unfinished tasks in
this gate split. Retain the no-vector exclusion in any future payload-based
grouping or adjudication path.
