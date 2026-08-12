# Standalone stale-observation inspect/purge — design note

**Status: proposal, not implemented.** No code, CLI surface, or behaviour changes with this
document. Every claim below cites code read on 2026-08-12 at `feat/face-pipeline-phase1`;
anything not directly readable there is marked *(inferred)*.

## The gap

A stale observation is a face point an earlier run stored that the current run did not
produce again — a face that fell below the review gate (no point at all now) or one whose
bbox moved because the detector changed, so the new observation lands on a different point
id (`src/scalar_forensic/faces/store.py:393-407`). If it was embedded, it is still in the
search space under thresholds the run no longer applies (`src/scalar_forensic/cli.py:1728-1731`).

Detection exists only inside an index run: `stale_face_points()` is called per medium in the
indexing loop (`src/scalar_forensic/cli.py:1635`), the operator is prompted once at the end
(`cli.py:1743`), and declining prints "Left in place … Re-run and confirm, or purge the
media." (`cli.py:1773-1777`).

That advice does not hold. Re-running under the same config seeds the skip set from
`processed_hashes(cfg.config_hash)` (`cli.py:477`), which matches on the medium's marker
(`store.py:267-284`); the medium is skipped, `stale_face_points()` never runs for it, and
**the operator is never asked again**. The keys survive only in the audit record — the
`index_run` event carries `n_stale_detected`, `n_stale_removed` and the full
`stale_observation_keys` list (`cli.py:1797-1819`) — so recovery today means reading
`face_audit.log` by hand. The only in-tool escape is `sfn-faces purge --media <sha256>`
(`cli.py:1870-1911`), which deletes the medium's *good* observations too.

## What the command inspects, and how it finds stale points

Reuse the existing detector rather than inventing a second definition of "stale":
re-derive each medium's produced point ids and diff against what is stored, exactly as
`stale_face_points(image_hash, produced_ids)` does (`store.py:393-424`).

Two ways to obtain `produced_ids`, and the choice is the main design fork:

1. **Re-detect** (faithful, expensive): run detection on the media again under the current
   config and diff. This is what the index path does and is the only way to catch a face
   that dropped below the review gate. Cost is a full detection pass over the named media.
2. **Config-hash diff** (cheap, partial): treat every `is_face` point whose
   `pipeline_config_hash` differs from the current `cfg.config_hash` as suspect. Cheap, but
   it over-reports — a threshold change rewrites points in place at the same id
   (`store.py:325-341`, `store.py:396-398`), so many differing-hash points are current, not
   stale. Usable for *inspect*, not as a delete predicate. *(Inferred: over-reporting
   follows from the id derivation; no test asserts the population size.)*

Note `pipeline_config_hash` is **not** a payload-indexed field — the indexed set is
`image_hash, image_path, video_hash, video_path, is_face, group_id, quality,
frame_timecode_ms` (`store.py:111-119`) — so a config-hash filter is an unindexed scan.
*(Inferred: Qdrant serves it correctly, just without index acceleration.)*

Scope should be the same shapes purge already accepts: `--media <sha256>` and `--all`
(`cli.py:1871-1878`).

## What it deletes, and what it must never delete

Deletes: only `is_face` observation points named by the diff, via
`delete_face_points(ids)` (`store.py:426-447`), then unlinks chip files filtered through
`unreferenced_chip_hashes()` — chips are content-addressed and shared between observations
with byte-identical crops (`store.py:351-380`), and the index path already does exactly this
(`cli.py:1752-1770`).

Must never delete:

- **`is_face_meta`**, the enablement record: examiner id, timestamp, authorization ref
  (`store.py:124-136`). `purge_all()` is filter-based specifically so it survives, and never
  calls `delete_collection` (`store.py:474-487`).
- **Per-medium markers (`is_face_marker`) and video rollups (`is_face_video_rollup`)** —
  they are not observations; deleting a marker would silently un-skip the medium and lose
  the run's counts. `stale_face_points()` is scoped to `is_face` for this reason
  (`store.py:404-406`). Commit `bb39d0e` is the mirror case: `purge_all()` had to *gain* the
  rollup clause, because a purge that promises to remove all biometric-derived data must
  take the rollups with it — the same points a *stale* purge must leave alone.
- **Current observations.** No `--yes`-by-default. The index path defaults its confirm to
  `False` and treats an EOF as "do not delete" (`cli.py:1742-1749`); this command must match.
- **Adjudications**, which reference `observation_key`, not point ids (`cli.py:1732-1735`).
  They are not deleted; a deleted observation's key simply stops resolving. That is a
  reporting obligation, not a deletion one.

## How it reports

Mirror the existing prompt block (`cli.py:1711-1735`) so the two surfaces read alike:
count; breakdown by `embedding_status`; breakdown by `pipeline_config_hash` (12-char prefix);
the explicit warning when any are `embedded` ("still returned by similarity search"); the
adjudication note. Add per-medium grouping, since this command can span the whole collection.
Inspect prints and exits 0 without touching anything.

## Examiner id and audit entry

Both required, for the same reason the existing paths require them. Every face command
already refuses to run without `SFN_EXAMINER_ID` via `face_startup_error()`
(`cli.py:1880-1888`), and `AuditLog.append()` takes `examiner_id` as a positional part of
every record (`faces/audit.py:20-30`). Deleting biometric-derived data unattributed would be
the only such act in the tool.

The audit entry should follow the `purge` event's shape — `scope`, `image_hash`, `n_points`,
`n_chip_files` (`cli.py:1942-1949`) — plus the two fields the index path proved necessary:
`n_stale_detected` separate from `n_stale_removed`, so a declined inspection is
distinguishable from a clean one (`cli.py:1810-1813`), and `stale_observation_keys`, which is
the record that makes a deleted key explainable afterwards. Suggest event type
`"purge_stale"`, distinct from `"purge"`, so retention questions can be answered by event
type alone. *(Inferred: Phase 1 event types are enumerated as enablement / index_run / purge
in `faces/audit.py:1-7`; adding one is a doc change there too.)*

## Smallest version that closes the gap

`sfn-faces stale --media <sha256>` — inspect-only by default, `--delete` to act, re-detection
for `produced_ids` (option 1), one medium at a time, reusing `stale_face_points()`,
`delete_face_points()` and `unreferenced_chip_hashes()` unchanged, and writing a
`purge_stale` audit event. That alone removes "recoverable only by hand from the audit
record": an operator who declined the prompt has a named, repeatable way back to the same
list. `--all` and the cheap config-hash inspect mode are additive later.

## Open questions for the maintainer

1. **Re-detect or config-hash diff?** Faithfulness against cost. A re-detect over a full case
   is an indexing-scale operation; the cheap mode cannot be a delete predicate.
2. **Should declining the in-run prompt be recorded on the medium's marker** (a
   `stale_declined_at` field), so a later run can re-raise it instead of skipping silently?
   That fixes the gap at its source and might make this command unnecessary — but it writes
   operator workflow state into evidentiary payloads, which is a policy call, not a code one.
3. **Does a standalone deletion command belong in Phase 1 at all**, given the deferral
   reasoning was "underspecified, adds user-facing surface"? Question 2's marker field is the
   smaller alternative.
4. **`--all` scope**: is a whole-collection stale sweep a thing an examiner should be able to
   do in one command, or does forcing per-medium scope carry useful friction?
5. **Event type**: new `purge_stale`, or an existing `purge` with `scope="stale"`? Affects how
   retention questions get answered from the log.
