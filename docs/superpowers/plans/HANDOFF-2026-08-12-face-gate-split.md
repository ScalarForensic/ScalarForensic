# Handoff — face review/embedding gate split

**Written:** 2026-08-12, mid-execution, at the maintainer's request before a break.
**Branch:** `feat/face-pipeline-phase1` (local only — never pushed)
**Plan:** `docs/superpowers/plans/2026-08-12-face-review-embedding-split.md`
**Spec:** `docs/superpowers/specs/2026-08-12-face-review-embedding-split-design.md`

## How to resume

Say "continue the face gate split" and point me at this file. Read the three
open decisions below first — two of them change what Task 8 and Task 12 say,
and I deliberately did not guess. Everything else is mechanical plan execution
and needs nothing from you.

Plan step checkboxes are ticked through Task 7. Tasks 8–12 are untouched.

## State: Tasks 0–7 complete, 441 tests passing

| Task | Commit | What landed |
|---|---|---|
| 0 preflight | — | Qdrant returned `{"collections":[]}`; migration window verified open |
| 1 settings | `f91a5e2` | `face_review_min_conf/size`, clamped not raising; `face_threshold_notes()` surfaced at 2 CLI sites + web lifespan |
| 2 review gate | `7b51bf9` | `review_gate()` — confidence + size, no pose |
| 3 provenance | `92ea221` | Both thresholds inside `config_hash`, added to `_SOFT_FIELDS` |
| 4 chip hashes | `c8ba4d4` | `aligned_chip_hash` / `review_chip_hash`, `write_review_chips`, `write_aligned_chips` returning both |
| — review fixes | `56043ab` | Group B review findings (see below) |
| 6 partition | `e20de65` | Three-way partition; vectorless review-only points; demotion wired at CLI |
| 7 counts | `f68f1ae` | `n_review_only`, `review_only_reasons`, `n_dropped_noncanonical` on markers + rollups |

Baseline was 398 passed. Now 441, zero skipped, `ruff check` and
`ruff format --check` clean on `src tests`.

Task 5 (`7a9ce8e`) landed between 4 and 6: `clear_face_vector`,
`unreferenced_chip_hashes`, and `_purge_by_filter` reading both hash domains.

## Decisions — resolved 2026-08-12

The three below were answered by the maintainer:

1. **Chip store scoping: Option A.** Document that `SFN_FACE_STORE_DIR` is set per case.
   No code change; Task 12 Step 2's placeholder wording stands.
2. **Chip-hash prefix: keep widened.** No revert.
3. **Validation run:** setup written up as
   `docs/superpowers/plans/SETUP-2026-08-12-face-validation-run.md`; the maintainer runs
   it when the models are in place. Not scheduled into task order.
4. **Task 6 review:** no separate pass — folded into the review checkpoint after Task 9,
   which therefore covers `56043ab..HEAD` (Tasks 6–9) as one read.

The original statements of 1–3 are kept below for the reasoning.

## Open decisions — these are yours

### 1. `SFN_FACE_STORE_DIR` scoping (blocks nothing, changes Task 12)

`SFN_FACE_COLLECTION` is per case; `face_store_dir` defaults to a single
`data/faces` for every case. Chips are content-addressed, so two cases holding
the same image share one file on disk. Purge's new reference check
(`unreferenced_chip_hashes`) scrolls only its own collection, so **purging case
A can unlink a chip case B still references.**

`check_compat` already refuses to mix biometric data across cases for vectors,
so the principle is established; the chip store just doesn't follow it.

- **Option A (currently in the plan):** document that operators must set
  `SFN_FACE_STORE_DIR` per case. Zero code change. Relies on operator discipline.
- **Option B:** default `face_store_dir` to a per-case subdirectory, e.g.
  `data/faces/{collection}`. Enforced by construction, but it is a config
  default change affecting any existing deployment, and needs its own task.

I wrote Option A into Task 12 Step 2 as a placeholder. Say the word and I'll
convert it to Option B as a new task before Task 8.

### 2. Whether the narrower chip-hash prefix was intentional (already changed — confirm or revert)

I widened `_domain_hash`'s prefix from `f"{h}x{w}:"` to cover dtype and the
full shape. Verified collisions under the old prefix, same domain:

```python
review_chip_hash(np.zeros((6,6), np.uint8))   == review_chip_hash(np.zeros((6,6,1), np.uint8))
review_chip_hash(np.zeros((2,2,4), np.uint8)) == review_chip_hash(np.zeros((2,2,1), np.uint32))
```

Neither is reachable today — every array is `HxWx3 uint8` out of
`load_for_detection` / `align_face`. I changed it anyway because from Task 6
these digests are payload fields and filenames, so widening later would be a
schema-and-filesystem migration. If you prefer the narrower prefix plus a
documented dtype/shape precondition, revert `chips.py:33-41` and drop
`test_hashes_separate_channel_count_and_dtype`.

### 3. Timing of the danny* validation run

The plan closes with a validation run against `danny*`. It needs a YuNet ONNX
(`scripts/download_models.py --yunet`) and an operator-supplied embedder, plus
`SFN_FACES_ENABLED=true` and `SFN_EXAMINER_ID`. I have not run it and cannot
supply the embedder. Tell me when you want it and whether the models are in
place.

## Do not run `--faces` or `sfn-faces purge` right now

This window is narrower than it was — Task 6 fixed the payload — but Task 8
has not yet made purge unlink through `unreferenced_chip_hashes`, so purge
still unlinks unconditionally and can remove a shared chip. Nothing is
indexed, so this costs nothing as long as no one indexes on this branch before
Task 8. The warning is written into the plan at Task 6's preamble.

## Findings worth remembering

**Two vacuous tests in the plan document, both from hand-written numeric
fixtures.** Task 2's pose test used "profile" landmarks whose nose sat exactly
at the eye midpoint — `pose_ratio` 0.0, a squashed frontal face. It would have
passed against a `review_gate` that *did* check pose. Replaced with landmarks
giving ratio ≈2.2 plus a companion assertion that `pre_align_gate` rejects the
same landmarks with `reason == "pose"`. Tasks 9–12 use the same hand-literal
technique; compute what a fixture implies before trusting it.

**A mock that hid the failure mode it was supposed to cover.** My first
`unreferenced_chip_hashes` tests patched `qdrant_scroll_all` and returned full
payload dicts. If the payload projection omitted `aligned_chip_hash`, real
Qdrant would return payloads without that key, every aligned PNG would look
unreferenced, and purge would unlink the PNGs authenticating surviving embedded
observations — and the test would still pass. Now asserts the scroll kwargs.
Caught by the Group B reviewer, not by me.

**The sentinel test is verified live, not just written.** Task 6's
interleaved-outcome test was mutation-tested against two realistic bugs:
reversed embedding pairing, and combined-list indexing (what a single `kept`
list would produce). Both fail the test. Worth redoing if anyone refactors
`process_image`.

## Tasks 8–12 execution notes (added 2026-08-12, second session)

- **Task 9 ordering lives in the route, not `list_faces`.** The plan put the
  embedded-first sort in `store.py`. It went into `routes/faces.py` instead
  (`_browse_order`), next to `_normalized`, because the same handler must also
  default `embedding_status` for pre-split payloads — that is an API contract
  concern, and splitting the two across layers would have made the ordering
  untestable from the route tests the plan itself specifies.
- **`chip_hash` was stale in three places.** The payload key became
  `aligned_chip_hash` / `review_chip_hash` in Task 4, but `routes/faces.py`,
  `index.html` and the route-test fixture still read `chip_hash`. The UI would
  have rendered a broken image for every face. Now covered by
  `test_face_grid_uses_the_review_hash_domain`.
- **Two live-Qdrant findings contradict the plan's Task 11 assumptions** (see
  the commit message on the Task 11 commit). Neither changes the design, both
  are now tests:
  1. A re-upsert with `vector={}` *already* drops the named vector on qdrant
     1.17 — the explicit `clear_face_vector` is belt-and-braces there. It stays:
     that is a storage-engine detail, not an API guarantee.
  2. `delete_vectors` on an id that does not exist returns **404, not a no-op**.
     The plan assumed a no-op. The CLI call site is safe (it upserts first, and
     the client's `upsert` defaults to `wait=True`), and the error is
     deliberately not swallowed — a silently ignored clear could leave a
     review-only observation searchable. Precondition now documented on
     `clear_face_vector`.
  The load-bearing claim itself holds: after demotion, vector search returns
  nothing and the payload survives intact.
- **Task 10 Step 4 (manual UI check) is not done and cannot be.** It needs
  indexed face data, which needs the models. Substituted: `node --check` on
  `faces.js`, a TestClient smoke check that `/`, `faces.js` and `style.css`
  serve, and four static-wiring assertions. The visual confirmation — review
  crops legible at native resolution, both populations distinguishable — is
  folded into the validation run, `SETUP-2026-08-12-face-validation-run.md` §7.

## Plan edits made during execution

The plan document is not as committed at `10ace0d`. Changes:

- Dropped `FaceIndexResult.orphaned_chip_hashes` — declared, populated by no
  task, read by none.
- Renamed `demoted_point_ids` → `review_only_point_ids`; it holds first-time
  review-only faces too, and Task 8 reports it. Calling first sightings
  "demoted" would misdescribe them in the audit record.
- Task 6 preamble: the interim-state warning above.
- Task 8: recorded purge's collection-scoped limit and its check-then-unlink
  race (a concurrent index run can leave a dangling chip reference); explained
  why the unlink loop lists both path builders.
- Task 11: added a live-Qdrant test that `delete_vectors` on absent point ids
  is a no-op. The demotion call site depends on it and I could only verify it
  from the client signature, not from server behaviour.
- Task 4 interface note: `chip_paths` returns a 3-tuple now spanning both hash
  domains, so `chip_paths(dir, aligned_hash)[1]` names a file that can never
  exist. Task 9 must choose the path builder from `embedding_status` rather
  than indexing blindly.
- Task 1's test path was wrong (`tests/test_config.py` → `tests/faces/test_config.py`);
  Task 2's `_det` helper collided with an existing one; Task 5's tests assumed
  a `fake_client` fixture this repo doesn't have. All adapted to repo idiom.

## What's next, in order

**Task 8** (CLI summary + audit record, ~90 plan lines) — needs decision 1 only
for its documentation wording; the code is independent. Then **9** (endpoints
and explainer), **10** (UI labelling), **11** (live-Qdrant test), **12** (docs).

My suggested review checkpoint: after Task 9, since Tasks 8–9 together decide
what the audit record and the API say about review-only observations, and that
is the pair a reviewer would have to read as one.
