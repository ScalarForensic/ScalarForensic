# Handoff — scalarforensic-com-c23, phase 8 browser side

## 1. The task

Phase 8 of `docs/specs/video-playback-transcode.md`, **browser side only**: make
the §7.2 label render the whole pipeline, give `closeFullJob()` a caller, and
resolve the two differently-worded download affordances — dispatch
`data/reports/dispatch-phase8-browser.md`, three PRs, merged by `m8`.

## 2. Done, and not done

| item | state | where |
|---|---|---|
| 1.1 the §7.2 label | **PR #197**, CI green at `fe2ee23`, live-checked | branch `feat/phase8-browser` |
| 1.2 `closeFullJob()` has no caller | **PR #198**, opened, CI not yet confirmed | branch `fix/close-player-stops-full-job` (off `5397ddb`) |
| 1.3 two download affordances | **NOT STARTED** | — |

Both branches are **pushed**. Neither is merged; `m8` merges, never the coder.

`fe2ee23` = `2d0b2ab` (the renderer) + the `chunk_seconds` row-name fix the live
check found. `090eeee` = the `closeVideoPlayback()` → `closeFullJob()` caller.

Bar measured in this worktree (`/tmp/c23-phase8`, `models/` copied in, clean
tree at each commit): JS suite **78 pass / 0 fail** on `feat/phase8-browser`,
**61 / 0** on `fix/close-player-stops-full-job` (it does not carry #197's
tests). `tests/test_video_playback.py` + `tests/test_static_wiring_web.py`:
**356 passed**. The full suite was *not* run — the two PRs touch
`web/static/` and `tests/js/` only, and CI runs it.

Mutations: 15 for #197 and 5 for #198, each applied individually, each red;
the tables are in the PR bodies, and M5 in #197 survived first and became a
named test. Harnesses (throwaway) are in this session's scratchpad, not the
repo.

## 3. The next concrete step

**Item 1.3, and it is the only thing left.** The panel shows two download
affordances in the same states:

- `index.html:1679` in the §6.3 refusal band — *"Download the original
  instead"*;
- `index.html:1795`, the panel's permanent escape route — *"Download original
  (clip.mov)"*.

(A third, `index.html:1745` in the stale-evidence band — *"Download the file as
it is now (clip.mov)"* — is **not** a duplicate: it is a different statement
about a file that is no longer the one that was indexed. Leave it.)

My reading, not yet implemented, for whoever picks it up: it is **one control in
two placements**, not two controls. §6.3 requires the refusal to offer the
original ("refused before it starts, with the estimate shown and Download
original offered"), and §7.5 requires the permanent escape route, so both
placements are specified — but *one act with two wordings* is the same defect
class this codebase already refuses for the contention and override sentences.
So: give it one label with one definition (a getter beside
`videoPlaybackDownloadUrl` in `computed.js`, e.g. `videoPlaybackDownloadLabel`
→ `` `Download original (${filename})` ``), render it in both places, and say in
the PR why the stale-evidence link keeps its own wording. Existing wiring test
`the §6.3 refusal shows both numbers and offers the original`
(`tests/js/markup_wiring.test.mjs`) and `the override control is offered, and
beside Download original` both read that region — check them before editing.

Then: `m8` is waiting on CI for #198, and #197 is accepted on substance and
held only for the green confirmation.

## 4. Things written down nowhere else

- **A live check on this panel needs the results view.** `phase = 'results'` and
  `appMode = 'search'`, or `.results-layout` is `display:none` and everything you
  inject measures 0 px. `results` is a list of **files** (`{file_id, filename,
  hits: [...]}`), not hits — `selectedFile.hits` is what `filteredHits` reads,
  and `selectedHit` is a getter over it (c18's note that assigning to it does
  nothing still holds). The player panel then lives in a `.meta-box` whose own
  `clientHeight` is ~129 px, so for a photograph you must open that scroll box
  and hide the empty `.image-box` above it; both are layout-only.
- **The injected fixture is not a probe.** `m8` read
  `c23-live-2-cache-hit-no-invocation.png` and asked why the record says the
  source has no audio while `STREAMS` says `hevc / aac`. Answer: both lines were
  injected, inconsistently, by me — the console 400s on `/api/video-*` prove
  nothing was probed. **Nobody has yet checked the label against a real encode**,
  and `has_audio` (`routes.py`, from `_validated_source`'s `info`) versus
  `playback-info`'s `audio_codec` is therefore *read* but not *measured*. That is
  a worthwhile check for whoever next has the CPU window and a video in an
  allowed root — it is the one claim in §7.2 that a canned payload cannot test.
- **A leaked interval hangs `node --test`, it does not fail it.** Under the M1/M3
  mutations for #198 the suite is killed on a timeout rather than reported red
  (the assertions do fail first). Any new test that starts the poll must stop it
  in a `finally`.
- **`gh pr view --json body -q .body` returned an empty string on this box**, and
  piping that into `gh pr edit --body-file` **wiped #197's body**. It was
  rewritten by hand from the create call. Use
  `gh api repos/.../pulls/N --jq .body` and check `wc -c` before you edit a body
  from a file.
- **I killed the operator's app.** `pkill -f sfn-web` matched the pre-existing
  instance on 8080 as well as mine on 8099. I restarted it from the shared
  checkout (`./run.sh sfn-web`, HTTP 200 on 8080) — but it is a *new process*,
  started by me, and anything the old one held in memory is gone. Kill by PID
  next time.
- The shared checkout was not touched: no branch switch, no commit, and
  `docs/CTO_LEDGER.md` is still the operator's uncommitted file.
