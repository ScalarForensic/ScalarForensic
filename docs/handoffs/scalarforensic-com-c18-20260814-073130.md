# Handoff — scalarforensic-com-c18 (JS test runner, phase 7 browser side)

2026-08-14, 07:31 UTC. Manager at retire: `scalarforensic-com-m7` (m6 retired mid-task).

## 1. What I was asked to do

Adopt a real JS test runner (operator ruled YES, spec §14), then build phase 7's
browser side on top of it — progress, cancel, navigate-away prompt, completion
notification, auto-switch, the §4.3 contention disclosure and the §6.3 refusal.

## 2. Done

**Merged.** PR #173, squash `1e34aef`. Node's built-in `node --test` via
`npm test`, **zero npm dependencies** (`package-lock.json` has one entry, its own
root). `tests/js/harness.mjs` reads the `<script src>` list out of
`static/index.html` and compiles+runs each file in a `node:vm` context, so a new
part file is parse-covered by the commit that adds its tag. Wired as a **step
inside `lint-and-test (3.12)`**, not a separate job — `main`'s ruleset requires
exactly that job and `qdrant-integration`, so a third job could be red and still
merge. The rationale comment in `ci.yml` is the part that stops someone tidying
it into a separate job later.

Accepted by mutation, not by installation: reintroducing the historical
`?? … ||` SyntaxError at `player.js:118` took the JS suite to 10 passed / 12
failed while the Python suite stayed fully green at 954/5 — the contrast that
justifies the dependency.

**Open, green, not merged: PR #183** (`feat/video-phase7-browser`, pushed to
origin, rebased on `32d9da8`). Phase 7 browser side. `npm test` 41 passed / 0
failed; `uv run pytest -q` 979 passed / 5 skipped, 74.51%, unchanged from main.
All six checks green. m7 merges it, not me.

## 3. Not done

- **`docs/specs/video-playback-transcode.md` §14 still says "there is no
  JavaScript test harness in this repository."** The file never reached my
  ownership row. The replacement text is §7 below — **paste it in, it exists
  nowhere else on disk that survives a reboot**.
- **`CLAUDE.md` has no `npm test` line.** Text also in §7.
- **#184's `override` surface is unrendered.** Contract in §6.
- Two cosmetic findings from the live check, in §5.

## 4. The disclosure invariant, as built — do not "simplify" it

The §4.3 contention notice is **the remedy, not a label**. `csm-b2` measured the
residual at n=6 per arm, CPU/libx264: **18.31 s unniced vs 16.83 s niced**. The
yield buys ~9% and **neither arm lands inside §4.2's 6–10 s window**, so the
slowdown is not fixed by yielding — it is disclosed. This sentence is the only
thing that tells an analyst why playback got worse, on both pipelines.

- The server sends `contention_notice` on **two** payloads on purpose:
  `video_playback/routes.py:480` (chunk response) and `jobs.py:270`
  (`Job.view()`), non-null only while `full-job-running`.
- Both land in **one state cell**, `contentionNotice`, written only by
  `_setContentionNotice` (`player.js`). That is what makes two carriers one
  disclosure instead of two that can disagree. **Deleting either write as
  "redundant" is the failure mode**; the comment above each says so.
- **Chunk-side is primary.** The notice renders beside the chunk spinner, and is
  *not* gated on `chunkIsEncoding`: the sentence is about the *next* chunk, so it
  must be readable while the current one still plays. A notice that appeared only
  in a full-job panel would explain the symptom where the person watching the
  spinner is not looking.
- **The sentence is never copied into JS or markup.** One definition, at
  `jobs.py:135`, for the same reason `_check_allowed_path` has one. A test
  asserts the string does not appear in `index.html`.
- **It is not an error.** `.vc-notice` is informational (`--semantic`, no ⚠, no
  red). #139 removed a fake error band from this UI and that precedent binds.
- **The guard is demonstrated, not assumed.** Deleting `_applyFullJobView`'s
  `_setContentionNotice(view.contention_notice)` takes the suite to **40 passed /
  1 failed**, failing exactly `the contention notice renders from a job-status
  poll — the second carrier`. Re-run that mutation if you touch either carrier.

The §6.3 refusal follows the same principle and is **three-valued**: `refused`
shows estimate and ceiling; `unknown` sends `estimate_bytes: null`, so it prints
no number and a different sentence. Collapsing them is #147's "unknown displayed
as mismatch", already rejected at three layers — this is the fourth.

## 5. The live browser check — run, and what it found

Done the way `CLAUDE.md`'s gotcha demands: every `<script src>` and the
stylesheet force-refetched with `fetch(url, {cache:'reload'})` (14 × HTTP 200)
before reloading, so nothing measured was stale. App served from the worktree on
port 8099 with `SFN_WEB_PORT=8099`; the pre-existing instance on 8080 was left
alone. **The CPU window is closed and the server is stopped.**

Rendering the `.vc-block` needs an analysis session; without Qdrant I injected
the *result* (`results`, `selectedFileId`, `hit.is_video_frame`,
`hit.scores.exact`) and let the real Alpine component render the real markup with
the real CSS. The markup and styling are genuine; the analysis is not. Note
`selectedHit` is a **getter** — assigning to it silently does nothing; you must
set `results` + `selectedFileId`.

Screenshots: `live-1-contention-notice.png`, `live-2-refusal-unknown.png` in my
scratchpad `out/` — **on the tmpfs, treat as gone.** The measurements are here:

| check | result |
|---|---|
| notice visible | yes, 25 px, text identical to the server constant |
| notice not an error | `rgb(74,158,224)` blue border vs `.vc-error` `rgb(192,57,43)`; no `vc-error` class; no ⚠ |
| bar geometry | fill/track = **0.4196** for `fraction: 0.42` |
| `fraction: null` | bar `0×0` (not drawn at all), label `"63 s elapsed"`, no `%` |
| §6.3 `refused` | estimate **and** ceiling shown, Download original offered |
| §6.3 `unknown` | no estimate printed, too-big sentence absent, Download original still offered |

Two cosmetic findings, both left unfixed on purpose (past my retire nudge):

1. **Unknown-refusal reads as a near-duplicate.** The server's `reason` and my
   explanatory sentence say almost the same thing back to back. Consider dropping
   the static sentence and letting the server's reason stand alone.
2. **Byte counts follow the host locale** (`9.000.000.000` on this de-DE box),
   because `_bytesLabel` uses `toLocaleString()` — matching `index.html:471`'s
   existing convention. Fine, but if a report ever quotes it, pin `'en-US'`.

## 6. #184's `override` surface — the next browser task, not mine

Merged server-side and **rendered nowhere**. Same rule as `contention_notice`:
render the field, never a copy of the sentence.

- Every job view carries `override`: examiner id, verdict, the estimate set
  aside, and `limit_bytes`.
- `jobs.py` defines an `OVERRIDE_NOTICE` constant — one definition, like
  `CONTENTION_NOTICE` at `jobs.py:135`.
- `playback-info` carries `full_copy.overridable`.

Build it the way §4 describes: one cell, one writer, both carriers if there are
two, and a mutation test that proves the guard fires.

## 7. Text to apply when the spec and CLAUDE.md reach you

**I wrote this block myself.** c17 promised it to me and retired before writing
it; its handoff records the promise, not the text. m7 has accepted that
correction.

### 7a. `docs/specs/video-playback-transcode.md` §14

Replace the block that starts **"There is no JavaScript test harness in this
repository, and phase 6 is the first phase whose weight is in the browser."** and
ends **"...it is named here so the next person weighs it deliberately instead of
rediscovering the gap."** with:

> **There is a JavaScript test harness as of 2026-08-14, and phase 6 was the
> first phase whose weight was in the browser.** The gap that motivated it,
> recorded here because it is a project-level fact and not a phase-6 caveat:
>
> - `player.js` shipped a `?? … ||` precedence `SyntaxError`, so the file did not
>   parse *at all*, and all fourteen text-level wiring tests passed against it.
>   **A wiring test cannot tell you the browser can run the file.** This is the
>   `/?cachebust=N` gotcha generalised — reading a file as text says nothing
>   about executing it.
> - The operator ruled the dependency YES on 2026-08-14. What landed is
>   `tests/js/`, run by Node's built-in test runner (`npm test` →
>   `node --test "tests/js/**/*.test.mjs"`, Node ≥22), as a **step inside the
>   already-required `lint-and-test (3.12)` job**. A separate job would not be a
>   required check under the repository ruleset, so it could be red and still
>   merge.
> - `tests/js/harness.mjs` reads the `<script src>` list out of
>   `static/index.html` and **compiles and runs** each file in a `node:vm`
>   context holding a small `window`. Compiling is the mechanism: a parse error
>   in any part file is a test failure. The component under test is assembled by
>   the shipped `static/app.js`, so the property-descriptor merge rule
>   (CLAUDE.md: never `Object.assign`) is asserted rather than assumed.
> - **The dependency cost is zero packages.** `package.json` declares no
>   dependencies and the committed `package-lock.json` has a single entry, its
>   own root. The airgap cost is therefore a pinned Node (24.14.0 in CI,
>   `engines: node >= 22`) and nothing to vendor. This was a deliberate choice
>   over Vitest/Jest+jsdom: the frontend is plain browser scripts registering
>   fragments on `window.__sfnParts` — no bundler, no module graph — and the
>   parts under test touch no DOM, so a several-hundred-package tree would have
>   bought nothing and cost a real `vendor/` regeneration.
> - **What it still does not cover: the DOM.** There is none. The markup inside
>   `.vc-block` — Alpine directives, the two `<video>` elements, CSS — stays
>   **wiring-pinned only**, because rendering it needs a full analysis session
>   with Qdrant and an indexed corpus. The harness pins the script half: that it
>   parses, and that its methods and getters compute what they claim to. The live
>   browser check with a forced re-fetch of every `<script src>` and the
>   stylesheet remains mandatory.
> - The Python wiring tests in `tests/test_video_playback.py` **stay**. They pin
>   that the markup and the script are wired together at all, which the JS
>   harness cannot see; deleting them would trade one blind spot for another.
> - The harness was accepted by mutation, not by installation: re-introducing the
>   historical `?? … ||` SyntaxError takes the JS suite from 22/22 to 10 passed /
>   12 failed, while the Python suite stays at its full 954 passed / 5 skipped,
>   73.09% — including all 276 tests in `tests/test_video_playback.py`. That
>   before/after is the whole argument for the harness in one line.

### 7b. `CLAUDE.md`, under `## Commands`, after the `uv run pytest -q` bullet

```
- `npm test` — the JS suite (`node --test`, Node ≥22, zero dependencies; 41 tests
  at PR #183). Runs as a step inside CI's `lint-and-test (3.12)` job. It
  *executes* every `<script src>` in `static/index.html` in a `node:vm` context,
  which is the only check that can catch a `SyntaxError` in a part file — 14
  text-level wiring tests once passed against a `player.js` that did not parse.
  No DOM: markup stays wiring-pinned. See `tests/js/harness.mjs`.
```

## 8. Written down nowhere else

- **An unpushed branch on this fleet lives on a tmpfs and dies with the box.**
  Worker worktrees are under `/tmp/claude-1000/…`, which hit 97–98% full twice
  today. Phase 7's server side (1625 lines) existed only there, with its author
  already killed, until it was pushed. Push WIP to origin when it becomes
  expensive to recreate, not when it is review-ready. Capacity and durability are
  different risks; reclaiming space closes only the first.
- **`git worktree` + `models/`:** symlink `models` as a *directory* containing
  inner symlinks, not as a symlink itself — `.gitignore`'s `models/` pattern does
  not match a symlink, so `git status --porcelain` stops being empty and no bar
  you measure can honestly be quoted.
- **A `uv sync` in a `/tmp` worktree costs 4.5 GB and can fill the ramdisk.** Run
  the suite with the main checkout's venv instead:
  `PYTHONPATH=$PWD/src /home/user01/Schreibtisch/gitea/ScalarForensic/.venv/bin/python -m pytest -q`
  (verified: worktree `src/` wins over the editable install).
- **Cross-realm equality in `tests/js/`:** objects the part files create live in
  the vm realm, so `assert.deepStrictEqual` against a literal fails on the
  prototype alone. Compare field by field.
- **`node --test <dir>` does not work on Node 24** — positional args are globs.
  Use `node --test "tests/js/**/*.test.mjs"`.
- **The harness's `window` stub is deliberately minimal.** A part reaching for a
  missing browser API throws, which is information. It gained recording
  `addEventListener`/`removeEventListener` in #183; a stub that swallowed the
  call would have produced a green `beforeunload` test asserting nothing — the
  same defect class as the fourteen wiring tests that passed on an unparseable
  file.

## 9. Next concrete step

1. Merge #183 (green; live check reported in §5).
2. Apply §7a and §7b when the spec and `CLAUDE.md` reach the next owner.
3. Build #184's `override` surface per §6.
