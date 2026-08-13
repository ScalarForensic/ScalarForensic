# Handoff — scalarforensic-cfm-c6 (2026-08-13 12:12 UTC)

## 1. Task
Operator UI change-set 2026-08-13 (4 items), spec + UPDATE section at
`/tmp/claude-1000/-home-user01-Schreibtisch-gitea-ScalarForensic/cf8c108a-aff7-481e-8bab-7e5989335a4e/scratchpad/ui-changeset-2026-08-13-faces.md`
(read the UPDATE section — it supersedes the original items 3/4 with the compare-endpoint interaction model).

## 2. Done / not done
Done, merged to main:
- **Items 1+2** (FACE header legend pill + sectioned query controls): PR #122, squash `860bee5`. Static-only. c4 asked to verify in operator Chrome; verification confirmed by m2 (deployed at main `e66fa78`, 618/5).
- **Backend deltas** (spec UPDATE): PR #124, squash `4bd9418`. `POST /api/faces/compare` (pairwise raw-cosine matrix, comparable faces only, counts for review-only omissions, uncalibrated calibration block, compat 409, audit-logged) and `point_ids` probes on `/api/faces/search` (uuid-validated; unknown id → 400; vectorless/review-only point → 400 same wording family as session refusal; `face_indices` now optional, neither → existing 400 "no face indices given"; point-probe hits carry `query_face_index: "pt:<id>"`; probe ids audit-logged). Tests: `tests/faces/test_face_probes_compare.py`. **Server restart still pending operator timing** — until then the new endpoints 404 on the live server.
- Both remote branches deleted after merge. Bars at merge time (worktree, no `models/`): #122 = 599/6, #124 = 610/6.

Not done — **items 3+4 frontend, ~60% written, WIP pushed to branch `ui/face-basket` (commit `1264006`, based on `4bd9418`)**:
- DONE in that commit: `state.js` (faceBasket, faceComparePairs/Counts/Loading/Error, faceCrossThreshold 0.0, removed `selectedQueryFaceIndices` from state), `computed.js` (getters `selectedQueryFaceIndices`, `selectedFacePointIds`, `faceCrossHighlight`; faces-only ordering in `filteredHits`), `faces.js` (basket ops `_basketAddQuery`/`toggleQueryFace`/`toggleHitFace`/`hitFaceSelected`/`basketToggleRow`/`basketRemoveRow`/`basketClear`/`selectAllQueryFaces`/`clearQueryFaceSelection` rewired over the basket; `runFaceSearch` sends `face_indices`+`point_ids`; `runFaceCompare(imageHash)` fails soft pre-restart; `faceCrossMatched`/`queryFaceCrossMatched`), and 10 new wiring tests in `tests/faces/test_static_wiring.py` (**red by design** — they assert the HTML/CSS that is not written yet).
- NOT started: `index.html` (basket box in left panel below Not Matched; ctrl+click handlers on both chip strips — query chip becomes an `<a target="_blank">` to `queryFaceChipUrl` like the match side; `face-cross-matched` class bindings both sides; `runFaceCompare` added twice to the faces-panel `x-init` — once direct, once in the `$watch('selectedHit', …)` callback; `faceCrossThreshold` slider row in the Face section of Query Controls) and `style.css` (`.face-basket {` with `overflow-y: auto`; `.face-basket-row`; `.face-cross-matched` using `var(--accent)`; make `.face-chip img` `object-fit: contain; max-width: 72px; width: auto` to match the query side, item 3e).

## 3. Next concrete step
`git worktree add <dir> ui/face-basket`, run `uv run pytest tests/faces/test_static_wiring.py -q --no-cov` — the failures are the exact TODO list (each new test names the missing HTML/CSS anchor). Write index.html + style.css until green, full suite (expect ≥618 passed), ruff format, PR per CLAUDE.md flow, then message scalarforensic-cfm-c4 to verify in operator Chrome (force-refetch ALL part files + style.css via `fetch(url,{cache:'reload'})`; cachebust busts HTML only). Note in the PR body: the compare/point_ids UI features only go live after the operator-timed server restart of #124; the JS degrades soft until then.

## 4. Not written down anywhere else
- Interaction decisions (agreed shape, from spec UPDATE): the basket is the *single* selection model — `selectedQueryFaceIndices` is now derived (basket rows side='query' ∧ fileId=selectedFileId ∧ selected). `loadQueryFaces` reseeds query-side rows for the current file (removes all side='query' rows, re-adds searchable as selected) — hit-side rows persist across file switches. Ctrl+click is idempotent select (spec: "sets selected=true"); deselection = basket checkmark; removal = ctrl+click on basket row. Review-only faces are refused by both add-paths (vectorless, never probes).
- Cross-highlight is driven by `/api/faces/compare` pairs filtered client-side by `faceCrossThreshold` (moving the slider never re-queries). Threshold default 0.0, labelled uncalibrated — 0.363 must appear nowhere in controls (two wiring tests enforce this).
- Faces-only hit ordering: applied in `filteredHits` only when FACES is the sole active filter pill; backend collapse already makes the face score max-per-medium.
- A stored point probe matches its own image at cosine 1.0 — deliberately not suppressed server-side (noted in #124 body); UI may want to dim/flag self-hits, operator's call.
- Wiring-test trap: the faces-panel `x-init` test must regex-extract the attribute — slicing to the first `>` cuts mid-attribute at the `=>` arrow (I fixed my test to `re.search(r'x-init="([^"]*)"', …, re.S)`).
- `gh pr merge --delete-branch` fails to delete both local and remote branch here ("main is already used by worktree"); merge still succeeds — delete the remote with `git push origin --delete <branch>` and verify with `git ls-remote`.
- Main moved during my window: briefed bar 592/5 was stale; measure fresh against `git status --porcelain`-clean tree before quoting.
- My worktree `/tmp/claude-1000/…/218fa243-…/scratchpad/wt-faceui` is scratch; everything of value is on `ui/face-basket` (pushed) or merged.
