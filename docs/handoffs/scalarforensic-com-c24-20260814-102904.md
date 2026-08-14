# Handoff — scalarforensic-com-c24, closing phase 8

## 1. The task

Close phase 8 and finish `docs/specs/video-playback-transcode.md`: the two
download affordances, the §7.2 label checked against a real encode, and then the
fold (spec §15 + four `CLAUDE.md` corrections) — dispatch
`data/reports/dispatch-phase8-close.md`.

## 2. Done, and not done

| task | state | where |
|---|---|---|
| 1 — the two download affordances | **merged**, `ea9ddab` (#201) | — |
| 2 — the §7.2 label against a real encode | **check done, two findings, both fixed** | PR **#202**, `fix/render-report-and-invocation` at `1106f20` |
| 3 — the fold (spec §15 + `CLAUDE.md`) | **NOT STARTED** | — |

#202 was green at `9d2a3d4` and carries one review commit on top (`1106f20`);
`m8` accepted it on substance and was holding it only for that commit. **A coder
never merges its own PR** — `m8`'s successor merges it.

Worktree `/tmp/sfn-c24` (off `origin/main`, `models/` copied in, `.env` copied
with `SFN_WEB_PORT=8123` appended). It is on tmpfs; both branches are pushed.

## 3. The next concrete step

**Task 3, the fold, and it is all that is left of the spec.** Only after #202 is
merged — never text describing something not on `main`.

1. **Spec §15**, last paragraph, still says *"Done: phases 1–3 and the carve"*.
   False since phase 4. Bring it to phases 1–8 complete. §11's module list is
   already correct and needs nothing.
2. **`CLAUDE.md`**, four things:
   1. the bar **re-measured on merged `main`**, in a worktree with `models/`
      copied in, quoting the sha and the tree. The file currently says
      991/5 at `fd31488`. **Never quote 771/5, and never my branch figures**
      (1,032/5 at `9d2a3d4`, 1,034/5 at `1106f20` — both measured on a branch);
   2. the `video_playback/` module list — it names none of `capability.py`,
      `encode.py`, `jobs.py`, `states.py`, `audit.py`;
   3. the audit log at `data/video_audit.log` under the default config, **beside**
      the cache directory and outside it, which is what makes it survive
      `sfn-video purge --all`. Confirmed by reading `audit_dir()` and by seeing
      the file land there. Put it next to the `sfn-faces purge` line;
   4. `sfn-video render` and `sfn-video purge` in the commands section.

Then say plainly in the PR whether the spec is finished. I believe it is, once
those two files are true.

## 4. Things written down nowhere else

- **The live check that produced the findings is cheap to repeat and you should
  repeat it rather than inject state.** Recipe, ~10 minutes:
  `.env` copied into the worktree with `SFN_WEB_PORT=8123`; a real HDR HEVC source
  (`/media/user01/SAM_870_SATA/Gitea_Backup/input_scalar/IMG_1018.MOV`, 16 s,
  hevc Main 10 + aac → `mode: transcode`); in the page set `phase='results'`,
  `appMode='search'` and one `results` entry whose hit carries **`scores`**
  (`filteredHits` does `'exact' in h.scores` and throws on a hit without it —
  that cost me two tries), then call the component's **own**
  `openVideoPlayback(path, 5000)` and `playChunkAt(5)`. Those two fetch and encode
  for real, so nothing is a fixture but the search result.
- **The record is below the fold and `fullPage` will not reach it.** The scroll
  container is `.image-panel-inner`. What worked: `position: fixed; top: 0` on
  `.vc-rendering` for the screenshot — layout-only, and disclosed as such.
- **The strongest evidence available for §7.2 is the sha256, and it is one
  command.** Copy the invocation off the screen, replace the last argument (the
  `.part` path) with a temp file, run it, and `sha256sum` it against the cache
  artifact. It matched byte for byte (`12e1fd9e…053f`, 5,668,062 B). That is the
  spec's promise demonstrated instead of asserted; do it again if the pipeline
  ever changes.
- **A mutation that survives is information about a missing *statement*, not
  always about a missing test.** M9 ("ignore `command_line`, always re-join")
  survived because both paths produce the same string for every current record —
  the difference is which carrier is *authoritative*. The test that killed it
  asserts that: rewrite the stored line in the log and it must be the one printed.
- **The audit log is per-CWD.** `video_cache_dir` defaults to `data/video_cache`
  *relative to the process CWD*, so the app and `sfn-video render` must run from
  the same directory or the CLI reads an empty log and prints a reproduction
  recipe. Running both from the worktree also kept the shared checkout's real
  `data/video_audit.log` untouched.
- **`data/reports/` on the shared checkout is the durable copy.** The worktree is
  tmpfs; I copied every artifact across:
  `c24-task2-label-vs-render.md` (the finding report),
  `c24-live-1-real-encode-record.png`, `c24-live-1-cli-render.txt`,
  `c24-live-2-cache-hit-no-invocation.png`,
  `c24-live-3-invocation-that-runs.png`, `c24-live-3-cli-render-after-fix.txt`,
  `c24-live-3-screen-record.json`.
- **The operator's app on :8080 was not touched.** Mine ran on 8123 and was killed
  by PID (`/tmp/c24-web.pid`). `pgrep -f sfn-web` shows the operator's PIDs
  alongside yours — read it, never `pkill` it.
- One agreement the fix does *not* remove, deliberately: the window prints `0`/`30`
  on screen and `0.0`/`30.0` on the CLI, because JSON hands JS a number and Python
  a float. Same value; unifying it would put float formatting in a third place.
- Ownership I hold and am releasing: `src/scalar_forensic/video_playback/`
  (prefix) and `tests/test_video_playback.py`. Both were unowned when `c22`
  retired; whoever takes the fold does not need them — the fold touches
  `docs/specs/video-playback-transcode.md` and `CLAUDE.md` only.
