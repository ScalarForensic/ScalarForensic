# Image-box parity — independent verification

Tree: `763665f0d26835216a36f57f699d51248ee450fc` (`fix(web): keep the query and
match image boxes the same size`), already committed by the time this pass
started (brief said "working tree, not committed" — stale by the time the
browser was free; confirmed via `git rev-parse HEAD` and `git status --short`
showing clean before this pass began).

This is an **independent** re-check of `docs/qa/2026-08-12-image-box-parity/`
(com-c1's own self-check, committed in the same commit). Different session,
different browser profile, fresh CSS load.

## Setup

- App: `http://127.0.0.1:8099` (env `webenv.sh`, collection `danny_validation`,
  faces enabled). Server was already running; not started by this pass.
- Browser: Playwright MCP (chrome-devtools MCP's profile was locked by a
  stale/foreign session at pid 332268 — not any tracked fleet session; filed
  as friction, see `cx f` entry referenced in the DONE report, not worked
  around by touching that lock).
- Cache: navigated with `/?cachebust=<epoch>`; **separately** re-pointed the
  `<link rel="stylesheet">` href to `/static/style.css?v=<Date.now()>` after
  load, since the query-string cachebust on `/` does not bust `style.css`
  (confirmed stale-CSS trap, same one com-c1's brief called out).
- Idle drop-zone: real `mousemove`/click flows kept losing the race against
  the 5 s screensaver fade mid-multi-step-tool-call, so file selection was
  driven by opening the native chooser via `$refs.fileInput.click()` in-page
  (still the real `<input type=file>` + Playwright's file chooser, not a
  fabricated upload) after forcing `dropZoneIdle = false` on the live Alpine
  component. Analysis was started via the real `startAnalysis()` method the
  Analyze button calls — same code path, click bypassed only because the
  overlay was intercepting pointer events by the time multi-step tool calls
  landed.

## Real cases (0/1/3 chips reached honestly, no mocking)

| viewport | case | query box | match box | equal? |
|---|---|---|---|---|
| 1920×1080 | danny1.jpeg, 3 real chips (UI shows "3 in this image") | 703.78 × 693.33 | 703.78 × 693.33 | **YES** |
| 1920×1080 | danny2.jpeg, 1 real chip | 703.78 × 693.33 | 703.78 × 693.33 | **YES** |
| 1920×1080 | danny2.jpeg, panel absent (`facesAvailable` forced `false`) | 703.78 × 693.33 | 703.78 × 693.33 | **YES** |

Note: the Qdrant `faces_danny_validation` collection has 4 raw points for
danny1.jpeg (checked via direct scroll on the collection), but the UI/API
(`/api/faces/by-image/<hash>`) returns and renders 3 — a dedup/pagination
detail of the faces API, not something this pass chased; the UI's own count
("3 in this image") is what was measured against.

## 0 chips and 12 chips — could not be reached honestly, so synthesized

Checked every one of the 12 indexed images in `danny_validation` via the
Qdrant `faces_danny_validation` collection scroll: **every image has ≥1 face**
(counts 1–4), so no real match hits a genuine 0-face state. No sequence of
uploads reaches 12 chips on one image either (max real count is 4 raw / 3
rendered, on danny1.jpeg).

Both were reached instead by intercepting `window.fetch` in-page so the
component's own `loadFacesForHit()` (the real method, unmodified) received a
synthetic `{faces: [...]}` body for `/api/faces/by-image/<hash>` — 0 entries,
and 12 entries (built by cycling danny1's 3 real `review_chip_hash` values so
the chip images actually render, not placeholder failures). This exercises
the real Alpine render path (`x-for` over `facesForHit`, `x-show` on length)
under the same DOM/CSS the real 0/12 cases would use — it is not a claim that
12 real faces were detected in one image.

| viewport | case | query box | match box | equal? |
|---|---|---|---|---|
| 1920×1080 | 0 chips (synthetic) | 703.78 × 693.33 | 703.78 × 693.33 | **YES** |
| 1920×1080 | 12 chips (synthetic) | 703.78 × 693.33 | 703.78 × 693.33 | **YES** |

## Second/third viewport sweep (requested by com-m2: fixed-% CSS should hold at any size)

Re-ran all 5 states (0/1/3/12 chips synthetic + panel absent) at two more
viewports on the same selected hit (danny2.jpeg), using the fetch-mock method
above for every chip count including 1 and 3, for consistency across the
sweep:

**1280×800**

| case | query box | match box | equal? |
|---|---|---|---|
| 0 chips | 383.78 × 483.33 | 383.78 × 483.33 | **YES** |
| 1 chip | 383.78 × 483.33 | 383.78 × 483.33 | **YES** |
| 3 chips | 383.78 × 483.33 | 383.78 × 483.33 | **YES** |
| 12 chips | 383.78 × 483.33 | 383.78 × 483.33 | **YES** |
| panel absent | 383.78 × 483.33 | 383.78 × 483.33 | **YES** |

**1440×720** (short viewport — the case most likely to expose a fixed-height
column clipping differently than a percentage-of-column height)

| case | query box | match box | equal? |
|---|---|---|---|
| 0 chips | 463.78 × 423.33 | 463.78 × 423.33 | **YES** |
| 1 chip | 463.78 × 423.33 | 463.78 × 423.33 | **YES** |
| 3 chips | 463.78 × 423.33 | 463.78 × 423.33 | **YES** |
| 12 chips | 463.78 × 423.33 | 463.78 × 423.33 | **YES** |
| panel absent | 463.78 × 423.33 | 463.78 × 423.33 | **YES** |

## Verdict

**PASS**, all 15 case×viewport combinations (5 cases × 3 viewports), both real
and synthetic. Parity holds structurally — `.cmp-inner > .image-box { flex:
none; height: 75% }` sizes off the column in every configuration tried, never
off face-panel content. No case broke at any size.

## Console / network

0 console messages (errors or warnings) across the whole session
(`browser_console_messages`, all levels, since last navigation — one
navigation for the whole pass). All real network requests (`/api/analyze`,
`/api/query`, `/api/faces/by-image/...`, metadata, static assets) returned
200; the synthetic 0/12-chip responses were served from an in-page
`window.fetch` override and never hit the network, by design.

## Evidence

- `1920x1080-3chips.png`, `1920x1080-1chip.png`, `1920x1080-panel-absent.png`,
  `1920x1080-0chips-synthetic.png`, `1920x1080-12chips-synthetic.png`
- `1280x800-12chips-synthetic.png`, `1440x720-12chips-synthetic.png`
  (one screenshot per smaller viewport; the other 4 states per viewport are
  numeric-only above — screenshots are a paid resource, and the getBoundingClientRect
  numbers are the actual assertion).

## Replayable spec

See `replay.md` in this directory — no e2e runner exists in this repo yet
(`DECIDE:`d to the manager); steps are ordered selector/action/expected lines
for a human or a future runner to replay by hand.
