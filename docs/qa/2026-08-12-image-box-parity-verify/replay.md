# Replay spec: image-box parity (Query vs Best Match)

No e2e runner exists in this repo. Ordered steps — selector · action ·
expected — for a human or a future runner. Written against HEAD
`763665f0d26835216a36f57f699d51248ee450fc`.

## Setup (once per session)

1. Start app per `./run.sh sfn-web` (or reuse a running instance) with faces
   enabled and a collection that has ≥1 image with ≥1 indexed face.
2. Navigate to `/?cachebust=<epoch-ms>` — plain reload is not enough, the app
   sends no cache headers on `/`.
3. **Re-bust the stylesheet separately**: `document.querySelector('link[href*="style.css"]').href = '/static/style.css?v=' + Date.now()` —
   the `/?cachebust=` query string does **not** propagate to `style.css`.
   Skipping this measures stale CSS and produces a false PASS or FAIL.
4. Resize viewport before uploading (measurements depend on it).

## Case: real 1/3-chip upload flow

1. `input[type=file]:not([webkitdirectory])` (hidden) · set files to a query
   image with N indexed faces · expected: file queued.
   - If the drop-zone screensaver (`.drop-zone.idle`, 5 s no-mousemove) has
     faded, `document.dispatchEvent(new MouseEvent('mousemove', {bubbles:
     true}))` first (must dispatch on `document`, not `window` — a `window`
     target does not reach a `document`-level listener), OR set
     `Alpine.$data(document.querySelector('#app')).dropZoneIdle = false`
     directly — both are legitimate, the fade is a real UX behavior and this
     replicates a user nudging the mouse.
2. Analyze button (`▶ Analyze`, calls `startAnalysis()`) · click, or call
   `Alpine.$data(app).startAnalysis()` directly if the idle overlay is still
   intercepting pointer events · expected: `phase` becomes `"results"`,
   `selectedHit` set.
3. `.faces-head .sub-val` · read text · expected: `"N in this image"`
   matching the image's real indexed-face count.
4. `.image-box` (there are exactly two: Query Image panel, Best Match panel)
   · `getBoundingClientRect()` on both · expected: `width` and `height`
   equal between the two, to within floating-point rounding.

## Case: panel absent

1. With a result showing, `Alpine.$data(app).facesAvailable = false` ·
   expected: `.faces-panel` no longer in the DOM.
2. Re-measure both `.image-box` rects · expected: still equal, and equal to
   the panel-present measurement at the same viewport (the box must not
   resize when the panel disappears).

## Case: 0 / 12 chips (synthetic — no real fixture reaches these)

Every indexed image in this repo's face-enabled test data has ≥1 real
detected face, so 0 chips cannot be produced by a real upload; 12 requires
more faces than any single test image has. Reached instead by intercepting
the network call the real code makes, so the render path under test is real:

```js
window.__origFetch = window.__origFetch || window.fetch.bind(window);
window.fetch = async (url, ...rest) => {
  if (typeof url === 'string' && url.includes('/api/faces/by-image/')) {
    return new Response(JSON.stringify({ faces: /* [] or 12 fake face objects */ }),
      { status: 200, headers: { 'Content-Type': 'application/json' } });
  }
  return window.__origFetch(url, ...rest);
};
await Alpine.$data(document.querySelector('#app'))
  .loadFacesForHit(Alpine.$data(document.querySelector('#app')).selectedHit?.image_hash);
```

Fake face objects need at minimum: `id` (unique, for `x-for :key`),
`review_chip_hash` (reuse a real one so the `<img>` actually loads),
`det_conf`, `quality`, `embedding_status`. Then re-measure `.image-box` as
above.

## Assertion (every case above)

```js
const [q, m] = [...document.querySelectorAll('.image-box')].map(b => b.getBoundingClientRect());
console.assert(q.width === m.width && q.height === m.height, 'image-box parity broken', q, m);
```

## Viewports to sweep

1920×1080, 1280×800, 1440×720 (short — most likely to expose a
fixed-height-column vs percentage-of-column difference). Re-run every case at
each.
