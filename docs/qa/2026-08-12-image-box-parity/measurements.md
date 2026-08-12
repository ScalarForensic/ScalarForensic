# Image-box parity — measured before/after

Live UI, `./run.sh sfn-web` on :8099 against `danny_validation` /
`faces_danny_validation`, query `analysis_test/danny1.jpeg` (3 review-only
observations), viewport 1920×1080, `.image-box` `getBoundingClientRect()`.

## Before (`9aec99a` layout: `.image-box { flex: 1 }`)

| case | Query Image box | Best Match box | difference |
|---|---|---|---|
| no `.faces-panel` | 580.7 × 706.1 | 580.7 × 706.1 | 0 |
| panel, 0 chips | 580.7 × 706.1 | 580.7 × 635.3 | **−70.8 px** |
| panel, 1 chip | 580.7 × 706.1 | 580.7 × 509.0 | **−197.1 px** |
| panel, 3 chips | 580.7 × 706.1 | 580.7 × 509.0 | **−197.1 px** |

The loss equals the panel's own height exactly (70.8 / 197.1), because both
columns are `.image-panel-inner` flex columns of the same height (939.6 px) with
the same fixed stack below the image (`image-subs` 127 + `emb-box` 26 +
`meta-box` 80 = 233 px). Whatever the face panel takes comes out of the one
flexible item, which is the match image box. 1 chip and 3 chips cost the same
because they share one row; a second row would cost more again.

## After (`.cmp-inner > .image-box { flex: none; height: 75% }`)

| case | Query Image box | Best Match box | difference |
|---|---|---|---|
| no `.faces-panel` | 575.3 × 704.7 | 575.3 × 704.7 | 0 |
| panel, 0 chips | 575.3 × 704.7 | 575.3 × 704.7 | 0 |
| panel, 1 chip | 575.3 × 704.7 | 575.3 × 704.7 | 0 |
| panel, 3 chips | 575.3 × 704.7 | 575.3 × 704.7 | 0 |
| panel, 12 chips (2 rows) | 575.3 × 704.7 | 575.3 × 704.7 | 0 |

Both boxes lose 5.4 px of width to `scrollbar-gutter: stable`, applied to both
columns so the one that scrolls cannot come out narrower than the one that does
not. The match column now scrolls: `scrollHeight` 1135 against a client height
of 939.6. Chip sizing rules are untouched — review-only chips are still
`max-width/height: 72px`, `object-fit: contain`, never upscaled.

Screenshot: `after-3-chips.png` (3 review-only chips, both images equal size).

Not verified here: other viewport sizes, and the triage panels (deliberately out
of scope — `.cmp-inner` is scoped to the two compared columns).
