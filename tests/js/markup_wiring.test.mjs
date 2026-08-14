// Markup wiring for the phase 7 browser surface.
//
// These are **text-level** assertions and they are labelled as such. They belong
// here rather than in `tests/test_video_playback.py` only because that file is
// another owner's; the honest description of what they buy is the same as any
// wiring test — they pin that a binding exists, not that it renders. The
// behaviour behind each binding is covered by `full_job.test.mjs`, and rendering
// is covered by the live browser check, which nothing here replaces.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { REPO_ROOT, scriptPaths } from './harness.mjs';

const html = fs.readFileSync(
  path.join(REPO_ROOT, 'src', 'scalar_forensic', 'web', 'static', 'index.html'),
  'utf8',
);

test('full_job.js is loaded, and before app.js', () => {
  const files = scriptPaths().map((f) => path.basename(f));
  assert.ok(files.includes('full_job.js'));
  assert.ok(
    files.indexOf('full_job.js') < files.indexOf('app.js'),
    'parts must load before the assembler (CLAUDE.md)',
  );
});

test('the §4.3 disclosure is bound to the state cell and not to a literal', () => {
  assert.match(html, /class="vc-notice" x-show="contentionNotice"\s+x-text="contentionNotice"/);
  // The sentence itself must not appear in the markup: one definition, at
  // jobs.py:135. A copy here would drift silently.
  assert.doesNotMatch(html, /shares this host's two encode workers/);
});

// #139's precedent: a disclosure styled as a failure tells the analyst something
// untrue about what happened. The notice and the refusal must not reuse the
// error band's class.
test('the disclosure and the refusal are not error bands', () => {
  const notice = html.slice(html.indexOf('class="vc-notice" x-show="contentionNotice"'));
  assert.doesNotMatch(notice.slice(0, 200), /vc-error/);
  const refusal = html.slice(html.indexOf('x-show="fullJobRefused"'));
  assert.doesNotMatch(refusal.slice(0, 600), /vc-error/);
});

test('the §6.3 refusal shows both numbers and offers the original', () => {
  const refusal = html.slice(
    html.indexOf('x-show="fullJobRefused"'),
    html.indexOf('x-show="fullJobOverridden"'),
  );
  assert.match(refusal, /fullJobEstimateLabel/);
  assert.match(refusal, /fullJobLimitLabel/);
  assert.match(refusal, /videoPlaybackDownloadUrl/);
  // The numbers row is gated, which is what distinguishes the two verdicts on
  // screen: `unknown` measured nothing, so it prints nothing.
  assert.match(refusal, /x-show="fullJobRefusedTooBig"/);
});

// c18's finding, fixed: the client's explanation of the `unknown` verdict said
// back what the server's own `reason` already says. One disclosure, one wording.
test('the unknown refusal does not restate the server\'s reason', () => {
  assert.doesNotMatch(html, /did not report the duration, bitrate and coded/);
});

// ── The §6.3 override (#184) ────────────────────────────────────────────────
test('the override control is offered, and beside Download original', () => {
  const refusal = html.slice(
    html.indexOf('x-show="fullJobRefused"'),
    html.indexOf('x-show="fullJobOverridden"'),
  );
  assert.match(refusal, /x-show="fullJobOverrideOffered"/);
  assert.match(refusal, /@click="startFullJob\(true\)"/);
  // Beside, not instead of: the refusal still offers the cheap answer, and it
  // comes first.
  assert.ok(
    refusal.indexOf('videoPlaybackDownloadUrl') < refusal.indexOf('fullJobOverrideOffered'),
    'the override must not replace Download original',
  );
  // Not a retry button: this one is an act recorded against a named examiner.
  assert.match(refusal, /class="vc-override-btn"/);
});

test('the override disclosure renders the server\'s fields, not a copy of its sentence', () => {
  const band = html.slice(html.indexOf('x-show="fullJobOverridden"'));
  assert.match(band.slice(0, 900), /x-text="fullJobOverrideNotice"/);
  assert.match(band.slice(0, 900), /x-text="fullJobOverrideExaminer"/);
  assert.match(band.slice(0, 900), /x-text="fullJobOverrideVerdict"/);
  // One definition, at jobs.py:148. Same rule as CONTENTION_NOTICE.
  assert.doesNotMatch(html, /The estimate is advisory; the cache ceiling is not/);
});

// #139's precedent again: an override is a record, not a failure and not a
// success. It renders beside `full-job-failed` when a `full-copy-overshoot`
// kills the overridden encode, so styling it as either would be a claim about
// an outcome it does not know.
test('the override disclosure is not an error band', () => {
  assert.match(html, /class="vc-notice" x-show="fullJobOverridden"/);
  const band = html.slice(html.indexOf('x-show="fullJobOverridden"'));
  assert.doesNotMatch(band.slice(0, 900), /vc-error/);
});

// The disclosure has to survive the tab that clicked it: playback-info carries
// the job, and the panel adopts it on open.
test('playback-info is adopted when the player opens', () => {
  const evidence = fs.readFileSync(
    path.join(REPO_ROOT, 'src', 'scalar_forensic', 'web', 'static', 'js', 'evidence.js'),
    'utf8',
  );
  assert.match(evidence, /this\._adoptPlaybackInfo\(body\)/);
});

// TEXT-LEVEL ON PURPOSE, and it is the only check that can see this on every
// runner. `full_job.test.mjs` asserts the *value* `9,000,000,000`, but a bare
// `toLocaleString()` produces exactly that string on an en-US host — so on CI
// the value test cannot tell a pinned locale from an unpinned one. The defect
// only appears on a de-DE workstation, which is where the evidence gets
// screenshotted. This assertion sees it anywhere.
test('byte counts pin a locale rather than taking the host\'s', () => {
  const js = fs.readFileSync(
    path.join(REPO_ROOT, 'src', 'scalar_forensic', 'web', 'static',
      'js', 'video_playback', 'full_job.js'),
    'utf8',
  );
  for (const [name, source] of [['full_job.js', js], ['index.html', html]]) {
    assert.doesNotMatch(
      source,
      /toLocaleString\(\)\}\s*bytes/,
      `${name} prints a byte count in the host locale`,
    );
  }
  assert.match(js, /toLocaleString\('en-US'\)/);
});

test('the progress bar is only drawn when there is a fraction', () => {
  assert.match(html, /class="vc-bar" x-show="fullJobPercent !== null"/);
});

// ── The §7.2 rendering record (phase 8, browser side) ───────────────────────
// Text-level, and labelled as such: these pin that the record is bound to the
// state cells. What the rows contain is `rendering.test.mjs`'s job, and that it
// renders on screen is the live check's.
test('rendering.js is loaded, and before app.js', () => {
  const files = scriptPaths().map((f) => path.basename(f));
  assert.ok(files.includes('rendering.js'));
  assert.ok(
    files.indexOf('rendering.js') < files.indexOf('app.js'),
    'parts must load before the assembler (CLAUDE.md)',
  );
});

test('both panels render the §7.2 record through the one shared renderer', () => {
  assert.match(html, /x-for="row in chunkRenderingRows"/);
  assert.match(html, /x-for="row in fullJobRenderingRows"/);
  assert.match(html, /x-text="chunkRenderingCommand"/);
  assert.match(html, /x-text="fullJobRenderingCommand"/);
  // Rows carry their own name from the payload; a hand-written <dt> list in the
  // markup would be the second renderer this file exists to prevent.
  // Scoped to the two records: `x-text="row.label"` also belongs to the face
  // basket, which is a different x-for over a different shape.
  const records = [...html.matchAll(/<dl class="vc-rendering-rows">[\s\S]*?<\/dl>/g)];
  assert.equal(records.length, 2);
  for (const [record] of records) {
    assert.match(record, /<dt x-text="row\.label">/);
    assert.match(record, /<dd x-text="row\.value">/);
  }
});

// The record is neither a failure nor a disclosure — it is what produced the
// bytes on screen. #139's precedent: styling it as an error would tell the
// analyst something untrue about what happened.
test('the rendering record is not an error band, and is open by default', () => {
  const record = html.slice(html.indexOf('class="vc-rendering"'));
  assert.doesNotMatch(record.slice(0, 700), /vc-error/);
  assert.equal((html.match(/class="vc-rendering"[^>]*open>/g) || []).length, 2);
});

// The invocation is gated on there being one. A cache hit sends `command: null`
// because no process ran for that response (audit.py), and an argv shown anyway
// would be a sentence about a process that never existed.
test('the invocation block is gated on the server having recorded one', () => {
  assert.match(html, /class="vc-rendering-cmd" x-show="chunkRenderingCommand"/);
  assert.match(html, /class="vc-rendering-cmd" x-show="fullJobRenderingCommand"/);
});

// The old label was three fields, and the GPU fallback was a client sentence
// glossing a boolean. Both are now rows of the record, rendered from the
// server's own `fell_back` / `fallback_reason`.
test('the fallback is the server\'s field and not a client gloss', () => {
  assert.doesNotMatch(html, /encoded on the CPU after the GPU declined/);
});
