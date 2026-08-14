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
    html.indexOf('x-show="fullJobFailed"'),
  );
  assert.match(refusal, /fullJobEstimateLabel/);
  assert.match(refusal, /fullJobLimitLabel/);
  assert.match(refusal, /fullJobRefusedUnknown/);
  assert.match(refusal, /videoPlaybackDownloadUrl/);
});

test('the progress bar is only drawn when there is a fraction', () => {
  assert.match(html, /class="vc-bar" x-show="fullJobPercent !== null"/);
});
