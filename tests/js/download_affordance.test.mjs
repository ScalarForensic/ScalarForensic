// The download affordance: one control, two placements (phase 8, item 1.3).
//
// §6.3 requires the refusal band to offer the original and §7.5 requires the
// panel's permanent escape route, so the control is rendered twice on purpose.
// What is *not* on purpose is two wordings for one act — the same defect class
// this codebase already refuses for the contention and override sentences, where
// a client paraphrase beside the server's own sentence is one disclosure said
// twice.
//
// The stale-evidence link is deliberately excluded: it offers the file as it is
// on disk *now*, which is no longer the file that was indexed, so it is a
// different statement about a different artifact and keeps its own wording.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { REPO_ROOT, loadComponent } from './harness.mjs';

const html = fs.readFileSync(
  path.join(REPO_ROOT, 'src', 'scalar_forensic', 'web', 'static', 'index.html'),
  'utf8',
);

// UNREACHABLE FROM TEXT: the label is an interpolation over payload state, so
// its value — that the analyst is told *which* file the click will fetch — can
// only be seen by evaluating it.
test('the label names the file it will download', () => {
  const component = loadComponent();
  component.videoPlayback = { filename: 'clip.mov', source_path: '/evidence/clip.mov' };
  assert.equal(component.videoPlaybackDownloadLabel, 'Download original (clip.mov)');
});

// The panel renders before `videoPlayback` is populated, and an escape route
// reading `Download original (undefined)` is worse than one that just names the
// act. Same rule as `videoPlaybackDownloadUrl` returning '' rather than a URL
// with `undefined` in it.
test('the label degrades to the bare act when there is no filename yet', () => {
  const component = loadComponent();
  component.videoPlayback = null;
  assert.equal(component.videoPlaybackDownloadLabel, 'Download original');
  component.videoPlayback = { source_path: '/evidence/clip.mov' };
  assert.equal(component.videoPlaybackDownloadLabel, 'Download original');
});

// TEXT-LEVEL ON PURPOSE: the defect being pinned is a *second wording in the
// markup*, which is by definition invisible to a value test — a duplicate
// literal in `index.html` computes nothing.
test('both placements bind the one label and neither restates it', () => {
  const refusal = html.slice(
    html.indexOf('x-show="fullJobRefused"'),
    html.indexOf('x-show="fullJobOverridden"'),
  );
  assert.match(refusal, /x-text="videoPlaybackDownloadLabel"/);
  // The escape route at the foot of the panel, outside the full-job block.
  const escape = html.slice(html.lastIndexOf('x-text="videoPlaybackDownloadLabel"'));
  assert.match(escape.slice(0, 200), /vc-open-btn|<\/a>/);
  assert.equal(
    (html.match(/x-text="videoPlaybackDownloadLabel"/g) || []).length,
    2,
    'the label is rendered in exactly the two placements §6.3 and §7.5 require',
  );
  // The two historical wordings, gone. A literal here would be the second
  // definition this test exists to prevent.
  assert.doesNotMatch(html, /Download the original instead/);
  assert.doesNotMatch(html, /Download original \(\$\{/);
});

// One definition, in `computed.js`. `full_job.js` and `player.js` must not grow
// a copy of the sentence.
test('the label has exactly one definition, and it is a getter', () => {
  const jsDir = path.join(REPO_ROOT, 'src', 'scalar_forensic', 'web', 'static', 'js');
  const sources = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.name.endsWith('.js')) sources.push([full, fs.readFileSync(full, 'utf8')]);
    }
  };
  walk(jsDir);
  const definers = sources.filter(([, src]) => /Download original/.test(src));
  assert.equal(definers.length, 1, `expected one definition, found ${definers.map(([f]) => f)}`);
  assert.equal(path.basename(definers[0][0]), 'computed.js');
  assert.match(definers[0][1], /get videoPlaybackDownloadLabel\(\)/);
});

// The stale-evidence band is not the same control and must not be folded into
// it: it offers a file that is no longer the indexed one, and saying "Download
// original" over it would be a false claim about what the bytes are.
test('the stale-evidence link keeps its own, different statement', () => {
  const band = html.slice(html.indexOf("videoPlaybackProvenance === 'stale'"));
  assert.match(band.slice(0, 600), /Download the file as it is now/);
  assert.doesNotMatch(band.slice(0, 600), /videoPlaybackDownloadLabel/);
});
