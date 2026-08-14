// Every browser script the page loads must compile and execute.
//
// This is the test the fourteen wiring tests could not be.  It covers the whole
// <script src> list and not just `player.js`: a SyntaxError in any part file
// breaks the same `sfn()` component, and the list is read out of `index.html`
// so a new part is covered on the commit that adds its tag.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';

import { REPO_ROOT, scriptPaths, runScript, browserContext, loadFrontend } from './harness.mjs';

test('every <script src> in index.html parses and executes', async (t) => {
  const files = scriptPaths();
  assert.ok(files.length >= 11, `expected the full part list, got ${files.length}`);
  for (const file of files) {
    // A fresh context per file: this test answers "does this file parse and
    // run", not "does the set compose" — that is the next test's job.
    await t.test(path.relative(REPO_ROOT, file), () => {
      runScript(file, browserContext());
    });
  }
});

test('the parts assemble into an sfn() component', () => {
  const { component } = loadFrontend();
  assert.ok(Object.keys(component).length > 100);
  assert.equal(component.chunk.state, 'idle');
});

// The merge rule from CLAUDE.md, asserted on the assembled object rather than
// trusted: `Object.assign` would evaluate every getter once at merge time and
// freeze the result as a plain value.  A wiring test can grep app.js for the
// string "defineProperties"; only an executing test can see that the property
// arrived as a getter.
test('computed properties survive the merge as getters, not evaluated values', () => {
  const { component } = loadFrontend();
  for (const name of ['chunkPlaybackOffered', 'chunkRetryOffered', 'chunkElapsedLabel']) {
    const descriptor = Object.getOwnPropertyDescriptor(component, name);
    assert.equal(typeof descriptor.get, 'function', `${name} is not a getter`);
    assert.equal(descriptor.value, undefined, `${name} was flattened to a value`);
  }
});
