import assert from 'node:assert/strict';
import { test } from 'node:test';
import { deferred, loadFrontend } from './harness.mjs';

function setup() {
  const { context, component: app } = loadFrontend();
  context.FormData = FormData;
  context.AbortController = AbortController;
  app.sessionId = 'session-a';
  app.analysisRunModes = ['exact'];
  app.loadHitTags = async () => {};
  return { context, app };
}
const payload = name => ({ results: [{ file_id: name, hits: [] }], provenance: { name }, embedding_models: { name } });

test('late response JSON cannot replace a newer query and superseded fetch is aborted', async () => {
  const { context, app } = setup();
  const oldJson = deferred();
  let firstSignal;
  let calls = 0;
  context.fetch = async (_url, options) => {
    if (++calls === 1) {
      firstSignal = options.signal;
      return { ok: true, json: () => oldJson.promise };
    }
    return { ok: true, json: async () => payload('new') };
  };
  const old = app.runQuery();
  await Promise.resolve();
  await app.runQuery();
  oldJson.resolve(payload('old'));
  await old;
  assert.equal(app.results[0].file_id, 'new');
  assert.equal(app.provenance.name, 'new');
  assert.equal(app.embeddingModels.name, 'new');
  assert.equal(firstSignal.aborted, true);
});

test('debounce invalidates an in-flight request before the timer fires', async () => {
  const { context, app } = setup();
  const pending = deferred();
  let signal;
  context.fetch = (_url, options) => { signal = options.signal; return pending.promise; };
  context.setTimeout = () => 1;
  const old = app.runQuery();
  app.debouncedQuery();
  pending.reject(new Error('obsolete failure'));
  await old;
  assert.equal(app.queryError, '');
  assert.equal(signal.aborted, true);
});

test('a response from a previous session cannot publish results or provenance', async () => {
  const { context, app } = setup();
  const pending = deferred();
  context.fetch = () => pending.promise;
  const old = app.runQuery();
  app.sessionId = 'session-b';
  pending.resolve({ ok: true, json: async () => payload('old-session') });
  await old;
  assert.equal(app.results.length, 0);
  assert.equal(app.provenance, null);
});
