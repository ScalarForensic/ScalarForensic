import assert from 'node:assert/strict';
import { test } from 'node:test';
import { deferred, loadFrontend } from './harness.mjs';

function setup() {
  const { component: app, context } = loadFrontend();
  context.FormData = FormData;
  Object.defineProperty(app, 'selectedHit', { value: { image_hash: 'a'.repeat(64) }, writable: true });
  app.facesAvailable = true;
  app.sessionId = 's';
  app.selectedFileId = 'f';
  app.queryFacesToken = 'issued-token';
  app.queryFaces = [{ index: 0, searchable: true, chip_url: '/full-review' }];
  app._basketAddQuery(app.queryFaces[0]);
  return { app, context };
}

test('uploaded search and comparison carry detection identity and retain provenance', async () => {
  const { app, context } = setup();
  const requests = [];
  context.fetch = async (url, options) => {
    requests.push(options.body);
    return { ok: true, json: async () => ({ hits: [], pairs: [], embedder: { embedder_model_name: 'model' }, compat: { ok: true, warnings: ['changed gate'] } }) };
  };
  await app.runFaceSearch();
  await app.runFaceCompare(app.selectedHit.image_hash);
  await app.openFaceStats();
  assert.equal(requests.length, 3);
  for (const body of requests) assert.equal(body.get('detection_token'), 'issued-token');
  assert.equal(app.faceSearchCompat.warnings[0], 'changed gate');
  assert.equal(app.faceCompareEmbedder.embedder_model_name, 'model');
});

for (const [method, field] of [['explainFace', 'faceExplain'], ['openFaceStats', 'faceStats'], ['openFaceAudit', 'faceAudit']]) {
  test(`${method} cannot restore state after leaving the session`, async () => {
    const { app, context } = setup();
    const response = deferred();
    context.fetch = async () => ({ ok: true, json: () => response.promise });
    const pending = app[method]('point');
    app.sessionId = null;
    response.resolve({ marker: 'old' });
    await pending;
    assert.equal(app[field], null);
  });
}

test('query basket uses the full session review crop', () => {
  const { app } = setup();
  assert.equal(app.faceBasket[0].reviewUrl, '/full-review');
});

test('comparison reports excluded faces even when all query faces are review-only', async () => {
  const { app, context } = setup();
  app.queryFaces = [{ index: 0, searchable: false }];
  context.fetch = async () => ({ ok: true, json: async () => ({
    pairs: [], n_query_comparable: 0, n_query_review_only: 1,
    n_match_comparable: 2, n_match_review_only: 3,
  }) });
  await app.runFaceCompare(app.selectedHit.image_hash);
  assert.equal(app.faceCompareCounts.queryComparable, 0);
  assert.equal(app.faceCompareCounts.queryReviewOnly, 1);
  assert.equal(app.faceCompareCounts.matchReviewOnly, 3);
});
