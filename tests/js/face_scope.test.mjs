import assert from 'node:assert/strict';
import { test } from 'node:test';
import { deferred, loadFrontend } from './harness.mjs';

function setup(hit) {
  const { context, component: app } = loadFrontend();
  context.FormData = FormData;
  Object.defineProperty(app, 'selectedHit', { configurable: true, writable: true, value: hit });
  app.facesAvailable = true;
  app.sessionId = 'session';
  app.selectedFileId = 'file';
  app.queryFaces = [{ searchable: true }];
  return { context, app };
}

for (const video of [false, true]) {
  test(`face request scope follows selected ${video ? 'video frame' : 'image'}`, async () => {
    const hit = { image_hash: 'a'.repeat(64), ...(video ? { video_hash: 'b'.repeat(64), frame_timecode_ms: 1200 } : {}) };
    const { context, app } = setup(hit);
    const requests = [];
    context.fetch = async (url, options) => {
      requests.push({ url: new URL(url, 'http://local'), body: options?.body });
      return { ok: true, json: async () => ({ faces: [], pairs: [] }) };
    };
    await app.loadFacesForHit(hit.image_hash);
    await app.runFaceCompare(hit.image_hash);
    await app.openFaceAudit();
    assert.equal(requests.length, 3);
    for (const request of requests) {
      const values = request.body || request.url.searchParams;
      assert.equal(values.get('video_hash'), video ? hit.video_hash : null);
      assert.equal(values.get('frame_timecode_ms'), video ? '1200' : null);
    }
  });
}

for (const method of ['loadFacesForHit', 'runFaceCompare']) {
  test(`${method} ignores a late response from the previous source`, async () => {
    const hit = { image_hash: 'a'.repeat(64), video_hash: 'b'.repeat(64), frame_timecode_ms: 1000 };
    const { context, app } = setup(hit);
    const old = deferred();
    let calls = 0;
    context.fetch = async () => ({ ok: true, json: () => ++calls === 1 ? old.promise : Promise.resolve({ faces: [{ id: 'new' }], pairs: [{ point_id: 'new' }] }) });
    const first = app[method](hit.image_hash);
    await Promise.resolve();
    await Promise.resolve();
    app.selectedHit = { ...hit, frame_timecode_ms: 2000 };
    await app[method](hit.image_hash);
    old.resolve({ faces: [{ id: 'old' }], pairs: [{ point_id: 'old' }] });
    await first;
    assert.equal(method === 'loadFacesForHit' ? app.facesForHit[0].id : app.faceComparePairs[0].point_id, 'new');
  });
}

test('face search cannot repopulate a cleared probe selection', async () => {
  const { context, app } = setup({ image_hash: 'a'.repeat(64) });
  Object.defineProperty(app, 'selectedQueryFaceIndices', { configurable: true, writable: true, value: [0] });
  Object.defineProperty(app, 'selectedFacePointIds', { configurable: true, writable: true, value: [] });
  const old = deferred();
  context.fetch = async () => ({ ok: true, json: () => old.promise });
  const pending = app.runFaceSearch();
  app.selectedQueryFaceIndices = [];
  await app.runFaceSearch();
  old.resolve({ hits: [{ score: 0.9, face: { point_id: 'old' } }] });
  await pending;
  assert.equal(app.faceHits.length, 0);
  assert.equal(app.faceSearchLoading, false);
});
