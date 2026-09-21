import test from 'node:test';
import assert from 'node:assert/strict';
import { loadFrontend, deferred } from './harness.mjs';

function setup() {
  const { context, component: c } = loadFrontend();
  context.FormData = FormData;
  context.TextDecoder = TextDecoder;
  context.AbortController = AbortController;
  c.sessionId = 'case-one';
  c.selectedFileId = 'file-one';
  c.results = [{ file_id: 'file-one', hits: [{ path: '/old.jpg', image_hash: 'old', scores: { exact: 1 } }] }];
  c.concepts = [{ tag_id: 'tag-one' }];
  c.selectedConceptId = 'tag-one';
  return { context, c };
}

const cases = [
  ['query metadata', c => c._loadQueryMeta(), 'queryMeta', { hash_sha256: 'private' }],
  ['match metadata', c => c._loadMetaFrom('/api/metadata?path=old'), 'matchMeta', { private: true }],
  ['timeline', c => c._loadVideoTimeline('old'), 'videoTimeline', { private: true }],
  ['semantic stats', c => c.calcSemanticStats(), 'semanticStats', { private: true }],
  ['query frames', c => c._loadQueryFrames('file-one'), 'videoFramesCache', { frames: [{ timecode_ms: 100 }] }],
  ['expanded frames', c => c.toggleVideoExpanded('file-one'), 'videoFramesCache', { frames: [{ timecode_ms: 100 }] }],
  ['classifications', c => c.loadHitTags(), 'hitTags', { by_hash: { private: ['tag-one'] } }],
  ['triage selection', c => c.selectTriageHit({ point_id: 1, path: '/old.jpg' }), 'triageMatchMeta', { private: true }],
  ['triage search', c => c.runTriage(), 'triageResults', { hits: [{ point_id: 1, path: '/old.jpg' }] }],
  ['explore search', c => c.runExplore(), 'triageResults', { hits: [{ point_id: 1, path: '/old.jpg' }] }],
  ['audit', c => c.openAudit(), 'auditQueryLibVersions', { private: true }],
];

for (const [name, run, field, payload] of cases) {
  test(`${name}: delayed JSON cannot restore old data after reset`, async () => {
    const { context, c } = setup();
    const json = deferred();
    let requests = 0;
    context.fetch = async () => { requests++; return { ok: true, json: () => json.promise }; };
    const pending = run(c);
    await Promise.resolve();
    await Promise.resolve();
    c.resetToUpload();
    const expected = JSON.stringify(c[field]);
    const before = requests;
    json.resolve(payload);
    await pending;
    assert.equal(JSON.stringify(c[field]), expected);
    assert.equal(requests, before, 'stale response launched another request');
    assert.equal(c.auditLoading, false);
    assert.equal(c.semanticStatsLoading, false);
    assert.equal(c.triageRunning, false);
  });
}

for (const [name, run, field, payload] of cases.slice(0, 4)) {
  test(`${name}: selection changes reject delayed JSON`, async () => {
    const { context, c } = setup();
    const json = deferred();
    context.fetch = async () => ({ ok: true, json: () => json.promise });
    const pending = run(c);
    await Promise.resolve();
    c.selectedFileId = 'file-two';
    c.selectedHitKey = 'new-hit';
    json.resolve(payload);
    await pending;
    assert.equal(c[field], null);
  });
}

test('same-target metadata requests keep the latest response', async () => {
  const { context, c } = setup();
  const old = deferred();
  context.fetch = async () => ({ ok: true, json: () => old.promise });
  const first = c._loadMetaFrom('/api/metadata?path=old');
  context.fetch = async () => ({ ok: true, json: async () => ({ width: 20 }) });
  await c._loadMetaFrom('/api/metadata?path=new');
  old.resolve({ width: 10 });
  await first;
  assert.equal(c.matchMeta.width, 20);
});

test('triage metadata follows the newly selected hit', async () => {
  const { context, c } = setup();
  const old = deferred();
  context.fetch = async () => ({ ok: true, json: () => old.promise });
  const first = c.selectTriageHit({ point_id: 1, path: '/old.jpg' });
  context.fetch = async () => ({ ok: true, json: async () => ({ width: 20 }) });
  await c.selectTriageHit({ point_id: 2, path: '/new.jpg' });
  old.resolve({ width: 10 });
  await first;
  assert.equal(c.triageMatchMeta.width, 20);
});

test('audit cannot launch preprocessing after reset during provenance JSON', async () => {
  const { context, c } = setup();
  c.embeddingModels = { altered: {} };
  const provenance = deferred();
  const entered = deferred();
  const urls = [];
  context.fetch = async url => {
    urls.push(url);
    return { ok: true, json: () => {
      if (url.includes('hit-provenance')) { entered.resolve(); return provenance.promise; }
      return Promise.resolve({ version: '1' });
    } };
  };
  const pending = c.openAudit();
  await entered.promise;
  c.resetToUpload();
  provenance.resolve({ altered: { sscd_n_crops: 5 } });
  await pending;
  assert.equal(c.auditHitProvenance, null);
  assert.equal(urls.length, 2);
});

test('analysis completion cannot reopen results after reset', async () => {
  const { context, c } = setup();
  c.pendingFiles = [new File(['x'], 'query.jpg')];
  c.selectedModes = ['exact'];
  const chunk = deferred();
  let reads = 0;
  context.fetch = async () => ({ ok: true, json: async () => ({ results: [] }),
    body: { getReader: () => ({ read: () => reads++ ? Promise.resolve({ done: true }) : chunk.promise, cancel: async () => {} }) } });
  const pending = c.startAnalysis();
  await Promise.resolve();
  c.resetToUpload();
  chunk.resolve({ done: false, value: new TextEncoder().encode('data: {"type":"done","session_id":"old"}\n') });
  await pending;
  assert.equal(c.phase, 'upload');
  assert.equal(c.sessionId, null);
});

for (const [name, run, field, payload] of cases) {
  test(`${name}: current successful responses still populate the view`, async () => {
    const { context, c } = setup();
    context.fetch = async () => ({ ok: true, json: async () => payload });
    await run(c);
    assert.ok(c[field]);
    if (name.includes('frames')) assert.equal(c.videoFramesCache['file-one'][0].timecode_ms, 100);
    else if (name === 'triage search' || name === 'explore search') assert.equal(c.triageResults[0].point_id, 1);
    else if (name === 'classifications') assert.equal(c.hitTags.private[0], 'tag-one');
    else assert.deepEqual(c[field], payload);
  });
}

test('expansion and selection share frame freshness for the same file', async () => {
  const { context, c } = setup();
  const old = deferred();
  context.fetch = async () => ({ ok: true, json: () => old.promise });
  const first = c._loadQueryFrames('file-one');
  context.fetch = async () => ({ ok: true, json: async () => ({ frames: [{ timecode_ms: 200 }] }) });
  await c.toggleVideoExpanded('file-one');
  old.resolve({ frames: [{ timecode_ms: 100 }] });
  await first;
  assert.equal(c.videoFramesCache['file-one'][0].timecode_ms, 200);
});

test('re-query replacing the implicit selected hit invalidates metadata', async () => {
  const { context, c } = setup();
  const old = deferred();
  context.fetch = async () => ({ ok: true, json: () => old.promise });
  const pending = c._loadMatchMeta();
  c.results = [{ file_id: 'file-one', hits: [{ path: '/new.jpg', image_hash: 'new', scores: { exact: 1 } }] }];
  old.resolve({ private: true });
  await pending;
  assert.equal(c.matchMeta, null);
});

test('older triage finally cannot clear a newer exploration spinner', async () => {
  const { context, c } = setup();
  const old = deferred();
  const fresh = deferred();
  context.fetch = async () => ({ ok: true, json: () => old.promise });
  const first = c.runTriage();
  context.fetch = async () => ({ ok: true, json: () => fresh.promise });
  const second = c.runExplore();
  old.resolve({ hits: [{ point_id: 1 }] });
  await first;
  assert.equal(c.triageRunning, true);
  assert.equal(c.triageResults.length, 0);
  fresh.resolve({ hits: [{ point_id: 2 }], strategy: 'centroid' });
  await second;
  assert.equal(c.triageRunning, false);
  assert.equal(c.triageResults[0].point_id, 2);
});

test('a stale hit selection cannot launch its timeline after metadata completes', async () => {
  const { context, c } = setup();
  const old = deferred();
  let requests = 0;
  context.fetch = async () => { requests++; return { ok: true, json: () => old.promise }; };
  const pending = c.selectHit({ path: '/frame.jpg', image_hash: 'frame', scores: { exact: 1 }, is_video_frame: true, video_hash: 'video' });
  c.resetToUpload();
  old.resolve({ width: 10 });
  await pending;
  assert.equal(requests, 1);
  assert.equal(c.videoTimeline, null);
});

test('classification awaiting concepts cannot start work after reset', async () => {
  const { context, c } = setup();
  const concepts = deferred();
  c.concepts = [];
  c._conceptsPromise = concepts.promise;
  let requests = 0;
  context.fetch = async () => { requests++; return { ok: true, json: async () => ({}) }; };
  const pending = c.loadHitTags();
  c.resetToUpload();
  c.concepts = [{ tag_id: 'new' }];
  concepts.resolve();
  await pending;
  assert.equal(requests, 0);
});

test('metadata remains valid when face scores derive a fresh hit object', async () => {
  const { context, c } = setup();
  c.faceHits = [{ image_hash: 'old', score: 0.9 }];
  context.fetch = async () => ({ ok: true, json: async () => ({ width: 30 }) });
  await c._loadMatchMeta();
  assert.equal(c.matchMeta?.width, 30);
});

for (const [name, run, errorField] of [
  ['match metadata', c => c._loadMetaFrom('/api/metadata?path=old'), 'matchMetaError'],
  ['semantic stats', c => c.calcSemanticStats(), 'semanticStatsError'],
  ['triage', c => c.runTriage(), 'triageError'],
  ['explore', c => c.runExplore(), 'triageError'],
  ['audit', c => c.openAudit(), 'auditError'],
]) {
  test(`${name}: stale failures cannot restore errors after reset`, async () => {
    const { context, c } = setup();
    const json = deferred();
    const entered = deferred();
    context.fetch = async () => ({ ok: true, json: () => { entered.resolve(); return json.promise; } });
    const pending = run(c);
    await entered.promise;
    c.resetToUpload();
    json.reject(new Error('old case failure'));
    await pending;
    assert.equal(c[errorField], null);
  });
}

test('reset during analysis query completion cannot reopen results', async () => {
  const { context, c } = setup();
  c.pendingFiles = [new File(['x'], 'query.jpg')];
  c.selectedModes = ['exact'];
  const query = deferred();
  const entered = deferred();
  context.fetch = async url => url === '/api/analyze'
    ? { body: { getReader: () => ({ read: async () => ({ done: false,
      value: new TextEncoder().encode('data: {"type":"done","session_id":"old"}\n') }), cancel: async () => {} }) } }
    : { ok: true, json: () => { entered.resolve(); return query.promise; } };
  const pending = c.startAnalysis();
  await entered.promise;
  c.resetToUpload();
  query.resolve({ results: [], provenance: { private: true } });
  await pending;
  assert.equal(c.phase, 'upload');
  assert.equal(c.provenance, null);
});
