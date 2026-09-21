// Behaviour of `js/video_playback/player.js`, asserted by running it.
//
// Each test below names why it is unreachable from a text-level wiring test.
// The rule of thumb: a wiring test can prove a token appears in a file; it
// cannot evaluate an expression, so anything whose truth is a *computed value*
// belongs here.
import test from 'node:test';
import assert from 'node:assert/strict';

import { deferred, loadFrontend } from './harness.mjs';

test('source seeking requests the containing window and positions within its video', async () => {
  const { context, component } = playerUnderTest();
  const requests = [];
  context.fetch = async url => {
    requests.push(url);
    return { ok: true, json: async () => ({ player_state: 'chunk-ready', chunk_start: 60,
      next_chunk_start: null, chunk_url: '/window60' }) };
  };
  try {
    await component.seekChunkTo(65.5);
    assert.ok(requests[0].includes('t=65.5'));
    assert.equal(component.chunk.start, 60);
    const element = { currentTime: 0 };
    component.seekChunkBuffer(element, component.chunk.buffer);
    assert.equal(element.currentTime, 5.5);
    const count = requests.length;
    await component.seekChunkTo(-1);
    await component.seekChunkTo(NaN);
    assert.equal(requests.length, count);
  } finally { component.closeChunkPlayback(); }
});

test('prefetch from a closed player cannot populate the same next timestamp in a new one', async () => {
  const { context, component } = playerUnderTest();
  const old = deferred();
  context.fetch = async url => ({ ok: true, json: () => url.includes('video-chunk') ? old.promise : {} });
  component.chunk.next = 30;
  const pending = component._prefetchNextChunk();
  component.closeChunkPlayback();
  component.chunk.next = 30;
  old.resolve({ chunk_url: '/old-case' });
  await pending;
  assert.equal(component.chunk.preload.url, '');
});

test('seeking twice within one cached window moves the already-loaded element', async () => {
  const { context, component } = playerUnderTest();
  component.$refs = { chunkA: { currentTime: 2 } };
  Object.assign(component.chunk, { state: 'chunk-ready', start: 60, url: '/window60', buffer: 0 });
  context.fetch = async () => ({ ok: true, json: async () => ({ chunk_start: 60,
    next_chunk_start: null, chunk_url: '/window60', player_state: 'chunk-ready' }) });
  try {
    await component.seekChunkTo(68);
    assert.equal(component.$refs.chunkA.currentTime, 8);
    await component.seekChunkTo(72);
    assert.equal(component.$refs.chunkA.currentTime, 12);
  } finally { component.closeChunkPlayback(); }
});

test('retry after a failed seek requests the failed position, not the last successful window', async () => {
  const { context, component } = playerUnderTest();
  component.chunk.start = 0;
  const positions = [];
  context.fetch = async url => {
    if (url.includes('video-chunk')) positions.push(new URL(url, 'http://test').searchParams.get('t'));
    return { ok: false, json: async () => ({ detail: { player_state: 'chunk-failed',
      retryable: true, retry_after_seconds: 0, reason: 'temporarily unavailable' } }) };
  };
  await component.seekChunkTo(95);
  await component.retryChunk();
  assert.deepEqual(positions, ['95', '95']);
  component.closeChunkPlayback();
});

test('an old window ending cannot overtake an explicit seek in flight', async () => {
  const { context, component } = playerUnderTest();
  const seekResponse = deferred();
  const positions = [];
  Object.assign(component.chunk, { state: 'chunk-ready', start: 0, next: 30, url: '/window0' });
  context.fetch = async url => {
    if (!url.includes('video-chunk')) return { ok: true, json: async () => ({}) };
    const seconds = Number(new URL(url, 'http://test').searchParams.get('t'));
    positions.push(seconds);
    return { ok: true, json: () => seconds === 95 ? seekResponse.promise :
      { player_state: 'chunk-ready', chunk_start: 30, next_chunk_start: null, chunk_url: '/window30' } };
  };
  const pending = component.seekChunkTo(95);
  try {
    await component.advanceToNextChunk(null, null);
    seekResponse.resolve({ player_state: 'chunk-ready', chunk_start: 90,
      next_chunk_start: null, chunk_url: '/window90' });
    await pending;
    assert.deepEqual(positions, [95]);
    assert.equal(component.chunk.start, 90);
    assert.equal(component.chunk.seekOffset, 5);
  } finally { component.closeChunkPlayback(); }
});

function playerUnderTest(playbackInfo = {}) {
  const { context, component } = loadFrontend();
  component.videoPlayback = {
    player_state: 'needs-transcode',
    source_path: '/evidence/clip.mov',
    video_sha256: 'a'.repeat(64),
    chunk_seconds: 30,
    duration_ms: 120_000,
    lease_seconds: 120,
    ...playbackInfo,
  };
  return { context, component };
}

// UNREACHABLE FROM TEXT: this is the exact expression that carried the historical
// `?? … ||` SyntaxError. Its correctness is a *value* — which of three sources
// the reason string comes from — and a grep for "detail" cannot evaluate it.
// §10.1 requires that no failure ever renders a blank reason.
test('_applyChunkFailure renders a FastAPI plain-string detail', () => {
  const { component } = playerUnderTest();
  component._applyChunkFailure({ status: 403, detail: 'Path is outside the allowed roots.' });
  assert.equal(component.chunk.state, 'chunk-failed');
  assert.equal(component.chunk.reason, 'Path is outside the allowed roots.');
  assert.equal(component.chunk.retryable, false);
});

test('_applyChunkFailure copies the server §10.1 row and never invents a state', () => {
  const { component } = playerUnderTest();
  component._applyChunkFailure({
    status: 503,
    detail: {
      player_state: 'capacity-exhausted',
      error: 'queue_full',
      reason: 'Too many encodes are already running.',
      retryable: true,
      retry_after_seconds: 30,
    },
  });
  assert.equal(component.chunk.state, 'capacity-exhausted');
  assert.equal(component.chunk.kind, 'queue_full');
  assert.equal(component.chunk.reason, 'Too many encodes are already running.');
  assert.equal(component.chunk.retryable, true);
  assert.equal(component.chunk.retryAfterS, 30);
  component.closeChunkPlayback(); // stop the countdown interval
});

test('_applyChunkFailure falls back to the HTTP status, never to a blank reason', () => {
  const { component } = playerUnderTest();
  component._applyChunkFailure({ status: 500, detail: null });
  assert.equal(component.chunk.state, 'chunk-failed');
  assert.match(component.chunk.reason, /HTTP 500/);
});

// UNREACHABLE FROM TEXT: a getter's value across a three-field state
// transition. The §10.1 anti-retry-storm rule is not "the file mentions
// retryable" — it is that the button stays off until the countdown reaches
// zero, which only evaluating the getter can show.
test('chunkRetryOffered stays false until the Retry-After countdown reaches zero', () => {
  const { component } = playerUnderTest();
  Object.assign(component.chunk, { state: 'chunk-failed', retryable: true, retryAfterS: 3 });
  assert.equal(component.chunkHasFailed, true);
  assert.equal(component.chunkRetryOffered, false);
  assert.equal(component.chunkRetryCountdown, 3);

  component.chunk.retryAfterS = 0;
  assert.equal(component.chunkRetryOffered, true);
  assert.equal(component.chunkRetryCountdown, 0);

  // A failure the server called permanent never offers a retry, at any count.
  component.chunk.retryable = false;
  assert.equal(component.chunkRetryOffered, false);
});

// UNREACHABLE FROM TEXT: the boundary swap is a five-field state transition
// (§4.2). A wiring test can see the word `buffer` in the file; only running
// `advanceToNextChunk` shows that the swap consumes the preload, advances the
// window, and computes the *next* window's end against the duration.
test('advanceToNextChunk swaps buffers and consumes the preload', async () => {
  const { context, component } = playerUnderTest();
  context.fetch = async () => ({
    ok: true,
    json: async () => ({ chunk_url: '/api/video-chunk?t=90', chunk_start: 90 }),
  });
  Object.assign(component.chunk, {
    state: 'chunk-ready',
    buffer: 0,
    start: 30,
    next: 60,
    url: '/api/video-chunk?t=30',
    preload: { start: 60, url: '/api/video-chunk?t=60' },
  });
  const played = [];
  const playing = { pause: () => played.push('pause') };
  const hidden = { play: async () => played.push('play') };

  await component.advanceToNextChunk(playing, hidden);

  assert.equal(component.chunk.buffer, 1, 'the on-screen element did not swap');
  assert.equal(component.chunk.start, 60);
  assert.equal(component.chunk.next, 90);
  assert.equal(component.chunk.url, '/api/video-chunk?t=60');
  // Field by field, not deepStrictEqual: objects the part file created live in
  // the vm realm, so their Object.prototype is not this module's and a strict
  // deep-equal against a literal fails on the prototype alone.
  assert.equal(component.chunk.preload.start, null);
  assert.equal(component.chunk.preload.url, '');
  assert.deepEqual(played, ['play', 'pause'], 'the hidden element must start before the old one stops');
});

// The final chunk must stop, not wrap. `next: null` is the only signal, and the
// duration arithmetic that produces it is the same expression tested above.
test('advanceToNextChunk on the final chunk does nothing', async () => {
  const { component } = playerUnderTest();
  Object.assign(component.chunk, { state: 'chunk-ready', buffer: 1, start: 90, next: null });
  await component.advanceToNextChunk(null, null);
  assert.equal(component.chunk.buffer, 1);
  assert.equal(component.chunk.start, 90);
});

// UNREACHABLE FROM TEXT: §5 forbids a fabricated percentage. The assertion is
// about the *shape of a formatted string*, which is produced by a getter.
test('chunkElapsedLabel reports elapsed seconds and no percentage', () => {
  const { component } = playerUnderTest();
  component.chunk.elapsedS = 8;
  assert.equal(component.chunkElapsedLabel, '8 s elapsed');
  assert.doesNotMatch(component.chunkElapsedLabel, /%/);
});

// The player is offered only on the server's verdict, and `chunk_seconds` is
// read off the wire with a documented default — not hard-coded twice.
test('chunkPlaybackOffered and chunkSeconds come off the wire', () => {
  const { component } = playerUnderTest({ player_state: 'playable', chunk_seconds: 10 });
  assert.equal(component.chunkPlaybackOffered, false);
  assert.equal(component.chunkSeconds, 10);

  component.videoPlayback.player_state = 'needs-transcode';
  assert.equal(component.chunkPlaybackOffered, true);

  delete component.videoPlayback.chunk_seconds;
  assert.equal(component.chunkSeconds, 30);
});
