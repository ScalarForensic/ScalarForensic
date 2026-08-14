// Behaviour of `js/video_playback/player.js`, asserted by running it.
//
// Each test below names why it is unreachable from a text-level wiring test.
// The rule of thumb: a wiring test can prove a token appears in a file; it
// cannot evaluate an expression, so anything whose truth is a *computed value*
// belongs here.
import test from 'node:test';
import assert from 'node:assert/strict';

import { loadFrontend } from './harness.mjs';

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
