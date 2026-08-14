// Behaviour of `js/video_playback/full_job.js` and the §4.3 disclosure it shares
// with `player.js`.
//
// Each test names why it is unreachable from a text-level wiring test. The rule:
// a wiring test proves a token appears in a file; anything whose truth is a
// *computed value* belongs here.
import test from 'node:test';
import assert from 'node:assert/strict';

import { loadFrontend } from './harness.mjs';

const NOTICE =
  "A full viewing copy is being produced. It shares this host's two encode workers " +
  'with playback, so the next chunk will take longer than usual to appear.';

function componentUnderTest() {
  const { context, component } = loadFrontend();
  component.videoPlayback = {
    player_state: 'needs-transcode',
    source_path: '/evidence/clip.mov',
    video_sha256: 'a'.repeat(64),
    chunk_seconds: 30,
    duration_ms: 600_000,
    lease_seconds: 120,
  };
  return { context, component };
}

function jobView(overrides = {}) {
  return {
    video_sha256: 'a'.repeat(64),
    player_state: 'full-job-running',
    cancelled: false,
    waiters: 1,
    frames: 300,
    out_seconds: 12,
    duration_seconds: 600,
    fraction: 0.02,
    elapsed_seconds: 8,
    written_bytes: 1_048_576,
    estimate_bytes: 900_000_000,
    limit_bytes: 4_000_000_000,
    rate: 1.5,
    eta_seconds: 400,
    eta_label: 'about 7 min remaining (extrapolated)',
    contention_notice: NOTICE,
    full_url: null,
    error: null,
    ...overrides,
  };
}

// ── The §4.3 disclosure ─────────────────────────────────────────────────────
// THE LOAD-BEARING TEST. Yielding buys ~9% (18.31 s vs 16.83 s median, n=6) and
// neither pipeline lands inside §4.2's 6–10 s window, so this sentence *is* the
// remedy. The server sends it on two payloads; both must reach the same cell, or
// one surface shows the disclosure and the other silently drops it.
//
// UNREACHABLE FROM TEXT: a wiring test can prove the markup mentions
// `contentionNotice`. Only executing the code can prove that the field arriving
// as a string produces the server's sentence and arriving as null produces an
// empty cell.
test('the contention notice renders from the chunk response, and clears when null', async () => {
  const { context, component } = componentUnderTest();
  let notice = NOTICE;
  context.fetch = async () => ({
    ok: true,
    json: async () => ({
      player_state: 'chunk-ready',
      chunk_start: 0,
      next_chunk_start: 30,
      chunk_url: '/api/video-chunk?start=0',
      contention_notice: notice,
    }),
  });

  await component.playChunkAt(0);
  assert.equal(component.contentionNotice, NOTICE);

  notice = null;
  await component.playChunkAt(30);
  assert.equal(component.contentionNotice, '');
  component.closeChunkPlayback();
});

test('the contention notice renders from a job-status poll — the second carrier', () => {
  const { component } = componentUnderTest();
  component._applyFullJobView(jobView());
  assert.equal(component.contentionNotice, NOTICE);

  // The server sends it only while running; a finished job clears it.
  component._applyFullJobView(jobView({ player_state: 'full-job-done', contention_notice: null }));
  assert.equal(component.contentionNotice, '');
});

// UNREACHABLE FROM TEXT, and a real hazard: a chunk failure while an export runs
// is exactly when the analyst most needs to know the encoders are shared. The
// disclosure must survive it.
test('a chunk failure does not clear a live contention notice', () => {
  const { component } = componentUnderTest();
  component._setContentionNotice(NOTICE);
  component._applyChunkFailure({ status: 503, detail: { player_state: 'chunk-failed' } });
  assert.equal(component.chunk.state, 'chunk-failed');
  assert.equal(component.contentionNotice, NOTICE);
});

test('the notice is never invented client-side', () => {
  const { component } = componentUnderTest();
  for (const value of [null, undefined, '', 0, false]) {
    component._setContentionNotice(value);
    assert.equal(component.contentionNotice, '');
  }
});

// ── Progress, and no fabricated percentage ──────────────────────────────────
// UNREACHABLE FROM TEXT: the shape of a formatted string, and the absence of a
// number. §5 and #139 forbid inventing a percentage; `fraction: null` must
// degrade to elapsed-only rather than to "0%".
test('a null fraction degrades to elapsed-only with no percentage anywhere', () => {
  const { component } = componentUnderTest();
  component._applyFullJobView(jobView({ fraction: null, elapsed_seconds: 12 }));
  assert.equal(component.fullJobPercent, null);
  assert.equal(component.fullJobProgressLabel, '12 s elapsed');
  assert.doesNotMatch(component.fullJobProgressLabel, /%/);
});

test('a reported fraction becomes a percentage and the ETA label is the server\'s', () => {
  const { component } = componentUnderTest();
  component._applyFullJobView(jobView({ fraction: 0.4567 }));
  assert.equal(component.fullJobPercent, 45.7);
  assert.match(component.fullJobProgressLabel, /^45\.7% · 8 s elapsed$/);
  assert.equal(component.fullJobEtaLabel, 'about 7 min remaining (extrapolated)');
  assert.equal(component.fullJobRateLabel, '1.50× realtime');
});

// ── The §6.3 refusal, three-valued ──────────────────────────────────────────
// UNREACHABLE FROM TEXT: which of two mutually exclusive getters is true for a
// given payload. §6.3 requires the estimate *shown* and the original offered.
test('a refused export shows the estimate and the ceiling', async () => {
  const { context, component } = componentUnderTest();
  context.fetch = async () => ({
    ok: false,
    status: 507,
    json: async () => ({
      detail: {
        error: 'full-copy-refused',
        player_state: 'capacity-exhausted',
        reason: 'The estimated viewing copy does not fit in the cache.',
        estimate_bytes: 9_000_000_000,
        limit_bytes: 4_000_000_000,
      },
    }),
  });

  await component.startFullJob();
  assert.equal(component.fullJobRefused, true);
  assert.equal(component.fullJobRefusedTooBig, true);
  assert.equal(component.fullJobRefusedUnknown, false);
  // Digits compared with grouping separators stripped: the label uses
  // `toLocaleString()`, matching how this UI already prints sizes
  // (index.html:471), so the separator is the runner's locale and not a
  // property of the code under test.
  const digits = (s) => s.replace(/[^0-9]/g, '');
  assert.ok(digits(component.fullJobEstimateLabel).includes('9000000000'));
  assert.ok(digits(component.fullJobLimitLabel).includes('4000000000'));
  assert.equal(component.fullJob.reason, 'The estimated viewing copy does not fit in the cache.');
});

// #147's defect, in this UI: `unknown` is not `refused`. The server sends
// `estimate_bytes: null`, so there is no number — and printing the too-big
// sentence, or a zero, would state something the system never measured.
test('an unknown-verdict refusal prints no estimate and not the too-big sentence', async () => {
  const { context, component } = componentUnderTest();
  context.fetch = async () => ({
    ok: false,
    status: 507,
    json: async () => ({
      detail: {
        error: 'full-copy-unknown',
        player_state: 'capacity-exhausted',
        reason: 'This source did not report the numbers an estimate needs.',
        estimate_bytes: null,
        limit_bytes: 4_000_000_000,
      },
    }),
  });

  await component.startFullJob();
  assert.equal(component.fullJobRefused, true);
  assert.equal(component.fullJobRefusedUnknown, true);
  assert.equal(component.fullJobRefusedTooBig, false, 'unknown must not claim a size');
  assert.equal(component.fullJobEstimateLabel, '', 'there is no estimate to print');
});

// ── Cancel, completion, auto-switch ─────────────────────────────────────────
// UNREACHABLE FROM TEXT: the server states that a cancel with other waiters does
// NOT stop the job, and the client must copy that rather than assume it stopped.
test('cancelling with other waiters leaves the job running', async () => {
  const { context, component } = componentUnderTest();
  component._applyFullJobView(jobView());
  context.fetch = async () => ({
    ok: true,
    json: async () => ({ outcome: 'detached', waiters: 2, player_state: 'full-job-running' }),
  });

  await component.cancelFullJob();
  assert.equal(component.fullJob.state, 'full-job-running');
  assert.equal(component.fullJob.waiters, 2);
  component.closeFullJob();
});

test('cancelling the last waiter returns the player to needs-transcode', async () => {
  const { context, component } = componentUnderTest();
  component._applyFullJobView(jobView());
  context.fetch = async () => ({
    ok: true,
    json: async () => ({ outcome: 'cancelled', waiters: 0, player_state: 'needs-transcode' }),
  });

  await component.cancelFullJob();
  assert.equal(component.fullJob.state, 'needs-transcode');
  assert.equal(component.fullJobRunning, false);
});

// UNREACHABLE FROM TEXT: the switch timestamp is computed from whichever chunk
// window is loaded, falling back to the panel's timecode. Jumping to zero would
// lose the frame the analyst waited for.
test('the completion switch keeps the current position', () => {
  const { component } = componentUnderTest();
  component.chunk.start = 120;
  component._applyFullJobView(jobView({
    player_state: 'full-job-done',
    contention_notice: null,
    full_url: '/api/video-full?path=x&fp=y',
  }));
  assert.equal(component.fullJobDone, true);
  assert.equal(component.fullJobSwitchAtSeconds, 120);

  component.chunk.start = null;
  component.videoPlaybackTimecodeMs = 45_000;
  assert.equal(component.fullJobSwitchAtSeconds, 45);
});

// ── The navigate-away prompt ────────────────────────────────────────────────
// UNREACHABLE FROM TEXT: that a listener was actually registered, and that it
// asks. Whether the browser then shows its dialog is the browser's call, gated on
// user interaction — that half belongs to the live check.
test('a running export arms a beforeunload handler and disarms it when the job ends', async () => {
  const { context, component } = componentUnderTest();
  context.fetch = async () => ({ ok: true, json: async () => jobView() });

  assert.equal(context.window.listeners.get('beforeunload'), undefined);
  await component.startFullJob();
  const armed = context.window.listeners.get('beforeunload');
  assert.equal(armed.length, 1, 'no beforeunload handler was registered');
  assert.equal(component.fullJobWarnsOnLeave, true);

  // It asks: preventDefault called and returnValue set, which is what a browser
  // reads as "prompt the user".
  const event = { prevented: false, preventDefault() { this.prevented = true; }, returnValue: null };
  armed[0](event);
  assert.equal(event.prevented, true);
  assert.equal(event.returnValue, '');

  component._stopFullJobPoll();
  assert.equal(context.window.listeners.get('beforeunload').length, 0, 'handler was not removed');
  assert.equal(component.fullJobWarnsOnLeave, true); // state is still running; the timer is not
});

test('an idle job does not warn on leave', () => {
  const { component } = componentUnderTest();
  assert.equal(component.fullJobWarnsOnLeave, false);
});
