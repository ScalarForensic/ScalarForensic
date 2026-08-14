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

// `jobs.OVERRIDE_NOTICE`, verbatim — the constant, not a paraphrase of it. The
// point of asserting it here is that the client renders the server's string; a
// paraphrase would pass while the UI showed a second wording of one disclosure.
const OVERRIDE_NOTICE =
  'The capacity estimate for this full viewing copy was refused and set aside by the ' +
  'examiner named here. The estimate is advisory; the cache ceiling is not. This export ' +
  'is still stopped if it passes the size a single rendering may occupy.';

function overrideRecord(overrides = {}) {
  return {
    examiner_id: 'examiner-7',
    verdict: 'refused',
    estimate_bytes: 9_000_000_000,
    limit_bytes: 4_000_000_000,
    notice: OVERRIDE_NOTICE,
    ...overrides,
  };
}

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
    override: null,
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
  // Exact, both figures, and no dependence on the runner's locale: the label
  // pins 'en-US' (see the locale test below), so this is the string an examiner
  // screenshots on any workstation.
  assert.equal(component.fullJobEstimateLabel, '8583.1 MB (9,000,000,000 bytes)');
  assert.equal(component.fullJobLimitLabel, '3814.7 MB (4,000,000,000 bytes)');
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

// ── The §6.3 examiner override (ruling 2026-08-14, #184) ────────────────────
// The server has sent all of this since 45fe545 and nothing displayed it. Each
// test below names the carrier it pins, because the failure mode here is the
// contention notice's: two carriers, one of them quietly dropped.

// UNREACHABLE FROM TEXT: that the byte label does not depend on the machine the
// test (or the browser) runs on. A wiring test sees `toLocaleString` either way;
// only executing it shows which separator comes out. On a de-DE host an
// unpinned label prints `9.000.000.000`, which a reader can take for nine.
test('byte labels are pinned to one locale, not the host\'s', () => {
  const { component } = componentUnderTest();
  component._applyFullJobView(jobView({ estimate_bytes: 9_000_000_000 }));
  assert.equal(component.fullJobEstimateLabel, '8583.1 MB (9,000,000,000 bytes)');
  assert.doesNotMatch(component.fullJobEstimateLabel, /9\.000\.000\.000/);
});

// CARRIER 1 of the offer: the 507 refusal detail (routes.py:676).
// UNREACHABLE FROM TEXT: whether a getter combining three conditions is true.
test('the override is offered from the refusal payload', async () => {
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
        overridable: true,
      },
    }),
  });

  await component.startFullJob();
  assert.equal(component.fullJobOverrideOffered, true);
});

// CARRIER 2 of the offer: playback-info's `full_copy.overridable`
// (routes.py:289). Deleting either write leaves one surface without the control.
test('the override offer is also read off playback-info — the second carrier', () => {
  const { component } = componentUnderTest();
  assert.equal(component.overrideOffer, false);
  component._adoptPlaybackInfo({ full_copy: { state: 'refused', overridable: true } });
  assert.equal(component.overrideOffer, true);
  // And it is not sticky: the next video decides for itself.
  component._adoptPlaybackInfo({ full_copy: { state: 'fits', overridable: false } });
  assert.equal(component.overrideOffer, false);
});

// The server refuses an override it cannot attribute (403 override-unattributed)
// and an override of a verdict that is not a §6.3 refusal. `capacity-exhausted`
// is also `queue-full`'s state (states.py:232) — a wait, not a verdict — so the
// state alone must not produce a button.
test('the override is not offered where the server would refuse it', async () => {
  const { context, component } = componentUnderTest();
  const refuse = (detail) => {
    context.fetch = async () => ({ ok: false, status: 507, json: async () => ({ detail }) });
  };

  refuse({
    error: 'full-copy-refused',
    player_state: 'capacity-exhausted',
    reason: 'x',
    overridable: false, // no SFN_EXAMINER_ID on this server
  });
  await component.startFullJob();
  assert.equal(component.fullJobRefused, true);
  assert.equal(component.fullJobOverrideOffered, false, 'unattributable override was offered');

  // queue-full: same §5 state, retryable, and nothing an override can set aside.
  refuse({ error: 'queue-full', player_state: 'capacity-exhausted', reason: 'x', overridable: true });
  await component.startFullJob();
  assert.equal(component.fullJobOverrideOffered, false, 'a wait was offered an override');
});

// The operator's 2026-08-14 narrowing, on the client half: an `unknown` verdict
// is refused with NO escape hatch, and the ruling is "refuse *and* offer Download
// original" — refusing without the hatch is a different, worse behaviour. The
// hatch rides on `fullJobRefused`: `index.html`'s `vc-warn-band` is shown by it
// and carries the Download original anchor unconditionally, beside the panel's
// permanent one. So this asserts the band appears and the override button does
// not, which is exactly the pair the narrowing changed.
test('an unknown verdict is refused with the download offer and no override', async () => {
  const { context, component } = componentUnderTest();
  context.fetch = async () => ({
    ok: false,
    status: 507,
    json: async () => ({
      detail: {
        error: 'full-copy-unknown',
        player_state: 'capacity-exhausted',
        reason: 'This container reports no duration, bitrate or height.',
        estimate_bytes: null,
        limit_bytes: 4_000_000_000,
        overridable: false,
      },
    }),
  });

  await component.startFullJob(true); // the examiner asking anyway
  assert.equal(component.fullJobRefused, true, 'the refusal band — and its Download original — is absent');
  assert.equal(component.fullJobOverrideOffered, false, 'an override was offered for `unknown`');
  // `refused` prints the estimate row and `unknown` must not: there is no number.
  assert.equal(component.fullJobRefusedTooBig, false);
});

// UNREACHABLE FROM TEXT: what URL the click actually requests. The override is
// one act on one video, so it rides on the request and is never a mode.
test('only the override click sends override=true', async () => {
  const { context, component } = componentUnderTest();
  const urls = [];
  context.fetch = async (url) => {
    urls.push(url);
    return { ok: true, json: async () => jobView() };
  };

  await component.startFullJob();
  await component.startFullJob(true);
  component._stopFullJobPoll();

  assert.doesNotMatch(urls[0], /override/);
  assert.match(urls[1], /[?&]override=true/);
});

// THE DISCLOSURE. UNREACHABLE FROM TEXT: that the field arriving on a view
// produces the server's sentence, the examiner and the verdict, and that it
// survives the job ending — §6.3 requires the record to outlive the job.
test('the override disclosure renders from a job view and survives the job ending', () => {
  const { component } = componentUnderTest();
  component._applyFullJobView(jobView({ override: overrideRecord() }));
  assert.equal(component.fullJobOverridden, true);
  assert.equal(component.fullJobOverrideExaminer, 'examiner-7');
  assert.equal(component.fullJobOverrideVerdict, 'refused');
  assert.equal(component.fullJobOverrideNotice, OVERRIDE_NOTICE);
  assert.equal(component.fullJobOverrideEstimateLabel, '8583.1 MB (9,000,000,000 bytes)');
  assert.equal(component.fullJobOverrideLimitLabel, '3814.7 MB (4,000,000,000 bytes)');

  component._applyFullJobView(jobView({
    player_state: 'full-job-done',
    contention_notice: null,
    override: overrideRecord(),
    full_url: '/api/video-full?path=x&fp=y',
  }));
  assert.equal(component.fullJobDone, true);
  assert.equal(component.fullJobOverridden, true, 'the record must outlive the job');
});

// The third carrier, and the one that makes the disclosure survive the *tab*:
// an analyst opening this video after the export finished reads who set the
// capacity gate aside. Without this, the record exists only where it was clicked.
test('a page opened after the export still shows the override — playback-info', () => {
  const { component } = componentUnderTest();
  component._adoptPlaybackInfo({
    full_copy: { state: 'refused', overridable: true },
    full_job: jobView({
      player_state: 'full-job-done',
      contention_notice: null,
      override: overrideRecord(),
      full_url: '/api/video-full?path=x&fp=y',
    }),
  });
  assert.equal(component.fullJobDone, true);
  assert.equal(component.fullJobOverridden, true);
  assert.equal(component.fullJobOverrideExaminer, 'examiner-7');
});

// A video with no job must not inherit the previous one's panel.
test('adopting a playback-info with no job clears the previous panel', () => {
  const { component } = componentUnderTest();
  component._applyFullJobView(jobView({ player_state: 'full-job-done', override: overrideRecord() }));
  component._adoptPlaybackInfo({ full_copy: { state: 'fits', overridable: false } });
  assert.equal(component.fullJob.state, 'idle');
  assert.equal(component.fullJobOverridden, false);
});

// REQUIREMENT 4 OF THE RULING. An override buys the chance to find out the
// estimate was wrong; it does not buy the right to fill the cache. The `.part`
// watch still kills the encode at the real ceiling, and that ending is a
// failure — the disclosure beside it may not make it read as a success.
test('an overridden job killed by the ceiling renders as a failure', () => {
  const { component } = componentUnderTest();
  component._applyFullJobView(jobView({
    player_state: 'full-job-failed',
    contention_notice: null,
    override: overrideRecord(),
    error: {
      error: 'full-copy-overshoot',
      player_state: 'full-job-failed',
      reason: 'The full viewing copy passed 3.7 GiB while encoding and was stopped.',
    },
  }));
  assert.equal(component.fullJobFailed, true);
  assert.equal(component.fullJobDone, false, 'a killed encode must not read as done');
  assert.equal(component.fullJobRefused, false, 'the job ran; it was not refused');
  assert.equal(component.fullJob.kind, 'full-copy-overshoot');
  // The record still stands: the gate was set aside, and that is true whether or
  // not the export then survived.
  assert.equal(component.fullJobOverridden, true);
});

// #147 again, on this surface: an overridden `unknown` set aside no number.
test('an overridden unknown verdict prints no estimate', () => {
  const { component } = componentUnderTest();
  component._applyFullJobView(jobView({
    override: overrideRecord({ verdict: 'unknown', estimate_bytes: null }),
  }));
  assert.equal(component.fullJobOverrideVerdict, 'unknown');
  assert.equal(component.fullJobOverrideEstimateLabel, '', 'there was no estimate to set aside');
  assert.equal(component.fullJobOverrideLimitLabel, '3814.7 MB (4,000,000,000 bytes)');
});

test('the override disclosure is never invented client-side, and does not stick', async () => {
  const { context, component } = componentUnderTest();
  for (const value of [null, undefined, '', 0, false, 'yes']) {
    component._setOverrideDisclosure(value);
    assert.equal(component.overrideDisclosure, null);
  }

  // A new export attempt starts with no record; only the server's answer creates
  // one, so a previous override cannot be shown against a job that had none.
  component._applyFullJobView(jobView({ override: overrideRecord() }));
  context.fetch = async () => ({ ok: true, json: async () => jobView() });
  await component.startFullJob();
  component._stopFullJobPoll();
  assert.equal(component.fullJobOverridden, false);

  // And on the path that has no job view to overwrite it: a start that is
  // refused never reaches `_applyFullJobView`, so without the reset the record
  // of the *previous* export stays on screen beside a refusal for a job that
  // does not exist — a disclosure attached to nothing.
  component._applyFullJobView(jobView({ override: overrideRecord() }));
  context.fetch = async () => ({
    ok: false,
    status: 503,
    json: async () => ({
      detail: { error: 'queue-full', player_state: 'capacity-exhausted', reason: 'busy' },
    }),
  });
  await component.startFullJob();
  assert.equal(component.fullJobRefused, true);
  assert.equal(component.fullJobOverridden, false, 'a refusal kept the previous override on screen');
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

// ── Closing the player closes the job panel ─────────────────────────────────
// UNREACHABLE FROM TEXT, and it was a live defect: `closeFullJob()` had no
// caller on the close path, so the 1 Hz status poll and the `beforeunload`
// handler both outlived the panel they belonged to. A grep can see the function
// exists; only running the close can show what it left behind.
test('closing the player stops the full-job poll and disarms the leave prompt', async () => {
  const { context, component } = componentUnderTest();
  let polls = 0;
  context.fetch = async () => {
    polls += 1;
    return { ok: true, json: async () => jobView() };
  };

  await component.startFullJob();
  assert.equal(component.fullJobRunning, true);
  assert.equal(context.window.listeners.get('beforeunload').length, 1);

  component.closeVideoPlayback();

  assert.equal(component.fullJob.state, 'idle', 'the panel kept the closed job');
  assert.equal(
    context.window.listeners.get('beforeunload').length,
    0,
    'the tab still warns about an export it no longer shows',
  );
  // The poll is a timer, so the proof is that it stops firing. Two intervals'
  // worth of wall clock with no further request.
  const after = polls;
  try {
    await new Promise((r) => setTimeout(r, 2100));
    assert.equal(polls, after, 'the status poll outlived the panel');
  } finally {
    // A timer this test is *asserting about* must not be left running: node's
    // runner does not exit while an interval is pending, so a regression here
    // would hang the suite instead of failing it.
    component.closeFullJob();
  }
});

// Closing the panel is not cancelling the job: the export is the server's and
// the DELETE is a button. The distinction matters — a close that cancelled would
// throw away minutes of encoding because an analyst folded a panel.
test('closing the player does not cancel the export', async () => {
  const { context, component } = componentUnderTest();
  const methods = [];
  context.fetch = async (url, opts) => {
    methods.push(opts?.method ?? 'GET');
    return { ok: true, json: async () => jobView() };
  };
  await component.startFullJob();
  const before = methods.length;
  component.closeVideoPlayback();
  component.closeFullJob(); // see above: never leave the poll timer behind
  assert.ok(!methods.slice(before).includes('DELETE'), 'closing the panel cancelled the job');
});

// The override disclosure belongs to the job and goes with it; the *offer*
// belongs to the video and is re-established by the next playback-info load, so
// a close must not leave a stale disclosure to be re-shown under a new video.
test('closing the player clears the §6.3 disclosure it was showing', async () => {
  const { context, component } = componentUnderTest();
  context.fetch = async () => ({
    ok: true,
    json: async () => jobView({ override: overrideRecord() }),
  });
  await component.startFullJob();
  assert.equal(component.fullJobOverridden, true);
  component.closeVideoPlayback();
  component.closeFullJob(); // see above: never leave the poll timer behind
  assert.equal(component.fullJobOverridden, false);
});
