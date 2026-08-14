// Behaviour of `js/video_playback/rendering.js`, asserted by running it.
//
// §7.2 requires the label to record the actual pipeline: hwaccel, decoder, the
// full filter chain with parameters, encoder and rate control, output
// resolution, ffmpeg version, and any audio transformation or omission. Until
// #193 the browser could not have shown that; since #193 the payload carries all
// of it and the label showed three fields. These tests pin what the label now
// renders, and — more importantly — that it cannot quietly stop rendering a
// field the server adds.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { REPO_ROOT, loadComponent } from './harness.mjs';

// The server's own sentence, read out of the module that defines it. Asserting
// the constant and never a paraphrase: if `audit.py` rewords it, this test
// follows, and the day the browser grows its own copy the last assertion in
// this file goes red.
const AUDIT_PY = fs.readFileSync(
  path.join(REPO_ROOT, 'src', 'scalar_forensic', 'video_playback', 'audit.py'),
  'utf8',
);
function pythonConstant(name) {
  const m = AUDIT_PY.match(new RegExp(`^${name} = \\(([\\s\\S]*?)\\)$`, 'm'));
  assert.ok(m, `${name} not found in audit.py`);
  return [...m[1].matchAll(/"([^"]*)"/g)].map((x) => x[1]).join('');
}
const AUDIO_REENCODED = pythonConstant('AUDIO_REENCODED');
const AUDIO_OMITTED = pythonConstant('AUDIO_OMITTED');

// One `Rendering.describe()` payload, in the shape the server actually sends:
// `Pipeline.describe()`'s fields (tuples already joined server-side) merged with
// the scope, the audio transformation, the thread cap, the window and the argv.
function payload(overrides = {}) {
  return {
    hwaccel: 'cuda',
    decoder: 'hevc_cuvid',
    filter_chain: 'zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709:m=bt709:r=tv,format=yuv420p',
    encoder: 'h264_nvenc',
    rate_control: '-rc vbr -cq 23',
    output_height: 720,
    chunk_seconds: 30,
    audio: '-c:a aac -b:a 128k',
    ffmpeg_version: '7.1.1',
    fingerprint: 'f'.repeat(64),
    tone_mapped: true,
    scope: 'chunk',
    audio_transformation: AUDIO_REENCODED,
    threads: null,
    start_seconds: 30.0,
    duration_seconds: 30.0,
    fell_back: false,
    fallback_reason: null,
    command: ['ffmpeg', '-ss', '30', '-i', '/evidence/clip.mov', '-t', '30', 'out.mp4'],
    // The server's own `shlex.join` of the invocation (audit.py). Quoted the way
    // a real payload is quoted — the filter chain is the argument that needs it,
    // and a fixture that never needed quoting would test the easy half.
    command_line:
      "ffmpeg -ss 30 -i /evidence/clip.mov -vf 'scale=-2:min(ih,720),tonemap=hable' -t 30 out.mp4",
    ...overrides,
  };
}

function rowsByKey(rows) {
  return Object.fromEntries(rows.map((r) => [r.key, r.value]));
}

// UNREACHABLE FROM TEXT: §7.2 is a statement about which *values* reach the
// screen. A grep can show `_renderingRows` is called; only running it can show
// the decoder and the rate control come out the other side.
test('every §7.2 requirement is rendered from the payload', () => {
  const c = loadComponent();
  const v = rowsByKey(c._renderingRows(payload()));
  assert.equal(v.hwaccel, 'cuda');
  assert.equal(v.decoder, 'hevc_cuvid');
  assert.match(v.filter_chain, /tonemap=hable/);
  assert.match(v.filter_chain, /zscale=t=bt709:m=bt709:r=tv/); // parameters, not just names
  assert.equal(v.encoder, 'h264_nvenc');
  assert.equal(v.rate_control, '-rc vbr -cq 23');
  assert.equal(v.output_height, '720');
  assert.equal(v.ffmpeg_version, '7.1.1');
  assert.equal(v.audio_transformation, AUDIO_REENCODED);
});

// Every field of `Pipeline` is in the fingerprint and therefore in the label
// (capability.py). The live check found `chunk_seconds` reaching the screen as a
// bare key because it had no row name — rendered, correctly, but unlabelled.
// This pins that every pipeline field the server sends has one.
test('every pipeline field the server sends is rendered under a name', () => {
  const c = loadComponent();
  const named = c._renderingRows(payload()).filter((r) => r.label !== r.key).map((r) => r.key);
  for (const key of Object.keys(payload())) {
    if (key === 'command' || key === 'command_line' || payload()[key] === null) continue;
    assert.ok(named.includes(key), `${key} reaches the screen without a row name`);
  }
});

test('the audio omission is the server sentence, not a client one', () => {
  const c = loadComponent();
  const v = rowsByKey(c._renderingRows(payload({ audio: '-an', audio_transformation: AUDIO_OMITTED })));
  assert.equal(v.audio_transformation, AUDIO_OMITTED);
  assert.equal(v.audio, '-an');
});

// A field the server adds after this file was written must appear anyway.
// `Pipeline.describe()` derives itself from `fields()` so the label cannot fall
// behind the fingerprint (capability.py); a hand-listed renderer here would
// undo that one layer later.
test('a payload field the renderer has never heard of is rendered, not dropped', () => {
  const c = loadComponent();
  const v = rowsByKey(c._renderingRows(payload({ colour_primaries: 'bt2020' })));
  assert.equal(v.colour_primaries, 'bt2020');
  const row = c._renderingRows(payload({ colour_primaries: 'bt2020' })).find(
    (r) => r.key === 'colour_primaries',
  );
  assert.equal(row.label, 'colour_primaries', 'an unknown key falls back to its own name');
});

test('the known fields lead, in §7.2 order, and the unknown one follows', () => {
  const c = loadComponent();
  const keys = c._renderingRows(payload({ zzz_new: 'x' })).map((r) => r.key);
  assert.ok(keys.indexOf('hwaccel') < keys.indexOf('decoder'));
  assert.ok(keys.indexOf('decoder') < keys.indexOf('filter_chain'));
  assert.ok(keys.indexOf('encoder') < keys.indexOf('rate_control'));
  assert.equal(keys[keys.length - 1], 'zzz_new');
  assert.equal(new Set(keys).size, keys.length, 'no key may be rendered twice');
});

// #147's rule, in this file: an absent value is not a zero and not the word
// "null". `threads` is null for a chunk — chunks are not thread-capped — and a
// row claiming a cap would describe an encode that did not happen.
test('a null field prints no row at all', () => {
  const c = loadComponent();
  const keys = c._renderingRows(payload()).map((r) => r.key);
  assert.ok(!keys.includes('threads'));
  assert.ok(!keys.includes('fallback_reason'));
  const v = rowsByKey(c._renderingRows(payload({ threads: 4 })));
  assert.equal(v.threads, '4');
});

// `false` is a statement and must survive. A renderer that skipped falsy values
// would answer "was this tone-mapped?" with silence on every SDR rendering.
test('a false boolean renders as no, and is never skipped as falsy', () => {
  const c = loadComponent();
  const v = rowsByKey(c._renderingRows(payload({ tone_mapped: false, fell_back: false })));
  assert.equal(v.tone_mapped, 'no');
  assert.equal(v.fell_back, 'no');
});

// §8's fallback, rendered as the server recorded it rather than as a client
// gloss on a boolean.
test('a GPU fallback shows the server reason beside the flag', () => {
  const c = loadComponent();
  const v = rowsByKey(
    c._renderingRows(payload({ fell_back: true, fallback_reason: 'nvenc session limit reached' })),
  );
  assert.equal(v.fell_back, 'yes');
  assert.equal(v.fallback_reason, 'nvenc session limit reached');
});

// audit.py: `command` is None for a rendering found in the cache, because no
// process ran for that response. The label must not imply one did.
test('a cache-served rendering shows no invocation', () => {
  const c = loadComponent();
  assert.equal(c._renderingCommand(payload({ command: null, command_line: null })), '');
  assert.equal(c._renderingCommand(null), '');
  assert.match(c._renderingCommand(payload()), /^ffmpeg -ss 30 -i /);
  // ...and neither carrier is smuggled into the rows.
  const keys = c._renderingRows(payload()).map((r) => r.key);
  assert.ok(!keys.includes('command'));
  assert.ok(!keys.includes('command_line'));
});

// THE FINDING OF 2026-08-14 (`data/reports/c24-task2-label-vs-render.md`), now a
// test. The block above calls the invocation "a line to copy" and §7.2 asks for a
// rendering a reviewer can reproduce; `command.join(' ')` produced a line that
// does not survive being copied, because the real filter chain carries
// `'min(ih,1080)'` and a shell hands ffmpeg `No such filter: '1080)'`. The label
// therefore renders the server's `shlex.join`, verbatim, and never rebuilds one.
test('the invocation is the server\'s quoted line, never a client join', () => {
  const c = loadComponent();
  const p = payload();
  assert.equal(c._renderingCommand(p), p.command_line);
  // The distinguishing case: an argv that needs quoting. A client join would put
  // the array's own spacing on screen and drop the quotes the shell needs.
  assert.notEqual(c._renderingCommand(p), p.command.join(' '));
  // A payload carrying only the argv is a server that predates `command_line`.
  // Nothing is printed rather than a line that does not run — the same rule as
  // the cache hit: no invocation is honest, a broken one is not.
  assert.equal(c._renderingCommand({ ...p, command_line: undefined }), '');
});

test('no payload, no rows', () => {
  const c = loadComponent();
  // `equal(rows.length, 0)` and not `deepEqual(rows, [])`: the array is built
  // inside the vm context, so it is not reference-equal to this file's Array.
  assert.equal(c._renderingRows(null).length, 0);
  assert.equal(c._renderingRows(undefined).length, 0);
  assert.equal(c._renderingRows('not an object').length, 0);
});

// ── The two panels that use it ───────────────────────────────────────────────

test('the chunk label renders the record from the chunk response payload', () => {
  const c = loadComponent();
  c.chunk.pipeline = payload();
  const v = rowsByKey(c.chunkRenderingRows);
  assert.equal(v.decoder, 'hevc_cuvid');
  assert.equal(v.audio_transformation, AUDIO_REENCODED);
  assert.match(c.chunkRenderingCommand, /^ffmpeg /);
  // The one-line summary stays a summary of the same payload, so the two can
  // never contradict each other.
  assert.equal(c.chunkPipelineLabel, 'h264_nvenc · cuda · tone-mapped to BT.709');
});

test('the full copy carries its own record, from every job view', () => {
  const c = loadComponent();
  c._applyFullJobView({
    player_state: 'full-job-done',
    full_url: '/api/video-full?path=x&fp=y',
    rendering: payload({ scope: 'full', threads: 4, start_seconds: null, duration_seconds: 61.5 }),
  });
  const v = rowsByKey(c.fullJobRenderingRows);
  assert.equal(v.scope, 'full');
  assert.equal(v.threads, '4', '§4.3 caps the full copy, and libx264 output depends on it');
  assert.equal(v.duration_seconds, '61.5');
  assert.match(c.fullJobRenderingCommand, /^ffmpeg /);
});

test('a running job has no record yet, and closing the panel takes it away', () => {
  const c = loadComponent();
  c._applyFullJobView({ player_state: 'full-job-running', rendering: null });
  assert.equal(c.fullJobRenderingRows.length, 0);
  assert.equal(c.fullJobRenderingCommand, '');

  c._applyFullJobView({ player_state: 'full-job-done', rendering: payload({ scope: 'full' }) });
  assert.equal(c.fullJobRenderingRows.length > 0, true);
  c.closeFullJob();
  assert.equal(c.fullJobRenderingRows.length, 0, 'a closed panel may not keep the last job record');
  // The cell has to come back *declared*, not merely absent. A reset that drops
  // the key leaves the rows getter right by accident — `undefined` renders as
  // nothing too — while the state object no longer matches the one the part
  // file declares, which is the shape every other cell here is written to keep.
  // This was a surviving mutation (M5) before it was a test.
  assert.ok('rendering' in c.fullJob, 'closeFullJob must restore the declared shape');
  assert.equal(c.fullJob.rendering, null);
});

// The rule from `contention_notice` and `OVERRIDE_NOTICE`, applied to §7.2's
// sentences: the browser renders the field and never a second copy of the
// wording. A copy in the static assets would drift the day audit.py is reworded
// and an examiner would have two versions of one disclosure.
test('neither audio sentence is copied into the frontend', () => {
  const staticDir = path.join(REPO_ROOT, 'src', 'scalar_forensic', 'web', 'static');
  const files = fs
    .readdirSync(staticDir, { recursive: true })
    .filter((f) => /\.(js|html|css)$/.test(f))
    .map((f) => fs.readFileSync(path.join(staticDir, f), 'utf8'));
  for (const sentence of [AUDIO_REENCODED, AUDIO_OMITTED]) {
    for (const source of files) {
      assert.ok(!source.includes(sentence), 'the sentence has one definition, in audit.py');
    }
  }
});
