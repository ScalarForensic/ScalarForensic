// ── The §7.2 rendering record, for both scopes ───────────────────────────────
//
// One `sfn()` part file, loaded before app.js like every other part and merged
// with property descriptors — never Object.assign, which would evaluate the
// getters below once at merge time instead of copying them (CLAUDE.md).
//
// WHAT §7.2 ASKS FOR.  "The label records the actual pipeline: hwaccel used,
// decoder, full filter chain with parameters, encoder and rate control, output
// resolution, ffmpeg version, and any audio transformation or omission."  All of
// it is on the wire already — `audit.Rendering.describe()` (#193) merges
// `Pipeline.describe()` with the scope, the audio transformation, the thread
// cap, the window and the argv that ran — and until this file existed the
// browser rendered three of those fields.
//
// WHY IT IS ONE FILE AND NOT A GETTER IN EACH PANEL.  A chunk and a full copy
// are the same kind of artifact with the same payload shape, and §7.2 is one
// requirement.  Two renderers would be two labels for one record, free to
// disagree about what an examiner is shown — the defect class this codebase
// already refuses for `CONTENTION_NOTICE` and `OVERRIDE_NOTICE`.
//
// WHY THE ROW LIST IS NOT HAND-WRITTEN.  `Pipeline.describe()` derives itself
// from `fields()` precisely so that a field added to the pipeline cannot go
// missing from the label (capability.py).  A hand-listed renderer here would
// undo that server-side care one layer later: the new field would change the
// fingerprint and the cache key while the label kept describing the old
// pipeline.  So `ORDER` is a *display order* and not a filter — every key of
// the payload is rendered, and one this file has never heard of is appended
// under its own name rather than dropped.
//
// WHAT IT MAY NOT DO.  It never writes a sentence the server already has a
// word for.  `audio_transformation` and `fallback_reason` are rendered
// verbatim; the row names are field names, not disclosures.  `command` is
// `null` for a rendering served from the cache — no process ran for that
// response — so its block is absent rather than reconstructed, which is the
// whole reason the server sends `null` instead of a plausible argv.
(window.__sfnParts = window.__sfnParts || []).push({
  // Display order, §7.2's own order where it states one. Keys absent from a
  // given payload are skipped; keys absent from *this list* are appended (see
  // `_renderingRows`), so the list is never load-bearing for completeness.
  _renderingOrder: [
    'scope',
    'hwaccel',
    'decoder',
    'filter_chain',
    'tone_mapped',
    'encoder',
    'rate_control',
    'output_height',
    'chunk_seconds',
    'audio',
    'audio_transformation',
    'threads',
    'start_seconds',
    'duration_seconds',
    'ffmpeg_version',
    'fell_back',
    'fallback_reason',
    'fingerprint',
  ],
  // Row names. Deliberately the *name of the field*, with its unit where the
  // value carries none — never a sentence about what the field means. A gloss
  // here would be a second wording of something the server already states, and
  // an unlabelled number in an evidence viewer is worse than a raw key.
  _renderingLabels: {
    scope: 'Scope',
    hwaccel: 'Hardware acceleration',
    decoder: 'Decoder',
    filter_chain: 'Filter chain',
    tone_mapped: 'Tone-mapped to BT.709',
    encoder: 'Encoder',
    rate_control: 'Rate control',
    output_height: 'Output height (px)',
    chunk_seconds: 'Chunk length (s)',
    audio: 'Audio arguments',
    audio_transformation: 'Audio',
    threads: 'Thread cap',
    start_seconds: 'Window start (s)',
    duration_seconds: 'Window length (s)',
    ffmpeg_version: 'ffmpeg',
    fell_back: 'Fell back to the CPU',
    fallback_reason: 'Fallback reason',
    fingerprint: 'Pipeline fingerprint',
  },

  // `null` is "the server did not state one", which is not the same as a value
  // and must not print as `null` or as an invented zero (#147). A boolean gets
  // yes/no because `false` is a statement and dropping it would leave "was this
  // tone-mapped?" unanswered on every SDR rendering.
  _renderingValue(value) {
    if (value === null || value === undefined || value === '') return null;
    if (typeof value === 'boolean') return value ? 'yes' : 'no';
    return String(value);
  },

  // The rows an examiner reads, in order, from one `Rendering.describe()`.
  // `command` is excluded here only because it is rendered as its own block —
  // an argv is a line to copy, not a table cell.
  _renderingRows(payload) {
    if (!payload || typeof payload !== 'object') return [];
    const rows = [];
    const seen = new Set(['command', 'command_line']);
    const push = (key) => {
      seen.add(key);
      const value = this._renderingValue(payload[key]);
      if (value === null) return;
      rows.push({ key, label: this._renderingLabels[key] ?? key, value });
    };
    for (const key of this._renderingOrder) {
      if (key in payload) push(key);
    }
    // Anything the server added since this file was written. Appended rather
    // than dropped: a label that silently omits a pipeline field is §7.2's
    // failure mode, and it must not be reachable by adding a field.
    for (const key of Object.keys(payload)) {
      if (!seen.has(key)) push(key);
    }
    return rows;
  },

  // The invocation that produced the bytes, as the server recorded it. Absent
  // for a cache hit, and absent is the honest state: `sfn-video render` is
  // where an invocation is retrieved from the log or rebuilt as a clearly
  // labelled reproduction recipe (audit.py).
  //
  // `command_line`, never `command.join(' ')`. §7.2 asks for an invocation a
  // reviewer can reproduce and the block above calls it "a line to copy"; joining
  // the argv on spaces produced a line that does not survive being copied,
  // because the filter chain carries `'min(ih,1080)'` and a shell hands ffmpeg
  // `No such filter: '1080)'` (measured 2026-08-14 against a real encode,
  // `data/reports/c24-task2-label-vs-render.md`). The quoting has one definition,
  // `shlex.join` in `audit.py`, for the same reason the contention sentence has
  // one: an argv printed two ways is one record with two readings, and only one
  // of them runs.
  _renderingCommand(payload) {
    const line = payload?.command_line;
    return typeof line === 'string' ? line : '';
  },
});
