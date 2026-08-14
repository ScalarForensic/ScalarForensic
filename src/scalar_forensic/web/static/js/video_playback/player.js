// ── Chunk playback: the double-buffered player (spec §4.2, §5, §10.1) ───────
//
// One `sfn()` part file, loaded before app.js like every other part and merged
// with property descriptors — never Object.assign, which would evaluate the
// getters below once at merge time instead of copying them (CLAUDE.md).
//
// WHY TWO <video> ELEMENTS AND NOT MEDIA SOURCE EXTENSIONS.  Each chunk is an
// ordinary independent MP4.  While one plays, the next loads hidden in the
// other; at the boundary they swap.  MSE was reviewed and rejected in §4.1 —
// independently encoded fragments do not share an initialisation segment, and
// making them do so is the unproven assumption v1 was built on.  The cost is a
// brief visible glitch at each boundary, which the operator accepted
// explicitly.  Do not "simplify" this into MSE; §4.1 records what evidence
// would be needed to revisit it, and "this is fiddly" is not it.
//
// WHAT THIS FILE MAY NOT DO.  It never decides a state.  Every §5 state comes
// off the wire — `player_state` on a success, `detail.player_state` on a
// failure — so the failure matrix (§10.1) has exactly one implementation, on
// the server.  A client that inferred "probably needs a transcode" from a
// status code would be the second implementation, and the two would drift.
(window.__sfnParts = window.__sfnParts || []).push({
  // ── State ────────────────────────────────────────────────────────────────
  // `state` starts at 'idle' and not at 'chunk-failed' or 'chunk-ready':
  // before anything has been asked, the player is not entitled to a verdict.
  // Same reason `videoPlayback.player_state` carries 'unknown' as a value
  // distinct from 'needs-transcode' (§5).
  chunk: {
    state: 'idle',
    reason: '',
    kind: '',
    retryable: false,
    retryAfterS: null,
    start: null,
    next: null,
    url: '',
    pipeline: null,
    fellBack: false,
    elapsedS: 0,
    buffer: 0,          // which of the two <video> elements is on screen
    preload: { start: null, url: '' },
    prefetchFailed: false,
  },
  // ── The §4.3 contention disclosure ───────────────────────────────────────
  // THIS IS THE REMEDY, NOT A LABEL.  Yielding was measured at ~9% (median
  // 18.31 s unniced vs 16.83 s niced, n=6, CPU/libx264) and *neither* arm lands
  // inside §4.2's 6–10 s window, so the slowdown an analyst sees while an export
  // runs is not fixed — it is disclosed.  This sentence is the only thing that
  // explains it, on both pipelines.
  //
  // It lives here and not inside `chunk` because it is a fact about the host,
  // not about this chunk: `closeChunkPlayback()`'s reset would clear something
  // still true, and `full_job.js` would have to reach into the chunk object to
  // write it.
  contentionNotice: '',
  _chunkLeaseTimer: null,
  _chunkElapsedTimer: null,
  _chunkRetryTimer: null,
  _chunkRequestId: 0,

  // ── Computed ─────────────────────────────────────────────────────────────
  get chunkPlaybackOffered() {
    return this.videoPlayback?.player_state === 'needs-transcode';
  },
  get chunkSeconds() {
    return this.videoPlayback?.chunk_seconds ?? 30;
  },
  get chunkIsEncoding() {
    return this.chunk.state === 'chunk-encoding';
  },
  get chunkIsPlaying() {
    return this.chunk.state === 'chunk-ready';
  },
  get chunkHasFailed() {
    return ['chunk-failed', 'capacity-exhausted', 'cache-disabled', 'unknown'].includes(
      this.chunk.state
    );
  },
  // A retry is offered only when the server said one could help AND its
  // Retry-After has elapsed.  §10.1: "nothing may retry-storm" — the countdown
  // is the whole mechanism, so it is a disabled button and never a timer that
  // fires a request on its own.
  get chunkRetryOffered() {
    return this.chunkHasFailed && this.chunk.retryable && this.chunk.retryAfterS === 0;
  },
  get chunkRetryCountdown() {
    return this.chunk.retryable && this.chunk.retryAfterS > 0 ? this.chunk.retryAfterS : 0;
  },
  // Elapsed seconds, never a percentage.  §5: "spinner with elapsed time; no
  // fabricated percentage" — there is no progress signal to derive one from,
  // and #139 removed the last invented one from this codebase.
  get chunkElapsedLabel() {
    return `${this.chunk.elapsedS.toFixed(0)} s elapsed`;
  },
  // The one-line summary. It is a *subset* of the §7.2 record below it and not
  // a substitute for it: the full record is `chunkRenderingRows`, rendered
  // open beneath this line. Both are built from the same `b.pipeline` payload,
  // so the summary cannot say something the record contradicts.
  get chunkPipelineLabel() {
    const p = this.chunk.pipeline;
    if (!p) return '';
    const bits = [p.encoder, p.hwaccel === 'none' ? 'software' : p.hwaccel];
    if (p.tone_mapped) bits.push('tone-mapped to BT.709');
    return bits.filter(Boolean).join(' · ');
  },
  // §7.2 in full, from `audit.Rendering.describe()` — the decoder, the filter
  // chain with its parameters, the rate control, the output height, the ffmpeg
  // build and what became of the audio. Shared with the full copy's panel
  // (`rendering.js`), because one requirement gets one renderer.
  get chunkRenderingRows() {
    return this._renderingRows(this.chunk.pipeline);
  },
  get chunkRenderingCommand() {
    return this._renderingCommand(this.chunk.pipeline);
  },
  get chunkWindowLabel() {
    if (this.chunk.start === null) return '';
    const end = this.chunk.start + this.chunkSeconds;
    return `${this.chunk.start.toFixed(0)}–${end.toFixed(0)} s of the source`;
  },

  // The disclosure's only writer.  The server sends `contention_notice` on two
  // payloads on purpose — the chunk response (`routes.py:480`) and the job status
  // view (`jobs.py:270`) — because the analyst meets the slowdown at the chunk
  // spinner, not in an export panel.  Both land in this one cell, so the two
  // carriers are one disclosure that cannot disagree rather than two that can.
  // Do not delete either write as redundant, and do not hard-code the sentence:
  // it has exactly one definition, server-side at `jobs.py:135`, for the same
  // reason `_check_allowed_path` has exactly one — a disclosure an examiner
  // defends may not have a second wording.
  _setContentionNotice(value) {
    this.contentionNotice = typeof value === 'string' && value ? value : '';
  },

  // ── Requesting a chunk ───────────────────────────────────────────────────
  async _requestChunk(seconds) {
    const path = this.videoPlayback?.source_path;
    if (!path) return { ok: false, detail: null };
    const indexed = this.videoPlayback?.video_sha256;
    const q =
      `path=${encodeURIComponent(path)}&t=${encodeURIComponent(seconds)}` +
      (indexed ? `&video_hash=${encodeURIComponent(indexed)}` : '');
    const r = await fetch(`/api/video-chunk?${q}`, { method: 'POST' });
    const body = await r.json().catch(() => ({}));
    return { ok: r.ok, body, detail: body?.detail ?? null, status: r.status };
  },

  // Every failure path lands here, so there is one place that maps a response
  // onto a state — and it copies the server's state rather than choosing one.
  _applyChunkFailure(result) {
    const d = result?.detail;
    this.chunk.state = d?.player_state ?? 'chunk-failed';
    this.chunk.kind = d?.error ?? '';
    // FastAPI's own HTTPExceptions carry a plain-string detail; ours carry the
    // §10.1 row. Both have to render, and neither may fall through to blank.
    const plain = typeof result?.detail === 'string' ? result.detail : '';
    this.chunk.reason =
      d?.reason ?? (plain || `The chunk request failed (HTTP ${result?.status ?? '?'}).`);
    this.chunk.retryable = Boolean(d?.retryable);
    this.chunk.retryAfterS = d?.retry_after_seconds ?? null;
    // Deliberately does not touch `contentionNotice`.  A failed chunk request
    // says nothing about whether an export is still running, and blanking the
    // disclosure on an unrelated failure would drop it at the moment it matters
    // most — a slow chunk that then failed is exactly when the analyst needs to
    // know an export is competing for the encoders.
    this._stopChunkElapsed();
    this._startChunkRetryCountdown();
  },

  async playChunkAt(seconds) {
    if (!this.chunkPlaybackOffered) return;
    const id = ++this._chunkRequestId;
    this.chunk.state = 'chunk-encoding';
    this.chunk.reason = '';
    this.chunk.kind = '';
    this._startChunkElapsed();
    let result;
    try {
      result = await this._requestChunk(seconds);
    } catch (e) {
      result = { ok: false, detail: { reason: e?.message || 'The chunk request failed.' } };
    }
    // A seek that overtook this request owns the player now.  Landing a stale
    // answer would show the analyst a chunk they navigated away from.
    if (id !== this._chunkRequestId) return;
    if (!result.ok) {
      this._applyChunkFailure(result);
      return;
    }
    this._stopChunkElapsed();
    const b = result.body;
    this.chunk.state = b.player_state ?? 'chunk-ready';
    this.chunk.start = b.chunk_start;
    this.chunk.next = b.next_chunk_start;
    this.chunk.url = b.chunk_url;
    this.chunk.pipeline = b.pipeline ?? null;
    this.chunk.fellBack = Boolean(b.fell_back);
    this.chunk.preload = { start: null, url: '' };
    this.chunk.prefetchFailed = false;
    this._setContentionNotice(b.contention_notice);
    this._startChunkLease();
    // §4.2: the next chunk is queued the moment this one is ready — not when
    // playback reaches the boundary — so the ~8 s encode is spent while the
    // analyst is watching rather than while they are waiting.
    this._prefetchNextChunk();
  },

  // Prefetch depth is exactly one (§4.2).  A prefetch that fails must not
  // change what the analyst sees: it is speculative work, and reporting it
  // would put a failure on screen for a request nobody made.  The boundary
  // handler asks again as a real request, and that one is allowed to fail
  // loudly.
  async _prefetchNextChunk() {
    const target = this.chunk.next;
    if (target === null || target === undefined) return;
    try {
      const result = await this._requestChunk(target);
      if (!result.ok) {
        this.chunk.prefetchFailed = true;
        return;
      }
      if (this.chunk.next !== target) return; // a seek moved us
      this.chunk.preload = { start: target, url: result.body.chunk_url };
      this.chunk.prefetchFailed = false;
    } catch {
      this.chunk.prefetchFailed = true;
    }
  },

  // ── The boundary swap ────────────────────────────────────────────────────
  // Called from the playing element's @ended.  A prefetch made useless by a
  // seek is left to finish rather than cancelled (§4.2) — it is one small job,
  // it is cached if the analyst comes back, and cancelling speculative work is
  // the ownership problem that made v1's scheduler complex.
  async advanceToNextChunk(playingEl, hiddenEl) {
    const target = this.chunk.next;
    if (target === null || target === undefined) return; // final chunk: stop
    if (this.chunk.preload.start === target && this.chunk.preload.url) {
      this.chunk.buffer = this.chunk.buffer === 0 ? 1 : 0;
      this.chunk.start = target;
      this.chunk.next = target + this.chunkSeconds < this.videoDurationSeconds
        ? target + this.chunkSeconds
        : null;
      this.chunk.url = this.chunk.preload.url;
      this.chunk.preload = { start: null, url: '' };
      if (hiddenEl) { try { await hiddenEl.play(); } catch { /* autoplay refused */ } }
      if (playingEl) { try { playingEl.pause(); } catch { /* already stopped */ } }
      this._prefetchNextChunk();
      return;
    }
    // The prefetch has not landed (slow encode, or it failed).  Ask for it as a
    // real request, which is allowed to show a real failure.
    await this.playChunkAt(target);
  },

  get videoDurationSeconds() {
    return (this.videoPlayback?.duration_ms ?? 0) / 1000;
  },

  // ── Seeking ──────────────────────────────────────────────────────────────
  // Inside the loaded chunk this is an ordinary currentTime move against a
  // faststart MP4.  Outside it, it is a new chunk at the new position — which
  // is what gives random access before any full copy exists (§4.2).
  async seekChunkTo(seconds, el) {
    if (!Number.isFinite(seconds) || seconds < 0) return;
    const start = this.chunk.start;
    if (start !== null && seconds >= start && seconds < start + this.chunkSeconds) {
      if (el) { try { el.currentTime = seconds - start; } catch { /* not seekable */ } }
      return;
    }
    await this.playChunkAt(seconds);
  },

  // ── The playback lease (§6.2) ────────────────────────────────────────────
  // HTTP is stateless and a FileResponse streams after its handler returns, so
  // between two chunk requests nothing tells eviction this video is on screen.
  // The heartbeat is that signal.  A quarter of the ttl means three beats can
  // be lost before the lease lapses.
  _startChunkLease() {
    if (this._chunkLeaseTimer) return;
    const ttl = this.videoPlayback?.lease_seconds ?? 120;
    const every = Math.max(5, Math.floor(ttl / 4)) * 1000;
    this._chunkLeaseTimer = setInterval(() => this._beatChunkLease(false), every);
    this._beatChunkLease(false);
  },
  _beatChunkLease(release) {
    const path = this.videoPlayback?.source_path;
    if (!path) return;
    const q = `path=${encodeURIComponent(path)}${release ? '&release=true' : ''}`;
    // keepalive so the release still leaves the page on a close or a reload.
    fetch(`/api/video-lease?${q}`, { method: 'POST', keepalive: true }).catch(() => {});
  },
  _stopChunkLease() {
    if (this._chunkLeaseTimer) { clearInterval(this._chunkLeaseTimer); this._chunkLeaseTimer = null; }
    // Explicit release rather than waiting out the ttl: a closed player should
    // stop protecting a video from eviction now, not in two minutes.
    this._beatChunkLease(true);
  },

  // ── Elapsed-time ticker and the retry countdown ──────────────────────────
  _startChunkElapsed() {
    this.chunk.elapsedS = 0;
    if (this._chunkElapsedTimer) clearInterval(this._chunkElapsedTimer);
    this._chunkElapsedTimer = setInterval(() => { this.chunk.elapsedS += 1; }, 1000);
  },
  _stopChunkElapsed() {
    if (this._chunkElapsedTimer) { clearInterval(this._chunkElapsedTimer); this._chunkElapsedTimer = null; }
  },
  _startChunkRetryCountdown() {
    if (this._chunkRetryTimer) { clearInterval(this._chunkRetryTimer); this._chunkRetryTimer = null; }
    if (!this.chunk.retryable || !this.chunk.retryAfterS) return;
    this._chunkRetryTimer = setInterval(() => {
      this.chunk.retryAfterS = Math.max(0, this.chunk.retryAfterS - 1);
      if (this.chunk.retryAfterS === 0) {
        clearInterval(this._chunkRetryTimer);
        this._chunkRetryTimer = null;
      }
      // Deliberately does not re-request: the analyst clicks Retry.  An
      // automatic retry here is the retry-storm §10.1 forbids.
    }, 1000);
  },

  retryChunk() {
    if (!this.chunkRetryOffered) return;
    this.playChunkAt(this.chunk.start ?? this.videoPlaybackTimecodeMs / 1000 ?? 0);
  },

  // ── Teardown ─────────────────────────────────────────────────────────────
  closeChunkPlayback() {
    this._chunkRequestId += 1; // orphan any in-flight response
    this._stopChunkElapsed();
    if (this._chunkRetryTimer) { clearInterval(this._chunkRetryTimer); this._chunkRetryTimer = null; }
    this._stopChunkLease();
    // Cleared on close, unlike on failure: nothing is on screen to explain, and
    // a stale sentence would outlive the export that justified it.  Both
    // carriers re-establish it — the next chunk response, or the next job poll.
    this.contentionNotice = '';
    this.chunk = {
      state: 'idle', reason: '', kind: '', retryable: false, retryAfterS: null,
      start: null, next: null, url: '', pipeline: null, fellBack: false,
      elapsedS: 0, buffer: 0, preload: { start: null, url: '' }, prefetchFailed: false,
    };
  },
});
