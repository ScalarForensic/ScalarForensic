// ── The full-video job: progress, cancel, completion, auto-switch (§4.3, §5) ─
//
// One `sfn()` part file, loaded before app.js and merged with property
// descriptors — never Object.assign, which would evaluate the getters below once
// at merge time instead of copying them (CLAUDE.md).
//
// WHAT THIS FILE MAY NOT DO.  It never decides a state and it never writes a
// sentence.  `player_state` comes off the wire on every poll, the §6.3 refusal
// carries the server's own `reason`, and the §4.3 disclosure is the server's
// string rendered verbatim.  §10.1 has one implementation and it is not here.
//
// WHY POLLING AND NOT A WEBSOCKET OR SSE.  The job is minutes long and the
// interesting quantity changes once a second; a poll is one line of state and
// survives a reload, a proxy and an isolated LAN with no sticky sessions.  The
// server already models "joined an existing job" through `waiters`, so a second
// tab is a second poller and not a second job.
(window.__sfnParts = window.__sfnParts || []).push({
  // `state` starts at 'idle', not at a verdict: before anything has been asked
  // the job has no state, the same reason `chunk.state` does (§5).
  fullJob: {
    state: 'idle',
    reason: '',
    kind: '',
    fraction: null,
    etaLabel: null,
    rate: null,
    elapsedS: 0,
    outSeconds: 0,
    durationSeconds: 0,
    writtenBytes: 0,
    estimateBytes: null,
    limitBytes: null,
    waiters: 0,
    url: '',
    switched: false,
  },
  _fullJobPollTimer: null,
  _fullJobLeaveHandler: null,

  // ── Computed ─────────────────────────────────────────────────────────────
  get fullJobRunning() {
    return this.fullJob.state === 'full-job-running';
  },
  get fullJobDone() {
    return this.fullJob.state === 'full-job-done';
  },
  get fullJobFailed() {
    return this.fullJob.state === 'full-job-failed';
  },
  // The §6.3 refusal is its own thing and not a failure of a running job: the
  // job never started. `capacity-exhausted` is the state the server states.
  get fullJobRefused() {
    return this.fullJob.state === 'capacity-exhausted';
  },
  get fullJobOffered() {
    return this.chunkPlaybackOffered && this.fullJob.state === 'idle';
  },

  // Progress, and no fabricated percentage.  `fraction` is null until the
  // encoder has reported enough to derive one; §5 and #139 both forbid inventing
  // a number to fill the gap, so the bar simply has no width yet and the label
  // falls back to elapsed seconds.
  get fullJobPercent() {
    const f = this.fullJob.fraction;
    return f === null || f === undefined ? null : Math.round(f * 1000) / 10;
  },
  get fullJobProgressLabel() {
    const pct = this.fullJobPercent;
    if (pct === null) return `${this.fullJob.elapsedS.toFixed(0)} s elapsed`;
    return `${pct.toFixed(1)}% · ${this.fullJob.elapsedS.toFixed(0)} s elapsed`;
  },
  // The ETA label is the server's string (`jobs.eta_label`), which says it is an
  // extrapolation.  Rendered, never computed here: two ETA formatters would
  // disagree and the analyst would not know which one to believe.
  get fullJobEtaLabel() {
    return this.fullJob.etaLabel || '';
  },
  get fullJobRateLabel() {
    const r = this.fullJob.rate;
    return r === null || r === undefined ? '' : `${r.toFixed(2)}× realtime`;
  },

  // ── The §6.3 refusal ─────────────────────────────────────────────────────
  // Three-valued on purpose (`cache.py`): `fits` / `refused` / `unknown`, and
  // `unknown` refuses too.  "This video is too big for the cache" and "this file
  // would not say how big it is" are different sentences, and collapsing them
  // reproduces #147's "unknown displayed as mismatch" — rejected at three layers
  // already.  On `unknown` the server sends `estimate_bytes: null`, so there is
  // no number to print and printing a zero would read as measured.
  get fullJobRefusedUnknown() {
    return this.fullJobRefused && this.fullJob.kind === 'full-copy-unknown';
  },
  get fullJobRefusedTooBig() {
    return this.fullJobRefused && this.fullJob.estimateBytes !== null;
  },
  // Both figures, the way this UI already prints sizes (index.html:471): a human
  // number an analyst reads and the exact byte count they can quote.  §6.3 says
  // the estimate must be *shown*; a rounded "3 GB" alone is not a number anyone
  // can check.
  _bytesLabel(n) {
    if (n === null || n === undefined) return '';
    return `${(n / 1048576).toFixed(1)} MB (${n.toLocaleString()} bytes)`;
  },
  get fullJobEstimateLabel() {
    return this._bytesLabel(this.fullJob.estimateBytes);
  },
  get fullJobLimitLabel() {
    return this._bytesLabel(this.fullJob.limitBytes);
  },

  // ── Starting, polling, cancelling ────────────────────────────────────────
  _fullJobQuery() {
    const path = this.videoPlayback?.source_path;
    if (!path) return null;
    const indexed = this.videoPlayback?.video_sha256;
    return (
      `path=${encodeURIComponent(path)}` +
      (indexed ? `&video_hash=${encodeURIComponent(indexed)}` : '')
    );
  },

  _applyFullJobView(view) {
    this.fullJob.state = view.player_state ?? 'full-job-failed';
    this.fullJob.fraction = view.fraction ?? null;
    this.fullJob.etaLabel = view.eta_label ?? null;
    this.fullJob.rate = view.rate ?? null;
    this.fullJob.elapsedS = view.elapsed_seconds ?? 0;
    this.fullJob.outSeconds = view.out_seconds ?? 0;
    this.fullJob.durationSeconds = view.duration_seconds ?? 0;
    this.fullJob.writtenBytes = view.written_bytes ?? 0;
    this.fullJob.estimateBytes = view.estimate_bytes ?? null;
    this.fullJob.limitBytes = view.limit_bytes ?? null;
    this.fullJob.waiters = view.waiters ?? 0;
    this.fullJob.url = view.full_url ?? '';
    this.fullJob.reason = view.error?.reason ?? '';
    this.fullJob.kind = view.error?.error ?? '';
    // The other carrier of the §4.3 disclosure. Same single cell as the chunk
    // response writes (player.js `_setContentionNotice`) — one disclosure, not
    // two that can disagree.
    this._setContentionNotice(view.contention_notice);
  },

  _applyFullJobRefusal(detail, status) {
    this.fullJob.state = detail?.player_state ?? 'capacity-exhausted';
    this.fullJob.kind = detail?.error ?? '';
    const plain = typeof detail === 'string' ? detail : '';
    this.fullJob.reason =
      detail?.reason ?? (plain || `The export could not start (HTTP ${status ?? '?'}).`);
    this.fullJob.estimateBytes = detail?.estimate_bytes ?? null;
    this.fullJob.limitBytes = detail?.limit_bytes ?? null;
    this._stopFullJobPoll();
  },

  async startFullJob() {
    const q = this._fullJobQuery();
    if (!q) return;
    this.fullJob.switched = false;
    let r;
    try {
      r = await fetch(`/api/video-full?${q}`, { method: 'POST' });
    } catch (e) {
      this._applyFullJobRefusal({ reason: e?.message || 'The export request failed.' }, null);
      return;
    }
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      this._applyFullJobRefusal(body?.detail ?? null, r.status);
      return;
    }
    this._applyFullJobView(body);
    this._startFullJobPoll();
    this._armLeavePrompt();
  },

  _startFullJobPoll() {
    if (this._fullJobPollTimer) return;
    this._fullJobPollTimer = setInterval(() => this.pollFullJob(), 1000);
  },
  _stopFullJobPoll() {
    if (this._fullJobPollTimer) {
      clearInterval(this._fullJobPollTimer);
      this._fullJobPollTimer = null;
    }
    this._disarmLeavePrompt();
  },

  async pollFullJob() {
    const q = this._fullJobQuery();
    if (!q) return;
    let body;
    try {
      const r = await fetch(`/api/video-job-status?${q}`);
      body = await r.json();
    } catch {
      return; // a dropped poll is not a job failure; the next one asks again
    }
    // `state: "none"` means the runner has no job for this source — it finished
    // and was reaped, or the server restarted. Not a failure, and not a reason
    // to overwrite a terminal state we already have.
    if (body?.state === 'none') {
      this._stopFullJobPoll();
      return;
    }
    this._applyFullJobView(body);
    if (this.fullJobDone || this.fullJobFailed) this._stopFullJobPoll();
    if (this.fullJobDone) this._offerAutoSwitch();
  },

  async cancelFullJob() {
    const q = this._fullJobQuery();
    if (!q) return;
    this._stopFullJobPoll();
    let body = {};
    try {
      const r = await fetch(`/api/video-full?${q}`, { method: 'DELETE' });
      body = await r.json().catch(() => ({}));
    } catch {
      /* the job is the server's; a failed cancel leaves the poll stopped */
    }
    // The server states the state here too: a cancel with other waiters does not
    // stop the job, and it says so by returning 'full-job-running'.
    this.fullJob.state = body?.player_state ?? 'idle';
    this.fullJob.waiters = body?.waiters ?? 0;
    if (this.fullJob.state === 'full-job-running') this._startFullJobPoll();
  },

  // ── Completion: notify, then switch at the current timestamp ──────────────
  // The switch keeps the analyst's position.  Jumping to zero on completion
  // would lose the frame they were looking at, which for a long source is the
  // whole reason they waited for the export.
  _offerAutoSwitch() {
    if (this.fullJob.switched || !this.fullJob.url) return;
    this.fullJob.switched = true;
  },
  get fullJobSwitchAtSeconds() {
    return this.chunk.start === null
      ? (this.videoPlaybackTimecodeMs ?? 0) / 1000
      : this.chunk.start;
  },

  // ── The navigate-away prompt ──────────────────────────────────────────────
  // A running export is server-side work the analyst cannot see from another
  // page, and leaving abandons a job that is still holding an encode worker.
  // The predicate is a getter so it can be tested; whether the browser actually
  // shows its dialog is the browser's call and gated on user interaction, which
  // is why the live check still matters.
  get fullJobWarnsOnLeave() {
    return this.fullJobRunning;
  },
  _armLeavePrompt() {
    if (this._fullJobLeaveHandler) return;
    this._fullJobLeaveHandler = (e) => {
      if (!this.fullJobWarnsOnLeave) return undefined;
      e.preventDefault();
      e.returnValue = '';
      return '';
    };
    window.addEventListener('beforeunload', this._fullJobLeaveHandler);
  },
  _disarmLeavePrompt() {
    if (!this._fullJobLeaveHandler) return;
    window.removeEventListener('beforeunload', this._fullJobLeaveHandler);
    this._fullJobLeaveHandler = null;
  },

  closeFullJob() {
    this._stopFullJobPoll();
    this.fullJob = {
      state: 'idle', reason: '', kind: '', fraction: null, etaLabel: null, rate: null,
      elapsedS: 0, outSeconds: 0, durationSeconds: 0, writtenBytes: 0,
      estimateBytes: null, limitBytes: null, waiters: 0, url: '', switched: false,
    };
  },
});
