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
    // §7.2's record of the copy this job produced (`jobs.py:321`). `null` while
    // the job runs: there is no encode to describe yet, and a label naming a
    // pipeline before one has produced anything is a claim, not a record.
    rendering: null,
  },
  // ── The §6.3 override (ruling 2026-08-14, #184) ──────────────────────────
  // Two cells, each with exactly one writer, for the same reason
  // `contentionNotice` has one (player.js): the server sends each of these on
  // more than one payload, and two carriers writing two cells is two answers
  // that can disagree.
  //
  // `overrideOffer` — whether the server would honour an override for *this
  // video*. It is a fact about the video and not about a job, so it does not
  // live inside `fullJob`. Carriers: `playback-info.full_copy.overridable`
  // (routes.py:289) and the 507 refusal detail's `overridable`
  // (routes.py:676). Both are already `verdict.overridable AND an examiner id`
  // — the server refuses an override it cannot attribute (403
  // `override-unattributed`), so a button offered without this would produce a
  // 403 and that is worse than no button.
  overrideOffer: false,
  // `overrideDisclosure` — the record of an override that was performed:
  // `{examiner_id, verdict, estimate_bytes, limit_bytes, notice}`, or null.
  // The server puts it on *every* job view, during the job and after it, so an
  // analyst opening the page once the export has finished still sees that a
  // capacity gate was set aside and by whom. Carriers: the POST response, every
  // status poll, and `playback-info.full_job` — all three land in
  // `_applyFullJobView`, which is why there is one writer and not three.
  overrideDisclosure: null,
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
  // §7.2 for the full copy, through the one renderer both scopes share
  // (`rendering.js`). Empty until the encode has produced something to
  // describe, which is what keeps the panel from labelling a job as though it
  // were an artifact.
  get fullJobRenderingRows() {
    return this._renderingRows(this.fullJob.rendering);
  },
  get fullJobRenderingCommand() {
    return this._renderingCommand(this.fullJob.rendering);
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
  //
  // The locale is pinned rather than the host's.  A bare `toLocaleString()`
  // prints `9.000.000.000` on a de-DE box and `9,000,000,000` on an en-US one,
  // so the same evidence screenshotted on two workstations carries two
  // different-looking numbers — and a reader who reads the first as a decimal
  // point reads it as nine. What an examiner exhibits may not depend on which
  // machine rendered it.
  _bytesLabel(n) {
    if (n === null || n === undefined) return '';
    return `${(n / 1048576).toFixed(1)} MB (${n.toLocaleString('en-US')} bytes)`;
  },
  get fullJobEstimateLabel() {
    return this._bytesLabel(this.fullJob.estimateBytes);
  },
  get fullJobLimitLabel() {
    return this._bytesLabel(this.fullJob.limitBytes);
  },

  // ── The §6.3 override: the offer, and the disclosure ─────────────────────
  // The control is offered for the two refusals an override can actually set
  // aside, named explicitly rather than inferred from `capacity-exhausted`:
  // `queue-full` is that state too (states.py:232) and is a wait, not a
  // verdict — an override would neither help it nor be honoured. Naming the
  // kinds keeps a later refusal in that state from silently acquiring a button.
  get fullJobOverrideOffered() {
    return (
      this.fullJobRefused &&
      ['full-copy-refused', 'full-copy-unknown'].includes(this.fullJob.kind) &&
      this.overrideOffer
    );
  },
  // The disclosure outlives the job (§6.3, and jobs.py:141): true while the
  // export runs, after it finishes, and beside the copy it produced.
  get fullJobOverridden() {
    return this.overrideDisclosure !== null;
  },
  // The server's sentence, rendered — never a copy. One definition, at
  // `jobs.py:148`, for the same reason CONTENTION_NOTICE has one.
  get fullJobOverrideNotice() {
    return this.overrideDisclosure?.notice ?? '';
  },
  get fullJobOverrideExaminer() {
    return this.overrideDisclosure?.examiner_id ?? '';
  },
  // The verdict token as the server recorded it (`refused` / `unknown`), not a
  // sentence about it: this is the value an examiner defending the override
  // will find in the log line at `routes.py:693`, and it has to match.
  get fullJobOverrideVerdict() {
    return this.overrideDisclosure?.verdict ?? '';
  },
  // Three-valued here too. An overridden `unknown` set aside no number, so
  // `estimate_bytes` is null and nothing is printed — printing a zero would
  // claim a measurement that was never made (#147).
  get fullJobOverrideEstimateLabel() {
    return this._bytesLabel(this.overrideDisclosure?.estimate_bytes);
  },
  get fullJobOverrideLimitLabel() {
    return this._bytesLabel(this.overrideDisclosure?.limit_bytes);
  },

  // The two writers.  Do not inline either into its callers: that is how the
  // carriers stop agreeing.
  _setOverrideOffer(value) {
    this.overrideOffer = value === true;
  },
  _setOverrideDisclosure(value) {
    this.overrideDisclosure = value && typeof value === 'object' ? value : null;
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
    // §7.2: the full copy is a rendering too, and the analyst who watches it
    // is entitled to the same record as the one who watches a chunk. Carried
    // by every job view — the POST response, each poll and playback-info's
    // `full_job` — so a page opened after the export finished still shows what
    // produced the copy it is offering to play.
    this.fullJob.rendering = view.rendering ?? null;
    this.fullJob.reason = view.error?.reason ?? '';
    this.fullJob.kind = view.error?.error ?? '';
    // The other carrier of the §4.3 disclosure. Same single cell as the chunk
    // response writes (player.js `_setContentionNotice`) — one disclosure, not
    // two that can disagree.
    this._setContentionNotice(view.contention_notice);
    // The §6.3 disclosure, from whichever of the three carriers produced this
    // view — the POST response, a status poll, or playback-info's `full_job`.
    // Deleting this write leaves the override invisible on every surface.
    this._setOverrideDisclosure(view.override);
  },

  _applyFullJobRefusal(detail, status) {
    this.fullJob.state = detail?.player_state ?? 'capacity-exhausted';
    this.fullJob.kind = detail?.error ?? '';
    const plain = typeof detail === 'string' ? detail : '';
    this.fullJob.reason =
      detail?.reason ?? (plain || `The export could not start (HTTP ${status ?? '?'}).`);
    this.fullJob.estimateBytes = detail?.estimate_bytes ?? null;
    this.fullJob.limitBytes = detail?.limit_bytes ?? null;
    // The refusal's own carrier of the offer (routes.py:676). The second one is
    // playback-info; both land in the one cell, so a page that was opened before
    // the refusal and one that was reloaded after it show the same control.
    this._setOverrideOffer(detail?.overridable);
    this._stopFullJobPoll();
  },

  // `override` is an argument and never a mode: it is one deliberate act on one
  // video by one examiner, and the server records it that way (§6.3). A sticky
  // flag would carry the last click onto the next video.
  async startFullJob(override = false) {
    const q = this._fullJobQuery();
    if (!q) return;
    this.fullJob.switched = false;
    // A new attempt has not been overridden until the server says it was. The
    // disclosure is re-established from the response, so a stale one cannot
    // survive onto a job that carries no override.
    this._setOverrideDisclosure(null);
    let r;
    try {
      r = await fetch(`/api/video-full?${q}${override ? '&override=true' : ''}`, { method: 'POST' });
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

  // ── What a freshly opened panel already knows ────────────────────────────
  // §9: playback-info carries `full_job` so a reloaded page rejoins the running
  // export instead of offering to start a second one — and, for §6.3, so the
  // override disclosure outlives the job that carried it. An analyst who opens
  // this video an hour after the export finished still reads who set the
  // capacity gate aside; a disclosure that existed only in the tab that clicked
  // is not disclosure.
  //
  // Called on every playback-info load, including when there is no job: that is
  // what clears the previous video's panel rather than showing its state under
  // this video's name.
  _adoptPlaybackInfo(info) {
    this._setOverrideOffer(info?.full_copy?.overridable);
    const view = info?.full_job;
    if (!view) {
      this.closeFullJob();
      return;
    }
    this._applyFullJobView(view);
    if (this.fullJobRunning) {
      this._startFullJobPoll();
      this._armLeavePrompt();
    }
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
    // The disclosure belongs to the job and goes with it; `overrideOffer` does
    // not — it is a property of the video, re-established by whichever
    // playback-info load put this panel on screen.
    this._setOverrideDisclosure(null);
    this.fullJob = {
      state: 'idle', reason: '', kind: '', fraction: null, etaLabel: null, rate: null,
      elapsedS: 0, outSeconds: 0, durationSeconds: 0, writtenBytes: 0,
      estimateBytes: null, limitBytes: null, waiters: 0, url: '', switched: false,
      rendering: null,
    };
  },
});
