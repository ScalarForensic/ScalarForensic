// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
(window.__sfnParts = window.__sfnParts || []).push({
    // ── Analysis ──────────────────────────────────────────────────────────
    async startAnalysis() {
      if (!this.pendingFiles.length || !this.selectedModes.length) return;
      this.analysisRunModes = [...this.selectedModes];
      this.phase = 'analyzing';
      this.progress = { current: 0, total: this.pendingFiles.length, filename: '', frameCount: 0 };

      const fd = new FormData();
      for (const f of this.pendingFiles) fd.append('files', f, f.webkitRelativePath || f.name);
      fd.append('modes', this.selectedModes.join(','));

      // Release file objects — server now holds temp copies
      this.pendingFiles = [];

      const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let evt;
          try { evt = JSON.parse(line.slice(6)); } catch { continue; }
          if (['progress','file_done','error'].includes(evt.type))
            this.progress = { current: evt.current, total: evt.total, filename: evt.filename, frameCount: 0 };
          if (evt.type === 'video_progress')
            this.progress = { ...this.progress, frameCount: evt.current };
          if (evt.type === 'done') {
            this.sessionId = evt.session_id;
            await this.runQuery();
            this.phase = 'results';
            const first = this.matchedFiles[0] ?? this.results[0];
            if (first) this.selectFile(first.file_id);
          }
        }
      }
    },

    // ── Query ─────────────────────────────────────────────────────────────
    async runQuery() {
      if (!this.sessionId) return;
      const fd = new FormData();
      fd.append('session_id', this.sessionId);
      fd.append('modes', this.analysisRunModes.join(','));
      fd.append('threshold_altered', this.thresholdAltered);
      fd.append('threshold_semantic', this.thresholdSemantic);
      fd.append('limit', this.limit);
      fd.append('unify', this.hitsUnified ? 'true' : 'false');
      fd.append('include_reference', this.includeReference ? 'true' : 'false');
      this.queryError = '';
      this.provenance = null;
      try {
        const resp = await fetch('/api/query', { method: 'POST', body: fd });
        if (!resp.ok) {
          this.queryError = `Query failed (HTTP ${resp.status}) — check Qdrant connection`;
          return;
        }
        const d = await resp.json();
        this.results = Array.isArray(d.results) ? d.results : [];
        this.provenance = d.provenance ?? null;
        this.embeddingModels = d.embedding_models ?? {};

        // Fire-and-forget tag classification for badge display
        this.loadHitTags().catch(() => {});

        // Re-sync selected hit after re-query
        if (this.selectedFileId) {
          const f = this.results.find(r => r.file_id === this.selectedFileId);
          if (f && this.selectedHitKey &&
              !f.hits.find(h => this.hitKey(h) === this.selectedHitKey)) {
            // Previously selected hit dropped from results — auto-select new best.
            this.selectedHitKey = null;
            const bestHit = this.selectedHit;
            if (bestHit) {
              this.selectedHitKey = this.hitKey(bestHit);
              this.matchSrc = `/api/hit-image?path=${encodeURIComponent(bestHit.path)}`;
            } else {
              this.matchSrc = null;
            }
            this.matchZoom = 1.0;
          }
          await this._loadMatchMeta();
        }
      } catch (e) {
        this.queryError = e?.message || 'Query failed — check Qdrant connection';
      }
    },

    debouncedQuery() {
      clearTimeout(this._queryTimer);
      this._queryTimer = setTimeout(() => this.runQuery(), 300);
    },

    // ── Selection ─────────────────────────────────────────────────────────
    async selectFile(fileId) {
      clearTimeout(this._matchImgTimer);

      this.selectedFileId = fileId;
      this.selectedHitKey = null;
      this.selectedFrameTimecode = null;
      this.selectedMatchedFrameKey = null;
      this.queryMeta = null;
      this.matchMeta = null; this.matchMetaError = null;
      this.queryZoom = 1.0;
      this.matchZoom = 1.0;
      this.queryFrames = null;
      this.queryFrameIdx = 0;
      this.expandedHits = {};

      // For video uploads, seed the slideshow; for images use the raw file.
      if (this.isVideoUpload && this.sessionId) {
        this.querySrc = null;  // shown as placeholder until first frame loads
        this._loadQueryFrames(fileId);
      } else {
        this.querySrc = this.sessionId ? `/api/query-image/${this.sessionId}/${fileId}` : null;
      }

      // Auto-select best hit: set matchSrc immediately, or clear it if no hits.
      const bestHit = this.selectedHit; // filteredHits[0] since selectedHitKey is null
      if (bestHit) {
        this.selectedHitKey = this.hitKey(bestHit);
        this.matchSrc = `/api/hit-image?path=${encodeURIComponent(bestHit.path)}`;
      } else {
        this.matchSrc = null;
      }

      await Promise.all([this._loadQueryMeta(), this._loadMatchMeta()]);
    },

    async _loadQueryFrames(fileId) {
      if (!this.sessionId) return;
      try {
        const r = await fetch(`/api/query-frames/${this.sessionId}/${fileId}`);
        if (!r.ok) return;
        const d = await r.json();
        const frames = d.frames ?? [];
        this.videoFramesCache = { ...this.videoFramesCache, [fileId]: frames };
        // Guard: user may have navigated away before the fetch completed
        if (this.selectedFileId === fileId) {
          this.queryFrames = frames;
          // Only update querySrc when in parent mode (not when a specific frame child is selected)
          if (this.selectedFrameTimecode === null && frames.length) {
            // Show best-matching frame in parent mode, fall back to first frame
            const result = this.results.find(res => res.file_id === fileId);
            const bestTimecode = result ? this._bestFrameTimecodeForResult(result) : null;
            const showTimecode = bestTimecode ?? frames[0].timecode_ms;
            const idx = frames.findIndex(f => f.timecode_ms === showTimecode);
            if (idx >= 0) this.queryFrameIdx = idx;
            this.querySrc = `/api/query-frame/${this.sessionId}/${fileId}?timecode_ms=${showTimecode}`;
          }
        }
      } catch { /* fall through — querySrc stays null */ }
    },

    goQueryFrame(idx) {
      if (!this.queryFrames || idx < 0 || idx >= this.queryFrames.length) return;
      this.queryFrameIdx = idx;
      const frame = this.queryFrames[idx];
      this.querySrc = `/api/query-frame/${this.sessionId}/${this.selectedFileId}?timecode_ms=${frame.timecode_ms}`;
      // If a frame child is selected, keep frame-child mode in sync with nav
      if (this.selectedFrameTimecode !== null) {
        this.selectedFrameTimecode = frame.timecode_ms;
        this.selectedMatchedFrameKey = null;
        // Auto-select best hit for the new frame
        this.selectedHitKey = null;
        const bestHit = this.filteredHits[0];
        if (bestHit) {
          this.selectedHitKey = this.hitKey(bestHit);
          this.matchSrc = `/api/hit-image?path=${encodeURIComponent(bestHit.path)}`;
          this._loadMatchMeta();
        } else {
          this.matchSrc = null;
        }
      }
    },

    toggleHitExpanded(key) {
      this.expandedHits = { ...this.expandedHits, [key]: !this.expandedHits[key] };
    },

    async toggleVideoExpanded(fileId) {
      const isExpanding = !this.videoExpanded[fileId];
      this.videoExpanded = { ...this.videoExpanded, [fileId]: isExpanding };
      // Lazy-load frame list when expanding for the first time
      if (isExpanding && !this.videoFramesCache[fileId] && this.sessionId) {
        const r = await fetch(`/api/query-frames/${this.sessionId}/${fileId}`).catch(() => null);
        if (r?.ok) {
          const d = await r.json();
          const frames = d.frames ?? [];
          this.videoFramesCache = { ...this.videoFramesCache, [fileId]: frames };
          if (this.selectedFileId === fileId) this.queryFrames = frames;
        }
      }
    },

    async selectQueryFrame(fileId, timecode_ms) {
      clearTimeout(this._matchImgTimer);
      if (this.selectedFileId !== fileId) {
        // Switching to a different video file — full reset
        this.selectedFileId = fileId;
        this.selectedHitKey = null;
        this.queryMeta = null;
        this.matchMeta = null; this.matchMetaError = null;
        this.queryZoom = 1.0;
        this.matchZoom = 1.0;
        this.expandedHits = {};
        this.queryFrames = this.videoFramesCache[fileId] ?? null;
        if (!this.queryFrames && this.sessionId) this._loadQueryFrames(fileId);
      }
      this.selectedFrameTimecode = timecode_ms;
      this.selectedMatchedFrameKey = null;
      this.querySrc = `/api/query-frame/${this.sessionId}/${fileId}?timecode_ms=${timecode_ms}`;
      // Sync frame-nav index
      const frames = this.queryFrames ?? this.videoFramesCache[fileId];
      if (frames) {
        const idx = frames.findIndex(f => f.timecode_ms === timecode_ms);
        if (idx >= 0) this.queryFrameIdx = idx;
      }
      // Auto-select best hit for this frame
      const bestHit = this.filteredHits[0];
      if (bestHit) {
        this.selectedHitKey = this.hitKey(bestHit);
        this.matchSrc = `/api/hit-image?path=${encodeURIComponent(bestHit.path)}`;
      } else {
        this.selectedHitKey = null;
        this.matchSrc = null;
      }
      // Only (re)load query metadata when switching to a new file; frame
      // navigation within the same video must not re-read the full upload.
      const loads = [this._loadMatchMeta()];
      if (this.queryMeta === null) loads.push(this._loadQueryMeta());
      await Promise.all(loads);
    },

    async selectMatchedFrame(hit, mf) {
      this.selectedHitKey = this.hitKey(hit);
      this.selectedMatchedFrameKey = `${hit.path}:${mf.timecode_ms}`;
      this.matchZoom = 1.0;
      this.matchMeta = null; this.matchMetaError = null;
      // mf.path is this frame's own stored JPEG (not the representative's), so
      // /api/hit-image serves it at native resolution.  No thumbnail fallback:
      // a 128x96 thumb upscaled into the compare pane looks like the frame but
      // is not it, and a missing path is missing data — show the placeholder.
      this.matchSrc = mf.path
        ? `/api/hit-image?path=${encodeURIComponent(mf.path)}`
        : '/static/vector-fallback.svg';
      if (hit.video_hash) {
        await this._loadFrameMeta(hit.video_hash, mf.timecode_ms);
      }
    },

    async selectHit(hit) {
      clearTimeout(this._matchImgTimer);
      this.selectedHitKey = this.hitKey(hit);
      this.selectedMatchedFrameKey = null;
      this.matchZoom = 1.0;
      this.matchMeta = null; this.matchMetaError = null;
      this.videoTimeline = null;
      // The open player belongs to the hit that was selected before this one.
      this.closeVideoPlayback();
      // hit.path is the stored frame JPEG for a video-frame hit (the indexer
      // indexes the extracted frame file, so image_path *is* the frame), and
      // the frame store is inside the allowed path set.  /api/thumbnail serves
      // the 128x96 index thumb — in the big pane that is an upscaled blur of
      // the very artefact the examiner is being asked to compare.
      this.matchSrc = `/api/hit-image?path=${encodeURIComponent(hit.path)}`;
      if (hit.is_video_frame && hit.video_hash) {
        await this._loadFrameMeta(hit.video_hash, hit.frame_timecode_ms ?? 0);
        this._loadVideoTimeline(hit.video_hash);
      } else {
        await this._loadMatchMetaFor(hit.path);
      }
    },
});
