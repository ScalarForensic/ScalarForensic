function sfn() {
  return {
    // ── Constants ──────────────────────────────────────────────────────────
    LIST_CAP: 200,

    // ── State ─────────────────────────────────────────────────────────────
    phase: 'upload',
    pendingFiles: [],
    dragOver: false,
    dropZoneIdle: false,
    _idleTimer: null,
    _resetIdle: null,
    selectedModes: [],
    availableModes: [],
    analysisRunModes: [],      // modes actually used in the last analysis
    serverError: '',
    progress: { current: 0, total: 0, filename: '' },
    sessionId: null,
    results: [],
    selectedFileId: null,
    selectedHitKey: null,
    thresholdAltered: 0.75,
    thresholdSemantic: 0.55,
    limit: 10,

    // Image sources with TTL management
    querySrc: null,
    matchSrc: null,
    _queryImgTimer: null,
    _matchImgTimer: null,

    // In-panel zoom
    queryZoom: 1.0,
    matchZoom: 1.0,

    // Lightbox
    lightboxSrc: null,
    lightboxZoom: 1.0,

    // Metadata
    queryMeta: null,
    matchMeta: null,
    matchMetaError: null,
    queryError: '',
    _queryTimer: null,

    // Query-video slideshow
    queryFrames: null,    // [{frame_index, timecode_ms, frame_hash}, …] from /api/query-frames
    queryFrameIdx: 0,     // index into queryFrames currently shown

    // Video file-item expansion state in left panel  {fileId: boolean}
    videoExpanded: {},
    // Cached frame lists per video file  {fileId: [{frame_index, timecode_ms, frame_hash}]}
    videoFramesCache: {},
    // When a specific query frame child is selected, its timecode_ms; null when parent selected
    selectedFrameTimecode: null,
    // When a specific dataset video frame row is selected in the hits panel
    selectedMatchedFrameKey: null,

    // Hit card expansion state  {hitKey: true|false}
    expandedHits: {},

    // Video timeline
    videoTimeline: null,

    // Semantic stats modal
    semanticStats: null,
    semanticStatsLoading: false,
    semanticStatsError: null,
    showSemanticStats: false,

    // Forensic audit modal
    showAudit: false,
    auditLoading: false,
    auditError: null,
    auditQueryLibVersions: null,
    auditHitProvenance: null,
    auditQueryPreproc: null,
    auditHitPreproc: null,

    // Provenance
    provenance: null,
    showProvenance: false,
    embeddingModels: {},
    showEmbModels: false,
    showPasteArea: false,
    pasteBuffer: '',

    // Hits filtering
    hitsFilterAltered: true,
    hitsFilterSemantic: true,
    hitsUnified: true,
    includeReference: false,
    hasReferenceCollection: false,

    // ── Tag Triage ────────────────────────────────────────────────────────
    appMode: 'search',   // 'search' | 'triage'

    // Tag store (shared between modes)
    concepts: [],
    conceptsLoading: false,
    _conceptsPromise: null,
    selectedConceptId: null,

    // Triage run state
    triageRunning: false,
    triageResults: [],
    triageError: null,
    triageLimit: 50,
    triageReverse: false,
    triageTarget: 'dataset',       // 'dataset' | 'reference' | 'uploaded'
    triageCosineThreshold: 0.5,    // cosine floor used in Recommend mode (tag has no negatives)
    exploreStrategy: null,         // 'random' | 'context' — returned by last /api/explore call

    // Triage hit selection
    selectedTriageHitId: null,
    triageMatchSrc: null,
    triageMatchMeta: null,
    triageMatchMetaError: null,
    triageMatchZoom: 1.0,

    // Cache: {pointId: {path, image_hash}} — populated from triage runs.
    // All writes go through _putTriageHitCache so reverse-index invalidation
    // is reliable even when an entry is replaced in place (same key count).
    triageHitCache: {},
    _triageHitCacheVersion: 0,
    // Reverse index: {image_hash: pointId} — rebuilt lazily in searchHitMark
    // when the cache version advances past the cached version.
    _hashToPid: {},
    _hashToPidCacheVersion: -1,
    // Optimistic mark overrides: {tagId: {pointId: 'positive'|'negative'}}
    markedOverrides: {},

    // Tag create form
    showConceptCreate: false,
    newConceptName: '',
    newConceptNotes: '',
    newConceptPositive: '',
    newConceptNegative: '',
    conceptCreateError: null,
    conceptCreateLoading: false,

    // Tag inline edit
    conceptEditing: false,
    editNotes: '',
    conceptEditError: null,
    conceptEditLoading: false,

    // Search-mode mark: image_hash currently being looked up
    markingSearchHit: null,
    // Set-anchor in-flight flag
    settingAnchor: false,

    // Global active tag for +/- keyboard marking and hit-card buttons
    activeTagId: '',

    // Tag membership badges: {image_hash: [tag_name, ...]}
    hitTags: {},

    // ── Computed ──────────────────────────────────────────────────────────
    get progressPct() {
      // Total may be null on error events — treat as no-progress rather than dividing by zero.
      return !this.progress.total
        ? 0
        : Math.round(this.progress.current / this.progress.total * 100);
    },
    get selectedFile() {
      return this.results.find(r => r.file_id === this.selectedFileId) ?? null;
    },
    get isVideoUpload() {
      const f = this.selectedFile;
      return f ? this.isVideoFile(f.filename) : false;
    },
    get currentQueryFrame() {
      if (!this.queryFrames || !this.queryFrames.length) return null;
      return this.queryFrames[this.queryFrameIdx] ?? null;
    },
    // Timecode of the currently shown query frame, or null when not in video mode.
    // Used to dim hits that weren't generated by this frame (parent mode only).
    get activeQueryTimecode() {
      if (!this.isVideoUpload) return null;
      if (this.selectedFrameTimecode !== null) return this.selectedFrameTimecode;
      return this.currentQueryFrame ? this.currentQueryFrame.timecode_ms : null;
    },
    get matchedFiles() {
      return this.results.filter(r => r.hits.length > 0 && !(r.errors?.length > 0));
    },
    get errorFiles() {
      return this.results.filter(r => r.errors?.length > 0);
    },
    get unmatchedFiles() {
      return this.results.filter(r => r.hits.length === 0 && !(r.errors?.length > 0));
    },
    get selectedHit() {
      const hits = this.filteredHits;
      if (this.selectedHitKey) {
        const h = hits.find(h => this.hitKey(h) === this.selectedHitKey);
        if (h) return h;
      }
      return hits[0] ?? null;
    },
    get filteredHits() {
      if (!this.selectedFile) return [];
      let hits = this.selectedFile.hits.filter(h => {
        // Keep if any active-mode score is present
        if ('exact' in h.scores) return true;
        if ('altered' in h.scores && this.hitsFilterAltered) return true;
        if ('semantic' in h.scores && this.hitsFilterSemantic) return true;
        return false;
      });
      // When a specific query frame child is selected, show only that frame's hits
      if (this.selectedFrameTimecode !== null) {
        hits = hits.filter(h => h.query_timecodes?.includes(this.selectedFrameTimecode));
      }
      return hits;
    },
    get selectedConcept() {
      return this.concepts.find(c => c.tag_id === this.selectedConceptId) ?? null;
    },
    tagHealth(tag) {
      // Thresholds derived from high-dimensional metric learning practice.
      // See docs/tag-triage.md §Collection health for the rationale.
      const p = tag.positive_ids.length;
      const n = tag.negative_ids.length;
      if (p < 50 || n < 20) return 'critical';
      if (p < 200 || n < 100) return 'marginal';
      if (p / n > 5 || n / p > 5) return 'marginal';
      return 'healthy';
    },
    get tagHealthByName() {
      const m = {};
      for (const c of this.concepts) m[c.name] = this.tagHealth(c);
      return m;
    },
    get exploreStrategyHint() {
      const c = this.selectedConcept;
      if (!c) return '';
      if (c.positive_ids.length > 0 && c.negative_ids.length > 0)
        return 'Boundary mode: surfaces points near the +/− decision boundary';
      return 'Random mode: returns a uniform sample — mark +/− to enable boundary mode';
    },
    get selectedTriageHit() {
      if (this.selectedTriageHitId) {
        const h = this.triageResults.find(h => String(h.point_id ?? h.file_id) === this.selectedTriageHitId);
        if (h) return h;
      }
      return this.triageResults[0] ?? null;
    },
    get effectiveMarks() {
      if (!this.selectedConceptId) return {};
      const c = this.selectedConcept;
      const marks = {};
      if (c) {
        for (const id of c.positive_ids) marks[String(id)] = 'positive';
        for (const id of c.negative_ids) marks[String(id)] = 'negative';
      }
      const ov = this.markedOverrides[this.selectedConceptId] ?? {};
      return { ...marks, ...ov };
    },
    get auditIntegrity() {
      if (!this.auditQueryLibVersions) return null;
      const rows = this._auditLibRows();
      const libMismatch = rows.filter(r => !r.match && r.indexed !== null).length;
      const modelMismatch = ['altered', 'semantic'].filter(m => this._modelMismatch(m)).length;
      return { libMismatch, modelMismatch, clean: libMismatch === 0 && modelMismatch === 0 };
    },

    // ── Helpers ───────────────────────────────────────────────────────────
    hitKey(hit) {
      // Stable key derived from the hit itself — works for both unified and unmerged.
      // Unified hits have multiple score keys (e.g. "exact,altered"); unmerged hits have one.
      return hit.path + '\x00' + Object.keys(hit.scores).sort().join(',');
    },
    shortName(filename) {
      return filename.replace(/\\/g, '/').split('/').pop();
    },
    folderOf(filename) {
      const parts = filename.replace(/\\/g, '/').split('/');
      return parts.length > 1 ? parts.slice(0, -1).join('/') : '';
    },
    isVideoFile(filename) {
      if (!filename) return false;
      const dot = filename.lastIndexOf('.');
      const ext = dot >= 0 ? filename.slice(dot).toLowerCase() : '';
      return ['.mp4','.avi','.mov','.mkv','.wmv','.flv','.webm','.m4v','.mpg','.mpeg','.3gp','.ts','.mts'].includes(ext);
    },
    frameHitCount(r, timecode_ms) {
      return r.hits.filter(h => h.query_timecodes?.includes(timecode_ms)).length;
    },
    _bestFrameTimecodeForResult(r) {
      // The timecode that generated the highest-scored hit.
      // Prefer the backend-provided best_query_timecode_ms; after merges
      // query_timecodes[0] is not guaranteed to be the highest-scored frame.
      let bestTimecode = null, bestScore = -1;
      for (const hit of r.hits) {
        const candidate = Number.isFinite(hit.best_query_timecode_ms)
          ? hit.best_query_timecode_ms
          : (hit.query_timecodes?.length ? hit.query_timecodes[0] : null);
        if (candidate == null) continue;
        const maxScore = Object.values(hit.scores).reduce((a, b) => Math.max(a, b), 0);
        if (maxScore > bestScore) { bestScore = maxScore; bestTimecode = candidate; }
      }
      return bestTimecode;
    },
    _bestHitImageHash(r) {
      return r.hits[0]?.image_hash ?? null;
    },
    isSelectedMatchedFrame(hit, mf) {
      return this.selectedMatchedFrameKey === `${hit.path}:${mf.timecode_ms}`;
    },
    openLightbox(src) {
      if (!src) return;
      this.lightboxSrc = src;
      this.lightboxZoom = 1.0;
    },
    closeLightbox() {
      this.lightboxSrc = null;
    },

    _metaItems(meta, other) {
      if (!meta) return [];
      const eq  = (a, b) => a != null && b != null && String(a).trim() === String(b).trim();
      const num = (a, b, tol = 0) => a != null && b != null && Math.abs(Number(a) - Number(b)) <= tol;
      const items = [];

      if (meta.size_bytes != null) {
        const rel = other?.size_bytes != null
          ? Math.abs(meta.size_bytes - other.size_bytes) / Math.max(meta.size_bytes, other.size_bytes)
          : 1;
        items.push({ label: 'Size', value: this._fmtSize(meta.size_bytes),
          match: rel === 0, partial: rel > 0 && rel < 0.05 });
      }
      if (meta.format)
        items.push({ label: 'Format', value: meta.format,
          match: eq(meta.format, other?.format), partial: false });
      if (meta.make || meta.model) {
        const makeMatch  = eq(meta.make,  other?.make);
        const modelMatch = eq(meta.model, other?.model);
        items.push({ label: 'Camera', value: [meta.make, meta.model].filter(Boolean).join(' '),
          match: makeMatch && modelMatch, partial: makeMatch && !modelMatch });
      }
      if (meta.datetime) {
        const [dateA, timeA] = (meta.datetime ?? '').split(' ');
        const [dateB, timeB] = (other?.datetime ?? '').split(' ');
        items.push({ label: 'Date', value: meta.datetime,
          match: eq(dateA, dateB) && eq(timeA, timeB),
          partial: eq(dateA, dateB) && !eq(timeA, timeB) });
      }
      if (meta.gps_lat !== undefined && meta.gps_lat !== null)
        items.push({
          label: 'GPS', value: `${meta.gps_lat.toFixed(5)}, ${meta.gps_lon.toFixed(5)}`,
          match:   num(meta.gps_lat, other?.gps_lat, 0.0001) && num(meta.gps_lon, other?.gps_lon, 0.0001),
          partial: num(meta.gps_lat, other?.gps_lat, 0.01)   && num(meta.gps_lon, other?.gps_lon, 0.01),
        });
      if (meta.software)
        items.push({ label: 'Software', value: meta.software,
          match: eq(meta.software, other?.software), partial: false });
      return items;
    },

    _fmtSize(b) {
      if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
      if (b >= 1e3) return `${Math.round(b / 1e3)} KB`;
      return `${b} B`;
    },

    // ── Lifecycle ─────────────────────────────────────────────────────────
    async init() {
      try {
        const d = await fetch('/api/collections').then(r => r.json());
        this.availableModes = d.modes;
        // EXACT is always selected; add other available modes
        this.selectedModes = ['exact', ...d.modes.filter(m => m !== 'exact')];
        this.hasReferenceCollection = d.has_reference ?? false;
        if (d.error) this.serverError = d.error;
      } catch {
        this.availableModes = ['exact'];
        this.selectedModes = ['exact'];
        this.serverError = 'Backend unreachable — only exact hash matching is available.';
      }
      // Load concepts in background (needed for stats-bar concept selector)
      this.loadConcepts().catch(() => {});
      // Screensaver: fade drop-zone out after 5 s of no pointer/keyboard/drag activity
      this._startIdleScreensaver();
    },

    toggleMode(mode) {
      if (mode === 'exact') return; // mandatory, cannot deselect
      if (this.selectedModes.includes(mode)) {
        this.selectedModes = this.selectedModes.filter(m => m !== mode);
      } else {
        this.selectedModes.push(mode);
      }
      if (this.phase === 'results') this.runQuery();
    },

    addFiles(fileList) {
      const exts = new Set([
        '.jpg','.jpeg','.png','.bmp','.tiff','.tif','.webp',
        '.gif','.jp2','.ico','.psd','.heic','.heif',
        '.mp4','.avi','.mov','.mkv','.wmv','.flv','.webm',
        '.m4v','.mpg','.mpeg','.3gp','.ts','.mts',
      ]);
      for (const f of fileList) {
        const dot = f.name.lastIndexOf('.');
        const ext = dot >= 0 ? f.name.slice(dot).toLowerCase() : '';
        if (exts.has(ext) && !this.pendingFiles.some(p => p.name === f.name && p.size === f.size))
          this.pendingFiles.push(f);
      }
    },

    onDrop(e) {
      this.dragOver = false;
      this.addFiles(e.dataTransfer.files);
    },

    // ── Idle screensaver ───────────────────────────────────────────────────
    _startIdleScreensaver() {
      const IDLE_MS = 5000;
      const IDLE_EVENTS = ['mousemove','mousedown','keydown','touchstart','touchmove','dragenter','dragover'];
      let lastReset = 0;
      const resetIdle = () => {
        const now = Date.now();
        // Suppress Alpine reactivity + timer churn on high-frequency events
        // (mousemove can fire 100+ times/s). Only reschedule if ≥100 ms have
        // elapsed since the last reset, OR the drop-zone is already faded out.
        if (!this.dropZoneIdle && now - lastReset < 100) return;
        lastReset = now;
        this.dropZoneIdle = false;
        clearTimeout(this._idleTimer);
        this._idleTimer = setTimeout(() => { this.dropZoneIdle = true; }, IDLE_MS);
      };
      this._resetIdle = resetIdle;
      for (const ev of IDLE_EVENTS) document.addEventListener(ev, resetIdle, { passive: true });
      resetIdle(); // start the initial countdown
    },

    _stopIdleScreensaver() {
      clearTimeout(this._idleTimer);
      this.dropZoneIdle = false;
      if (this._resetIdle) {
        const IDLE_EVENTS = ['mousemove','mousedown','keydown','touchstart','touchmove','dragenter','dragover'];
        for (const ev of IDLE_EVENTS) document.removeEventListener(ev, this._resetIdle);
        this._resetIdle = null;
      }
    },

    // ── Analysis ──────────────────────────────────────────────────────────
    async startAnalysis() {
      if (!this.pendingFiles.length || !this.selectedModes.length) return;
      // Tear down idle screensaver — stop timer and remove all listeners
      this._stopIdleScreensaver();
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
      this.matchSrc = mf.frame_hash
        ? `/api/thumbnail/${mf.frame_hash}`
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
      if (hit.is_video_frame && hit.image_hash) {
        this.matchSrc = `/api/thumbnail/${hit.image_hash}`;
      } else {
        this.matchSrc = `/api/hit-image?path=${encodeURIComponent(hit.path)}`;
      }
      if (hit.is_video_frame && hit.video_hash) {
        await this._loadFrameMeta(hit.video_hash, hit.frame_timecode_ms ?? 0);
        this._loadVideoTimeline(hit.video_hash);
      } else {
        await this._loadMatchMetaFor(hit.path);
      }
    },

    // ── Metadata loading ──────────────────────────────────────────────────
    async _loadQueryMeta() {
      if (!this.sessionId || !this.selectedFileId) return;
      try {
        this.queryMeta = await fetch(
          `/api/query-metadata/${this.sessionId}/${this.selectedFileId}`
        ).then(r => r.json());
        this.loadHitTags().catch(() => {});
      } catch { this.queryMeta = null; }
    },

    async _loadMatchMeta() {
      const hit = this.selectedHit;
      if (hit) {
        if (hit.is_video_frame && hit.video_hash) {
          await this._loadFrameMeta(hit.video_hash, hit.frame_timecode_ms ?? 0);
        } else {
          await this._loadMatchMetaFor(hit.path);
        }
      } else {
        this.matchMeta = null; this.matchMetaError = null;
      }
    },

    async _loadMatchMetaFor(path) {
      this.matchMetaError = null;
      try {
        const r = await fetch(`/api/metadata?path=${encodeURIComponent(path)}`);
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          this.matchMetaError = body.detail ?? `HTTP ${r.status}`;
          this.matchMeta = null;
        } else {
          this.matchMeta = await r.json();
        }
      } catch (e) {
        this.matchMeta = null;
        this.matchMetaError = e?.message || 'Metadata request failed';
      }
    },

    async _loadFrameMeta(videoHash, timecodeMs) {
      this.matchMetaError = null;
      try {
        const r = await fetch(`/api/frame-metadata?video_hash=${encodeURIComponent(videoHash)}&timecode_ms=${timecodeMs}`);
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          this.matchMetaError = body.detail ?? `HTTP ${r.status}`;
          this.matchMeta = null;
        } else {
          this.matchMeta = await r.json();
        }
      } catch (e) {
        this.matchMeta = null;
        this.matchMetaError = e?.message || 'Metadata request failed';
      }
    },

    async _loadVideoTimeline(videoHash) {
      try {
        this.videoTimeline = await fetch(
          `/api/video-timeline?video_hash=${encodeURIComponent(videoHash)}`
        ).then(r => r.json());
      } catch { this.videoTimeline = null; }
    },

    // ── Semantic stats ────────────────────────────────────────────────────
    async calcSemanticStats() {
      if (!this.sessionId || !this.selectedFileId) return;
      this.semanticStats = null;
      this.semanticStatsError = null;
      this.semanticStatsLoading = true;
      this.showSemanticStats = true;
      try {
        const resp = await fetch(`/api/semantic-stats/${this.sessionId}/${this.selectedFileId}`);
        if (!resp.ok) {
          const d = await resp.json().catch(() => ({}));
          this.semanticStatsError = d.detail || `Stats request failed (HTTP ${resp.status})`;
        } else {
          this.semanticStats = await resp.json();
        }
      } catch (e) {
        this.semanticStatsError = e?.message || 'Stats request failed';
      } finally {
        this.semanticStatsLoading = false;
      }
    },

    // ── Provenance ────────────────────────────────────────────────────────
    async copyProvenance() {
      if (!this.provenance) return;
      try { await navigator.clipboard.writeText(JSON.stringify(this.provenance, null, 2)); }
      catch { /* clipboard unavailable */ }
    },

    applyProvenance() {
      let p;
      try { p = JSON.parse(this.pasteBuffer); } catch { return; }
      if (Array.isArray(p.modes)) {
        const validModes = p.modes.filter(m => this.availableModes.includes(m));
        const normalizedModes = this.availableModes.includes('exact') && !validModes.includes('exact')
          ? ['exact', ...validModes]
          : validModes;
        this.analysisRunModes = normalizedModes;
        this.selectedModes = normalizedModes;
      }
      if (typeof p.threshold_altered === 'number')
        this.thresholdAltered = Math.min(1, Math.max(0, p.threshold_altered));
      if (typeof p.threshold_semantic === 'number')
        this.thresholdSemantic = Math.min(1, Math.max(0, p.threshold_semantic));
      if (typeof p.limit === 'number')
        this.limit = Math.min(50, Math.max(1, Math.round(p.limit)));
      this.showPasteArea = false;
      this.pasteBuffer = '';
      this.runQuery();
    },

    // ── Forensic Audit ────────────────────────────────────────────────────
    async openAudit() {
      if (!this.sessionId || !this.selectedFileId || !this.selectedHit) return;
      this.showAudit = true;
      this.auditLoading = true;
      this.auditError = null;
      this.auditQueryLibVersions = null;
      this.auditHitProvenance = null;
      this.auditQueryPreproc = null;
      this.auditHitPreproc = null;
      try {
        // Phase 1: library versions + hit provenance
        const hash = this.selectedHit.image_hash;
        const [lvResp, hpResp] = await Promise.all([
          fetch('/api/library-versions'),
          hash ? fetch(`/api/hit-provenance?image_hash=${encodeURIComponent(hash)}`) : Promise.resolve(null),
        ]);
        if (!lvResp.ok) throw new Error(`Library versions request failed (HTTP ${lvResp.status})`);
        this.auditQueryLibVersions = await lvResp.json();
        if (hpResp?.ok) this.auditHitProvenance = await hpResp.json();
        else if (hpResp) this.auditHitProvenance = {};

        // Phase 2: preprocessing previews (non-fatal if unavailable)
        const hasAlter = !!this.embeddingModels?.altered;
        const hasSeman = !!this.embeddingModels?.semantic;
        if (hasAlter || hasSeman) {
          const hitNcrops   = this.auditHitProvenance?.altered?.sscd_n_crops ?? 1;
          const hitDinoSize = this.auditHitProvenance?.semantic?.normalize_size ?? 224;
          const hitPath     = this.selectedHit?.path;

          const hitParams = new URLSearchParams({ path: hitPath ?? '' });
          hitParams.set('sscd_n_crops',      hasAlter ? String(hitNcrops)   : '0');
          hitParams.set('dino_normalize_size', hasSeman ? String(hitDinoSize) : '0');

          const tc = this.currentQueryFrame?.timecode_ms;
          const queryUrl = `/api/query-preprocessed/${this.sessionId}/${this.selectedFileId}`
                         + (tc != null ? `?timecode_ms=${tc}` : '');
          const hitUrl = hitPath ? `/api/hit-preprocessed?${hitParams}` : null;

          const [qpResp, hpPrep] = await Promise.all([
            fetch(queryUrl),
            hitUrl ? fetch(hitUrl) : Promise.resolve(null),
          ]);
          if (qpResp.ok) {
            this.auditQueryPreproc = await qpResp.json();
          } else {
            console.error(`query-preprocessed failed: HTTP ${qpResp.status}`, queryUrl);
          }
          if (hpPrep?.ok) {
            this.auditHitPreproc = await hpPrep.json();
          } else if (hpPrep) {
            console.error(`hit-preprocessed failed: HTTP ${hpPrep.status}`, hitUrl);
          }
        }
      } catch (e) {
        this.auditError = e?.message || 'Audit data fetch failed';
      } finally {
        this.auditLoading = false;
      }
    },

    async exportAuditJson() {
      try {
        await navigator.clipboard.writeText(JSON.stringify(this._buildAuditReport(), null, 2));
      } catch { /* clipboard unavailable */ }
    },

    _dbLibVersions() {
      if (!this.auditHitProvenance) return {};
      for (const prov of Object.values(this.auditHitProvenance)) {
        if (prov?.library_versions) return prov.library_versions;
      }
      return {};
    },

    _auditLibRows() {
      const qLibs = this.auditQueryLibVersions ?? {};
      const dbLibs = this._dbLibVersions();
      const all = [...new Set([...Object.keys(qLibs), ...Object.keys(dbLibs)])];
      return all.map(lib => ({
        lib,
        query: qLibs[lib] ?? null,
        indexed: dbLibs[lib] ?? null,
        match: qLibs[lib] === dbLibs[lib],
      }));
    },

    _modelMismatch(mode) {
      const qHash = this.embeddingModels[mode]?.hash;
      const dbHash = this.selectedHit?.model_provenance?.[mode]?.hash;
      return !!(qHash && dbHash && qHash !== dbHash);
    },

    _firstIndexedAt() {
      if (!this.auditHitProvenance) return null;
      for (const prov of Object.values(this.auditHitProvenance)) {
        if (prov?.indexed_at) return prov.indexed_at;
      }
      return null;
    },

    _scoreRows() {
      return Object.entries(this.selectedHit?.scores ?? {}).map(([mode, sim]) => ({
        mode,
        similarity: sim,
        distance: mode === 'exact' ? 0 : 1 - sim,
        note: mode === 'exact' ? 'identical file'
            : sim >= 0.99 ? 'near-identical'
            : sim >= 0.90 ? 'very high similarity'
            : sim >= 0.75 ? 'high similarity'
            : sim >= 0.55 ? 'moderate similarity'
            :               'low similarity',
      }));
    },

    _buildAuditReport() {
      const hit = this.selectedHit;
      const scores = {};
      for (const row of this._scoreRows()) {
        scores[row.mode] = { similarity: row.similarity, cosine_distance: parseFloat(row.distance.toFixed(6)) };
      }
      const libRows = this._auditLibRows();
      const libComp = {};
      for (const r of libRows) libComp[r.lib] = { query: r.query, indexed: r.indexed, match: r.match };
      const modelComp = {};
      for (const mode of ['altered', 'semantic']) {
        const qHash = this.embeddingModels[mode]?.hash ?? null;
        const dbHash = hit?.model_provenance?.[mode]?.hash ?? null;
        if (qHash !== null || dbHash !== null)
          modelComp[mode] = { query_hash: qHash, indexed_hash: dbHash, match: qHash === dbHash };
      }
      const notes = [];
      for (const r of libRows)
        if (!r.match && r.indexed !== null)
          notes.push(`${r.lib}: query process has ${r.query}, indexed with ${r.indexed}`);
      for (const [mode, info] of Object.entries(modelComp))
        if (!info.match)
          notes.push(`${mode} model hash: query ${info.query_hash?.slice(0,16)}… vs indexed ${info.indexed_hash?.slice(0,16)}…`);
      return {
        sfn_report_version: '1.0',
        report_generated_utc: new Date().toISOString(),
        analysis_image: {
          filename: this.selectedFile?.filename ?? null,
          sha256: this.queryMeta?.hash_sha256 ?? null,
          md5: this.queryMeta?.hash_md5 ?? null,
          dimensions_px: this.queryMeta?.width ? `${this.queryMeta.width}x${this.queryMeta.height}` : null,
          size_bytes: this.queryMeta?.size_bytes ?? null,
          exif_present: this.queryMeta?.exif ?? null,
          exif_geo_data: this.queryMeta?.exif_geo_data ?? null,
          embedding_models: this.embeddingModels,
          library_versions: this.auditQueryLibVersions,
        },
        result_image: {
          path: hit?.path ?? null,
          sha256: hit?.image_hash ?? null,
          dimensions_px: this.matchMeta?.width ? `${this.matchMeta.width}x${this.matchMeta.height}` : null,
          size_bytes: this.matchMeta?.size_bytes ?? null,
          exif_present: this.matchMeta?.exif ?? null,
          exif_geo_data: this.matchMeta?.exif_geo_data ?? null,
          similarity_scores: scores,
          indexing_provenance: this.auditHitProvenance,
        },
        search_parameters: {
          timestamp_utc: this.provenance?.timestamp ?? null,
          modes: this.provenance?.modes ?? null,
          threshold_altered: this.provenance?.threshold_altered ?? null,
          threshold_semantic: this.provenance?.threshold_semantic ?? null,
          result_limit: this.provenance?.limit ?? null,
        },
        integrity_assessment: {
          model_hashes: modelComp,
          library_versions: libComp,
          forensically_clean: this.auditIntegrity?.clean ?? null,
          discrepancy_notes: notes,
        },
      };
    },

    // ── Concept Triage ────────────────────────────────────────────────────
    async switchToTriage() {
      this.appMode = 'triage';
      if (!this.concepts.length && !this.conceptsLoading) await this.loadConcepts();
    },

    async enterTriageDirect() {
      this.phase = 'results';
      await this.switchToTriage();
    },

    async loadConcepts() {
      this.conceptsLoading = true;
      this._conceptsPromise = (async () => {
        try {
          const d = await fetch('/api/tags').then(r => r.json());
          this.concepts = d.tags ?? [];
        } catch { /* Qdrant may not be up yet */ }
        finally { this.conceptsLoading = false; this._conceptsPromise = null; }
      })();
      await this._conceptsPromise;
    },

    async createConcept() {
      if (!this.newConceptName.trim()) return;
      this.conceptCreateLoading = true;
      this.conceptCreateError = null;
      const fd = new FormData();
      fd.append('name', this.newConceptName.trim());
      fd.append('notes', this.newConceptNotes.trim());
      fd.append('positive_ids', this.newConceptPositive.trim());
      fd.append('negative_ids', this.newConceptNegative.trim());
      fd.append('target_id', '');
      try {
        const resp = await fetch('/api/tag', { method: 'POST', body: fd });
        if (!resp.ok) {
          const d = await resp.json().catch(() => ({}));
          this.conceptCreateError = d.detail ?? `HTTP ${resp.status}`;
          return;
        }
        const tag = await resp.json();
        const idx = this.concepts.findIndex(c => c.tag_id === tag.tag_id);
        if (idx >= 0) this.concepts[idx] = tag; else this.concepts.unshift(tag);
        this.selectedConceptId = tag.tag_id;
        this.showConceptCreate = false;
        this.newConceptName = '';
        this.newConceptNotes = '';
        this.newConceptPositive = '';
        this.newConceptNegative = '';
        this.loadConceptExamples(tag);
      } catch (e) {
        this.conceptCreateError = e?.message || 'Failed to create tag';
      } finally {
        this.conceptCreateLoading = false;
      }
    },

    async deleteConcept(tagId) {
      const resp = await fetch(`/api/tag/${tagId}`, { method: 'DELETE' }).catch(() => null);
      if (!resp || resp.ok || resp.status === 404) {
        this.concepts = this.concepts.filter(c => c.tag_id !== tagId);
        if (this.selectedConceptId === tagId) {
          this.selectedConceptId = null;
          this.triageResults = [];
          this.selectedTriageHitId = null;
        }
      }
    },

    async selectConcept(tagId) {
      this.selectedConceptId = tagId;
      this.activeTagId = tagId;
      this.conceptEditing = false;
      this.triageResults = [];
      this.selectedTriageHitId = null;
      this.triageMatchSrc = null;
      this.triageMatchMeta = null;
      const c = this.selectedConcept;
      if (c) this.loadConceptExamples(c);
    },

    // Single write path for triageHitCache so the version counter and
    // reactivity stay in sync with the data.  Pass payload=null to refresh
    // reactivity without changing the value.
    _putTriageHitCache(id, payload) {
      const sid = String(id);
      if (payload !== null) {
        this.triageHitCache = { ...this.triageHitCache, [sid]: payload };
      } else {
        this.triageHitCache = { ...this.triageHitCache };
      }
      this._triageHitCacheVersion++;
    },

    async loadConceptExamples(concept) {
      const allIds = [
        ...concept.positive_ids,
        ...concept.negative_ids,
        ...(concept.target_id ? [concept.target_id] : []),
      ];
      for (const id of allIds) {
        const sid = String(id);
        if (this.triageHitCache[sid]) {
          this._putTriageHitCache(sid, null);  // trigger reactivity
        }
      }
      // Fetch any unknown IDs
      const toFetch = allIds.filter(id => !this.triageHitCache[String(id)]);
      await Promise.all(toFetch.map(async id => {
        try {
          const r = await fetch(`/api/point-payload?point_id=${encodeURIComponent(id)}`);
          if (r.ok) {
            const d = await r.json();
            this._putTriageHitCache(id, { path: d.image_path, image_hash: d.image_hash });
          } else {
            this._putTriageHitCache(id, { path: null, image_hash: null });
          }
        } catch {
          this._putTriageHitCache(id, { path: null, image_hash: null });
        }
      }));
    },

    async runTriage() {
      if (!this.selectedConceptId) return;
      if (this.triageTarget === 'uploaded' && !this.sessionId) {
        this.triageError = 'No query session active. Upload a query image first.';
        return;
      }
      this.triageRunning = true;
      this.triageError = null;
      this.triageResults = [];
      this.selectedTriageHitId = null;
      this.triageMatchSrc = null;
      this.triageMatchMeta = null;
      this.exploreStrategy = null;
      const fd = new FormData();
      fd.append('tag_id', this.selectedConceptId);
      fd.append('limit', this.triageLimit);
      try {
        let resp;
        fd.append('cosine_threshold', this.triageCosineThreshold);
        if (this.triageTarget === 'uploaded') {
          fd.append('session_id', this.sessionId);
          resp = await fetch('/api/triage/query-images', { method: 'POST', body: fd });
        } else {
          // Triage against the reference collection is intentionally unsupported
          // (tag IDs reference case-collection points and do not resolve in the
          // reference collection); the Run Triage button is hidden for that target.
          fd.append('reverse', this.triageReverse ? 'true' : 'false');
          resp = await fetch('/api/triage', { method: 'POST', body: fd });
        }
        if (!resp.ok) {
          const d = await resp.json().catch(() => ({}));
          this.triageError = d.detail ?? `HTTP ${resp.status}`;
          return;
        }
        const d = await resp.json();
        this.triageResults = d.hits ?? [];
        if (d.tag) {
          const idx = this.concepts.findIndex(c => c.tag_id === d.tag.tag_id);
          if (idx >= 0) this.concepts[idx] = d.tag; else this.concepts.unshift(d.tag);
        }
        if (this.triageTarget !== 'uploaded') {
          for (const h of this.triageResults) {
            const sid = String(h.point_id);
            if (!this.triageHitCache[sid]) {
              this._putTriageHitCache(sid, { path: h.path, image_hash: h.image_hash });
            }
          }
        }
        if (this.triageResults.length > 0) await this.selectTriageHit(this.triageResults[0]);
      } catch (e) {
        this.triageError = e?.message || 'Triage failed — check Qdrant connection';
      } finally {
        this.triageRunning = false;
      }
    },

    async runExplore() {
      if (!this.selectedConceptId) return;
      this.triageRunning = true;
      this.triageError = null;
      this.triageResults = [];
      this.selectedTriageHitId = null;
      this.triageMatchSrc = null;
      this.triageMatchMeta = null;
      const fd = new FormData();
      fd.append('tag_id', this.selectedConceptId);
      fd.append('limit', this.triageLimit);
      if (this.triageTarget === 'reference') fd.append('collection', 'reference');
      try {
        const resp = await fetch('/api/explore', { method: 'POST', body: fd });
        if (!resp.ok) {
          const d = await resp.json().catch(() => ({}));
          this.triageError = d.detail ?? `HTTP ${resp.status}`;
          return;
        }
        const d = await resp.json();
        this.exploreStrategy = d.strategy ?? null;
        this.triageResults = d.hits ?? [];
        if (d.tag) {
          const idx = this.concepts.findIndex(c => c.tag_id === d.tag.tag_id);
          if (idx >= 0) this.concepts[idx] = d.tag; else this.concepts.unshift(d.tag);
        }
        for (const h of this.triageResults) {
          const sid = String(h.point_id);
          if (!this.triageHitCache[sid])
            this._putTriageHitCache(sid, { path: h.path, image_hash: h.image_hash });
        }
        if (this.triageResults.length > 0) await this.selectTriageHit(this.triageResults[0]);
      } catch (e) {
        this.triageError = e?.message || 'Explore failed — check Qdrant connection';
      } finally {
        this.triageRunning = false;
      }
    },

    async selectTriageHit(hit) {
      this.selectedTriageHitId = String(hit.point_id ?? hit.file_id);
      this.triageMatchZoom = 1.0;
      this.triageMatchMeta = null;
      this.triageMatchMetaError = null;
      if (hit.path) {
        const metaPath = hit.is_video_frame && hit.video_path
          ? `${hit.video_path}::frame_000000_t=${hit.frame_timecode_ms ?? 0}ms`
          : hit.path;
        this.triageMatchSrc = `/api/hit-image?path=${encodeURIComponent(hit.path)}`;
        try {
          const r = await fetch(`/api/metadata?path=${encodeURIComponent(metaPath)}`);
          if (r.ok) this.triageMatchMeta = await r.json();
          else this.triageMatchMetaError = `HTTP ${r.status}`;
        } catch (e) { this.triageMatchMetaError = e?.message; }
      } else if (hit.file_id && this.sessionId) {
        this.triageMatchSrc = `/api/query-image/${this.sessionId}/${hit.file_id}`;
      }
    },

    async _applyMark(tagId, pointId, role) {
      // Optimistic update
      this.markedOverrides = {
        ...this.markedOverrides,
        [tagId]: { ...(this.markedOverrides[tagId] ?? {}), [String(pointId)]: role },
      };
      const fd = new FormData();
      fd.append('point_id', String(pointId));
      fd.append('role', role);
      try {
        const resp = await fetch(`/api/tag/${tagId}/mark`, { method: 'POST', body: fd });
        if (resp.ok) {
          const updated = await resp.json();
          const idx = this.concepts.findIndex(c => c.tag_id === tagId);
          if (idx >= 0) this.concepts[idx] = updated; else this.concepts.unshift(updated);
        } else {
          const ov = { ...this.markedOverrides[tagId] };
          delete ov[String(pointId)];
          this.markedOverrides = { ...this.markedOverrides, [tagId]: ov };
        }
      } catch {
        const ov = { ...(this.markedOverrides[tagId] ?? {}) };
        delete ov[String(pointId)];
        this.markedOverrides = { ...this.markedOverrides, [tagId]: ov };
      }
    },

    async markTriageHit(hit, role) {
      if (!this.selectedConceptId || !hit.point_id) return;
      await this._applyMark(this.selectedConceptId, hit.point_id, role);
    },

    async unmarkTriageHit(hit) {
      if (!this.selectedConceptId || !hit.point_id) return;
      const tagId = this.selectedConceptId;
      const sid = String(hit.point_id);
      const ov = { ...(this.markedOverrides[tagId] ?? {}) };
      delete ov[sid];
      this.markedOverrides = { ...this.markedOverrides, [tagId]: ov };
      const fd = new FormData();
      fd.append('point_id', sid);
      try {
        const resp = await fetch(`/api/tag/${tagId}/unmark`, { method: 'POST', body: fd });
        if (resp.ok) {
          const updated = await resp.json();
          const idx = this.concepts.findIndex(c => c.tag_id === tagId);
          if (idx >= 0) this.concepts[idx] = updated;
        }
      } catch { /* ignore */ }
    },

    async unmarkConceptPoint(pointId) {
      if (!this.selectedConceptId) return;
      const tagId = this.selectedConceptId;
      const sid = String(pointId);
      const ov = { ...(this.markedOverrides[tagId] ?? {}) };
      delete ov[sid];
      this.markedOverrides = { ...this.markedOverrides, [tagId]: ov };
      const fd = new FormData();
      fd.append('point_id', sid);
      try {
        const resp = await fetch(`/api/tag/${tagId}/unmark`, { method: 'POST', body: fd });
        if (resp.ok) {
          const updated = await resp.json();
          const idx = this.concepts.findIndex(c => c.tag_id === tagId);
          if (idx >= 0) this.concepts[idx] = updated;
        }
      } catch { /* ignore */ }
    },

    async setAnchor(hit) {
      if (!this.selectedConceptId || !hit?.point_id) return;
      const tagId = this.selectedConceptId;
      const isCurrentAnchor = String(this.selectedConcept?.target_id) === String(hit.point_id);
      const fd = new FormData();
      fd.append('target_id', isCurrentAnchor ? '' : String(hit.point_id));
      this.settingAnchor = true;
      try {
        const resp = await fetch(`/api/tag/${tagId}/set-target`, { method: 'POST', body: fd });
        if (resp.ok) {
          const updated = await resp.json();
          const idx = this.concepts.findIndex(c => c.tag_id === tagId);
          if (idx >= 0) this.concepts[idx] = updated;
          if (updated.target_id) {
            const tid = String(updated.target_id);
            if (!this.triageHitCache[tid]) await this.loadConceptExamples(updated);
          }
        }
      } catch { /* ignore */ }
      finally { this.settingAnchor = false; }
    },

    async clearAnchor() {
      if (!this.selectedConceptId) return;
      const fd = new FormData();
      fd.append('target_id', '');
      this.settingAnchor = true;
      try {
        const resp = await fetch(`/api/tag/${this.selectedConceptId}/set-target`, { method: 'POST', body: fd });
        if (resp.ok) {
          const updated = await resp.json();
          const idx = this.concepts.findIndex(c => c.tag_id === updated.tag_id);
          if (idx >= 0) this.concepts[idx] = updated;
        }
      } catch { /* ignore */ }
      finally { this.settingAnchor = false; }
    },

    startEditConcept() {
      const c = this.selectedConcept;
      if (!c) return;
      this.editNotes = c.notes ?? '';
      this.conceptEditError = null;
      this.conceptEditing = true;
    },

    async saveEditConcept() {
      if (!this.selectedConceptId) return;
      this.conceptEditLoading = true;
      this.conceptEditError = null;
      const c = this.selectedConcept;
      const fd = new FormData();
      fd.append('name', c.name);
      fd.append('notes', this.editNotes.trim());
      fd.append('positive_ids', c.positive_ids.join(','));
      fd.append('negative_ids', c.negative_ids.join(','));
      fd.append('target_id', c.target_id != null ? String(c.target_id) : '');
      try {
        const resp = await fetch('/api/tag', { method: 'POST', body: fd });
        if (!resp.ok) {
          const d = await resp.json().catch(() => ({}));
          this.conceptEditError = d.detail ?? `HTTP ${resp.status}`;
          return;
        }
        const updated = await resp.json();
        const idx = this.concepts.findIndex(c => c.tag_id === updated.tag_id);
        if (idx >= 0) this.concepts[idx] = updated; else this.concepts.unshift(updated);
        this.conceptEditing = false;
      } catch (e) {
        this.conceptEditError = e?.message || 'Save failed';
      } finally {
        this.conceptEditLoading = false;
      }
    },

    exportTriageResults() {
      if (!this.triageResults.length) return;
      const tag = this.selectedConcept;
      const lines = this.triageResults.map(h => JSON.stringify({
        tag_id: tag?.tag_id ?? null,
        tag_name: tag?.name ?? null,
        point_id: h.point_id ?? h.file_id ?? null,
        triplet_score: h.triplet_score ?? null,
        raw_score: h.raw_score ?? null,
        path: h.path ?? null,
        image_hash: h.image_hash ?? null,
        is_video_frame: h.is_video_frame ?? false,
        video_path: h.video_path ?? null,
        frame_timecode_ms: h.frame_timecode_ms ?? null,
      }));
      const blob = new Blob([lines.join('\n') + '\n'], { type: 'application/x-ndjson' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      a.download = `triage-${(tag?.name ?? 'export').replace(/[^a-z0-9_-]/gi, '_')}-${ts}.jsonl`;
      a.click();
      URL.revokeObjectURL(url);
    },

    triageHitMark(hit) {
      return this.effectiveMarks[String(hit.point_id)] ?? null;
    },

    triageHitThumb(hit) {
      if (hit.file_id && this.sessionId && hit.point_id == null) return `/api/query-image/${this.sessionId}/${hit.file_id}`;
      if (hit.image_hash) return `/api/thumbnail/${hit.image_hash}`;
      if (hit.file_id && this.sessionId) return `/api/query-image/${this.sessionId}/${hit.file_id}`;
      return '/static/vector-fallback.svg';
    },

    conceptExampleThumb(pointId) {
      const cached = this.triageHitCache[String(pointId)];
      if (cached?.image_hash) return `/api/thumbnail/${cached.image_hash}`;
      return '/static/vector-fallback.svg';
    },

    conceptExampleSrc(pointId) {
      const cached = this.triageHitCache[String(pointId)];
      if (cached?.path) return `/api/hit-image?path=${encodeURIComponent(cached.path)}`;
      return null;
    },

    // Arrow-key navigation + +/- keyboard marking for search and triage hits
    async handleKeydown(e) {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      if (this.phase !== 'results') return;

      if (this.appMode === 'search') {
        const hits = this.filteredHits;
        if (!hits.length) return;
        const idx = hits.findIndex(h => this.hitKey(h) === this.selectedHitKey);
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          const next = idx < hits.length - 1 ? idx + 1 : idx;
          if (next !== idx) { this.selectHit(hits[next]); await this.$nextTick(); document.querySelector('.hit-card.selected')?.scrollIntoView({block:'nearest'}); }
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          const prev = idx > 0 ? idx - 1 : 0;
          if (prev !== idx) { this.selectHit(hits[prev]); await this.$nextTick(); document.querySelector('.hit-card.selected')?.scrollIntoView({block:'nearest'}); }
        } else if ((e.key === '+' || e.key === '=') && this.activeTagId && idx >= 0) {
          e.preventDefault();
          await this.markSearchHit(hits[idx], 'positive', this.activeTagId);
          const next = idx < hits.length - 1 ? idx + 1 : idx;
          if (next !== idx) { this.selectHit(hits[next]); await this.$nextTick(); document.querySelector('.hit-card.selected')?.scrollIntoView({block:'nearest'}); }
        } else if (e.key === '-' && this.activeTagId && idx >= 0) {
          e.preventDefault();
          await this.markSearchHit(hits[idx], 'negative', this.activeTagId);
          const next = idx < hits.length - 1 ? idx + 1 : idx;
          if (next !== idx) { this.selectHit(hits[next]); await this.$nextTick(); document.querySelector('.hit-card.selected')?.scrollIntoView({block:'nearest'}); }
        }

      } else if (this.appMode === 'triage') {
        const hits = this.triageResults;
        if (!hits.length) return;
        const cur = this.selectedTriageHit;
        const idx = cur ? hits.findIndex(h => String(h.point_id ?? h.file_id) === String(cur.point_id ?? cur.file_id)) : -1;
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          const next = idx < hits.length - 1 ? idx + 1 : idx;
          if (next !== idx) { await this.selectTriageHit(hits[next]); await this.$nextTick(); document.querySelector('.triage-hit-card.selected')?.scrollIntoView({block:'nearest'}); }
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          const prev = idx > 0 ? idx - 1 : 0;
          if (prev !== idx) { await this.selectTriageHit(hits[prev]); await this.$nextTick(); document.querySelector('.triage-hit-card.selected')?.scrollIntoView({block:'nearest'}); }
        } else if ((e.key === '+' || e.key === '=') && idx >= 0) {
          e.preventDefault();
          await this.markTriageHit(hits[idx], 'positive');
          const next = idx < hits.length - 1 ? idx + 1 : idx;
          if (next !== idx) { await this.selectTriageHit(hits[next]); await this.$nextTick(); document.querySelector('.triage-hit-card.selected')?.scrollIntoView({block:'nearest'}); }
        } else if (e.key === '-' && idx >= 0) {
          e.preventDefault();
          await this.markTriageHit(hits[idx], 'negative');
          const next = idx < hits.length - 1 ? idx + 1 : idx;
          if (next !== idx) { await this.selectTriageHit(hits[next]); await this.$nextTick(); document.querySelector('.triage-hit-card.selected')?.scrollIntoView({block:'nearest'}); }
        }
      }
    },

    // Mark a search result hit for a given tag (looks up point_id first).
    // The point-id endpoint searches the case collection then the reference
    // collection, so reference hits can be marked too.
    async markSearchHit(hit, role, tagId) {
      if (!tagId || !hit.image_hash) return;
      this.markingSearchHit = hit.image_hash;
      try {
        const r = await fetch(`/api/point-id?image_hash=${encodeURIComponent(hit.image_hash)}`);
        if (!r.ok) return;
        const { point_id } = await r.json();
        // Cache before _applyMark so the optimistic markedOverrides update
        // is immediately visible via searchHitMark (which looks up by point_id).
        this._putTriageHitCache(point_id, { path: hit.path, image_hash: hit.image_hash });
        await this._applyMark(tagId, point_id, role);
      } catch { /* ignore */ }
      finally { this.markingSearchHit = null; }
    },

    searchHitMark(hit) {
      if (!this.activeTagId || !hit.image_hash) return null;
      // Rebuild the reverse index only when the cache version advances.
      // Using the version counter (rather than Object.keys(...).length)
      // catches in-place replacements where the key count is unchanged.
      if (this._hashToPidCacheVersion !== this._triageHitCacheVersion) {
        this._hashToPid = Object.fromEntries(
          Object.entries(this.triageHitCache)
            .filter(([, v]) => v?.image_hash)
            .map(([pid, v]) => [v.image_hash, pid])
        );
        this._hashToPidCacheVersion = this._triageHitCacheVersion;
      }
      const pid = this._hashToPid[hit.image_hash];
      if (!pid) return null;
      const ov = (this.markedOverrides[this.activeTagId] ?? {})[String(pid)];
      if (ov !== undefined) return ov;
      const c = this.concepts.find(c => c.tag_id === this.activeTagId);
      if (!c) return null;
      if ((c.positive_ids ?? []).map(String).includes(String(pid))) return 'positive';
      if ((c.negative_ids ?? []).map(String).includes(String(pid))) return 'negative';
      return null;
    },

    // Load tag classifications for current search results + query image
    async loadHitTags() {
      if (!this.concepts.length) {
        if (this._conceptsPromise) await this._conceptsPromise;
        if (!this.concepts.length) return;
      }
      const hashes = this.results.flatMap(r => r.hits.map(h => h.image_hash).filter(Boolean));
      if (this.queryMeta?.hash_sha256) hashes.push(this.queryMeta.hash_sha256);

      const fetches = [];
      const uniqueHashes = [...new Set(hashes)];
      for (let i = 0; i < uniqueHashes.length; i += 256) {
        const batch = uniqueHashes.slice(i, i + 256);
        fetches.push(
          fetch('/api/tags/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_hashes: batch, cosine_threshold: this.triageCosineThreshold }),
          }).then(r => r.ok ? r.json() : {}).then(d => d.by_hash ?? {}).catch(() => ({}))
        );
      }
      // Session-based classify for uploaded query images not in the dataset
      if (this.sessionId) {
        fetches.push(
          fetch('/api/tags/classify-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: this.sessionId, cosine_threshold: this.triageCosineThreshold }),
          }).then(r => r.ok ? r.json() : {}).then(d => d.by_hash ?? {}).catch(() => ({}))
        );
      }
      if (!fetches.length) return;
      try {
        const results = await Promise.all(fetches);
        const merged = Object.assign({}, ...results);
        this.hitTags = { ...this.hitTags, ...merged };
      } catch { /* non-fatal */ }
    },

    // ── Reset ─────────────────────────────────────────────────────────────
    resetToUpload() {
      clearTimeout(this._queryTimer);
      clearTimeout(this._matchImgTimer);
      clearTimeout(this._queryImgTimer);
      this.phase = 'upload';
      this.pendingFiles = [];
      this.results = [];
      this.selectedFileId = null;
      this.selectedHitKey = null;
      this.sessionId = null;
      this.querySrc = null;
      this.matchSrc = null;
      this.queryMeta = null;
      this.matchMeta = null; this.matchMetaError = null;
      this.queryError = '';
      this.queryFrames = null;
      this.queryFrameIdx = 0;
      this.expandedHits = {};
      this.provenance = null;
      this.showProvenance = false;
      this.showPasteArea = false;
      this.pasteBuffer = '';
      this.analysisRunModes = [];
      this.queryZoom = 1.0;
      this.matchZoom = 1.0;
      this.lightboxSrc = null;
      this.hitsFilterAltered = true;
      this.hitsFilterSemantic = true;
      this.hitsUnified = true;
      this.semanticStats = null;
      this.semanticStatsLoading = false;
      this.semanticStatsError = null;
      this.showSemanticStats = false;
      this.showAudit = false;
      this.auditLoading = false;
      this.auditError = null;
      this.auditQueryLibVersions = null;
      this.auditHitProvenance = null;
      this.auditQueryPreproc = null;
      this.auditHitPreproc = null;
      // Triage state
      this.appMode = 'search';
      this.triageResults = [];
      this.selectedTriageHitId = null;
      this.triageMatchSrc = null;
      this.triageMatchMeta = null;
      this.triageMatchZoom = 1.0;
      this.triageRunning = false;
      this.triageError = null;
      this.triageTarget = 'dataset';
      this.triageCosineThreshold = 0.5;
      this.markedOverrides = {};
      this.hitTags = {};
      this.activeTagId = '';
      // Re-initialize idle screensaver for this upload session
      this._startIdleScreensaver();
    },
  };
}

