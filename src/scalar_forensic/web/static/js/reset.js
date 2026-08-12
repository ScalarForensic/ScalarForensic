// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
(window.__sfnParts = window.__sfnParts || []).push({
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
    },
});
