// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
(window.__sfnParts = window.__sfnParts || []).push({
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
      this.checkFacesAvailability().catch(() => {});
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
});
