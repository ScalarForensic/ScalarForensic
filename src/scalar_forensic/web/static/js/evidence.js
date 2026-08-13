// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
(window.__sfnParts = window.__sfnParts || []).push({
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

    // ── Source-video playback ─────────────────────────────────────────────
    // Opening the player asks the server what it would serve *before* asking
    // for the bytes, so the viewing-copy label is on screen from the first
    // frame rather than appearing once the rewrap finishes.
    async openVideoPlayback(videoPath, timecodeMs) {
      if (!videoPath) return;
      this.videoPlayback = null;
      this.videoPlaybackError = null;
      this.videoPlaybackLoading = true;
      this.videoPlaybackTimecodeMs = timecodeMs ?? 0;
      try {
        // The indexed video_hash rides along so the server — which is the only
        // side that can hash the file as it is now — decides whether this is
        // still the file that was indexed.  The answer comes back as
        // stale_evidence: true / false / null, and null means "not checked".
        const indexed = this.selectedHit?.video_hash;
        const q = indexed ? `&video_hash=${encodeURIComponent(indexed)}` : '';
        const r = await fetch(`/api/video-playback-info?path=${encodeURIComponent(videoPath)}${q}`);
        const body = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.videoPlaybackError = body.detail ?? `HTTP ${r.status}`;
        } else {
          this.videoPlayback = body;
        }
      } catch (e) {
        this.videoPlaybackError = e?.message || 'Playback request failed';
      } finally {
        this.videoPlaybackLoading = false;
      }
    },

    closeVideoPlayback() {
      // Before the payload goes, not after: closeChunkPlayback() releases the
      // §6.2 lease, and the release needs the source_path that lives on it.
      // Dropping the payload first would leave the video protected from
      // eviction until the ttl ran out.
      this.closeChunkPlayback();
      this.videoPlayback = null;
      this.videoPlaybackError = null;
      this.videoPlaybackLoading = false;
    },

    // Deep-link the player to the frame that produced the hit.  Seeking is only
    // legal once the browser has the duration, which is why this hangs off
    // loadedmetadata; the seek itself rides a range request against the
    // faststart MP4, so nothing downloads the whole clip first.
    seekVideoToHit(el) {
      const seconds = (this.videoPlaybackTimecodeMs ?? 0) / 1000;
      if (!el || !Number.isFinite(seconds) || seconds <= 0) return;
      try {
        el.currentTime = seconds;
      } catch {
        /* Seeking can still be refused on a stream with no seekable range. */
      }
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
});
