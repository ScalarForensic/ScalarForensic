// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
(window.__sfnParts = window.__sfnParts || []).push({
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
});
