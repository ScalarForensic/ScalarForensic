// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
//
// Face modality (spec: docs/specs/face-pipeline.md).  Browse only: these are
// face *observations*, never identifications.  Copy in the UI says "face
// observations" / "similar faces" — never "identified person".
(window.__sfnParts = window.__sfnParts || []).push({
    // ── Error text ────────────────────────────────────────────────────────
    // FastAPI's `detail` is a string for our own HTTPExceptions but an array of
    // validation objects for a 422, which renders as "[object Object]" — the
    // examiner is then told nothing at all about why a search failed.  Flatten
    // both shapes to one line.
    faceErrText(detail, fallback) {
      if (typeof detail === 'string' && detail) return detail;
      if (Array.isArray(detail)) {
        const parts = detail
          .map(d => {
            if (typeof d === 'string') return d;
            const where = Array.isArray(d?.loc) ? d.loc.filter(x => x !== 'body').join('.') : '';
            const msg = d?.msg || JSON.stringify(d);
            return where ? `${where}: ${msg}` : msg;
          })
          .filter(Boolean);
        if (parts.length) return parts.join('; ');
      }
      if (detail && typeof detail === 'object') return JSON.stringify(detail);
      return fallback;
    },

    // ── Availability ──────────────────────────────────────────────────────
    async checkFacesAvailability() {
      try {
        const body = await fetch('/api/faces/availability').then(r => r.json());
        this.facesAvailable = body.faces_available === true;
        this.facesReason = body.reason || '';
        this.facesNote = body.note || '';
      } catch {
        this.facesAvailable = false;
        this.facesReason = 'Face availability check failed.';
      }
    },

    // ── Per-hit face observations ─────────────────────────────────────────
    async loadFacesForHit(imageHash) {
      if (!this.facesAvailable || !imageHash) { this.facesForHit = []; return; }
      this.facesLoading = true;
      try {
        const body = await fetch(`/api/faces/by-image/${imageHash}`).then(r => r.json());
        this.facesForHit = Array.isArray(body.faces) ? body.faces : [];
      } catch {
        this.facesForHit = [];
      } finally {
        this.facesLoading = false;
      }
    },

    // ── Chip URLs ─────────────────────────────────────────────────────────
    // Grid renders thumbnails; clicking opens the full-resolution review chip.
    // The two chip kinds live in separate hash domains: the aligned PNG is
    // addressed by aligned_chip_hash, every review artefact by review_chip_hash.
    // Passing the wrong one names a file that can never exist.
    faceChipUrl(chipHash) { return `/api/faces/chip/${chipHash}`; },
    faceReviewUrl(chipHash) { return `/api/faces/chip/${chipHash}/review`; },
    faceThumbUrl(chipHash) { return `/api/faces/chip/${chipHash}/thumb`; },

    // ── Review-only vs comparable ─────────────────────────────────────────
    // Review-only observations are kept for examination by eye and are never
    // compared against anything.  The distinction must be visible, not a tooltip.
    faceIsReviewOnly(face) { return face?.embedding_status === 'review_only'; },

    faceStatusLabel(face) {
      return this.faceIsReviewOnly(face) ? 'review only — not comparable' : 'comparable';
    },

    // Why it was not embedded, when the record says.  Never guessed here: the
    // server derives it from the stored payload.
    faceExclusionLabel(face) {
      const reason = face?.embedding_exclusion_reason;
      return this.faceIsReviewOnly(face) && reason ? `${reason} below threshold` : '';
    },

    // ── Pipeline explainer ────────────────────────────────────────────────
    // "How was this face processed?" — assembled server-side from persisted
    // data only, so it describes what happened at index time.
    async explainFace(pointId) {
      this.faceExplain = null;
      this.faceExplainOpen = true;
      try {
        this.faceExplain = await fetch(`/api/faces/explain/${pointId}`).then(r => r.json());
      } catch {
        this.faceExplainError = 'Could not load the processing details for this face.';
      }
    },

    closeFaceExplain() {
      this.faceExplainOpen = false;
      this.faceExplain = null;
      this.faceExplainError = '';
    },

    faceQualityLabel(face) {
      const q = face?.quality;
      return typeof q === 'number' ? q.toFixed(2) : '—';
    },

    // ── Query-side faces (session-scoped, never persisted) ────────────────
    // The vectors stay in the server-side Session; the browser only ever holds
    // face *indices*.  A face that failed the embedding gate is vectorless and
    // therefore cannot be a probe — toggleQueryFace refuses it.
    // Every detection carries the file it ran on and a request ordinal.  Two
    // files in one session means two POSTs in flight, and the first one can
    // land last: applying it would leave the earlier file's faces on screen
    // while the chip URLs address the file now selected — a 404 when the new
    // file has fewer faces, and another person's crop when it has more.
    async loadQueryFaces() {
      if (!this.facesAvailable || !this.sessionId || !this.selectedFileId) return;
      const fileId = this.selectedFileId;
      const seq = ++this._queryFacesSeq;
      // Superseded = a newer request exists, which now owns the loading flag
      // and the panel.  Current also requires the file to still be selected.
      const superseded = () => seq !== this._queryFacesSeq;
      const current = () => !superseded() && fileId === this.selectedFileId;
      this.queryFacesLoading = true;
      this.queryFacesError = '';
      this.queryFaces = [];
      this.queryFacesStale = {};
      this.queryFacesTruncated = false;
      try {
        const fd = new FormData();
        fd.append('session_id', this.sessionId);
        fd.append('file_id', fileId);
        const resp = await fetch('/api/faces/query-faces', { method: 'POST', body: fd });
        const body = await resp.json();
        if (!current()) return;
        if (!resp.ok) {
          this.queryFacesError = this.faceErrText(body.detail, 'face detection failed');
          return;
        }
        this.queryFaces = Array.isArray(body.faces) ? body.faces : [];
        this.queryFacesTruncated = body.truncated === true;
        // Pre-select every searchable face: the examiner de-selects, rather than
        // starting from an empty selection that looks like "no faces found".
        // Query rows always describe the *current* file — stale ones would
        // silently probe with another image's faces.  Hit rows persist.
        this.faceBasket = this.faceBasket.filter(r => r.side !== 'query');
        for (const f of this.queryFaces) if (f.searchable) this._basketAddQuery(f);
      } catch (e) {
        if (current()) this.queryFacesError = String(e);
      } finally {
        if (!superseded()) this.queryFacesLoading = false;
      }
      // A dropped response must not drive the searches either: they read the
      // basket, which still describes the file that is actually selected.
      if (!current()) return;
      this.runFaceSearch();
      this.runFaceCompare(this.selectedHit?.image_hash);
    },

    // The server stamps each chip URL with the file and the detection
    // generation it was issued under.  Rebuilding it here from the *current*
    // selection is what let a stale index address the wrong file's faces.
    queryFaceChipUrl(face) {
      return face?.chip_url || '/static/vector-fallback.svg';
    },

    // A chip that fails to load is either gone (404) or superseded (409).
    // Either way the tile must not stay blank next to a live identity label.
    markQueryFaceStale(face) {
      if (face && face.index != null) this.queryFacesStale[face.index] = true;
    },

    queryFaceSelected(index) {
      return this.selectedQueryFaceIndices.includes(index);
    },

    // ── Selection basket (change-set 2026-08-13 item 3) ───────────────────
    // Ctrl+click on a chip *selects* (idempotent — deselection lives on the
    // basket row, removal on ctrl+click of the row).  Both add-paths refuse a
    // review-only face: it has no vector, so it can never be a probe, and a
    // basket row that could not be searched would misstate the selection.
    _basketAddQuery(face) {
      const key = `q:${this.selectedFileId}:${face.index}`;
      const row = this.faceBasket.find(r => r.key === key);
      if (row) { row.selected = true; return; }
      this.faceBasket.push({
        key, side: 'query', fileId: this.selectedFileId, faceIndex: face.index,
        pointId: null, imageHash: null,
        thumbUrl: this.queryFaceChipUrl(face),
        reviewUrl: this.queryFaceChipUrl(face),
        label: `query · face ${face.index + 1}`,
        selected: true,
      });
    },

    toggleQueryFace(face) {
      if (!face.searchable) return;   // review-only faces have no vector
      this._basketAddQuery(face);
      this.runFaceSearch();
    },

    toggleHitFace(face) {
      if (this.faceIsReviewOnly(face)) return;   // vectorless, never a probe
      const pid = String(face.id);
      const key = `p:${pid}`;
      const row = this.faceBasket.find(r => r.key === key);
      if (row) { row.selected = true; }
      else {
        this.faceBasket.push({
          key, side: 'hit', fileId: null, faceIndex: null,
          pointId: pid, imageHash: face.image_hash ?? this.selectedHit?.image_hash ?? null,
          thumbUrl: this.faceThumbUrl(face.review_chip_hash),
          reviewUrl: this.faceReviewUrl(face.review_chip_hash),
          label: `match · ${(face.image_hash ?? this.selectedHit?.image_hash ?? pid).slice(0, 8)}`,
          selected: true,
        });
      }
      this.runFaceSearch();
    },

    hitFaceSelected(face) {
      const row = this.faceBasket.find(r => r.key === `p:${String(face.id)}`);
      return row ? row.selected : false;
    },

    basketToggleRow(row) {
      row.selected = !row.selected;
      this.runFaceSearch();
    },

    basketRemoveRow(row) {
      const i = this.faceBasket.indexOf(row);
      if (i !== -1) this.faceBasket.splice(i, 1);
      this.runFaceSearch();
    },

    basketClear() {
      this.faceBasket = [];
      this.runFaceSearch();
    },

    selectAllQueryFaces() {
      for (const f of this.queryFaces) if (f.searchable) this._basketAddQuery(f);
      this.runFaceSearch();
    },

    clearQueryFaceSelection() {
      for (const r of this.faceBasket) {
        if (r.side === 'query' && r.fileId === this.selectedFileId) r.selected = false;
      }
      this.runFaceSearch();
    },

    // ── Cross-file face search ────────────────────────────────────────────
    // Aggregated many-to-many from the basket: every selected face — session
    // query faces and stored points alike — probes the collection, one kNN
    // per probe, and the backend collapses to the best-scoring observation
    // per medium.  Hit order is therefore max-score-per-hit.
    async runFaceSearch() {
      const idxs = this.selectedQueryFaceIndices;
      const pids = this.selectedFacePointIds;
      if (!this.facesAvailable || (!idxs.length && !pids.length)) {
        this.faceHits = [];
        this.faceMatchScores = {};
        return;
      }
      this.faceSearchLoading = true;
      this.faceSearchError = '';
      try {
        const fd = new FormData();
        fd.append('session_id', this.sessionId);
        fd.append('file_id', this.selectedFileId);
        fd.append('face_indices', idxs.join(','));
        fd.append('point_ids', pids.join(','));
        fd.append('limit', this.faceLimit);
        fd.append('threshold', this.faceThreshold);
        fd.append('exact', this.faceExactSearch ? 'true' : 'false');
        const resp = await fetch('/api/faces/search', { method: 'POST', body: fd });
        const body = await resp.json();
        if (!resp.ok) {
          this.faceSearchError = this.faceErrText(body.detail, 'face search failed');
          return;
        }
        this.faceHits = Array.isArray(body.hits) ? body.hits : [];
        this.faceCalibration = body.calibration || null;
        // Keyed as strings: /api/faces/by-image returns the raw point id while
        // the search returns str(id) — an unstringified key never matches.
        const scores = {};
        for (const h of this.faceHits) scores[String(h.face.point_id)] = h.score;
        this.faceMatchScores = scores;
      } catch (e) {
        this.faceSearchError = String(e);
      } finally {
        this.faceSearchLoading = false;
      }
    },

    // ── Pairwise compare against the selected match (item 3d) ─────────────
    // Auto-runs on match selection.  The response is the full matrix of raw
    // cosines over comparable faces; the operator's faceCrossThreshold floor
    // is applied client-side in faceCrossHighlight, so moving the slider
    // never re-queries.  Against a server without the endpoint (deployed
    // statics before restart) this fails soft: no highlight, error shown.
    async runFaceCompare(imageHash) {
      this.faceComparePairs = [];
      this.faceCompareCounts = null;
      this.faceCompareError = '';
      if (!this.facesAvailable || !this.sessionId || !this.selectedFileId || !imageHash) return;
      if (!this.queryFaces.some(f => f.searchable)) return;
      this.faceCompareLoading = true;
      try {
        const fd = new FormData();
        fd.append('session_id', this.sessionId);
        fd.append('file_id', this.selectedFileId);
        fd.append('image_hash', imageHash);
        const resp = await fetch('/api/faces/compare', { method: 'POST', body: fd });
        const body = await resp.json();
        if (!resp.ok) {
          this.faceCompareError = this.faceErrText(body.detail, 'face compare unavailable');
          return;
        }
        this.faceComparePairs = Array.isArray(body.pairs) ? body.pairs : [];
        this.faceCompareCounts = {
          queryComparable: body.n_query_comparable,
          queryReviewOnly: body.n_query_review_only,
          matchComparable: body.n_match_comparable,
          matchReviewOnly: body.n_match_review_only,
        };
      } catch (e) {
        this.faceCompareError = String(e);
      } finally {
        this.faceCompareLoading = false;
      }
    },

    faceCrossMatched(face) {
      return this.faceCrossHighlight.pointIds.has(String(face.id));
    },

    queryFaceCrossMatched(index) {
      return this.faceCrossHighlight.queryIndices.has(index);
    },

    // ── Per-model explainer surfaces (face pair) ──────────────────────────
    // The distribution is drawn by *one* probe: a mean over several faces would
    // describe no face in particular.  The first selected face is the probe,
    // and it is always a searchable one — the selection cannot hold any other.
    async openFaceStats() {
      const index = this.selectedQueryFaceIndices[0];
      this.showFaceStats = true;
      this.faceStatsError = '';
      if (index === undefined) {
        this.faceStats = null;
        this.faceStatsError = 'Select a query face first.';
        return;
      }
      this.faceStatsLoading = true;
      this.faceStats = null;
      try {
        const fd = new FormData();
        fd.append('session_id', this.sessionId);
        fd.append('file_id', this.selectedFileId);
        fd.append('face_index', index);
        const resp = await fetch('/api/faces/dist-stats', { method: 'POST', body: fd });
        const body = await resp.json();
        if (!resp.ok) { this.faceStatsError = this.faceErrText(body.detail, 'distribution query failed'); return; }
        this.faceStats = body;
      } catch (e) {
        this.faceStatsError = String(e);
      } finally {
        this.faceStatsLoading = false;
      }
    },

    closeFaceStats() {
      this.showFaceStats = false;
      this.faceStats = null;
      this.faceStatsError = '';
    },

    async openFaceAudit() {
      const hash = this.selectedHit?.image_hash;
      this.showFaceAudit = true;
      this.faceAuditError = '';
      if (!hash) {
        this.faceAudit = null;
        this.faceAuditError = 'Select a hit first.';
        return;
      }
      this.faceAuditLoading = true;
      this.faceAudit = null;
      try {
        const resp = await fetch(`/api/faces/audit?image_hash=${encodeURIComponent(hash)}`);
        const body = await resp.json();
        if (!resp.ok) { this.faceAuditError = this.faceErrText(body.detail, 'audit lookup failed'); return; }
        this.faceAudit = body;
      } catch (e) {
        this.faceAuditError = String(e);
      } finally {
        this.faceAuditLoading = false;
      }
    },

    closeFaceAudit() {
      this.showFaceAudit = false;
      this.faceAudit = null;
      this.faceAuditError = '';
    },

    // Mirrors debouncedQuery() in analysis.js: one search per drag, not one per
    // slider step.  The selection handlers call runFaceSearch() directly — the
    // two paths are disjoint, so a drag never fires both.
    debouncedFaceQuery() {
      clearTimeout(this._faceQueryTimer);
      this._faceQueryTimer = setTimeout(() => this.runFaceSearch(), 300);
    },

    faceIsMatched(face) {
      return Object.prototype.hasOwnProperty.call(this.faceMatchScores, String(face.id));
    },

    faceMatchScore(face) {
      const s = this.faceMatchScores[String(face.id)];
      return typeof s === 'number' ? s.toFixed(4) : '';
    },

    queryFaceStatusLabel(face) {
      if (face.searchable) return 'searchable';
      const why = face.embedding_exclusion_reason;
      return why ? `not searchable — ${why} below threshold` : 'not searchable';
    },
});
