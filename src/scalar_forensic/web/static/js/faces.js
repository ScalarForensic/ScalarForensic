// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
//
// Face modality (spec: docs/specs/face-pipeline.md).  Browse only: these are
// face *observations*, never identifications.  Copy in the UI says "face
// observations" / "similar faces" — never "identified person".
(window.__sfnParts = window.__sfnParts || []).push({
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
    async loadQueryFaces() {
      if (!this.facesAvailable || !this.sessionId || !this.selectedFileId) return;
      this.queryFacesLoading = true;
      this.queryFacesError = '';
      this.queryFaces = [];
      this.selectedQueryFaceIndices = [];
      this.queryFacesTruncated = false;
      try {
        const fd = new FormData();
        fd.append('session_id', this.sessionId);
        fd.append('file_id', this.selectedFileId);
        const resp = await fetch('/api/faces/query-faces', { method: 'POST', body: fd });
        const body = await resp.json();
        if (!resp.ok) {
          this.queryFacesError = body.detail || 'face detection failed';
          return;
        }
        this.queryFaces = Array.isArray(body.faces) ? body.faces : [];
        this.queryFacesTruncated = body.truncated === true;
        // Pre-select every searchable face: the examiner de-selects, rather than
        // starting from an empty selection that looks like "no faces found".
        this.selectedQueryFaceIndices = this.queryFaces
          .filter(f => f.searchable).map(f => f.index);
      } catch (e) {
        this.queryFacesError = String(e);
      } finally {
        this.queryFacesLoading = false;
      }
    },

    queryFaceChipUrl(index) {
      return `/api/faces/query-chip/${this.sessionId}/${this.selectedFileId}/${index}`;
    },

    queryFaceSelected(index) {
      return this.selectedQueryFaceIndices.includes(index);
    },

    toggleQueryFace(face) {
      if (!face.searchable) return;   // review-only faces have no vector
      const i = this.selectedQueryFaceIndices.indexOf(face.index);
      if (i === -1) this.selectedQueryFaceIndices.push(face.index);
      else this.selectedQueryFaceIndices.splice(i, 1);
    },

    selectAllQueryFaces() {
      this.selectedQueryFaceIndices = this.queryFaces
        .filter(f => f.searchable).map(f => f.index);
    },

    clearQueryFaceSelection() {
      this.selectedQueryFaceIndices = [];
    },

    queryFaceStatusLabel(face) {
      if (face.searchable) return 'searchable';
      const why = face.embedding_exclusion_reason;
      return why ? `not searchable — ${why} below threshold` : 'not searchable';
    },
});
