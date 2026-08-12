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
    faceChipUrl(chipHash) { return `/api/faces/chip/${chipHash}`; },
    faceReviewUrl(chipHash) { return `/api/faces/chip/${chipHash}/review`; },
    faceThumbUrl(chipHash) { return `/api/faces/chip/${chipHash}/thumb`; },

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
});
