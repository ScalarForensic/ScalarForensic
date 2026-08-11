// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
(window.__sfnParts = window.__sfnParts || []).push({
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
});
