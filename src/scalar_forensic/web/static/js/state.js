// Part of the sfn() Alpine component — merged by app.js. See app.js for load order.
(window.__sfnParts = window.__sfnParts || []).push({
    // ── Constants ──────────────────────────────────────────────────────────
    LIST_CAP: 200,

    // ── State ─────────────────────────────────────────────────────────────
    phase: 'upload',
    pendingFiles: [],
    dragOver: false,
    selectedModes: [],
    availableModes: [],
    analysisRunModes: [],      // modes actually used in the last analysis
    serverError: '',
    progress: { current: 0, total: 0, filename: '' },
    sessionId: null,
    results: [],
    selectedFileId: null,

    // ── Face modality (optional; browse only) ──────────────────────────────
    facesAvailable: false,
    facesReason: '',
    facesNote: '',
    facesForHit: [],
    facesLoading: false,
    faceExplain: null,
    faceExplainOpen: false,
    faceExplainError: '',
    selectedHitKey: null,

    // Query-side faces (session-scoped; detected on upload, never indexed)
    queryFaces: [],
    queryFacesLoading: false,
    queryFacesError: '',
    queryFacesTruncated: false,

    // Face selection basket (change-set 2026-08-13).  One row per face the
    // examiner picked, from either side; a row is selected or unselected, and
    // removal takes it out of the array — those are the three states.  The
    // probe lists the search sends are *derived* from this (computed.js).
    // Row: { key, side ('query'|'hit'), fileId, faceIndex, pointId, imageHash,
    //        thumbUrl, reviewUrl, label, selected }
    faceBasket: [],

    // Pairwise compare against the selected match (auto-run on selection).
    // faceCrossThreshold is the operator's display floor for the both-sides
    // highlight.  It stays 0.0 for the same reason faceThreshold does: this
    // deployment has no calibrated threshold, and a non-zero default would
    // manufacture one (spec §10).
    faceComparePairs: [],
    faceCompareCounts: null,
    faceCompareLoading: false,
    faceCompareError: '',
    faceCrossThreshold: 0.0,

    // Cross-file face search.  faceThreshold is a *display floor* on the raw
    // cosine and stays 0.0: this deployment has no calibrated threshold, and a
    // non-zero default would manufacture one (spec §10).
    faceHits: [],
    faceSearchLoading: false,
    faceSearchError: '',
    faceCalibration: null,
    faceMatchScores: {},        // {point_id (string): raw cosine}
    hitsFilterFaces: true,
    faceLimit: 10,
    faceThreshold: 0.0,
    faceExactSearch: true,
    _faceQueryTimer: null,

    // Per-model explainer surfaces (face pair; the DINO pair is above)
    faceStats: null,
    faceStatsLoading: false,
    faceStatsError: '',
    showFaceStats: false,
    faceAudit: null,
    faceAuditLoading: false,
    faceAuditError: '',
    showFaceAudit: false,

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
});
