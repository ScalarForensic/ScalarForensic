"""Runtime configuration loaded from environment variables and an optional .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

_VALID_DEDUP_MODES = frozenset({"hash", "filepath", "both"})

_DEFAULT_HASH_CACHE_PATH = "data/hash_cache.db"

# Mirrors YuNetDetector's default score_threshold (faces/detect.py); faces
# below it never reach any gate.  Duplicated rather than imported: config.py
# must not import cv2 transitively.
_DETECTOR_SCORE_FLOOR = 0.5

# Environment variable name for the allow-online flag.  Both the CLI and the
# web entry-point write this variable before constructing Settings() so that
# every per-request Settings() instance created by FastAPI handlers also sees
# the flag without an explicit argument.
ENV_ALLOW_ONLINE = "SFN_ALLOW_ONLINE"


class Settings:
    """All SFN_* runtime settings.

    Resolution order (highest to lowest priority):
    1. Existing process environment variables
    2. Values from the .env file (if found)
    3. Hardcoded defaults
    """

    def __init__(self, env_file: Path | None = None) -> None:
        resolved = (env_file or Path(".env")).resolve()
        self._env_file: Path | None = resolved if resolved.exists() else None
        load_dotenv(self._env_file, override=False)

        self.qdrant_url: str = os.environ.get("SFN_QDRANT_URL", "http://localhost:6333")
        self.collection: str = os.environ.get("SFN_COLLECTION", "sfn")
        self.model_dino: str = os.environ.get("SFN_MODEL_DINO", "facebook/dinov2-large")
        self.model_sscd: str = os.environ.get(
            "SFN_MODEL_SSCD", "models/sscd_disc_mixup.torchscript.pt"
        )
        self.normalize_size: int = self._parse_int("SFN_NORMALIZE_SIZE", 224)
        # None means "auto": the CLI will calibrate on first run and cache the result.
        # The web pipeline reads the cache; both fall back to 32 if no cache exists.
        # Set SFN_BATCH_SIZE explicitly to override auto/cached behavior.
        self.batch_size: int | None = self._parse_optional_int("SFN_BATCH_SIZE")
        self.device: str = os.environ.get("SFN_DEVICE", "auto")
        self.input_dir: Path | None = self._parse_optional_path("SFN_INPUT_DIR")
        self.duplicate_check_mode: str = self._parse_dedup_mode()
        self.extract_exif: bool = self._parse_bool("SFN_EXTRACT_EXIF", default=False)

        # --- Thumbnail cache ---
        # 128×128 JPEG thumbnails are written during indexing and served at
        # /api/thumbnail/{sha256} by sfn-web.  Defaults to data/thumbnails
        # (relative to CWD).  Set SFN_THUMBNAIL_DIR= (empty) to disable.
        self.thumbnail_dir: Path | None = self._parse_optional_path(
            "SFN_THUMBNAIL_DIR", "data/thumbnails"
        )
        self.thumbnail_size: int = self._parse_int("SFN_THUMBNAIL_SIZE", 128)

        # --- Video frame store ---
        # Long-side-capped video frames are stored here during
        # indexing, keyed by frame hash, so thumbnails can be regenerated later when
        # the source video is no longer present.  Defaults to data/frames.
        # Set SFN_FRAME_STORE_DIR= (empty) to disable.
        self.frame_store_dir: Path | None = self._parse_optional_path(
            "SFN_FRAME_STORE_DIR", "data/frames"
        )
        # Maximum pixel dimension (long side) at which frames are stored.
        # Keeping this smaller than the source resolution reduces disk use while
        # still producing high-quality thumbnails.  Default: 512.
        self.frame_store_size: int = self._parse_int("SFN_FRAME_STORE_SIZE", 512)

        # --- Network policy ---
        # Default: offline — no outward connections to HuggingFace or any other service.
        # Set to true (or pass --allow-online) only for first-time model downloads.
        self.allow_online: bool = self._parse_bool(ENV_ALLOW_ONLINE, default=False)

        # --- Qdrant auth (optional) ---
        self.qdrant_api_key: str | None = os.environ.get("SFN_QDRANT_API_KEY") or None

        # --- Tag Triage (Qdrant Discovery API) ---
        # Sidecar collection storing payload-only points — one per "tag".
        # A tag is a named set of positive and negative reference point IDs
        # (and optionally a target anchor) used as input to Qdrant's Discovery
        # and Recommendation APIs.  Kept in a separate collection from the case
        # vectors so tags can outlive any single case and be reused.
        self.tags_collection: str = os.environ.get("SFN_TAGS_COLLECTION", "sfn_tags")

        # Optional read-only reference collection holding vectors of externally
        # labelled reference material.  When set, tag triage queries may pass
        # lookup_from=LookupLocation(collection=<this>, vector=<name>) so the
        # case collection never has to ingest those vectors.  Unset by default.
        self.reference_collection: str | None = os.environ.get("SFN_REFERENCE_COLLECTION") or None

        # --- Remote embeddings endpoint (optional, OpenAI-compatible) ---
        self.embedding_endpoint: str | None = os.environ.get("SFN_EMBEDDING_ENDPOINT") or None
        self.embedding_api_key: str | None = os.environ.get("SFN_EMBEDDING_API_KEY") or None
        self.embedding_model: str | None = os.environ.get("SFN_EMBEDDING_MODEL") or None
        self.embedding_dim: int = self._parse_int("SFN_EMBEDDING_DIM", 0)

        # --- Video frame extraction ---
        # SFN_VIDEO_FPS: how many frames to extract per second of video (default 1).
        # SFN_VIDEO_MAX_FRAMES: hard cap on frames yielded per video file (0 = no cap).
        self.video_fps: float = self._parse_float("SFN_VIDEO_FPS", 1.0)
        self.video_max_frames: int = self._parse_int("SFN_VIDEO_MAX_FRAMES", 500)
        if self.video_max_frames < 0:
            raise ValueError("SFN_VIDEO_MAX_FRAMES must be >= 0 (use 0 for no cap)")

        # --- Hash cache ---
        # Persistent on-disk SHA-256 cache keyed by (path, mtime_ns, size).
        # Eliminates redundant disk reads for files whose content has not changed
        # since the last indexing run.  Defaults to data/hash_cache.db
        # (relative to CWD).
        # Set SFN_HASH_CACHE_PATH= (empty) to disable.
        self.hash_cache_path: Path | None = self._parse_optional_path(
            "SFN_HASH_CACHE_PATH", _DEFAULT_HASH_CACHE_PATH
        )

        # --- Ingestion reports & per-run manifests ---
        # CSV ingestion reports and the paired <stem>.manifest.json land here
        # by default; the CLI's --report flag still overrides the CSV path per
        # run.  Not disableable: every ingestion run writes its report.
        report_dir = self._parse_optional_path("SFN_REPORT_DIR", "data/reports")
        if report_dir is None:
            raise ValueError(
                "SFN_REPORT_DIR must not be empty — ingestion reports are always written "
                "(unset it to use the default data/reports)"
            )
        self.report_dir: Path = report_dir

        # --- SSCD multi-crop ensemble ---
        # SFN_SSCD_N_CROPS controls how many spatial crops are taken per image when
        # embedding with SSCD.  Allowed values:
        #   1 — center crop only.
        #   5 (default) — center crop + four corner crops; better recall for
        #                 off-centre subjects (surveillance stills, padded composites).
        #                 Requires ~1.5× the GPU time of n_crops=1 per SSCD batch.
        #                 Changing this value invalidates embeddings from previous
        #                 indexing runs.
        self.sscd_n_crops: int = self._parse_int("SFN_SSCD_N_CROPS", 5)
        if self.sscd_n_crops not in (1, 5):
            raise ValueError(
                f"SFN_SSCD_N_CROPS={self.sscd_n_crops!r} is invalid. "
                "Allowed values: 1 (center crop only) or 5 (center + 4 corners)."
            )

        # --- Web server ---
        self.web_host: str = os.environ.get("SFN_WEB_HOST", "0.0.0.0")
        self.web_port: int = self._parse_int("SFN_WEB_PORT", 8080)
        if not (1 <= self.web_port <= 65535):
            raise ValueError(f"SFN_WEB_PORT={self.web_port!r} must be between 1 and 65535")
        # Maximum total upload body per /api/analyze request (bytes).
        # Default: 2 GiB — sufficient for large video files.
        # Set SFN_MAX_UPLOAD_BYTES=0 to disable the cap.
        self.max_upload_bytes: int = self._parse_int("SFN_MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024)
        if self.max_upload_bytes < 0:
            raise ValueError("SFN_MAX_UPLOAD_BYTES must be >= 0 (0 disables the cap)")
        # Session idle timeout (seconds). The background reaper runs every 60 s
        # and deletes sessions idle for longer than this value.
        self.session_ttl_seconds: int = self._parse_int("SFN_SESSION_TTL_SECONDS", 3600)
        if self.session_ttl_seconds < 1:
            raise ValueError("SFN_SESSION_TTL_SECONDS must be >= 1")
        # Maximum number of concurrent active sessions. Requests that would
        # exceed this limit receive HTTP 503. Use 0 to disable the cap.
        self.max_active_sessions: int = self._parse_int("SFN_MAX_ACTIVE_SESSIONS", 32)
        if self.max_active_sessions < 0:
            raise ValueError("SFN_MAX_ACTIVE_SESSIONS must be >= 0 (0 disables the cap)")

        # --- Face modality (optional; spec docs/specs/face-pipeline.md) ---
        # Disabled by default.  Enabling requires a detector model, an
        # embedder model + manifest, and SFN_EXAMINER_ID; validated by
        # face_startup_error() so entry points fail fast with guidance.
        self.faces_enabled: bool = self._parse_bool("SFN_FACES_ENABLED", default=False)
        self.face_detector: str = os.environ.get("SFN_FACE_DETECTOR", "yunet")
        if self.face_detector != "yunet":
            raise ValueError(
                f"SFN_FACE_DETECTOR={self.face_detector!r} is invalid. Supported: yunet"
            )
        self.face_detector_model: Path | None = self._parse_optional_path("SFN_FACE_DETECTOR_MODEL")
        self.face_embedder_model: Path | None = self._parse_optional_path("SFN_FACE_EMBEDDER_MODEL")
        self.face_collection: str = (
            os.environ.get("SFN_FACE_COLLECTION") or f"{self.collection}_faces"
        )
        self.face_store_dir: Path | None = self._parse_optional_path(
            "SFN_FACE_STORE_DIR", "data/faces"
        )
        self.face_detect_max_size: int = self._parse_int("SFN_FACE_DETECT_MAX_SIZE", 1600)
        if self.face_detect_max_size < 64:
            raise ValueError("SFN_FACE_DETECT_MAX_SIZE must be >= 64")
        self.face_min_conf: float = self._parse_float("SFN_FACE_MIN_CONF", 0.8)
        if not (0.0 < self.face_min_conf <= 1.0):
            raise ValueError("SFN_FACE_MIN_CONF must be in (0, 1]")
        # 40 by maintainer decision 2026-08-12, lowered from 64.  This is the
        # EMBED floor: it decides whether a face gets a vector and enters search,
        # so it is the gate with evidential consequence.  40 is the largest value
        # that admits the measured 40.1 / 46.9 / 40.8 px cohort (runbook
        # 2026-08-12).  Lowering it does NOT retroactively vectorise already
        # indexed observations — a collection must be re-indexed to feel it.
        self.face_min_size: int = self._parse_int("SFN_FACE_MIN_SIZE", 40)
        if self.face_min_size < 1:
            raise ValueError("SFN_FACE_MIN_SIZE must be >= 1")
        # Review path (spec: 2026-08-12 gate-split design).  Admits faces for
        # hand examination only — never for embedding.  Clamped, never raising:
        # this block parses even when faces are disabled and Settings() is
        # built per request, so a default must not invalidate an explicit value.
        self._face_threshold_notes: list[str] = []
        review_conf = self._parse_float("SFN_FACE_REVIEW_MIN_CONF", 0.6)
        if not 0 < review_conf <= 1:
            raise ValueError("SFN_FACE_REVIEW_MIN_CONF must be in (0, 1]")
        # 24 by maintainer decision 2026-08-12, lowered from 48.  Retention only:
        # a review-only observation is vectorless and cannot produce a machine
        # match, so this floor carries no evidential consequence.
        review_size = self._parse_int("SFN_FACE_REVIEW_MIN_SIZE", 24)
        if review_size < 1:
            raise ValueError("SFN_FACE_REVIEW_MIN_SIZE must be >= 1")
        if review_conf > self.face_min_conf:
            self._face_threshold_notes.append(
                f"SFN_FACE_REVIEW_MIN_CONF ({review_conf}) exceeds SFN_FACE_MIN_CONF "
                f"({self.face_min_conf}); clamped to {self.face_min_conf}. The review "
                "gate can never be stricter than the embedding gate."
            )
            review_conf = self.face_min_conf
        if review_size > self.face_min_size:
            self._face_threshold_notes.append(
                f"SFN_FACE_REVIEW_MIN_SIZE ({review_size}) exceeds SFN_FACE_MIN_SIZE "
                f"({self.face_min_size}); clamped to {self.face_min_size}."
            )
            review_size = self.face_min_size
        if review_conf < _DETECTOR_SCORE_FLOOR:
            self._face_threshold_notes.append(
                f"SFN_FACE_REVIEW_MIN_CONF ({review_conf}) is below the detector's own "
                f"score threshold ({_DETECTOR_SCORE_FLOOR}); no face below that ever "
                "reaches the gate, so the lower value has no effect."
            )
        self.face_review_min_conf: float = review_conf
        self.face_review_min_size: int = review_size
        self.face_min_sharpness: float = self._parse_float("SFN_FACE_MIN_SHARPNESS", 25.0)
        self.face_max_clipped: float = self._parse_float("SFN_FACE_MAX_CLIPPED", 0.6)
        if not (0.0 < self.face_max_clipped <= 1.0):
            raise ValueError("SFN_FACE_MAX_CLIPPED must be in (0, 1]")
        self.face_max_pose: float = self._parse_float("SFN_FACE_MAX_POSE", 0.35)
        self.face_crop_dilation: float = self._parse_float("SFN_FACE_CROP_DILATION", 0.25)
        if not (0.0 < self.face_crop_dilation <= 0.5):
            raise ValueError("SFN_FACE_CROP_DILATION must be in (0, 0.5]")
        # Browse thumbnail long side (px).  Derived, non-evidentiary artefact
        # (spec §7.3) — regenerable, so it is not a comparability field.
        self.face_thumb_size: int = self._parse_int("SFN_FACE_THUMB_SIZE", 256)
        if self.face_thumb_size < 1:
            raise ValueError("SFN_FACE_THUMB_SIZE must be >= 1")
        # Cap on faces taken from ONE uploaded query image (spec §8, Phase 1b).
        # Detection is truncated at this many retained faces rather than
        # refused: a crowd scene must still yield selectable probes.  Query
        # scope only — it never affects what an index run stores, so it is not
        # a comparability field.
        self.face_query_max_faces: int = self._parse_int("SFN_FACE_QUERY_MAX_FACES", 25)
        if self.face_query_max_faces < 1:
            raise ValueError("SFN_FACE_QUERY_MAX_FACES must be >= 1")
        self.examiner_id: str | None = os.environ.get("SFN_EXAMINER_ID") or None

    def _parse_float(self, key: str, default: float) -> float:
        raw = os.environ.get(key)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{key}={raw!r} is not a valid float") from None
        if value <= 0:
            raise ValueError(f"{key}={raw!r} must be a positive number")
        return value

    def _parse_optional_int(self, key: str) -> int | None:
        raw = os.environ.get(key)
        if raw is None:
            return None
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{key}={raw!r} is not a valid integer") from None
        if value <= 0:
            raise ValueError(f"{key}={raw!r} must be a positive integer")
        return value

    def _parse_int(self, key: str, default: int) -> int:
        raw = os.environ.get(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{key}={raw!r} is not a valid integer") from None

    def _parse_bool(self, key: str, default: bool) -> bool:
        raw = os.environ.get(key)
        if raw is None:
            return default
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(f"{key}={raw!r} must be 'true' or 'false'")

    def _parse_optional_path(self, key: str, default: str | None = None) -> Path | None:
        raw = os.environ.get(key)
        if raw is not None:
            return Path(raw) if raw else None
        if default is None:
            return None
        p = Path(default)
        if not p.is_absolute():
            if self._env_file is not None:
                p = self._env_file.parent / p
            else:
                p = Path.cwd() / p
        return p

    def _parse_dedup_mode(self) -> str:
        raw = os.environ.get("SFN_DUPLICATE_CHECK_MODE", "hash")
        mode = raw.strip().lower()
        if mode not in _VALID_DEDUP_MODES:
            raise ValueError(
                f"SFN_DUPLICATE_CHECK_MODE={raw!r} is invalid. "
                f"Choose one of: {', '.join(sorted(_VALID_DEDUP_MODES))}"
            )
        return mode

    def resolve_embedding_model(self, local_model: str) -> str:
        """Return the effective model path/ID to pass to load_embedder.

        When a remote embedding endpoint is configured the model must be
        explicitly named via SFN_EMBEDDING_MODEL; raises ValueError otherwise.
        When using local inference, *local_model* (the per-backend default or
        explicit SFN_MODEL_SSCD/SFN_MODEL_DINO value) is returned as-is.
        """
        if self.embedding_endpoint:
            if not self.embedding_model:
                raise ValueError(
                    "SFN_EMBEDDING_MODEL must be set when SFN_EMBEDDING_ENDPOINT is configured."
                )
            return self.embedding_model
        return local_model

    def apply_network_policy(self) -> None:
        """Enforce the network policy for HuggingFace libraries.

        When *allow_online* is False (the default), unconditionally sets
        ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` so the HuggingFace
        SDK never attempts any network request — even if those variables were
        already present in the shell environment with a different value.

        When *allow_online* is True, removes those variables so that
        ``--allow-online`` reliably enables downloads even if the shell had them
        set to a blocking value.

        Configured endpoints (Qdrant, remote embedder) are unaffected: those use
        ``urllib`` / ``qdrant-client`` directly and do not consult these variables.
        """
        if self.allow_online:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def face_threshold_notes(self) -> list[str]:
        """Non-fatal notices about face threshold clamping (spec §Config)."""
        return list(self._face_threshold_notes)

    def face_startup_error(self) -> str | None:
        """Actionable error if the face modality is enabled but unusable, else None.

        Checked at entry points (CLI --faces, sfn-web lifespan) so
        misconfiguration fails at startup, not at first detection.
        """
        if not self.faces_enabled:
            return None
        problems: list[str] = []
        if self.face_detector_model is None or not self.face_detector_model.exists():
            problems.append(
                f"  - SFN_FACE_DETECTOR_MODEL={str(self.face_detector_model)!r} not found.\n"
                "    Fetch the YuNet ONNX (MIT) once and point this at the local file."
            )
        if self.face_embedder_model is None or not self.face_embedder_model.exists():
            problems.append(
                f"  - SFN_FACE_EMBEDDER_MODEL={str(self.face_embedder_model)!r} not found.\n"
                "    ScalarForensic ships no recognition weights (see INSTALL.md, licensing);\n"
                "    supply an ONNX model plus its .manifest.json."
            )
        elif not Path(str(self.face_embedder_model) + ".manifest.json").exists():
            problems.append(
                f"  - Manifest not found: {self.face_embedder_model}.manifest.json\n"
                "    Every embedder model needs a manifest (see docs/specs/face-pipeline.md §6.3)."
            )
        if not self.examiner_id:
            problems.append(
                "  - SFN_EXAMINER_ID is required while faces are enabled (self-asserted\n"
                "    examiner identity, stamped on adjudications and audit-log entries)."
            )
        if not problems:
            return None
        return "Face modality is enabled (SFN_FACES_ENABLED=true) but not usable:\n" + "\n".join(
            problems
        )

    def offline_model_error(self, *, need_dino: bool = False) -> str | None:
        """Return a user-facing error string if a model is not locally accessible, else None.

        Called before model loading to fail fast with an actionable message.
        Returns None when *allow_online* is True or a remote embedder endpoint is
        configured (remote embedder replaces local models entirely).

        The check is path existence, not a Hub-ID heuristic — a non-existent
        relative path like ``models/dinov2-large`` and a Hub ID like
        ``facebook/dinov2-large`` both require the same remediation steps.

        SSCD is always loaded from a local TorchScript file and never contacts
        HuggingFace Hub, so it is not checked here.
        """
        if self.allow_online or self.embedding_endpoint:
            return None
        if need_dino and not Path(self.model_dino).exists():
            return (
                f"DINOv2 model not found: SFN_MODEL_DINO={self.model_dino!r}\n"
                "  Online access is disabled (default). To fix:\n"
                "  - Download the model once (requires internet), then update .env:\n"
                "      uv run python scripts/download_models.py --dino\n"
                "      # then set SFN_MODEL_DINO=models/dinov2-large in .env\n"
                "  - Or allow a one-time download via the flag:\n"
                "      sfn --allow-online <image-dir> --dino\n"
                "      sfn-web --allow-online"
            )
        return None
