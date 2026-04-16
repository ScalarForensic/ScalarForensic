"""Runtime configuration loaded from environment variables and an optional .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

_VALID_DEDUP_MODES = frozenset({"hash", "filepath", "both"})

_DEFAULT_HASH_CACHE_PATH = "data/hash_cache.db"


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
        self.allow_online: bool = self._parse_bool("SFN_ALLOW_ONLINE", default=False)

        # --- Qdrant auth (optional) ---
        self.qdrant_api_key: str | None = os.environ.get("SFN_QDRANT_API_KEY") or None

        # --- Vector visualization ---
        # Maximum number of points fetched per collection for the 3-D background viz.
        self.viz_max_points: int = self._parse_int("SFN_VIZ_MAX_POINTS", 5000)
        # Optional path to write a self-contained viz HTML file on startup.
        # When set, the file is (re-)written every time the server starts so it
        # always reflects the current indexed collection.
        # Example: SFN_VIZ_EXPORT_PATH=~/.local/share/sfn/viz.html
        self.viz_export_path: Path | None = self._parse_optional_path("SFN_VIZ_EXPORT_PATH")

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

        # --- SSCD multi-crop ensemble ---
        # SFN_SSCD_N_CROPS controls how many spatial crops are taken per image when
        # embedding with SSCD.  Allowed values:
        #   1 (default) — center crop only; matches historical behaviour.
        #   5           — center crop + four corner crops; recommended for forensic
        #                 use cases where subjects may be off-centre (surveillance
        #                 stills, padded composites).  Requires ~5× the GPU compute
        #                 of n_crops=1 per SSCD batch.  Changing this value
        #                 invalidates embeddings from previous indexing runs.
        self.sscd_n_crops: int = self._parse_int("SFN_SSCD_N_CROPS", 1)
        if self.sscd_n_crops not in (1, 5):
            raise ValueError(
                f"SFN_SSCD_N_CROPS={self.sscd_n_crops!r} is invalid. "
                "Allowed values: 1 (center crop only) or 5 (center + 4 corners)."
            )

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
        if not default:
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
