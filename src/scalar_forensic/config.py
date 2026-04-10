"""Runtime configuration loaded from environment variables and an optional .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

_VALID_DEDUP_MODES = frozenset({"hash", "filepath", "both"})


def _is_hf_hub_id(model_name: str) -> bool:
    """True when *model_name* looks like a HuggingFace Hub ID rather than a local path.

    Hub IDs have the form ``org/model-name`` (exactly one slash, no leading dot).
    Local paths either exist on disk, are absolute, or start with ``./`` / ``../``.
    """
    p = Path(model_name)
    # is_absolute() is a pure string check — run it before exists() to avoid I/O
    # for Hub IDs like "facebook/dinov2-large" which can never be absolute.
    if p.is_absolute() or p.exists():
        return False
    parts = model_name.split("/")
    return len(parts) == 2 and not model_name.startswith(".")


class Settings:
    """All SFN_* runtime settings.

    Resolution order (highest to lowest priority):
    1. Existing process environment variables
    2. Values from the .env file (if found)
    3. Hardcoded defaults
    """

    def __init__(self, env_file: Path | None = None) -> None:
        resolved = env_file or Path(".env")
        self._env_file: Path | None = resolved if resolved.exists() else None
        load_dotenv(self._env_file, override=False)

        self.qdrant_url: str = os.environ.get("SFN_QDRANT_URL", "http://localhost:6333")
        self.collection_dino: str = os.environ.get("SFN_COLLECTION_DINO", "sfn-dinov2")
        self.collection_sscd: str = os.environ.get("SFN_COLLECTION_SSCD", "sfn-sscd")
        self.model_dino: str = os.environ.get("SFN_MODEL_DINO", "facebook/dinov2-large")
        self.model_sscd: str = os.environ.get(
            "SFN_MODEL_SSCD", "models/sscd_disc_mixup.torchscript.pt"
        )
        self.normalize_size: int = self._parse_int("SFN_NORMALIZE_SIZE", 512)
        self.batch_size: int = self._parse_int("SFN_BATCH_SIZE", 32)
        self.device: str = os.environ.get("SFN_DEVICE", "auto")
        self.input_dir: Path | None = self._parse_optional_path("SFN_INPUT_DIR")
        self.duplicate_check_mode: str = self._parse_dedup_mode()
        self.extract_exif: bool = self._parse_bool("SFN_EXTRACT_EXIF", default=False)

        # --- Network policy ---
        # Default: offline — no outward connections to HuggingFace or any other service.
        # Set to true (or pass --allow-online) only for first-time model downloads.
        self.allow_online: bool = self._parse_bool("SFN_ALLOW_ONLINE", default=False)

        # --- Qdrant auth (optional) ---
        self.qdrant_api_key: str | None = os.environ.get("SFN_QDRANT_API_KEY") or None

        # --- Remote embeddings endpoint (optional, OpenAI-compatible) ---
        self.embedding_endpoint: str | None = os.environ.get("SFN_EMBEDDING_ENDPOINT") or None
        self.embedding_api_key: str | None = os.environ.get("SFN_EMBEDDING_API_KEY") or None
        self.embedding_model: str | None = os.environ.get("SFN_EMBEDDING_MODEL") or None
        self.embedding_dim: int = self._parse_int("SFN_EMBEDDING_DIM", 0)

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

    def _parse_optional_path(self, key: str) -> Path | None:
        raw = os.environ.get(key)
        return Path(raw) if raw else None

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

        When *allow_online* is False (the default), sets ``HF_HUB_OFFLINE=1``
        and ``TRANSFORMERS_OFFLINE=1`` so the HuggingFace SDK never attempts
        any network request.  Call this as early as possible — before any model
        loading code runs.

        Configured endpoints (Qdrant, remote embedder) are unaffected: those use
        ``urllib`` / ``qdrant-client`` directly and do not consult these variables.
        """
        if not self.allow_online:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def offline_model_error(self, *, need_dino: bool = False) -> str | None:
        """Return a user-facing error string if a model requires online access, else None.

        Called before model loading to fail fast with an actionable message.
        Returns None when *allow_online* is True or a remote embedder endpoint is
        configured (remote embedder replaces local models entirely).

        SSCD is always loaded from a local TorchScript file and never contacts
        HuggingFace Hub, so it is not checked here.
        """
        if self.allow_online or self.embedding_endpoint:
            return None
        if need_dino and _is_hf_hub_id(self.model_dino):
            return (
                f"SFN_MODEL_DINO={self.model_dino!r} is a HuggingFace Hub ID, "
                "but online access is disabled (default).\n"
                "  To fix, download the model once (requires internet):\n"
                "    uv run python scripts/download_models.py --dino\n"
                "  Then point to the local snapshot in .env:\n"
                "    SFN_MODEL_DINO=models/dinov2-large\n"
                "  To temporarily allow a one-time download instead:\n"
                "    sfn --allow-online  /  sfn-web --allow-online"
            )
        return None
