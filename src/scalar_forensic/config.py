"""Runtime configuration loaded from environment variables and an optional .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

_VALID_DEDUP_MODES = frozenset({"hash", "filepath", "both"})


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
        self.model_sscd: str = os.environ.get("SFN_MODEL_SSCD", "models/sscd_disc_mixup.torchscript.pt")
        self.normalize_size: int = self._parse_int("SFN_NORMALIZE_SIZE", 512)
        self.batch_size: int = self._parse_int("SFN_BATCH_SIZE", 32)
        self.device: str = os.environ.get("SFN_DEVICE", "auto")
        self.input_dir: Path | None = self._parse_optional_path("SFN_INPUT_DIR")
        self.duplicate_check_mode: str = self._parse_dedup_mode()
        self.extract_exif: bool = self._parse_bool("SFN_EXTRACT_EXIF", default=False)

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
