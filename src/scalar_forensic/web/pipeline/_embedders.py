"""Process-wide embedder cache shared by the analysis and query phases."""

from __future__ import annotations

from scalar_forensic.config import Settings
from scalar_forensic.embedder import AnyEmbedder, load_embedder

# ---------------------------------------------------------------------------
# Embedder cache — models are expensive to load; keep them alive per process
# ---------------------------------------------------------------------------

_embedder_cache: dict[str, AnyEmbedder] = {}


def _get_embedder(key: str, settings: Settings) -> AnyEmbedder:
    if key not in _embedder_cache:
        use_sscd = key == "sscd"
        local_model = settings.model_sscd if use_sscd else settings.model_dino
        effective_model = settings.resolve_embedding_model(local_model)
        _embedder_cache[key] = load_embedder(
            model=effective_model,
            use_sscd=use_sscd,
            device=settings.device,
            normalize_size=settings.normalize_size,
            remote_endpoint=settings.embedding_endpoint,
            remote_api_key=settings.embedding_api_key,
            embedding_dim=settings.embedding_dim,
            local_files_only=not settings.allow_online,
            n_crops=settings.sscd_n_crops,
        )
    return _embedder_cache[key]
