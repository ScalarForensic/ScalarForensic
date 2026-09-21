"""Process-wide embedder cache shared by the analysis and query phases."""

from __future__ import annotations

import threading
from functools import wraps

from scalar_forensic.config import Settings
from scalar_forensic.embedder import AnyEmbedder, load_embedder
from scalar_forensic.safeguards import CompatibilityError

_embedder_cache: dict[str, AnyEmbedder] = {}
_cache_configs: dict[str, tuple] = {}
_cache_lock = threading.RLock()


def _configuration(key: str, settings: Settings) -> tuple:
    use_sscd = key == "sscd"
    local_model = settings.model_sscd if use_sscd else settings.model_dino
    return (
        settings.resolve_embedding_model(local_model),
        use_sscd,
        settings.device,
        settings.normalize_size,
        settings.embedding_endpoint,
        settings.embedding_api_key,
        settings.embedding_dim,
        settings.allow_online,
        settings.sscd_n_crops,
    )


def _raise_if_cache_drifted(key: str, identity: tuple) -> None:
    """Refuse a request whose embedder settings differ from the cached ones.

    Caller must hold ``_cache_lock``.  One definition so the drift message
    an examiner sees cannot depend on which entry point noticed it.
    """
    if key in _embedder_cache and _cache_configs.get(key) != identity:
        raise CompatibilityError(
            "The cached embedder configuration differs from this request; "
            "restart the web process before changing model or preprocessing settings."
        )


def validate_cached_embedder_settings(settings: Settings, needed) -> None:
    """Validate existing caches without loading models or disclosing credentials."""
    with _cache_lock:
        for key in needed:
            _raise_if_cache_drifted(key, _configuration(key, settings))


def _get_embedder(key: str, settings: Settings) -> AnyEmbedder:
    identity = _configuration(key, settings)
    effective_model, use_sscd = identity[:2]
    with _cache_lock:
        _raise_if_cache_drifted(key, identity)
        if key in _embedder_cache:
            return _embedder_cache[key]
        model = load_embedder(
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
        inference_lock = threading.Lock()
        original_embed = model.embed_images

        @wraps(original_embed)
        def embed_images(images):
            with inference_lock:
                return original_embed(images)

        model.embed_images = embed_images
        _embedder_cache[key] = model
        _cache_configs[key] = identity
        return model
