from headroom.backends.base import (
    Backend, BackendError, ContextExceededError, ModelMismatchError, is_warm, warm_up,
)
from headroom.schemas.config import BackendSettings


def make_backend(settings: BackendSettings):
    """The adapter for the experiment's [backend] kind. Use as an async context manager."""
    if settings.kind == "ollama":
        from headroom.backends.ollama import OllamaBackend
        return OllamaBackend(settings)
    if settings.kind == "llamacpp":
        from headroom.backends.llamacpp import LlamaCppBackend
        return LlamaCppBackend(settings)
    raise ValueError(f"unknown backend kind {settings.kind!r}")


__all__ = [
    "Backend", "BackendError", "ContextExceededError", "ModelMismatchError", "is_warm", "make_backend", "warm_up",
]
