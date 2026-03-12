"""Backend factory for FastVLM runtime implementations."""

from __future__ import annotations

import importlib.util
import logging

from lumen_resources.lumen_config import BackendSettings

from .base import BaseFastVLMBackend

logger = logging.getLogger(__name__)


class RuntimeKind:
    """Runtime kinds supported by FastVLM backends."""

    ONNX = "onnx"


DEFAULT_MAX_NEW_TOKENS = 512


_BACKEND_REGISTRY: dict[str, type[BaseFastVLMBackend]] = {}


def register_backend(kind: str, backend_class: type[BaseFastVLMBackend]) -> None:
    """Register a backend implementation for a runtime kind."""
    _BACKEND_REGISTRY[kind] = backend_class


def get_available_backends() -> list[str]:
    """Return available runtime kinds based on installed dependencies."""
    available: list[str] = []

    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            from .onnxrt_backend import FastVLMONNXBackend

            register_backend(RuntimeKind.ONNX, FastVLMONNXBackend)
            available.append(RuntimeKind.ONNX)
        except ImportError:
            pass

    return available


def create_backend(
    backend_config: BackendSettings | None,
    resources,
    runtime: str,
    prefer_fp16: bool = False,
) -> BaseFastVLMBackend:
    """Create a FastVLM backend from config-facing runtime settings."""
    get_available_backends()

    runtime_normalized = runtime.lower()
    if runtime_normalized not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Runtime '{runtime}' is not available. Available runtimes: {available}"
        )

    if runtime_normalized == RuntimeKind.ONNX:
        from .onnxrt_backend import FastVLMONNXBackend

        providers = (
            getattr(backend_config, "onnx_providers", None) if backend_config else None
        )
        device_preference = (
            getattr(backend_config, "device", "cpu") if backend_config else "cpu"
        )

        return FastVLMONNXBackend(
            resources=resources,
            device_preference=device_preference,
            providers=providers,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            prefer_fp16=prefer_fp16,
        )

    raise ValueError(f"Unknown runtime: {runtime}")
