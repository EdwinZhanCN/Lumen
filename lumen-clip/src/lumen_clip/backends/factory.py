"""
Backend factory for creating backend instances based on configuration.

This module provides a factory pattern implementation to dynamically create
backend instances based on the configuration, using lazy imports to avoid
requiring optional dependencies at import time.
"""

from __future__ import annotations

import importlib.util
import logging

from lumen_resources.lumen_config import BackendSettings

from .backend_exceptions import BackendDependencyError
from .base import BaseClipBackend, RuntimeKind
from .onnxrt_backend import ONNXRTBackend

logger = logging.getLogger(__name__)

# Module-level backend registry and initialization flag
_BACKEND_REGISTRY: dict[RuntimeKind, type[BaseClipBackend]] = {}
_INITIALIZED: bool = False


def register_backend(kind: RuntimeKind, backend_class: type[BaseClipBackend]) -> None:
    """Register a backend class for a given runtime kind."""
    _BACKEND_REGISTRY[kind] = backend_class


def _ensure_backends_registered() -> None:
    """
    Register all available backends using lazy imports (idempotent).

    This function checks for available optional dependencies using importlib.util,
    and only imports backends for dependencies that are available. It is called
    automatically by get_available_backends() and create_backend().

    The function is idempotent - safe to call multiple times. After the first call,
    it returns immediately without re-scanning dependencies.
    """
    global _INITIALIZED

    if _INITIALIZED:
        return

    # ONNXRT is always available (base dependency)
    _BACKEND_REGISTRY[RuntimeKind.ONNXRT] = ONNXRTBackend

    # Torch is optional - check if torch package is available
    if importlib.util.find_spec("torch") is not None:
        try:
            from .torch_backend import TorchBackend

            _BACKEND_REGISTRY[RuntimeKind.TORCH] = TorchBackend
        except ImportError as e:
            logger.warning(f"Torch backend unavailable due to import error: {e}")

    # RKNN is optional - check if rknnlite package is available
    if importlib.util.find_spec("rknnlite") is not None:
        try:
            from .rknn_backend import RKNNBackend

            _BACKEND_REGISTRY[RuntimeKind.RKNN] = RKNNBackend
        except ImportError as e:
            logger.warning(f"RKNN backend unavailable due to import error: {e}")

    _INITIALIZED = True


def get_available_backends() -> list[RuntimeKind]:
    """
    Get list of available runtime kinds (cached).

    The first call triggers backend discovery and registration.
    Subsequent calls return cached results for performance.

    Returns:
        List of RuntimeKind enum values representing available backends.
    """
    _ensure_backends_registered()
    return list(_BACKEND_REGISTRY.keys())


def reload_backends() -> list[RuntimeKind]:
    """
    Reload backend registry (call after installing new dependencies).

    Clears the backend registry cache and re-scans for available
    dependencies. Call this after installing optional backends
    (e.g., pip install lumen-clip[torch]) to make them available.

    Returns:
        List of RuntimeKind enum values after reload.
    """
    global _INITIALIZED, _BACKEND_REGISTRY

    logger.info("Reloading backend registry...")
    _INITIALIZED = False
    _BACKEND_REGISTRY.clear()
    _ensure_backends_registered()

    available = list(_BACKEND_REGISTRY.keys())
    logger.info(f"Backend registry reloaded. Available backends: {[k.value for k in available]}")
    return available


def create_backend(
    backend_config: BackendSettings,
    resources,
    runtime: RuntimeKind,
    precision: str | None = None,
) -> BaseClipBackend:
    """
    Create a backend instance based on the configuration.

    Args:
        backend_config: Backend configuration containing runtime kind, device, etc.
        resources: Model resources containing model files and configurations
        runtime: The runtime kind to use
        precision: Model precision for ONNX file selection (e.g., "fp32", "fp16", "int8", "q4fp16").
                   If None, uses default precision (fp32). Only applies to ONNX and RKNN runtimes.

    Returns:
        A backend instance

    Raises:
        BackendDependencyError: If requested runtime requires optional dependencies not installed
        ValueError: If the specified runtime is not recognized
    """
    # Ensure all backends are registered
    _ensure_backends_registered()

    if runtime not in _BACKEND_REGISTRY:
        raise BackendDependencyError(runtime.value)

    _backend_class = _BACKEND_REGISTRY[runtime]

    # Create backend instance based on runtime
    if runtime == RuntimeKind.TORCH:
        from .torch_backend import TorchBackend

        return TorchBackend(
            resources=resources,
            device_preference=backend_config.device,
            max_batch_size=backend_config.batch_size,
        )
    elif runtime == RuntimeKind.ONNXRT:
        # Get ONNX-specific settings from backend_config if available
        providers = getattr(backend_config, "onnx_providers", None)

        return ONNXRTBackend(
            resources=resources,
            providers=providers,
            device_preference=backend_config.device,
            max_batch_size=backend_config.batch_size,
            precision=precision,
        )
    elif runtime == RuntimeKind.RKNN:
        from .rknn_backend import RKNNBackend

        return RKNNBackend(
            resources=resources,
            device_preference=backend_config.device,
            max_batch_size=backend_config.batch_size,
        )
    else:
        raise ValueError(f"Unknown runtime: {runtime}")
