"""
Backend factory for creating backend instances based on configuration.

This module provides a factory pattern implementation to dynamically create
backend instances based on the configuration, without importing all backends
unconditionally.
"""

from __future__ import annotations

import importlib.util
import logging

from lumen_resources.lumen_config import BackendSettings

from .base import BaseClipBackend

logger = logging.getLogger(__name__)


class RuntimeKind:
    """Runtime kinds for CLIP backends."""

    ONNXRT = "onnxrt"
    TORCH = "torch"
    RKNN = "rknn"


# Global registry for backends
_BACKEND_REGISTRY: dict[str, type[BaseClipBackend]] = {}


def register_backend(kind: str, backend_class: type[BaseClipBackend]) -> None:
    """Register a backend class for a given runtime kind."""
    _BACKEND_REGISTRY[kind] = backend_class


def get_available_backends() -> list[str]:
    """Get a list of available runtime kinds."""
    available = []

    # Check ONNXRT (base dependency)
    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            from .onnxrt_backend import ONNXRTBackend

            register_backend(RuntimeKind.ONNXRT, ONNXRTBackend)
            available.append(RuntimeKind.ONNXRT)
        except ImportError:
            pass

    # Check PyTorch (optional dependency)
    if importlib.util.find_spec("torch") is not None:
        try:
            from .torch_backend import TorchBackend

            register_backend(RuntimeKind.TORCH, TorchBackend)
            available.append(RuntimeKind.TORCH)
        except ImportError:
            pass

    # Check RKNN (optional dependency, Linux only)
    if importlib.util.find_spec("rknnlite") is not None:
        try:
            from .rknn_backend import RKNNBackend

            register_backend(RuntimeKind.RKNN, RKNNBackend)
            available.append(RuntimeKind.RKNN)
        except ImportError:
            pass

    return available


def create_backend(
    backend_config: BackendSettings,
    resources,
    runtime: str,
    precision: str | None = None,
) -> BaseClipBackend:
    """
    Create a backend instance based on the configuration.

    Args:
        backend_config: Backend configuration containing runtime kind, device, etc.
        resources: Model resources containing model files and configurations
        runtime: The runtime kind to use (e.g., "onnxrt", "torch", "rknn")
        precision: Model precision for ONNX file selection (e.g., "fp32", "fp16", "int8", "q4fp16").
                   If None, uses default precision (fp32). Only applies to ONNX and RKNN runtimes.

    Returns:
        A backend instance

    Raises:
        ValueError: If the specified runtime is not available
        ImportError: If required dependencies are missing
    """

    # Ensure all backends are registered
    get_available_backends()

    # Normalize runtime name
    runtime_normalized = runtime.lower()

    if runtime_normalized not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Runtime '{runtime}' is not available. Available runtimes: {available}"
        )

    # Create backend instance based on runtime
    if runtime_normalized == RuntimeKind.ONNXRT:
        from .onnxrt_backend import ONNXRTBackend

        providers = getattr(backend_config, "onnx_providers", None)

        return ONNXRTBackend(
            resources=resources,
            providers=providers,
            device_preference=backend_config.device,
            max_batch_size=backend_config.batch_size,
            precision=precision,
        )
    elif runtime_normalized == RuntimeKind.TORCH:
        from .torch_backend import TorchBackend

        return TorchBackend(
            resources=resources,
            device_preference=backend_config.device,
            max_batch_size=backend_config.batch_size,
        )
    elif runtime_normalized == RuntimeKind.RKNN:
        from .rknn_backend import RKNNBackend

        return RKNNBackend(
            resources=resources,
            device_preference=backend_config.device,
            max_batch_size=backend_config.batch_size,
        )
    else:
        raise ValueError(f"Unknown runtime: {runtime}")
