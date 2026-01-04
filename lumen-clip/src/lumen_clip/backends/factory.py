"""
Backend factory for creating backend instances based on configuration.

This module provides a factory pattern implementation to dynamically create
backend instances based on the configuration, without importing all backends
unconditionally.
"""

from __future__ import annotations

import logging

from lumen_resources.lumen_config import BackendSettings

from .base import BaseClipBackend, RuntimeKind
from .onnxrt_backend import ONNXRTBackend

logger = logging.getLogger(__name__)

# Global registry for backends
_BACKEND_REGISTRY: dict[RuntimeKind, type[BaseClipBackend]] = {
    RuntimeKind.ONNXRT: ONNXRTBackend,
}


def register_backend(kind: RuntimeKind, backend_class: type[BaseClipBackend]) -> None:
    """Register a backend class for a given runtime kind."""
    _BACKEND_REGISTRY[kind] = backend_class


def get_available_backends() -> list[RuntimeKind]:
    """Get a list of available runtime kinds."""
    available = []

    # Check ONNXRT (always available since it's in base dependencies)
    try:
        import onnxruntime

        available.append(RuntimeKind.ONNXRT)
    except ImportError:
        pass

    # Check PyTorch (optional dependency)
    try:
        import torch  # type: ignore

        # Try to import TorchBackend
        from .torch_backend import TorchBackend

        register_backend(RuntimeKind.TORCH, TorchBackend)
        available.append(RuntimeKind.TORCH)
    except ImportError:
        pass

    # Check RKNN (optional dependency, Linux only)
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
    runtime: RuntimeKind,
    prefer_fp16: bool = False,
) -> BaseClipBackend:
    """
    Create a backend instance based on the configuration.

    Args:
        backend_config: Backend configuration containing runtime kind, device, etc.
        resources: Model resources containing model files and configurations
        runtime: The runtime kind to use
        prefer_fp16: Whether to prefer FP16 precision for ONNX models

    Returns:
        A backend instance

    Raises:
        ValueError: If the specified runtime is not available
        ImportError: If required dependencies are missing
    """

    # Ensure all backends are registered
    get_available_backends()

    if runtime not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Runtime '{runtime}' is not available. "
            f"Available runtimes: {[k.value for k in available]}"
        )

    backend_class = _BACKEND_REGISTRY[runtime]

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
            prefer_fp16=prefer_fp16,
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
