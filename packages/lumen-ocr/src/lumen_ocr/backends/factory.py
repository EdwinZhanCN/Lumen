"""
Backend factory for creating OCR backend instances based on configuration.

This module provides a factory pattern implementation to dynamically create
backend instances based on the configuration, without importing all backends
unconditionally.
"""

from __future__ import annotations

import importlib.util
import logging

from lumen_resources.lumen_config import BackendSettings

from .base import BaseOcrBackend

logger = logging.getLogger(__name__)


class RuntimeKind:
    """Runtime kinds for OCR backends."""

    ONNXRT = "onnxrt"
    PADDLE = "paddle"
    TORCH = "torch"
    RKNN = "rknn"


# Global registry for backends
_BACKEND_REGISTRY: dict[str, type[BaseOcrBackend]] = {}


def register_backend(kind: str, backend_class: type[BaseOcrBackend]) -> None:
    """Register a backend class for a given runtime kind."""
    _BACKEND_REGISTRY[kind] = backend_class


def get_available_backends() -> list[str]:
    """Get a list of available runtime kinds."""
    available = []

    # Check ONNXRT (base dependency)
    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            from .onnxrt_backend import OnnxOcrBackend

            register_backend(RuntimeKind.ONNXRT, OnnxOcrBackend)
            available.append(RuntimeKind.ONNXRT)
        except ImportError:
            pass

    # Check PyTorch (optional dependency)
    # if importlib.util.find_spec("torch") is not None:
    #     try:
    #         from .torch_backend import TorchOcrBackend

    #         register_backend(RuntimeKind.TORCH, TorchOcrBackend)
    #         available.append(RuntimeKind.TORCH)
    #     except ImportError:
    #         pass

    # Check RKNN (optional dependency, Linux only)
    # if importlib.util.find_spec("rknnlite") is not None:
    #     try:
    #         from .rknn_backend import RknnOcrBackend

    #         register_backend(RuntimeKind.RKNN, RknnOcrBackend)
    #         available.append(RuntimeKind.RKNN)
    #     except ImportError:
    #         pass

    return available


def create_backend(
    backend_config: BackendSettings,
    resources,
    runtime: str,
    prefer_fp16: bool = False,
) -> BaseOcrBackend:
    """
    Create an OCR backend instance based on the configuration.

    Args:
        backend_config: Backend configuration containing runtime kind, device, etc.
        resources: Model resources containing model files and configurations
        runtime: The runtime kind to use (e.g., "onnx", "paddle", "torch", "rknn")
        prefer_fp16: Whether to prefer FP16 precision for ONNX/RKNN models

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
    if runtime_normalized == "onnx":
        runtime_normalized = RuntimeKind.ONNXRT

    if runtime_normalized not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Runtime '{runtime}' is not available. Available runtimes: {available}"
        )

    # Create backend instance based on runtime
    if runtime_normalized == RuntimeKind.ONNXRT:
        from .onnxrt_backend import OnnxOcrBackend

        providers = getattr(backend_config, "onnx_providers", None)

        return OnnxOcrBackend(
            resources=resources,
            providers=providers,
            device_preference=backend_config.device,
            prefer_fp16=prefer_fp16,
        )
    # elif runtime_normalized == RuntimeKind.PADDLE:
    #     from .paddle_backend import PaddleOcrBackend

    #     return PaddleOcrBackend(
    #         resources=resources,
    #         device_preference=backend_config.device,
    #     )
    # elif runtime_normalized == RuntimeKind.TORCH:
    #     from .torch_backend import TorchOcrBackend

    #     return TorchOcrBackend(
    #         resources=resources,
    #         device_preference=backend_config.device,
    #     )
    # elif runtime_normalized == RuntimeKind.RKNN:
    #     from .rknn_backend import RknnOcrBackend

    #     return RknnOcrBackend(
    #         resources=resources,
    #         device_preference=backend_config.device,
    #         prefer_fp16=prefer_fp16,
    #     )
    else:
        raise ValueError(f"Unknown runtime: {runtime}")
