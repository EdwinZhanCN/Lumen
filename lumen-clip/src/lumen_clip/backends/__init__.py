"""
Backends package for CLIP-like model runtimes.

Exports:
- BaseClipBackend: abstract interface for runtime-agnostic backends
- BackendInfo: metadata container describing runtime, device, and model info
- RuntimeKind: enumeration of supported runtime families
- ONNXRTBackend: ONNX Runtime implementation (always available)
- create_backend: factory function to create backends based on configuration
- get_available_backends: function to list available runtime kinds
- reload_backends: function to reload backend registry after installing dependencies

Optional backends (TorchBackend, RKNNBackend) are not imported at module level.
They are dynamically loaded via create_backend() when dependencies are available.
"""

from .base import BackendInfo, BaseClipBackend, RuntimeKind
from .factory import create_backend, get_available_backends, reload_backends
from .onnxrt_backend import ONNXRTBackend

__all__ = [
    "BaseClipBackend",
    "BackendInfo",
    "RuntimeKind",
    "ONNXRTBackend",
    "create_backend",
    "get_available_backends",
    "reload_backends",
]
