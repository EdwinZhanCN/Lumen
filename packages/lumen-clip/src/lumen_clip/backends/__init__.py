"""
Backends package for CLIP-like model runtimes.

Exports:
- BaseClipBackend: abstract interface for runtime-agnostic backends
- BackendInfo: metadata container describing runtime, device, and model info
- RuntimeKind: runtime kind constants (class with string attributes)
- ONNXRTBackend: ONNX Runtime implementation (always available)
- create_backend: factory function to create backends based on configuration
- get_available_backends: function to list available runtime kinds

Optional backends (TorchBackend, RKNNBackend) are not imported at module level.
They are dynamically loaded via create_backend() when dependencies are available.
"""

from .base import BackendInfo, BaseClipBackend
from .factory import RuntimeKind, create_backend, get_available_backends
from .onnxrt_backend import ONNXRTBackend

__all__ = [
    "BaseClipBackend",
    "BackendInfo",
    "RuntimeKind",
    "ONNXRTBackend",
    "create_backend",
    "get_available_backends",
]
