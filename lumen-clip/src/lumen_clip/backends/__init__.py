"""
Backends package for CLIP-like model runtimes.

Exports:
- BaseClipBackend: abstract interface for runtime-agnostic backends
- BackendInfo: metadata container describing runtime, device, and model info
- RuntimeKind: enumeration of supported runtime families
- ONNXRTBackend: ONNX Runtime implementation scaffold of BaseClipBackend
- RKNNBackend: RKNN backend shim (Linux-only optional; provided via separate module)
- create_backend: factory function to create backends based on configuration
"""

from .base import BackendInfo, BaseClipBackend, RuntimeKind
from .factory import create_backend, get_available_backends
from .onnxrt_backend import ONNXRTBackend

# Optional backends
try:
    from .rknn_backend import RKNNBackend
except ImportError:
    RKNNBackend = None

# TorchBackend is not imported by default to avoid requiring torch dependency
# It will be dynamically imported by the factory when needed

__all__ = [
    "BaseClipBackend",
    "BackendInfo",
    "RuntimeKind",
    "ONNXRTBackend",
    "RKNNBackend",
    "create_backend",
    "get_available_backends",
]
