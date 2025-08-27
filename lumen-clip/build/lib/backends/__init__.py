"""
Backends package for CLIP-like model runtimes.

Exports:
- BaseClipBackend: abstract interface for runtime-agnostic backends
- BackendInfo: metadata container describing runtime, device, and model info
- RuntimeKind: enumeration of supported runtime families
- TorchBackend: PyTorch/OpenCLIP implementation of BaseClipBackend
- ONNXRTBackend: ONNX Runtime implementation scaffold of BaseClipBackend
- RKNNBackend: RKNN backend shim (Linux-only optional; provided via separate module)
"""

from .base import BaseClipBackend, BackendInfo, RuntimeKind
from .torch_backend import TorchBackend
from .onnxrt_backend import ONNXRTBackend
from .rknn_backend import RKNNBackend

__all__ = [
    "BaseClipBackend",
    "BackendInfo",
    "RuntimeKind",
    "TorchBackend",
    "ONNXRTBackend",
    "RKNNBackend",
]
