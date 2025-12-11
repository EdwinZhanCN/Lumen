"""
Lumen Face Recognition Service.

A Lumen ML general_face providing face detection and recognition capabilities
using the standardized Lumen architecture and protobuf interface.

Features:
- Face detection with confidence filtering and NMS
- Face embedding extraction with optional alignment
- Face comparison and matching
- Support for buffalo_l model (RetinaFace + ArcFace)
- Multi-runtime support (ONNX, RKNN)
- mDNS general_face discovery
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lumen-face")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

__author__ = "Edwin Zhan"
__email__ = "support@lumilio.org"

from .cli import main

__all__ = ["main"]
