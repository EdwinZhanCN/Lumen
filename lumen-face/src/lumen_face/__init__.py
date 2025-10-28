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

__version__ = "0.1.0"
__author__ = "Lumilio Team"
__email__ = "support@lumilio.org"
