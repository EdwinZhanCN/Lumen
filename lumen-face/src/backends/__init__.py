"""
Backends package for lumen-face.

This package exposes the common backend base interfaces:
- RuntimeKind: enumeration of supported runtime kinds
- BackendInfo: lightweight backend capabilities/info
- BaseFaceBackend: abstract base class for face backends
"""

from .base import RuntimeKind, BackendInfo, BaseFaceBackend

__all__ = ["RuntimeKind", "BackendInfo", "BaseFaceBackend"]
