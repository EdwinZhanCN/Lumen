"""
Typed placeholder RKNN backend.

This module provides a typed shim for RKNNBackend that subclasses BaseClipBackend
so that static type checkers see the expected interface. All methods raise
ImportError at runtime because RKNN is Linux-only and depends on rknn-toolkit2.

How to enable RKNN (Linux only):
- Install with the Linux-only extra:
    uv pip install '.[gpu,rknn]'  # see Dockerfiles/cuda|rocm
- Set backend at runtime:
    BIOCLIP_BACKEND=rknn or CLIP_BACKEND=rknn
- Provide required model paths/env as documented in README.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseClipBackend, BackendInfo

__all__ = ["RKNNBackend"]


class RKNNBackend(BaseClipBackend):
    """
    Linux-only RKNN backend shim.

    This class exists to satisfy type checkers by providing the full
    BaseClipBackend interface. All methods raise ImportError to indicate
    the RKNN backend is unavailable in the current environment/build.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
        model_id: Optional[str] = None,
        rknn_model_path: Optional[str] = None,
        target: str = "rk3588",
        device_preference: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            model_id=model_id,
            device_preference=device_preference,
            max_batch_size=max_batch_size,
            cache_dir=cache_dir,
        )
        raise ImportError(
            "RKNNBackend is a Linux-only optional backend and is not available in this build.\n"
            "- rknn-toolkit2 provides wheels only for Linux (manylinux) targets.\n"
            "- Use the CUDA/ROCm Docker images which install the Linux-only 'rknn' extra, e.g.:\n"
            "    docker build -f Dockerfiles/cuda/Dockerfile -t lumen-clip:cuda .\n"
            "    docker build -f Dockerfiles/rocm/Dockerfile -t lumen-clip:rocm .\n"
            "- At runtime, set BIOCLIP_BACKEND=rknn or CLIP_BACKEND=rknn to select the RKNN backend."
        )

    def initialize(self) -> None:
        raise ImportError(
            "RKNNBackend.initialize is unavailable: RKNN is Linux-only. "
            "Use a Linux build with the 'rknn' extra to enable this backend."
        )

    def text_to_vector(self, text: str) -> np.ndarray:
        raise ImportError(
            "RKNNBackend.text_to_vector is unavailable: RKNN is Linux-only. "
            "Use a Linux build with the 'rknn' extra to enable this backend."
        )

    def image_to_vector(self, image_bytes: bytes) -> np.ndarray:
        raise ImportError(
            "RKNNBackend.image_to_vector is unavailable: RKNN is Linux-only. "
            "Use a Linux build with the 'rknn' extra to enable this backend."
        )

    def get_info(self) -> BackendInfo:
        raise ImportError(
            "RKNNBackend.get_info is unavailable: RKNN is Linux-only. "
            "Use a Linux build with the 'rknn' extra to enable this backend."
        )
