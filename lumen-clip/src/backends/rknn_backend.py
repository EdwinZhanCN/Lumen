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

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
from numpy.typing import NDArray

from .base import BaseClipBackend, BackendInfo

if TYPE_CHECKING:
    from resources import ModelResources

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
        resources: "ModelResources",
        device_preference: str | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            resources,
            device_preference,
            max_batch_size,
        )
        raise ImportError(
            "RKNNBackend is a Linux-only optional backend and is not available in this build.\n"
            "- rknn-toolkit2 provides wheels only for Linux (manylinux) targets.\n"
            "- Use the CUDA/ROCm Docker images which install the Linux-only 'rknn' extra, e.g.:\n"
            "    docker build -f Dockerfiles/cuda/Dockerfile -t lumen-clip:cuda .\n"
            "    docker build -f Dockerfiles/rocm/Dockerfile -t lumen-clip:rocm .\n"
            "- At runtime, set BIOCLIP_BACKEND=rknn or CLIP_BACKEND=rknn to select the RKNN backend."
        )

    @override
    def initialize(self) -> None:
        raise ImportError(
            "RKNNBackend.initialize is unavailable: RKNN is Linux-only. "
            "Use a Linux build with the 'rknn' extra to enable this backend."
        )

    @override
    def text_to_vector(self, text: str) -> NDArray[np.float32]:
        raise ImportError(
            "RKNNBackend.text_to_vector is unavailable: RKNN is Linux-only. "
            "Use a Linux build with the 'rknn' extra to enable this backend."
        )

    @override
    def image_to_vector(self, image_bytes: bytes) -> NDArray[np.float32]:
        raise ImportError(
            "RKNNBackend.image_to_vector is unavailable: RKNN is Linux-only. "
            "Use a Linux build with the 'rknn' extra to enable this backend."
        )

    @override
    def get_info(self) -> BackendInfo:
        raise ImportError(
            "RKNNBackend.get_info is unavailable: RKNN is Linux-only. "
            "Use a Linux build with the 'rknn' extra to enable this backend."
        )
