"""
Base backend interfaces for CLIP-like models.

This module defines an abstract, runtime-agnostic interface that backends
(Torch, ONNX Runtime, RKNN, etc.) must implement so the rest of the system
can remain unchanged. A backend is responsible for:

- Model lifecycle (initialize/close)
- Converting raw inputs to unit-normalized embedding vectors:
  - text_to_vector(str) -> np.ndarray[float32, (D,)]
  - image_to_vector(bytes) -> np.ndarray[float32, (D,)]
  - optionally: image_batch_to_vectors(list[bytes]) -> np.ndarray[float32, (N, D)]
- Reporting runtime/device metadata via get_info()

Notes:
- Vectors returned by a backend MUST be unit-normalized (L2 = 1.0) and dtype float32.
- Batch method is optional; a default sequential fallback is provided.
- Classification helpers (cosine-sim + softmax) are provided as utilities to
  keep managers lean and backend-agnostic.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .backend_exceptions import (
    BackendError,
    InferenceError,
)

if TYPE_CHECKING:
    from lumen_clip.resources.loader import ModelResources


__all__ = [
    "RuntimeKind",
    "BackendInfo",
    "BaseClipBackend",
]


class RuntimeKind(str, enum.Enum):
    """Enumerates the primary runtime families for backends."""

    TORCH = "torch"
    ONNXRT = "onnx"
    RKNN = "rknn"


@dataclass
class BackendInfo:
    """
    Describes the active backend configuration and model metadata.

    All fields are optional except `runtime`. Populate as much as possible so the
    gRPC capability message can surface accurate details.
    """

    runtime: str
    device: str | None = None  # e.g., "cuda:0", "mps", "cpu", "rk3588-npu"
    model_id: str | None = (
        None  # a stable identifier (e.g., "ViT-B-32_laion2b_s34b_b79k")
    )
    model_name: str | None = None  # human-friendly name ("ViT-B-32")
    pretrained: str | None = None  # e.g., "laion2b_s34b_b79k"
    version: str | None = None  # backend or model version string
    image_embedding_dim: int | None = None
    text_embedding_dim: int | None = None
    precisions: list[str] = field(default_factory=list)  # e.g., ["fp32","fp16","int8"]
    max_batch_size: int | None = None  # backend hint (if any)
    supports_image_batch: bool = False
    extra: dict[str, str | None] = field(
        default_factory=dict
    )  # arbitrary key/value pairs

    def as_dict(self) -> dict[str, object]:
        """Convert to a plain dict (safe for JSON serialization)."""
        return {
            "runtime": self.runtime,
            "device": self.device,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "version": self.version,
            "image_embedding_dim": self.image_embedding_dim,
            "text_embedding_dim": self.text_embedding_dim,
            "precisions": list(self.precisions),
            "max_batch_size": self.max_batch_size,
            "supports_image_batch": self.supports_image_batch,
            "extra": dict(self.extra),
        }


class BaseClipBackend(abc.ABC):
    """
    Abstract base for CLIP-like backends.

    Implementations MUST:
    - Perform model loading in initialize()
    - Produce unit-normalized float32 vectors for text/image encoding
    - Provide accurate BackendInfo via get_info()

    Implementations MAY:
    - Override image_batch_to_vectors for true batched execution
    - Expose runtime-specific knobs via `extra` in BackendInfo
    """

    def __init__(
        self,
        resources: ModelResources,
        device_preference: str | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        """
        Construct a backend with model resources.

        Args:
            resources: ModelResources object containing all model files and configs
            device_preference: Hint for device selection (e.g., "cuda", "mps", "cpu").
            max_batch_size: Hint for batching; implementation may clamp lower/higher.
        """
        self._initialized: bool = False
        self.resources = resources
        self._device_pref: str | None = device_preference
        self._max_batch_size: int | None = max_batch_size

    # ---------- Lifecycle ----------

    @abc.abstractmethod
    def initialize(self) -> None:
        """Load weights and prepare runtime resources. Must be idempotent."""
        raise NotImplementedError

    def close(self) -> None:
        """Release runtime resources. Optional override."""
        # Default no-op
        return

    @property
    def is_initialized(self) -> bool:
        """Whether initialize() has successfully completed."""
        return self._initialized

    # ---------- Encoding API (unit-normalized float32) ----------

    @abc.abstractmethod
    def text_to_vector(self, text: str) -> NDArray[np.float32]:
        """
        Encode a text string to a unit-normalized embedding vector.

        Returns:
            np.ndarray with shape (D,) and dtype float32, L2-normalized to 1.0
        """
        raise NotImplementedError

    @abc.abstractmethod
    def image_to_vector(self, image_bytes: bytes) -> NDArray[np.float32]:
        """
        Encode image bytes to a unit-normalized embedding vector.

        Returns:
            np.ndarray with shape (D,) and dtype float32, L2-normalized to 1.0
        """
        raise NotImplementedError

    def image_batch_to_vectors(self, images: Sequence[bytes]) -> NDArray[np.float32]:
        """
        Encode a list of image bytes into a batch of unit-normalized vectors.

        Defaults to a sequential fallback using image_to_vector(). Backends that
        support true batched execution should override for better throughput.

        Returns:
            np.ndarray with shape (N, D) and dtype float32, each row L2-normalized.
        """
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        try:
            vecs: list[NDArray[np.float32]] = []
            for img in images:
                try:
                    vec = self.image_to_vector(img)
                    vecs.append(vec)
                except (BackendError, RuntimeError) as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to encode one image in batch: {e}")
                    continue

            if not vecs:
                raise InferenceError("All images in batch failed to encode")

            # Validate consistent dims
            dim = vecs[0].shape[0]
            out = np.stack(vecs, axis=0).astype(np.float32, copy=False)
            if out.ndim != 2 or out.shape[1] != dim:
                raise InferenceError(
                    "image_batch_to_vectors: inconsistent vector dimensions."
                )
            return out

        except Exception as e:
            raise InferenceError(f"Batch image encoding failed: {e}") from e

    def text_batch_to_vectors(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """
        Encode a list of text strings into a batch of unit-normalized vectors.

        Defaults to a sequential fallback using text_to_vector(). Backends that
        support true batched execution should override for better throughput.

        Returns:
            np.ndarray with shape (N, D) and dtype float32, each row L2-normalized.
        """
        if not texts:
            raise InferenceError("All texts in batch failed to encode, empty input.")

        try:
            vecs: list[NDArray[np.float32]] = []
            for text in texts:
                try:
                    vec = self.text_to_vector(text)
                    vecs.append(vec)
                except (BackendError, RuntimeError) as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to encode one text in batch: {e}")
                    continue

            if not vecs:
                raise InferenceError("All texts in batch failed to encode")

            # Validate consistent dims
            dim = vecs[0].shape[0]
            out = np.stack(vecs, axis=0).astype(np.float32, copy=False)
            if out.ndim != 2 or out.shape[1] != dim:
                raise InferenceError(
                    "text_batch_to_vectors: inconsistent vector dimensions."
                )
            return out

        except Exception as e:
            raise InferenceError(f"Batch text encoding failed: {e}") from e

    # ---------- Metadata ----------

    @abc.abstractmethod
    def get_info(self) -> BackendInfo:
        """
        Return a BackendInfo describing runtime, device, model identifiers,
        embedding dimensions, precision supports, batchability, etc.
        """
        raise NotImplementedError

    def get_temperature(self) -> float | None:
        """
        Get model temperature (logit scale) for classification calibration.

        This is an optional method that backends can override to provide
        their model's temperature/logit_scale parameter, which is used for
        better probability calibration in classification tasks.

        Returns:
            Temperature value if supported by backend, None otherwise.
            Typical CLIP models use values between 1.0 and 10.0.

        Note:
            Default implementation returns None. Concrete backends should
            override this if they can expose the temperature parameter.
        """
        return None

    # ---------- Utilities (reusable across backends/managers) ----------

    @staticmethod
    def unit_normalize(
        vec: NDArray[np.float32], axis: int = -1, eps: float = 1e-8
    ) -> NDArray[np.float32]:
        """
        L2-normalize vectors along the given axis.

        Args:
            vec: Input array
            axis: Axis to normalize over
            eps: Numerical stability epsilon

        Returns:
            Unit-normalized array (float32)
        """
        v = vec.astype(np.float32, copy=False)
        norm = np.linalg.norm(v, ord=2, axis=axis, keepdims=True)
        norm = np.maximum(norm, eps)
        return v / norm
