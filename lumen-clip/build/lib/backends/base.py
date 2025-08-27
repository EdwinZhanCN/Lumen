"""
Base backend interfaces for CLIP-like models.

This module defines an abstract, runtime-agnostic interface that backends
(Torch, ONNX Runtime, RKNN, etc.) must implement so the rest of the system
can remain unchanged. A backend is responsible for:

- Model lifecycle (initialize/close)
- Converting raw inputs to unit-normalized embedding vectors:
  - text_to_vector(str) -> np.ndarray[float32, (D,)]
  - image_to_vector(bytes) -> np.ndarray[float32, (D,)]
  - optionally: image_batch_to_vectors(List[bytes]) -> np.ndarray[float32, (N, D)]
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "RuntimeKind",
    "BackendInfo",
    "BaseClipBackend",
]


class RuntimeKind(str, enum.Enum):
    """Enumerates the primary runtime families for backends."""

    TORCH = "torch"
    ONNXRT = "onnxrt"
    RKNN = "rknn"
    TENSORRT = "tensorrt"
    COREML = "coreml"
    DIRECTML = "directml"
    OPENVINO = "openvino"
    CPU = "cpu"  # generic CPU runtime if needed
    UNKNOWN = "unknown"


@dataclass
class BackendInfo:
    """
    Describes the active backend configuration and model metadata.

    All fields are optional except `runtime`. Populate as much as possible so the
    gRPC capability message can surface accurate details.
    """
    runtime: str
    device: Optional[str] = None  # e.g., "cuda:0", "mps", "cpu", "rk3588-npu"
    model_id: Optional[str] = None  # a stable identifier (e.g., "ViT-B-32_laion2b_s34b_b79k")
    model_name: Optional[str] = None  # human-friendly name ("ViT-B-32")
    pretrained: Optional[str] = None  # e.g., "laion2b_s34b_b79k"
    version: Optional[str] = None  # backend or model version string
    image_embedding_dim: Optional[int] = None
    text_embedding_dim: Optional[int] = None
    precisions: List[str] = field(default_factory=list)  # e.g., ["fp32","fp16","int8"]
    max_batch_size: Optional[int] = None  # backend hint (if any)
    supports_image_batch: bool = False
    extra: Dict[str, str] = field(default_factory=dict)  # arbitrary key/value pairs

    def as_dict(self) -> Dict[str, object]:
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
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
        model_id: Optional[str] = None,
        device_preference: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Construct a backend with optional hints.

        Args:
            model_name: Logical model architecture name (e.g., "ViT-B-32").
            pretrained: Pretrained tag/weights identifier (e.g., "laion2b_s34b_b79k").
            model_id: Combined unique identifier; if not provided, implementations
                      can derive one from (model_name, pretrained).
            device_preference: Hint for device selection (e.g., "cuda", "mps", "cpu").
            max_batch_size: Hint for batching; implementation may clamp lower/higher.
            cache_dir: Optional writable directory for any cached assets.
        """
        self._initialized: bool = False
        self._model_name = model_name
        self._pretrained = pretrained
        self._model_id = model_id
        self._device_pref = device_preference
        self._max_batch_size = max_batch_size
        self._cache_dir = cache_dir

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
    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Encode a text string to a unit-normalized embedding vector.

        Returns:
            np.ndarray with shape (D,) and dtype float32, L2-normalized to 1.0
        """
        raise NotImplementedError

    @abc.abstractmethod
    def image_to_vector(self, image_bytes: bytes) -> np.ndarray:
        """
        Encode image bytes to a unit-normalized embedding vector.

        Returns:
            np.ndarray with shape (D,) and dtype float32, L2-normalized to 1.0
        """
        raise NotImplementedError

    def image_batch_to_vectors(self, images: Sequence[bytes]) -> np.ndarray:
        """
        Encode a list of image bytes into a batch of unit-normalized vectors.

        Defaults to a sequential fallback using image_to_vector(). Backends that
        support true batched execution should override for better throughput.

        Returns:
            np.ndarray with shape (N, D) and dtype float32, each row L2-normalized.
        """
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        vecs = [self.image_to_vector(img) for img in images]
        # Validate consistent dims
        dim = vecs[0].shape[0]
        out = np.stack(vecs, axis=0).astype(np.float32, copy=False)
        if out.ndim != 2 or out.shape[1] != dim:
            raise RuntimeError("image_batch_to_vectors: inconsistent vector dimensions.")
        return out

    # ---------- Metadata ----------

    @abc.abstractmethod
    def get_info(self) -> BackendInfo:
        """
        Return a BackendInfo describing runtime, device, model identifiers,
        embedding dimensions, precision supports, batchability, etc.
        """
        raise NotImplementedError

    # ---------- Utilities (reusable across backends/managers) ----------

    @staticmethod
    def unit_normalize(vec: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
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

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable softmax for numpy arrays.
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    @staticmethod
    def classify_with_text_embeddings(
        image_vec: np.ndarray,
        text_embeddings: np.ndarray,
        labels: Sequence[str],
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """
        Compute label probabilities from an image vector and a matrix of label text embeddings.

        Args:
            image_vec: Shape (D,), unit-normalized
            text_embeddings: Shape (N, D), unit-normalized rows
            labels: Sequence of N labels corresponding to the rows of text_embeddings
            top_k: Number of predictions to return (clamped to N)
            temperature: Optional scaling on logits (e.g., 100.0 for CLIP-like sharpening)

        Returns:
            List of (label, probability) tuples, length top_k
        """
        if text_embeddings.ndim != 2 or image_vec.ndim != 1:
            raise ValueError("Invalid shapes: expected image_vec (D,), text_embeddings (N, D).")

        # Ensure float32 and unit-normalized
        img = BaseClipBackend.unit_normalize(image_vec.astype(np.float32, copy=False))
        txt = BaseClipBackend.unit_normalize(text_embeddings.astype(np.float32, copy=False), axis=1)

        # Cosine similarities -> logits
        logits = np.dot(img, txt.T)  # shape (N,)
        if temperature != 1.0:
            logits = logits * float(temperature)

        probs = BaseClipBackend.softmax(logits, axis=0)  # shape (N,)
        n = probs.shape[0]
        k = int(max(1, min(top_k, n)))

        # Arg-partition top-k
        idxs = np.argpartition(-probs, k - 1)[:k]
        # Sort selected indices by descending prob
        idxs = idxs[np.argsort(-probs[idxs])]
        return [(str(labels[i]), float(probs[i])) for i in idxs]
