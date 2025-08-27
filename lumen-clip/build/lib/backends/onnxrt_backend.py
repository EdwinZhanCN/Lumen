"""
ONNX Runtime backend skeleton for CLIP-like models.

This class implements the BaseClipBackend interface surface and metadata
reporting, but intentionally raises NotImplementedError for the core encoding
methods. It serves as scaffolding for a future full implementation.

Usage notes:
- This skeleton is NOT functional for inference yet.
- Implement initialize(), text_to_vector(), image_to_vector(), and optionally
  image_batch_to_vectors() to wire up actual runtimes.
- get_info() returns runtime-specific metadata to support capability reporting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseClipBackend, BackendInfo
import onnxruntime as ort

__all__ = ["ONNXRTBackend"]


class ONNXRTBackend(BaseClipBackend):
    """
    Skeleton ONNX Runtime backend for CLIP-like models.

    Expected responsibilities for a complete implementation:
    - Load ONNX models for image and text encoders into separate or shared sessions
      (e.g., image_encoder.onnx and text_encoder.onnx) in initialize().
    - Perform preprocessing for image bytes to the expected model input tensor format
      and tokenization for text input, matching the training pipeline.
    - Run inference with onnxruntime.InferenceSession.run(...) and return
      L2-normalized float32 embeddings.

    Args:
        model_name: Logical architecture string (used for metadata).
        pretrained: Pretrained tag or revision (used for metadata).
        model_id: Stable identifier; if None, derived from model_name/pretrained.
        onnx_image_path: Filesystem path to image encoder ONNX model.
        onnx_text_path: Filesystem path to text encoder ONNX model.
        providers: ORT execution providers preference list.
        provider_options: Per-provider configuration list matching providers length.
        device_preference: Optional hint ("cuda", "cpu", "directml").
        max_batch_size: Optional hint for batching.
        cache_dir: Optional cache directory.

    Notes:
        - For CUDA: providers could be ["CUDAExecutionProvider", "CPUExecutionProvider"].
        - For CPU: providers could be ["CPUExecutionProvider"].
        - For DirectML (Windows): ["DmlExecutionProvider", "CPUExecutionProvider"].
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
        model_id: Optional[str] = None,
        onnx_image_path: Optional[str] = None,
        onnx_text_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict[str, Any]]] = None,
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
        self._onnx_image_path = onnx_image_path
        self._onnx_text_path = onnx_text_path
        self._providers = providers or self._default_providers(device_preference)
        self._provider_options = provider_options

        # Sessions to be created in initialize()
        self._sess_image: Optional[Any] = None
        self._sess_text: Optional[Any] = None

        # Derive model_id if missing
        if self._model_id is None:
            base = (self._model_name or "onnx-model")
            pt = (self._pretrained or "default")
            self._model_id = f"{base}_{pt}"

    def initialize(self) -> None:
        """
        Prepare ORT sessions. This skeleton marks the backend initialized but does
        not actually load models. Replace with real session creation:
            sess = ort.InferenceSession(path, sess_options, providers=..., provider_options=...)
        """
        # Example:
        # so = ort.SessionOptions()
        # self._sess_image = ort.InferenceSession(self._onnx_image_path, so, providers=self._providers, provider_options=self._provider_options)
        # self._sess_text = ort.InferenceSession(self._onnx_text_path, so, providers=self._providers, provider_options=self._provider_options)
        self._initialized = True

    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Tokenize and encode text into a unit-normalized float32 vector.
        """
        raise NotImplementedError(
            "ONNXRTBackend.text_to_vector is not implemented. "
            "Implement tokenization and session run to produce a (D,) float32 vector."
        )

    def image_to_vector(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image bytes and encode into a unit-normalized float32 vector.
        """
        raise NotImplementedError(
            "ONNXRTBackend.image_to_vector is not implemented. "
            "Implement image preprocessing and session run to produce a (D,) float32 vector."
        )

    def get_info(self) -> BackendInfo:
        """
        Report ONNX Runtime metadata and configuration hints.
        """
        version = getattr(ort, "__version__", None)
        device = self._infer_device_from_providers(self._providers)
        return BackendInfo(
            runtime="onnxrt",
            device=device,
            model_id=self._model_id,
            model_name=self._model_name,
            pretrained=self._pretrained,
            version=version,
            image_embedding_dim=None,
            text_embedding_dim=None,
            precisions=["fp32"],  # update if using fp16/int8 quantized models
            max_batch_size=self._max_batch_size,
            supports_image_batch=False,  # set True if implementing batched path
            extra={
                "providers": ",".join(self._providers) if self._providers else "",
                "onnx_image_path": str(self._onnx_image_path),
                "onnx_text_path": str(self._onnx_text_path),
            },
        )

    @staticmethod
    def _default_providers(device_pref: Optional[str]) -> List[str]:
        pref = (device_pref or "").lower().strip()
        if pref == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if pref == "directml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        # Default CPU
        return ["CPUExecutionProvider"]

    @staticmethod
    def _infer_device_from_providers(providers: Optional[Sequence[str]]) -> str:
        provs = [p.lower() for p in (providers or [])]
        if any("cuda" in p for p in provs):
            return "cuda"
        if any("dml" in p for p in provs):
            return "directml"
        return "cpu"
