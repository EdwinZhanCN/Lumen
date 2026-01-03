"""
OCR Model Manager

This module provides the high-level manager for OCR models, handling
initialization, resource loading, and inference delegation to backends.
It follows the standard Lumen architecture for model management.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lumen_resources.lumen_config import ModelConfig

from ..backends.backend_exceptions import (
    BackendNotInitializedError,
    ModelLoadingError,
)
from ..backends.base import BackendInfo, BaseOcrBackend, OcrResult
from ..backends.onnxrt_backend import OnnxOcrBackend
from ..resources.loader import ModelResources, ResourceLoader

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Type-safe model information for OCR models.

    Provides a consistent interface for accessing model metadata and status.
    """

    model_name: str
    model_id: str
    is_initialized: bool
    backend_info: Optional[BackendInfo] = None
    extra_metadata: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "is_initialized": self.is_initialized,
            "backend_info": self.backend_info.as_dict() if self.backend_info else None,
            "extra_metadata": self.extra_metadata,
        }


class OcrModelManager:
    """
    Manager for OCR models.

    This class orchestrates the lifecycle of OCR models, including:
    - Loading resources via ResourceLoader
    - Instantiating the appropriate backend (ONNX, etc.)
    - Delegating inference requests
    - Managing model state and metadata
    """

    def __init__(
        self,
        config: ModelConfig,
        cache_dir: str,
        providers: list[str] | None = None,
        device_preference: Optional[str] = None,
    ):
        """
        Initialize the manager.

        Args:
            config: Model configuration from lumen_config.yaml.
            cache_dir: Directory where model files are stored.
            device_preference: Optional device hint (e.g., "cuda", "cpu").
        """
        self.config = config
        self.providers = providers
        self.cache_dir = cache_dir
        self.device_preference = device_preference

        self._backend: BaseOcrBackend
        self._resources: ModelResources
        self._initialized = False

    def initialize(self) -> None:
        """
        Load resources and initialize the backend.

        This method is idempotent. It loads the necessary model files using
        ResourceLoader and sets up the inference backend.

        Raises:
            ModelLoadingError: If initialization fails.
        """
        if self._initialized:
            return

        logger.info(f"Initializing OCR model: {self.config.model}")

        try:
            # 1. Load Resources
            self._resources = ResourceLoader.load_model_resource(
                self.cache_dir, self.config
            )

            # 2. Instantiate Backend based on runtime
            runtime = self.config.runtime.value
            if runtime == "onnx":
                # Determine precision preference from ModelConfig
                prefer_fp16 = (
                    self.config.precision in ["fp16", "q4fp16"]
                    if self.config.precision
                    else False
                )

                self._backend = OnnxOcrBackend(
                    resources=self._resources,
                    providers=self.providers,
                    device_preference=self.device_preference,
                    prefer_fp16=prefer_fp16,
                )
            # Future support for other runtimes (e.g., rknn, torch) can be added here
            else:
                raise NotImplementedError(
                    f"Runtime '{runtime}' is not supported by OcrModelManager yet."
                )

            # 3. Initialize Backend
            self._backend.initialize()
            self._initialized = True

            logger.info(f"Successfully initialized OCR model: {self.config.model}")

        except Exception as e:
            logger.error(f"Failed to initialize OCR model: {e}")
            raise ModelLoadingError(f"Initialization failed: {e}") from e

    def predict(
        self,
        image_bytes: bytes,
        det_threshold: float = 0.3,
        rec_threshold: float = 0.5,
        use_angle_cls: bool = False,
        **kwargs: Any,
    ) -> List[OcrResult]:
        """
        Perform end-to-end OCR on the input image.

        Args:
            image_bytes: Raw image data.
            det_threshold: Confidence threshold for text detection.
            rec_threshold: Confidence threshold for text recognition.
            use_angle_cls: Whether to use angle classification.
            **kwargs: Additional backend-specific parameters.

        Returns:
            List[OcrResult]: Detected text regions and content.

        Raises:
            BackendNotInitializedError: If called before initialize().
            InferenceError: If inference fails.
        """
        if not self._initialized or not self._backend:
            raise BackendNotInitializedError("Model manager not initialized")

        return self._backend.predict(
            image_bytes=image_bytes,
            det_threshold=det_threshold,
            rec_threshold=rec_threshold,
            use_angle_cls=use_angle_cls,
            **kwargs,
        )

    def get_info(self) -> ModelInfo:
        """Get current model information and status."""
        backend_info = self._backend.get_info()
        # Extract extra metadata from resources if available
        extra = None
        if self._resources and self._resources.model_info.extra_metadata:
            extra = self._resources.model_info.extra_metadata

        return ModelInfo(
            model_name=self.config.model,
            model_id=f"{self.config.model}_{backend_info.runtime}",
            is_initialized=self._initialized,
            backend_info=backend_info,
            extra_metadata=extra,
        )

    def __repr__(self) -> str:
        return (
            f"<OcrModelManager(model={self.config.model}, "
            f"runtime={self.config.runtime.value}, init={self._initialized})>"
        )
