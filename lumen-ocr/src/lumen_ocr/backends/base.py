"""
Base Backend for OCR Service

This module defines the abstract base class for OCR backends, following Lumen's
development architecture. It provides standard interfaces for text detection
and recognition pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .backend_exceptions import (
    BackendNotInitializedError,
    InvalidInputError,
)


@dataclass
class BackendInfo:
    """Runtime configuration and model metadata for OCR backends.

    Attributes:
        runtime: Runtime framework name (e.g., "onnx", "paddle", "tensorrt").
        device: Target device identifier.
        det_model_id: Identifier for the detection model.
        rec_model_id: Identifier for the recognition model.
        cls_model_id: Identifier for the classification model (optional).
        model_name: Human-readable combined model name.
        version: Backend or model version string.
        precisions: Supported numerical precisions.
        max_batch_size: Maximum batch size supported.
        extra: Additional metadata.
    """

    runtime: str
    device: str | None = None
    det_model_id: str | None = None
    rec_model_id: str | None = None
    cls_model_id: str | None = None
    model_name: str | None = None
    version: str | None = None
    precisions: list[str] = field(default_factory=list)
    max_batch_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Convert to a plain dict (safe for JSON serialization)."""
        return {
            "runtime": self.runtime,
            "device": self.device,
            "det_model_id": self.det_model_id,
            "rec_model_id": self.rec_model_id,
            "cls_model_id": self.cls_model_id,
            "model_name": self.model_name,
            "version": self.version,
            "precisions": list(self.precisions),
            "max_batch_size": self.max_batch_size,
            "extra": self.extra,
        }


@dataclass
class OcrResult:
    """Single text detection and recognition result.

    Attributes:
        box: List of (x, y) coordinates defining the text bounding polygon.
             Typically 4 points for a rotated rectangle: [TL, TR, BR, BL].
        text: The recognized text string.
        confidence: The confidence score of the recognition (0.0 to 1.0).
    """

    box: list[tuple[int, int]]
    text: str
    confidence: float


class BaseOcrBackend(ABC):
    """Abstract base class defining the OCR backend interface.

    This class establishes a contract for OCR backends, which typically consist
    of a text detection model and a text recognition model (and optionally
    an angle classification model).
    """

    def __init__(self):
        """Initialize the backend in an uninitialized state."""
        self._initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend with model loading and device setup.

        This method must be called before any inference operations.
        """
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if the backend is properly initialized."""
        return self._initialized

    @abstractmethod
    def get_info(self) -> BackendInfo:
        """Get comprehensive runtime information about the backend configuration."""
        pass

    @abstractmethod
    def predict(
        self,
        image_bytes: bytes,
        det_threshold: float = 0.3,
        rec_threshold: float = 0.5,
        use_angle_cls: bool = False,
        **kwargs: Any,
    ) -> list[OcrResult]:
        """Perform end-to-end OCR on the input image.

        Args:
            image_bytes: Raw image bytes.
            det_threshold: Confidence threshold for text detection.
            rec_threshold: Confidence threshold for text recognition.
            use_angle_cls: Whether to use angle classification if available.
            **kwargs: Backend-specific parameters (e.g., DBNet parameters like
                      box_thresh, unclip_ratio).

        Returns:
            list[OcrResult]: List of detected text regions with content and confidence.

        Raises:
            BackendNotInitializedError: If backend is not initialized.
            InvalidInputError: If input image is invalid.
            InferenceError: If inference fails.
        """
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")
