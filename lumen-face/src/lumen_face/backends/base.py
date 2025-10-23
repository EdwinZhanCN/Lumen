"""
Base Backend for Face Recognition Service

This module defines the abstract base class for face recognition backends,
following Lumen's development architecture. It provides standard interfaces
for face detection and embedding extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .backend_exceptions import (
    BackendNotInitializedError,
    InvalidInputError,
)


@dataclass
class BackendInfo:
    """
    Describes the active backend configuration and model metadata for face recognition.

    All fields are optional except `runtime`. Populate as much as possible so the
    gRPC capability message can surface accurate details.
    """

    runtime: str
    device: str | None = None  # e.g., "cuda:0", "mps", "cpu", "rk3588-npu"
    model_id: str | None = None  # a stable identifier (e.g., "retinaface_resnet50")
    model_name: str | None = None  # human-friendly name ("RetinaFace")
    pretrained: str | None = None  # e.g., "resnet50"
    version: str | None = None  # backend or model version string
    face_embedding_dim: int | None = None  # e.g., 512 for face embeddings
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
            "face_embedding_dim": self.face_embedding_dim,
            "precisions": list(self.precisions),
            "max_batch_size": self.max_batch_size,
            "supports_image_batch": self.supports_image_batch,
            "extra": dict(self.extra),
        }


@dataclass
class FaceDetection:
    """
    Data structure for face detection results.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        landmarks: Optional facial landmarks (5 or 68 points).
        confidence: Detection confidence score.
    """

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    landmarks: list[tuple[float, float]] | None = None  # 5/68关键点
    confidence: float = 0.0


class FaceRecognitionBackend(ABC):
    """
    Abstract base class for face recognition backends.

    This class defines the standard interface for face detection and embedding
    extraction, ensuring consistency across different runtime implementations.
    """

    def __init__(self):
        self._initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the backend with model loading and device setup.

        Raises:
            ModelLoadingError: If model loading fails.
            DeviceUnavailableError: If requested device is not available.
        """
        self._initialized = True

    def is_initialized(self) -> bool:
        """
        Check if the backend is properly initialized.

        Returns:
            bool: True if initialized, False otherwise.
        """
        return self._initialized

    @abstractmethod
    def get_runtime_info(self) -> BackendInfo:
        """
        Get runtime information including device, precision, and other details.

        Returns:
            BackendInfo: Runtime information.
        """
        pass

    @abstractmethod
    def image_to_faces(self, image_bytes: bytes) -> list[FaceDetection]:
        """
        Detect faces in the input image bytes.

        Applies confidence threshold filtering and NMS deduplication as per contract.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            list[FaceDetection]: List of detected faces.

        Raises:
            BackendNotInitializedError: If backend is not initialized.
            InvalidInputError: If input is invalid.
            InferenceError: If inference fails.
        """
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")
        pass

    @abstractmethod
    def face_to_embedding(
        self,
        face_image: bytes | None = None,
        cropped_face_array: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
        | None = None,
        landmarks: list[tuple[float, float]] | None = None,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        """
        Extract face embedding from face image or cropped array.

        Returns a unit-normalized 512-dimensional vector suitable for 1:1 matching.

        Args:
            face_image: Raw face image bytes (alternative to cropped_face_array).
            cropped_face_array: Pre-cropped face array (alternative to face_image).
            landmarks: Optional landmarks for alignment.

        Returns:
            np.ndarray: Unit-normalized embedding vector of shape (512,).

        Raises:
            BackendNotInitializedError: If backend is not initialized.
            InvalidInputError: If neither face_image nor cropped_face_array is provided.
            InferenceError: If inference fails.
        """
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")
        if face_image is None and cropped_face_array is None:
            raise InvalidInputError(
                "Either face_image or cropped_face_array must be provided"
            )
        pass
