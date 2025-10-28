"""
Base Backend for Face Recognition Service

This module defines the abstract base class for face recognition backends,
following Lumen's development architecture. It provides standard interfaces
for face detection and embedding extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .backend_exceptions import (
    BackendNotInitializedError,
    InvalidInputError,
)


@dataclass
class BackendInfo:
    """Runtime configuration and model metadata for face recognition backends.

    This dataclass captures comprehensive information about the backend runtime,
    model configuration, and capabilities. It's used to populate gRPC capability
    messages and provide clients with detailed service information.

    Attributes:
        runtime: Runtime framework name (e.g., "onnx", "rknn", "torch").
        device: Target device identifier (e.g., "cuda:0", "mps", "cpu", "rk3588-npu").
        model_id: Stable model identifier for reference and caching.
        model_name: Human-readable model name (e.g., "RetinaFace", "ArcFace").
        pretrained: Pretrained dataset or method name (e.g., "resnet50", "buffalo_l").
        version: Backend or model version string for compatibility tracking.
        face_embedding_dim: Dimension of face embedding vectors (typically 512).
        precisions: Supported numerical precisions (e.g., ["fp32", "fp16", "int8"]).
        max_batch_size: Maximum batch size supported by the backend (1 for most face models).
        supports_image_batch: Whether the backend can process multiple images simultaneously.
        extra: Additional metadata as key-value pairs for extensibility.

    Note:
        Only the `runtime` field is required. All other fields should be populated
        as comprehensively as possible to enable accurate capability reporting.
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
    """Face detection result containing bounding box, landmarks, and confidence.

    This dataclass encapsulates the output of face detection algorithms, providing
    all necessary information for subsequent face recognition or processing steps.

    Attributes:
        bbox: Bounding box coordinates as (x1, y1, x2, y2) where (x1, y1) is the
            top-left corner and (x2, y2) is the bottom-right corner. Coordinates
            are in absolute pixel values relative to the original image.
        landmarks: Optional facial landmark points. Can contain 5-point landmarks
            (typical for buffalo_l: left eye, right eye, nose tip, left mouth corner,
            right mouth corner) or 68-point landmarks for more detailed analysis.
            Each point is expressed as (x, y) pixel coordinates.
        confidence: Detection confidence score ranging from 0.0 to 1.0, where higher
            values indicate greater confidence in the detection quality. Typical
            threshold values range from 0.5 to 0.9 depending on use case requirements.

    Note:
        The confidence score should be used to filter low-quality detections before
        passing to face recognition models to improve overall system accuracy.
    """

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    landmarks: list[tuple[float, float]] | None = None  # 5/68关键点
    confidence: float = 0.0


class FaceRecognitionBackend(ABC):
    """Abstract base class defining the face recognition backend interface.

    This class establishes a contract for all face recognition backends, ensuring
    consistent behavior across different runtime implementations (ONNX, RKNN,
    PyTorch, etc.). It provides the core methods for face detection and embedding
    extraction that all concrete backends must implement.

    The backend follows the Lumen architecture pattern where:
    1. Detection identifies faces in images with bounding boxes and landmarks
    2. Recognition extracts normalized embeddings for face matching
    3. Both operations support configurable parameters for different use cases

    Attributes:
        _initialized: Flag indicating whether the backend has been successfully
            initialized with loaded models and configured devices.

    Note:
        This is an abstract class - concrete implementations must inherit from it
        and implement all abstract methods. The initialization state tracking ensures
        that backend methods are not called before proper model loading.
    """

    def __init__(self):
        """Initialize the backend in an uninitialized state.

        Concrete backends must call `initialize()` to load models and complete
        the initialization process before performing inference operations.
        """
        self._initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend with model loading and device setup.

        This method must be called before any inference operations. It handles:
        - Loading face detection and recognition models from disk
        - Configuring compute devices (CPU, GPU, NPU, etc.)
        - Setting up runtime-specific optimizations
        - Validating model compatibility and capabilities

        Raises:
            ModelLoadingError: If model loading fails due to missing files,
                corrupted models, or incompatible formats.
            DeviceUnavailableError: If the requested device is not available
                or cannot be initialized by the runtime.

        Note:
            This method should be idempotent - calling it multiple times should
            not cause issues after successful initialization.
        """
        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the backend is properly initialized and ready for inference.

        Returns:
            bool: True if the backend has been successfully initialized with
                loaded models and configured devices, False otherwise.

        Note:
            This method should be called before inference operations to avoid
            BackendNotInitializedError exceptions.
        """
        return self._initialized

    @abstractmethod
    def get_runtime_info(self) -> BackendInfo:
        """Get comprehensive runtime information about the backend configuration.

        This method provides detailed information about the backend's current state,
        including device configuration, model details, supported capabilities, and
        performance characteristics. This information is used for:

        - gRPC capability reporting to clients
        - Debugging and troubleshooting
        - Performance optimization decisions
        - Compatibility checks

        Returns:
            BackendInfo: Comprehensive runtime information including device settings,
                model identifiers, supported precisions, and capability flags.

        Note:
            The returned BackendInfo should be as complete as possible to enable
            accurate capability reporting to clients and effective debugging.
        """
        pass

    @abstractmethod
    def image_to_faces(
        self,
        image_bytes: bytes,
        detection_confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        face_size_min: int = 50,
        face_size_max: int = 1000,
    ) -> list[FaceDetection]:
        """Detect faces in the input image with configurable filtering.

        This method performs face detection on the provided image bytes and returns
        a list of detected faces with bounding boxes, landmarks, and confidence scores.
        The detection pipeline includes:

        1. Image preprocessing and normalization
        2. Face detection using the backend's detection model
        3. Confidence threshold filtering
        4. Non-maximum suppression (NMS) to remove duplicate detections
        5. Size-based filtering to remove inappropriately sized faces

        Args:
            image_bytes: Raw image bytes in common formats (JPEG, PNG, BMP, etc.).
                The image should be decoded automatically by the backend.
            detection_confidence_threshold: Minimum confidence score (0.0-1.0) for
                a detection to be considered valid. Typical values: 0.5-0.9.
                Higher values reduce false positives but may miss some faces.
            nms_threshold: Non-maximum suppression threshold (0.0-1.0) for removing
                overlapping detections. Lower values are more aggressive in
                removing duplicates. Typical values: 0.3-0.5.
            face_size_min: Minimum face size in pixels (both width and height).
                Detections smaller than this are filtered out. Useful for removing
                false detections on small regions.
            face_size_max: Maximum face size in pixels (both width and height).
                Detections larger than this are filtered out. Useful for removing
                unrealistic detections that span most of the image.

        Returns:
            list[FaceDetection]: List of detected faces sorted by confidence
                (highest first). Each detection includes bounding box coordinates,
                optional landmark points, and confidence score. Empty list if no
                faces meet the filtering criteria.

        Raises:
            BackendNotInitializedError: If the backend has not been initialized.
            InvalidInputError: If image_bytes is empty, corrupted, or in an
                unsupported format.
            InferenceError: If the detection model fails during inference.

        Note:
            The coordinates are in absolute pixel values relative to the original
            image dimensions. The method should handle different input image formats
            and aspect ratios automatically.
        """
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")

    @abstractmethod
    def face_to_embedding(
        self,
        face_image: bytes | None = None,
        cropped_face_array: npt.NDArray[np.float32] | None = None,
        landmarks: list[tuple[float, float]] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Extract a normalized face embedding from face image data.

        This method processes a face image (either as raw bytes or pre-cropped array)
        and extracts a high-dimensional feature embedding that can be used for face
        recognition, verification, and clustering tasks. The embedding is L2-normalized
        to enable efficient cosine similarity comparisons.

        The embedding extraction pipeline includes:
        1. Optional face alignment using provided landmarks
        2. Image preprocessing and normalization
        3. Feature extraction using the recognition model
        4. L2 normalization of the output vector

        Args:
            face_image: Raw face image bytes in common formats (JPEG, PNG, etc.).
                Must contain a primarily cropped face image. Alternative to
                cropped_face_array - exactly one must be provided.
            cropped_face_array: Pre-cropped face array as numpy array with shape
                (H, W, 3) and values in [0, 255]. Alternative to face_image - exactly
                one must be provided.
            landmarks: Optional facial landmark points for alignment. If provided,
                the face will be aligned to a canonical pose before feature extraction.
                For buffalo_l models, this should be 5 points in the order:
                [left_eye, right_eye, nose_tip, left_mouth, right_mouth].

        Returns:
            npt.NDArray[np.float32]: L2-normalized embedding vector suitable for
                cosine similarity comparison. Shape is (embedding_dim,) where
                embedding_dim is typically 512 for most face recognition models.
                Values are normalized to unit length (L2 norm = 1.0).

        Raises:
            BackendNotInitializedError: If the backend has not been initialized.
            InvalidInputError: If neither face_image nor cropped_face_array is provided,
                or if both are provided, or if input format is invalid.
            InferenceError: If the recognition model fails during inference.

        Note:
            The L2 normalization enables direct cosine similarity comparisons:
            similarity = np.dot(embedding1, embedding2). Typical similarity thresholds
            for face verification range from 0.5 to 0.7 depending on the model and
            application requirements.
        """
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")
        if face_image is None and cropped_face_array is None:
            raise InvalidInputError(
                "Either face_image or cropped_face_array must be provided"
            )
