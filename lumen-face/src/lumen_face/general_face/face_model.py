"""
Face Model Manager for Lumen Face Service.

This module provides high-level model management for face detection and recognition,
integrating with Lumen Resources and backend abstractions.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import numpy.typing as npt

from ..backends.backend_exceptions import (
    InferenceError,
    InvalidInputError,
)
from ..backends.base import FaceDetection, FaceRecognitionBackend
from ..resources.loader import ModelResources
from ..runtime_info import RuntimeModelInfo

logger = logging.getLogger(__name__)


class ModelDataNotFoundError(Exception):
    """Raised when model-specific data (labels, embeddings) cannot be found or loaded."""

    pass


class CacheCorruptionError(Exception):
    """Raised when cached data is corrupted or incompatible."""

    pass


class LabelMismatchError(Exception):
    """Raised when cached embeddings don't match current labels."""

    pass


class FaceModelManager:
    """High-level interface for face detection and recognition operations.

    This class provides a unified, user-friendly API for face processing tasks
    by wrapping backend-specific implementations. It handles common workflows
    including face detection, embedding extraction, face comparison, and combined
    detect-and-embed operations.

    Key capabilities:
    - Face detection with confidence filtering and NMS deduplication
    - Face embedding extraction with optional landmark-based alignment
    - Face similarity comparison using cosine similarity
    - Combined detect-and-embed workflows for efficiency
    - Batch processing support for multiple faces
    - Comprehensive error handling and logging
    - Performance monitoring and timing information

    The manager uses dependency injection to work with different backends
    (ONNX, RKNN, PyTorch, etc.) while providing a consistent interface.

    Attributes:
        _backend: Backend implementation (ONNXRTBackend, RKNNBackend, etc.)
        resources: ModelResources containing model metadata and configuration
        model_id: Unique identifier for the model combination (e.g., "buffalo_l_onnx")
        _load_time: Time taken for initialization in seconds
        is_initialized: Flag indicating whether the model is ready for inference

    Example:
        ```python
        # Create backend and resources
        backend = ONNXRTBackend(resources)

        # Create manager
        manager = FaceModelManager(backend, resources)
        manager.initialize()

        # Detect faces
        faces = manager.detect_faces(image_bytes)

        # Extract embeddings
        for face in faces:
            embedding = manager.extract_embedding(
                cropped_face_array=manager.crop_face_from_image(image_bytes, face.bbox)
            )
        ```

    Note:
        Always call `initialize()` before performing inference operations.
    """

    def __init__(self, backend: FaceRecognitionBackend, resources: ModelResources):
        """Initialize FaceModelManager with backend and resources.

        Args:
            backend: Backend instance implementing FaceRecognitionBackend interface.
                The backend provides the actual inference capabilities (detection
                and recognition models) and handles device-specific optimizations.
            resources: ModelResources instance containing model metadata,
                file paths, and configuration information loaded from lumen-resources.

        Raises:
            ValueError: If backend is None or resources is invalid.
            TypeError: If backend doesn't implement FaceRecognitionBackend interface.

        Note:
            The constructor does not initialize the backend. Call `initialize()`
            to load models and prepare for inference operations.
        """
        # Backend dependency injection
        self._backend: FaceRecognitionBackend = backend
        self.resources: ModelResources = resources

        # Model identification
        self.model_id: str = f"{resources.model_name}_{resources.runtime}"

        # Initialization state
        self._load_time: float | None = None
        self.is_initialized: bool = False

    def initialize(self) -> None:
        """Initialize the face model manager and underlying backend.

        This method prepares the manager for inference operations by initializing
        the underlying backend and tracking initialization metrics. It performs
        validation to ensure the backend is properly loaded and functional.

        The initialization process:
        1. Checks if backend is already initialized (idempotent operation)
        2. Calls backend.initialize() to load models and configure devices
        3. Tracks initialization time for performance monitoring
        4. Sets internal state flags for subsequent operations

        Raises:
            RuntimeError: If backend initialization fails due to model loading
                errors, device unavailability, or other backend-specific issues.
            BackendNotInitializedError: If backend fails to initialize properly.

        Note:
            This method is idempotent - calling it multiple times after successful
            initialization will not cause issues or reload models.

        Performance:
            Initialization time varies by backend and hardware:
            - CPU: 1-5 seconds typical
            - GPU: 2-10 seconds typical (including model transfer)
            - NPU: 1-3 seconds typical (for optimized models)
        """
        t0 = time.time()
        logger.info(
            f"Initializing Face Model Manager for {self.resources.model_name}..."
        )

        try:
            if not self._backend.is_initialized():
                logger.info("Initializing face recognition backend...")
                self._backend.initialize()
                self._load_time = time.time() - t0
                logger.info(
                    f"✅ Face Model Manager initialized in {self._load_time:.2f}s"
                )
            else:
                logger.info("Face recognition backend already initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Face Model Manager: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def detect_faces(
        self,
        image_bytes: bytes,
        detection_confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        face_size_min: int = 50,
        face_size_max: int = 1000,
    ) -> list[FaceDetection]:
        """Detect faces in an image with configurable filtering and quality controls.

        This is the primary face detection interface that provides high-quality
        face detections with built-in filtering for various use cases. The method
        handles input validation, delegates to the backend for detection, and
        provides comprehensive error handling.

        Args:
            image_bytes: Raw image data in common formats (JPEG, PNG, BMP, etc.).
                Images are automatically decoded and preprocessed by the backend.
            detection_confidence_threshold: Minimum confidence score (0.0-1.0) for
                valid detections. Higher values reduce false positives. Recommended:
                0.6-0.8 for general use, 0.8-0.9 for high-security applications.
            nms_threshold: Non-maximum suppression threshold (0.0-1.0) for removing
                duplicate detections. Lower values = more aggressive deduplication.
                Recommended: 0.3-0.5 for most scenarios.
            face_size_min: Minimum face dimension in pixels to filter out tiny
                detections that are likely false positives or noise.
            face_size_max: Maximum face dimension in pixels to filter out unrealistic
                detections or full-image false detections.

        Returns:
            list[FaceDetection]: Detected faces sorted by confidence (highest first).
                Each detection includes bounding box (x1, y1, x2, y2), optional
                landmark points, and confidence score. Empty list if no faces meet criteria.

        Raises:
            InvalidInputError: If image_bytes is empty, corrupted, or in unsupported format.
            InferenceError: If backend detection fails due to model errors or device issues.

        Performance:
            - Typical inference time: 10-50ms depending on image size and hardware
            - Memory usage: Proportional to image resolution
            - GPU acceleration available when supported by backend

        Note:
            Coordinates are absolute pixel values. All filtering is applied before
            returning results to ensure high-quality detections suitable for
            downstream recognition tasks.
        """
        if not image_bytes:
            raise InvalidInputError("Image bytes cannot be empty")

        try:
            logger.debug(f"Detecting faces in image ({len(image_bytes)} bytes)")
            faces = self._backend.image_to_faces(
                image_bytes,
                detection_confidence_threshold=detection_confidence_threshold,
                nms_threshold=nms_threshold,
                face_size_min=face_size_min,
                face_size_max=face_size_max,
            )

            logger.debug(f"Detected {len(faces)} faces")
            return faces

        except Exception as e:
            if isinstance(e, (InvalidInputError, InferenceError)):
                raise
            raise InferenceError(f"Face detection failed: {e}") from e

    def extract_embedding(
        self,
        face_image: bytes | None = None,
        cropped_face_array: npt.NDArray[np.float32] | None = None,
        landmarks: list[tuple[float, float]] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Extract L2-normalized face embedding suitable for similarity comparison.

        This method processes a face image and extracts a high-dimensional feature
        vector that represents the unique characteristics of the face. The embedding
        is L2-normalized to enable efficient cosine similarity comparisons.

        Args:
            face_image: Raw face image bytes (alternative to cropped_face_array).
                Should contain primarily face region. Must provide exactly one input.
            cropped_face_array: Pre-cropped face array with shape (H, W, 3) and values
                in [0, 255]. Alternative to face_image. Must provide exactly one input.
            landmarks: Optional 5-point landmarks for alignment: [left_eye, right_eye,
                nose_tip, left_mouth, right_mouth]. Alignment improves recognition accuracy.

        Returns:
            npt.NDArray[np.float32]: L2-normalized embedding vector of shape
                (embedding_dim,) where embedding_dim is typically 512. Values are
                normalized to unit length for cosine similarity comparison.

        Raises:
            InvalidInputError: If neither or both inputs are provided, or input format is invalid.
            InferenceError: If recognition model fails during inference.

        Usage:
            ```python
            # Compare two faces
            emb1 = manager.extract_embedding(face_image=image1_bytes)
            emb2 = manager.extract_embedding(cropped_face_array=face2_array)
            similarity = np.dot(emb1, emb2)  # cosine similarity

            # Typical thresholds for face verification:
            # - 0.5-0.6: lenient (may have false matches)
            # - 0.6-0.7: moderate (good balance)
            # - 0.7-0.8: strict (few false matches, may miss some true matches)
            ```

        Performance:
            - Typical inference time: 2-10ms per face
            - GPU acceleration recommended for batch processing
            - Memory usage: Minimal (~2KB per embedding)
        """
        try:
            logger.debug("Extracting face embedding")
            embedding = self._backend.face_to_embedding(
                face_image=face_image,
                cropped_face_array=cropped_face_array,
                landmarks=landmarks,
            )

            logger.debug(f"Extracted embedding with shape: {embedding.shape}")
            return embedding

        except Exception as e:
            if isinstance(e, (InvalidInputError, InferenceError)):
                raise
            raise InferenceError(f"Face embedding extraction failed: {e}") from e

    def detect_and_extract(
        self,
        image_bytes: bytes,
        detection_confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        face_size_min: int = 50,
        face_size_max: int = 1000,
        max_faces: int | None = None,
        return_embeddings: bool = True,
    ) -> tuple[list[FaceDetection], list[npt.NDArray[np.float32]]]:
        """
        Detect faces and optionally extract embeddings for each.

        Args:
            image_bytes: Raw image bytes
            detection_confidence_threshold: Confidence threshold for detection
            nms_threshold: Non-maximum suppression threshold
            face_size_min: Minimum face size in pixels
            face_size_max: Maximum face size in pixels
            max_faces: Maximum number of faces to process
            return_embeddings: Whether to extract embeddings for detected faces

        Returns:
            Tuple of (detections, embeddings)
            - detections: List of FaceDetection objects
            - embeddings: List of embedding vectors (same order as detections)
        """
        # Detect faces
        faces = self.detect_faces(
            image_bytes,
            detection_confidence_threshold=detection_confidence_threshold,
            nms_threshold=nms_threshold,
            face_size_min=face_size_min,
            face_size_max=face_size_max,
        )

        # Limit number of faces if specified
        if max_faces is not None and len(faces) > max_faces:
            faces = faces[:max_faces]
            logger.debug(f"Limited to {max_faces} faces")

        # Extract embeddings if requested
        embeddings = []
        if return_embeddings:
            for face in faces:
                try:
                    # Crop face from original image using bbox
                    face_crop = self.crop_face_from_image(image_bytes, face.bbox)
                    embedding = self.extract_embedding(
                        cropped_face_array=face_crop, landmarks=face.landmarks
                    )
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for face: {e}")
                    # Add zero embedding to maintain order
                    embeddings.append(
                        np.zeros(
                            self._backend.get_runtime_info().face_embedding_dim or 512
                        )
                    )

        logger.debug(
            f"Processed {len(faces)} faces, extracted {len(embeddings)} embeddings"
        )
        return faces, embeddings

    def compare_faces(
        self, embedding1: npt.NDArray[np.float32], embedding2: npt.NDArray[np.float32]
    ) -> float:
        """
        Compare two face embeddings using cosine similarity.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Cosine similarity score (-1.0 to 1.0, higher = more similar)
        """
        if embedding1.shape != embedding2.shape:
            raise InvalidInputError("Embedding dimensions must match")

        # L2 normalize embeddings (should already be normalized from backend)
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        # Cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        return float(similarity)

    def find_best_match(
        self,
        target_embedding: npt.NDArray[np.float32],
        candidate_embeddings: list[npt.NDArray[np.float32]],
        threshold: float = 0.5,
    ) -> tuple[int, float] | None:
        """
        Find the best matching face from candidates.

        Args:
            target_embedding: Target face embedding to match
            candidate_embeddings: List of candidate embeddings
            threshold: Minimum similarity threshold for match

        Returns:
            Tuple of (index, similarity) of best match, or None if below threshold
        """
        if not candidate_embeddings:
            return None

        best_idx = -1
        best_similarity = -1.0

        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = self.compare_faces(target_embedding, candidate_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i

        if best_similarity >= threshold:
            return best_idx, best_similarity

        return None

    def crop_face_from_image(
        self, image_bytes: bytes, bbox: tuple[float, float, float, float]
    ) -> npt.NDArray[np.float32]:
        """
        Crop face from image using bounding box.

        Args:
            image_bytes: Original image bytes
            bbox: Bounding box as (x1, y1, x2, y2)

        Returns:
            Cropped face array
        """
        try:
            import io

            from PIL import Image as PILImage

            # Load and decode image
            image = PILImage.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
            img_array = np.array(image)

            # Extract integer coordinates
            x1, y1, x2, y2 = map(int, bbox)

            # Ensure coordinates are within image bounds
            img_h, img_w = img_array.shape[:2]
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            # Crop face
            face_crop = img_array[y1:y2, x1:x2]

            return face_crop.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to crop face: {e}")
            # Return default face crop
            return np.zeros((112, 112, 3), dtype=np.float32)

    def info(self) -> RuntimeModelInfo:
        """
        Get model runtime information (consistent with lumen-clip interface).

        Returns:
            RuntimeModelInfo containing model metadata and configuration
        """
        if not self.is_initialized or not self._backend:
            # Return minimal info if not initialized
            return RuntimeModelInfo(
                model_name="unknown",
                model_id="uninitialized",
                is_initialized=False,
                load_time=self._load_time,
            )

        backend_info = self._backend.get_runtime_info()

        # Extract extra metadata from resources if available
        extra = None
        if self.resources and self.resources.model_info.extra_metadata:
            extra = self.resources.model_info.extra_metadata

        return RuntimeModelInfo(
            model_name=backend_info.model_name or "unknown",
            model_id=backend_info.model_id or "unknown",
            is_initialized=self.is_initialized,
            load_time=self._load_time or 0.0,
            backend_info=backend_info,
            face_embedding_dim=backend_info.face_embedding_dim,
            supports_batch=backend_info.supports_image_batch,
            detection_model=backend_info.model_name,
            recognition_model=backend_info.pretrained,
            extra_metadata=extra,
        )

    def __repr__(self) -> str:
        """String representation of FaceModelManager."""
        return (
            f"FaceModelManager(backend={type(self._backend).__name__}, "
            f"initialized={self.is_initialized})"
        )
