"""
ONNXRTBackend: ONNX Runtime backend for Face Recognition models.

This backend integrates with Lumen Resources architecture:
- Loads models via ModelResources from lumen-resources
- Supports buffalo_l model with detection and recognition components
- Implements face detection with confidence filtering and NMS
- Implements face recognition with alignment support
- Supports configurable execution providers (CPU, CUDA, CoreML, etc.)

The backend follows Lumen's development architecture and uses standardized
model loading and configuration patterns.
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage

from ..resources.loader import ModelResources

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required for ONNXRTBackend. "
        + "Install with: pip install onnxruntime"
    ) from e

from .backend_exceptions import (
    BackendError,
    BackendNotInitializedError,
    InferenceError,
    InvalidInputError,
    ModelLoadingError,
)
from .base import BackendInfo, FaceDetection, FaceRecognitionBackend

logger = logging.getLogger(__name__)


class ONNXRTBackendError(BackendError):
    """Base class for ONNXRTBackend specific errors."""

    pass


class ONNXRTModelLoadingError(ONNXRTBackendError, ModelLoadingError):
    """Raised when ONNX model loading fails."""

    pass


class ONNXRTBackend(FaceRecognitionBackend):
    """ONNX Runtime implementation for face recognition backends.

    This backend provides face detection and recognition capabilities using ONNX Runtime
    as the inference engine. It integrates with Lumen Resources for standardized model
    loading and supports the buffalo_l model combination (RetinaFace + ArcFace).

    The backend automatically selects the best available execution providers and supports:
    - CPU execution (always available)
    - CUDA GPU acceleration (when available)
    - Apple CoreML acceleration (on macOS)
    - DirectML (Windows)
    - OpenVINO (Intel hardware)

    Supported models:
    - Detection: RetinaFace variants with 640x640 input resolution
    - Recognition: ArcFace variants with 112x112 input resolution
    - Output: 512-dimensional L2-normalized face embeddings

    Attributes:
        resources: ModelResources instance containing model metadata and file paths.
        _providers: List of ONNX Runtime execution providers in priority order.
        _sess_detection: ONNX inference session for face detection model.
        _sess_recognition: ONNX inference session for face recognition model.
        _detection_input_size: Input resolution expected by detection model.
        _recognition_input_size: Input resolution expected by recognition model.
        _embedding_dim: Dimension of output face embeddings (typically 512).
        _load_time_seconds: Time taken to initialize both models in seconds.

    Note:
        The backend requires specific model files: 'detection.fp32.onnx' and
        'recognition.fp32.onnx' in the model directory structure defined by
        ModelResources.
    """

    def __init__(
        self,
        resources: ModelResources,
        providers: list[str] | None = None,
        device_preference: str | None = None,
        max_batch_size: int | None = None,
        prefer_fp16: bool = True,
    ) -> None:
        """Initialize ONNXRTBackend with model resources and device configuration.

        Args:
            resources: ModelResources object from lumen_resources containing model
                metadata, file paths, and configuration information.
            providers: Optional list of ONNX Runtime execution providers. If None,
                automatically selects the best available providers based on hardware
                capabilities. Common providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'].
            device_preference: Optional device hint ('cuda', 'cpu', 'coreml', etc.).
                Used to prioritize specific execution providers when multiple are available.
            max_batch_size: Maximum batch size for inference (currently unused,
                retained for future batch processing support).
            prefer_fp16: Preference for FP16 precision when available (currently unused,
                retained for future precision optimization support).

        Raises:
            ValueError: If invalid device preference is provided.
            ImportError: If onnxruntime is not installed.

        Note:
            The constructor does not load models immediately. Call `initialize()` to
            load the detection and recognition models before performing inference.
        """
        super().__init__()

        self.resources = resources

        # Execution providers
        self._providers = providers or self._default_providers(device_preference)

        # Runtime objects
        self._sess_detection: ort.InferenceSession | None = None
        self._sess_recognition: ort.InferenceSession | None = None

        # Model metadata from resources
        self._detection_input_size = (640, 640)  # Default for RetinaFace
        self._recognition_input_size = (112, 112)  # Default for ArcFace
        self._embedding_dim = self.resources.get_embedding_dim() or 512
        self._load_time_seconds: float | None = None

        logger.info(f"ONNXRTBackend initialized for model: {self.resources.model_name}")

    def _default_providers(self, device_pref: str | None) -> list[str]:
        """Select optimal ONNX Runtime providers based on hardware availability.

        This method implements a hardware-aware provider selection strategy that
        prioritizes acceleration providers over CPU execution when available.
        The selection considers both performance and compatibility.

        Provider priority order (highest to lowest performance):
        1. CUDAExecutionProvider - NVIDIA GPU acceleration
        2. CoreMLExecutionProvider - Apple Silicon Neural Engine
        3. DmlExecutionProvider - Windows DirectML acceleration
        4. OpenVINOExecutionProvider - Intel CPU/GPU/VPU acceleration
        5. TensorrtExecutionProvider - NVIDIA TensorRT optimization
        6. CPUExecutionProvider - Fallback CPU execution

        Args:
            device_pref: Optional device hint string ('cuda', 'coreml', 'directml',
                'openvino', 'cpu'). When specified, this provider is moved to the
                highest priority position if available.

        Returns:
            list[str]: Ordered list of available execution providers. Always includes
                at least CPUExecutionProvider as a fallback option.

        Note:
            The method queries ONNX Runtime for available providers and filters
            the priority list accordingly. This ensures compatibility with the
            specific ONNX Runtime build and hardware configuration.
        """
        available = set(ort.get_available_providers())

        priority = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ]

        selected = [p for p in priority if p in available]

        # If device_pref specified, prioritize it if available
        if device_pref:
            pref_provider = {
                "cuda": "CUDAExecutionProvider",
                "coreml": "CoreMLExecutionProvider",
                "directml": "DmlExecutionProvider",
                "openvino": "OpenVINOExecutionProvider",
            }.get(device_pref.lower())
            if pref_provider and pref_provider in available:
                selected.insert(0, selected.pop(selected.index(pref_provider)))

        return selected if selected else ["CPUExecutionProvider"]

    def _infer_device_from_providers(self, providers: list[str]) -> str:
        """Infer device string from provider list."""
        provs = [p.lower() for p in providers]

        if any("cuda" in p for p in provs):
            return "cuda"
        if any("coreml" in p for p in provs):
            return "coreml"
        if any("dml" in p for p in provs):
            return "directml"
        if any("openvino" in p for p in provs):
            return "openvino"

        return "cpu"

    def initialize(self) -> None:
        """Load face detection and recognition models using ModelResources."""
        if self._initialized:
            return

        import time

        t0 = time.time()

        try:
            logger.info(f"Loading models for {self.resources.model_name}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Load face detection model
            detection_model_path = self.resources.get_model_file("detection.fp32.onnx")
            if not detection_model_path.exists():
                # Try other naming patterns
                for pattern in ["detection.onnx", "face_detection.fp32.onnx"]:
                    detection_model_path = self.resources.get_model_file(pattern)
                    if detection_model_path.exists():
                        break
                else:
                    raise ONNXRTModelLoadingError(
                        "Face detection model not found. Expected: detection.fp32.onnx"
                    )

            logger.info(f"Loading face detection model from {detection_model_path}")
            self._sess_detection = ort.InferenceSession(
                str(detection_model_path),
                sess_options,
                providers=self._providers,
            )

            # Load face recognition model
            recognition_model_path = self.resources.get_model_file(
                "recognition.fp32.onnx"
            )
            if not recognition_model_path.exists():
                # Try other naming patterns
                for pattern in ["recognition.onnx", "face_recognition.fp32.onnx"]:
                    recognition_model_path = self.resources.get_model_file(pattern)
                    if recognition_model_path.exists():
                        break
                else:
                    raise ONNXRTModelLoadingError(
                        "Face recognition model not found. Expected: recognition.fp32.onnx"
                    )

            logger.info(f"Loading face recognition model from {recognition_model_path}")
            self._sess_recognition = ort.InferenceSession(
                str(recognition_model_path),
                sess_options,
                providers=self._providers,
            )

            # Update input sizes from actual models
            if self._sess_detection:
                detection_input = self._sess_detection.get_inputs()[0]
                self._detection_input_size = (
                    int(detection_input.shape[2]),
                    int(detection_input.shape[3]),
                )

            if self._sess_recognition:
                recognition_input = self._sess_recognition.get_inputs()[0]
                self._recognition_input_size = (
                    int(recognition_input.shape[2]),
                    int(recognition_input.shape[3]),
                )

            self._load_time_seconds = time.time() - t0
            self._initialized = True

            logger.info(
                f"✅ ONNXRTBackend initialized in {self._load_time_seconds:.2f}s"
            )
            logger.info(f"   Providers: {self._providers}")
            logger.info(f"   Detection input: {self._detection_input_size}")
            logger.info(f"   Recognition input: {self._recognition_input_size}")
            logger.info(f"   Embedding dim: {self._embedding_dim}")

        except Exception as e:
            raise ONNXRTModelLoadingError(f"ONNX model loading failed: {e}") from e

    def get_runtime_info(self) -> BackendInfo:
        """Report ONNX Runtime metadata and configuration."""
        version = getattr(ort, "__version__", None)
        device = self._infer_device_from_providers(self._providers)

        return BackendInfo(
            runtime="onnx",
            device=device,
            model_id=self.resources.model_name,
            model_name=self.resources.model_info.description,
            pretrained="buffalo_l",
            version=str(version) if version else None,
            face_embedding_dim=self._embedding_dim,
            precisions=["fp32"],
            max_batch_size=1,  # buffalo_l typically processes one face at a time
            supports_image_batch=False,
            extra={
                "providers": ",".join(self._providers),
                "detection_model": "detection.fp32.onnx",
                "recognition_model": "recognition.fp32.onnx",
            },
        )

    def image_to_faces(
        self,
        image_bytes: bytes,
        detection_confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        face_size_min: int = 50,
        face_size_max: int = 1000,
    ) -> list[FaceDetection]:
        """Detect faces using buffalo_l detection model."""
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")

        if not image_bytes:
            raise InvalidInputError("image_bytes cannot be empty")

        try:
            # Decode image
            image = PILImage.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
            img_array = np.array(image)

            # Preprocess for detection
            detection_input = self._preprocess_detection(img_array)

            # Run face detection inference
            assert self._sess_detection is not None
            input_name = self._sess_detection.get_inputs()[0].name
            output_names = [
                output.name for output in self._sess_detection.get_outputs()
            ]

            outputs = self._sess_detection.run(
                output_names, {input_name: detection_input}
            )

            # Postprocess detection results for buffalo_l format
            faces = self._postprocess_detection_buffalo_l(
                outputs,
                img_array.shape[:2],
                detection_confidence_threshold,
                nms_threshold,
                face_size_min,
                face_size_max,
            )

            return faces

        except Exception as e:
            raise InferenceError(f"Face detection failed: {e}") from e

    def face_to_embedding(
        self,
        face_image: bytes | None = None,
        cropped_face_array: npt.NDArray[np.float32] | None = None,
        landmarks: list[tuple[float, float]] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Extract face embedding using buffalo_l recognition model."""
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")

        if face_image is None and cropped_face_array is None:
            raise InvalidInputError(
                "Either face_image or cropped_face_array must be provided"
            )

        try:
            # Convert input to numpy array
            if face_image is not None:
                face = PILImage.open(io.BytesIO(face_image))
                face = face.convert("RGB")
                face_array = np.array(face)
            else:
                face_array = cropped_face_array

            # Apply alignment if landmarks provided (5 points for buffalo_l)
            if landmarks is not None and len(landmarks) == 5:
                face_array = self._align_face_5points(face_array, landmarks)

            # Preprocess for recognition
            recognition_input = self._preprocess_recognition(face_array)

            # Run face recognition inference
            assert self._sess_recognition is not None
            input_name = self._sess_recognition.get_inputs()[0].name
            output_name = self._sess_recognition.get_outputs()[0].name

            outputs = self._sess_recognition.run(
                [output_name], {input_name: recognition_input}
            )
            embedding = np.asarray(outputs[0]).squeeze(0).astype(np.float32)

            # L2 normalize
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm

            return embedding

        except Exception as e:
            raise InferenceError(f"Face embedding extraction failed: {e}") from e

    def _preprocess_detection(
        self, image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        """Preprocess image for buffalo_l face detection."""
        h, w = self._detection_input_size

        # Resize image maintaining aspect ratio
        img_h, img_w = image.shape[:2]
        scale = min(h / img_h, w / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)

        image_resized = cv2.resize(image, (new_w, new_h))

        # Create padded image
        padded_image = np.full((h, w, 3), 128, dtype=np.uint8)  # Gray padding
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        padded_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            image_resized
        )

        # Convert to float32 and normalize
        image_float = padded_image.astype(np.float32) / 255.0

        # Convert from HWC to NCHW format and add batch dimension
        image_tensor = np.transpose(image_float, (2, 0, 1))  # HWC -> CHW
        image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension

        return image_tensor

    def _preprocess_recognition(
        self, face_image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        """Preprocess face image for buffalo_l face recognition."""
        h, w = self._recognition_input_size

        # Resize face image
        face_resized = cv2.resize(face_image, (w, h))

        # Convert to BGR (OpenCV default) if needed
        if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)

        # Convert to float32 and normalize to [0, 1]
        face_float = face_resized.astype(np.float32) / 255.0

        # Standard normalization for ArcFace
        face_normalized = (face_float - 0.5) / 0.5

        # Convert from HWC to NCHW format and add batch dimension
        face_tensor = np.transpose(face_normalized, (2, 0, 1))  # HWC -> CHW
        face_tensor = np.expand_dims(face_tensor, axis=0)  # Add batch dimension

        return face_tensor

    def _postprocess_detection_buffalo_l(
        self,
        outputs: list[npt.NDArray],
        original_shape: tuple[int, int],
        detection_confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        face_size_min: int = 50,
        face_size_max: int = 1000,
    ) -> list[FaceDetection]:
        """
        Postprocess buffalo_l detection results with confidence filtering and NMS.

        buffalo_l typically outputs:
        - face_boxes: [N, 4] (x1, y1, x2, y2)
        - face_scores: [N]
        - face_landmarks: [N, 10] (5 points * 2 coordinates)
        """
        if len(outputs) < 2:
            return []

        # buffalo_l output format may vary, adapt based on actual output structure
        if len(outputs) >= 3:
            boxes = outputs[0]  # [N, 4]
            scores = outputs[1]  # [N]
            landmarks = outputs[2]  # [N, 10]
        else:
            # Fallback for different output format
            logger.warning("Unexpected buffalo_l output format, using basic parsing")
            boxes = outputs[0]
            scores = np.ones(len(boxes)) * 0.8  # Default confidence
            landmarks = None

        if len(boxes) == 0:
            return []

        # Filter by confidence threshold
        valid_mask = scores >= detection_confidence_threshold
        if not np.any(valid_mask):
            return []

        filtered_boxes = boxes[valid_mask]
        filtered_scores = scores[valid_mask]
        filtered_landmarks = landmarks[valid_mask] if landmarks is not None else None

        # Scale boxes to original image size
        orig_h, orig_w = original_shape
        det_h, det_w = self._detection_input_size

        scale_x = orig_w / det_w
        scale_y = orig_h / det_h

        # Convert from normalized coordinates if needed
        if filtered_boxes.max() <= 1.0:  # Normalized coordinates
            filtered_boxes[:, [0, 2]] *= orig_w  # x coordinates
            filtered_boxes[:, [1, 3]] *= orig_h  # y coordinates
        else:  # Pixel coordinates in detection space
            filtered_boxes[:, [0, 2]] *= scale_x
            filtered_boxes[:, [1, 3]] *= scale_y

        # Clamp to image boundaries
        filtered_boxes[:, [0, 2]] = np.clip(filtered_boxes[:, [0, 2]], 0, orig_w)
        filtered_boxes[:, [1, 3]] = np.clip(filtered_boxes[:, [1, 3]], 0, orig_h)

        # Convert to (x1, y1, x2, y2) format and ensure valid boxes
        x1, y1, x2, y2 = (
            filtered_boxes[:, 0],
            filtered_boxes[:, 1],
            filtered_boxes[:, 2],
            filtered_boxes[:, 3],
        )

        # Filter by size constraints
        box_width = x2 - x1
        box_height = y2 - y1
        size_mask = (
            (box_width >= face_size_min)
            & (box_height >= face_size_min)
            & (box_width <= face_size_max)
            & (box_height <= face_size_max)
        )

        if not np.any(size_mask):
            return []

        final_mask = valid_mask.copy()
        final_mask[valid_mask] = size_mask

        final_boxes = filtered_boxes[size_mask]
        final_scores = filtered_scores[size_mask]
        final_landmarks = (
            filtered_landmarks[size_mask] if filtered_landmarks is not None else None
        )

        # Apply NMS
        if len(final_boxes) > 1:
            boxes_nms = np.column_stack(
                [x1[size_mask], y1[size_mask], x2[size_mask], y2[size_mask]]
            )
            indices = cv2.dnn.NMSBoxes(
                boxes_nms.tolist(),
                final_scores.tolist(),
                detection_confidence_threshold,
                nms_threshold,
            )

            if len(indices) > 0:
                indices = indices.flatten()
                final_boxes = final_boxes[indices]
                final_scores = final_scores[indices]
                final_landmarks = (
                    final_landmarks[indices] if final_landmarks is not None else None
                )
            else:
                return []

        # Convert to list of FaceDetection objects
        faces = []
        for i in range(len(final_boxes)):
            bbox = (
                float(final_boxes[i][0]),
                float(final_boxes[i][1]),
                float(final_boxes[i][2]),
                float(final_boxes[i][3]),
            )
            confidence = float(final_scores[i])

            # Convert landmarks from flat [10] to list of 5 tuples
            face_landmarks = None
            if final_landmarks is not None and len(final_landmarks[i]) == 10:
                landmark_coords = final_landmarks[i]
                face_landmarks = [
                    (float(landmark_coords[0]), float(landmark_coords[1])),  # Left eye
                    (float(landmark_coords[2]), float(landmark_coords[3])),  # Right eye
                    (float(landmark_coords[4]), float(landmark_coords[5])),  # Nose tip
                    (
                        float(landmark_coords[6]),
                        float(landmark_coords[7]),
                    ),  # Left mouth corner
                    (
                        float(landmark_coords[8]),
                        float(landmark_coords[9]),
                    ),  # Right mouth corner
                ]

            faces.append(
                FaceDetection(
                    bbox=bbox, landmarks=face_landmarks, confidence=confidence
                )
            )

        return faces

    def _align_face_5points(
        self,
        face_image: npt.NDArray[np.uint8],
        landmarks: list[tuple[float, float]],
    ) -> npt.NDArray[np.uint8]:
        """
        Align face using 5-point landmarks for buffalo_l.

        Standard 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        if len(landmarks) != 5:
            logger.warning(
                f"Expected 5 landmarks for alignment, got {len(landmarks)}. Skipping alignment."
            )
            return face_image

        # Standard 5-point positions for 112x112 aligned face (ArcFace standard)
        src_points = np.array(
            [
                [30.2946, 51.6963],  # Left eye
                [65.5318, 51.5014],  # Right eye
                [48.0252, 71.7366],  # Nose tip
                [33.5493, 92.3655],  # Left mouth corner
                [62.7299, 92.2041],  # Right mouth corner
            ],
            dtype=np.float32,
        )

        dst_points = np.array(landmarks, dtype=np.float32)

        # Calculate similarity transform
        transform = cv2.estimateAffinePartial2D(dst_points, src_points)[0]

        if transform is None:
            logger.warning(
                "Failed to calculate alignment transform. Using original face."
            )
            return face_image

        # Apply transformation
        h, w = self._recognition_input_size
        aligned_face = cv2.warpAffine(face_image, transform, (w, h))

        return aligned_face
