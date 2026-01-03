"""
face_service.py

A gRPC service for face detection and recognition, following the Lumen architecture.
It uses the streaming Inference protocol to expose tasks:
  - detect: Detect faces in an image with bounding boxes and landmarks.
  - embed: Extract face embeddings from face images.
  - detect_and_embed: Detect faces and extract embeddings in one call.
"""

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import grpc
from google.protobuf import empty_pb2
from lumen_resources import EmbeddingV1, FaceV1
from lumen_resources.lumen_config import BackendSettings, ModelConfig, Services
from lumen_resources.result_schemas.face_v1 import BboxItem, Face
from typing_extensions import override

import lumen_face.proto.ml_service_pb2 as pb
import lumen_face.proto.ml_service_pb2_grpc as rpc
from lumen_face.backends import FaceRecognitionBackend, ONNXRTBackend
from lumen_face.registry import TaskRegistry
from lumen_face.resources.loader import ModelResources, ResourceLoader

from .face_model import FaceModelManager

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Returns the current time in milliseconds."""
    return int(time.time() * 1000)


class GeneralFaceService(rpc.InferenceServicer):
    """gRPC service for face detection and recognition with streaming inference support.

    This service implements the Lumen Inference protocol to provide face detection,
    recognition, and combined detect-and-embed capabilities over gRPC. It supports
    bidirectional streaming for efficient processing of multiple requests and handles
    chunked data transfer for large images.

    Supported tasks:
    - detect: Face detection with bounding boxes, landmarks, and confidence scores
    - embed: Face embedding extraction with optional landmark-based alignment
    - detect_and_embed: Combined detection and embedding for single-call workflows

    Service features:
    - Configurable detection thresholds and NMS parameters
    - Automatic image preprocessing and format handling
    - Error handling with detailed gRPC status codes
    - Performance timing and logging
    - Capability reporting for client discovery

    Attributes:
        SERVICE_NAME: Unique service identifier "face-general" for discovery
        model: FaceModelManager instance providing core face processing capabilities
        is_initialized: Flag indicating whether the service is ready for inference

    Example:
        ```python
        # Server setup
        service = GeneralFaceService.from_config(model_config, cache_dir, backend_settings)
        service.initialize()

        # Add to gRPC server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        rpc.add_InferenceServicer_to_server(service, server)
        ```

    Note:
        The service requires proper initialization before handling inference requests.
        Call `initialize()` to load models and prepare for serving.
    """

    SERVICE_NAME = "face-general"

    def __init__(
        self, backend: FaceRecognitionBackend, resources: ModelResources
    ) -> None:
        """Initialize GeneralFaceService with backend and model resources.

        Args:
            backend: Face recognition backend instance (e.g., ONNXRTBackend) that
                provides the actual inference capabilities for detection and recognition.
            resources: ModelResources containing model metadata, file paths, and
                configuration information loaded from lumen-resources.

        Raises:
            ValueError: If backend or resources are None.
            TypeError: If backend doesn't implement FaceRecognitionBackend interface.

        Note:
            The constructor creates the FaceModelManager but doesn't initialize it.
            Call `initialize()` to load models before serving requests.
        """
        self.model = FaceModelManager(backend=backend, resources=resources)
        self.registry = TaskRegistry()
        self.is_initialized = False

        # Initialize task registry
        self._setup_task_registry()

    @classmethod
    def from_config(
        cls,
        service_config: Services,
        cache_dir: Path,
    ):
        """Create GeneralFaceService from service configuration.

        This factory method creates a fully configured service instance by:
        1. Extracting model configuration from service_config.models
        2. Loading model resources using the validated configuration
        3. Selecting appropriate backend based on runtime settings
        4. Configuring backend with device and performance settings
        5. Creating and returning the initialized service

        Args:
            service_config: Services config from lumen_config (services.face).
            cache_dir: Directory path for model caching and temporary files.
                Models will be downloaded and stored here if not present.

        Returns:
            GeneralFaceService: Fully configured service instance ready for
                initialization. The caller must still call `initialize()` to load models.

        Raises:
            ConfigError: If configuration is invalid or missing required fields.
            ResourceNotFoundError: If model files cannot be found or downloaded.
            RuntimeNotSupportedError: If specified runtime is not available.

        Example:
            ```python
            # Load from config file
            config = load_and_validate_config("config.yaml")
            service_config = config.services["face"]

            # Create service
            service = GeneralFaceService.from_config(
                service_config=service_config,
                cache_dir=Path("~/.cache/lumen"),
            )
            service.initialize()
            ```

        Note:
            This method handles resource loading and backend selection but doesn't
            initialize the models. Call `initialize()` to complete the setup.
        """
        from lumen_face.resources.exceptions import ConfigError

        # Extract model_config from service_config.models
        # Supports keys: "general", "face", "recognition"
        model_config = None
        for key in ["general", "face", "recognition"]:
            if key in service_config.models:
                model_config = service_config.models[key]
                break

        if model_config is None:
            # Fall back to first available model
            if not service_config.models:
                raise ValueError("No models configured for Face service")
            model_key = next(iter(service_config.models.keys()))
            logger.info(f"Using model '{model_key}' for Face service")
            model_config = service_config.models[model_key]

        # Get backend_settings from service_config
        backend_settings = service_config.backend_settings

        # Load resources using the validated model_config
        logger.info(f"Loading resources for General Face model: {model_config.model}")
        resources = ResourceLoader.load_model_resource(cache_dir, model_config)

        # Create backend based on runtime
        runtime = model_config.runtime.value
        device_pref = (
            getattr(backend_settings, "device", "cpu") if backend_settings else "cpu"
        )
        max_batch_size = (
            getattr(backend_settings, "batch_size", 1) if backend_settings else 1
        )

        # Determine precision preference from ModelConfig
        # Only applies to Runtime.onnx and Runtime.rknn
        prefer_fp16 = False
        if model_config.precision and runtime in ["onnx", "rknn"]:
            prefer_fp16 = model_config.precision in ["fp16", "q4fp16"]

        if runtime == "onnx":
            backend = ONNXRTBackend(
                resources=resources,
                device_preference=device_pref,
                max_batch_size=max_batch_size,
                prefer_fp16=prefer_fp16,
            )
        else:
            raise ConfigError(f"Unsupported runtime: {runtime}")

        # Create service
        service = cls(backend, resources)

        return service

    def _setup_task_registry(self) -> None:
        """Initialize the task registry with all supported tasks.

        This method serves as the single source of truth for task definitions.
        All task names, handlers, and metadata are registered here.
        """
        self.registry.set_service_name("lumen-face")

        # Register face detection task
        self.registry.register_task(
            name="face_detect",
            handler=self._handle_detect,
            description="Face detection with bounding boxes and landmarks",
            input_mimes=["image/jpeg", "image/png"],
            output_mime="application/json;schema=face_v1",
            metadata={},
        )

        # Register face embedding task
        self.registry.register_task(
            name="face_embed",
            handler=self._handle_embed,
            description="Face embedding extraction with optional alignment",
            input_mimes=["image/jpeg", "image/png"],
            output_mime="application/json;schema=embedding_v1",
            metadata={},
        )

        # Register combined detect and embed task
        self.registry.register_task(
            name="face_detect_and_embed",
            handler=self._handle_detect_and_embed,
            description="Combined face detection and embedding extraction",
            input_mimes=["image/jpeg", "image/png"],
            output_mime="application/json;schema=face_v1",
            metadata={},
        )

        logger.info(
            f"Task registry initialized with {len(self.registry.list_task_names())} tasks"
        )

    def initialize(self) -> None:
        """Loads the model and prepares it for inference."""
        logger.info("Initializing FaceModelManager...")
        self.model.initialize()
        self.is_initialized = True
        info = self.model.get_info()
        logger.info(
            "Face model ready: %s with %s (loaded in %.2fs)",
            info.model_name,
            info.backend_info or "unknown",
            info.load_time,
        )

    # -------- gRPC Service Methods ----------

    @override
    def Infer(
        self, request_iterator: Iterable[pb.InferRequest], context: grpc.ServicerContext
    ):
        """
        Handles the bidirectional streaming inference RPC. It routes incoming requests
        to the appropriate task handler.
        """
        if not self.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        buffers: dict[str, bytearray] = {}  # Buffers for reassembling chunked requests

        for req in request_iterator:
            cid = req.correlation_id or f"cid-{_now_ms()}"
            t0 = _now_ms()
            try:
                # 1. Reassemble payload if it was sent in chunks
                payload, ready = self._assemble(cid, req, buffers)

                if not ready:
                    continue  # Wait for more chunks

                if payload is None:
                    logger.error(
                        "Payload assembly returned None for %s despite ready flag; skipping request",
                        cid,
                    )
                    buffers.pop(cid, None)
                    continue

                # 2. Route to the correct handler using TaskRegistry
                try:
                    meta = dict(req.meta)

                    handler = self.registry.get_handler(req.task)
                    result_bytes, result_mime, extra_meta = handler(
                        payload, req.payload_mime, meta
                    )
                except ValueError as e:
                    # Task not found in registry
                    raise ValueError(
                        f"Unsupported task: {req.task}. Available tasks: {self.registry.list_task_names()}"
                    ) from e

                # 3. Stream response back
                yield pb.InferResponse(
                    correlation_id=cid,
                    is_final=True,
                    result=result_bytes,
                    result_mime=result_mime,
                    meta={
                        **extra_meta,
                        "processing_time_ms": str(_now_ms() - t0),
                    },
                )

                # 4. Clean up buffer for this correlation_id
                buffers.pop(cid, None)

            except Exception as e:
                logger.error(f"Error processing request {cid}: {e}", exc_info=True)
                yield pb.InferResponse(
                    correlation_id=cid,
                    is_final=True,
                    error=pb.Error(
                        code=pb.ErrorCode.ERROR_CODE_INTERNAL,
                        message=str(e),
                    ),
                )
                buffers.pop(cid, None)

    @override
    def GetCapabilities(
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> pb.Capability:
        """
        Returns the service capabilities including supported tasks and model info.
        """
        if not self.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        # Get model info
        model_info = self.model.get_info()
        backend_info = model_info.backend_info

        # Use registry to build capability automatically
        extra_metadata = {
            "model_name": model_info.model_name,
            "model_id": model_info.model_id,
            "face_embedding_dim": str(backend_info.get("face_embedding_dim", 512)),
            "supports_landmarks": str(
                backend_info.get("supports_landmarks", False)
            ).lower(),
        }

        return self.registry.build_capability(
            service_name=self.SERVICE_NAME,
            model_id=model_info.model_id,
            runtime=backend_info.get("runtime", "unknown"),
            precisions=backend_info.get("precisions", ["fp32"]),
            extra_metadata=extra_metadata,
        )

    @override
    def StreamCapabilities(self, request, context) -> Iterable[pb.Capability]:
        """Streams the capabilities of the service."""
        """
        Returns the service capabilities including supported tasks and model info.
        """
        if not self.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        # Get model info
        model_info = self.model.get_info()
        backend_info = model_info.backend_info

        # Use registry to build capability automatically
        extra_metadata = {
            "model_name": model_info.model_name,
            "model_id": model_info.model_id,
            "face_embedding_dim": str(backend_info.get("face_embedding_dim", 512)),
            "supports_landmarks": str(
                backend_info.get("supports_landmarks", False)
            ).lower(),
        }

        yield self.registry.build_capability(
            service_name=self.SERVICE_NAME,
            model_id=model_info.model_id,
            runtime=backend_info.get("runtime", "unknown"),
            precisions=backend_info.get("precisions", ["fp32"]),
            extra_metadata=extra_metadata,
        )

    @override
    def Health(self, request, context):
        """Simple health check endpoint."""
        return empty_pb2.Empty()

    # -------- Task Handlers ----------

    def _handle_detect(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handle face detection task."""
        # Extract parameters from meta
        detection_confidence_threshold = float(
            meta.get("detection_confidence_threshold", "0.7")
        )
        nms_threshold = float(meta.get("nms_threshold", "0.4"))
        face_size_min = int(meta.get("face_size_min", "50"))
        face_size_max = int(meta.get("face_size_max", "1000"))

        # Detect faces
        faces = self.model.detect_faces(
            payload,
            detection_confidence_threshold=detection_confidence_threshold,
            nms_threshold=nms_threshold,
            face_size_min=face_size_min,
            face_size_max=face_size_max,
        )

        result = FaceV1(
            faces=[
                Face(
                    bbox=[BboxItem(root=coord) for coord in face.bbox],
                    confidence=face.confidence,
                    landmarks=(
                        [coord for point in face.landmarks for coord in point]
                        if face.landmarks
                        else None
                    ),
                )
                for face in faces
            ],
            count=len(faces),
            model_id=self.model.get_info().model_id,
        ).model_dump()

        result_bytes = json.dumps(result).encode("utf-8")

        return (
            result_bytes,
            "application/json;schema=face_v1",
            {"face_count": str(len(faces))},
        )

    def _handle_embed(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handle face embedding task."""
        # Extract landmarks from meta if provided
        landmarks = None
        if "landmarks" in meta:
            try:
                landmarks_data = json.loads(meta["landmarks"])
                landmarks = [(pt["x"], pt["y"]) for pt in landmarks_data]
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning(
                    "Invalid landmarks format in meta, proceeding without alignment"
                )

        # Extract embedding
        embedding = self.model.extract_embedding(
            face_image=payload, landmarks=landmarks
        )

        info = self.model.get_info()
        model_id = info.model_id

        result = EmbeddingV1(
            vector=embedding.tolist(),
            dim=len(embedding),
            model_id=model_id,
        )

        result_bytes = json.dumps(result).encode("utf-8")

        return (
            result_bytes,
            "application/json;schema=embedding_v1",
            {"dim": str(len(embedding))},
        )

    def _handle_detect_and_embed(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handle detect and embed task."""
        # Extract parameters from meta with robust parsing that tolerates float-like strings
        detection_confidence_threshold = float(
            meta.get("detection_confidence_threshold", "0.7")
        )
        nms_threshold = float(meta.get("nms_threshold", "0.4"))

        def _parse_int_like(value: str, default: int) -> int:
            try:
                # Allow values like "50", "50.0", or numeric types
                return int(float(value))
            except (ValueError, TypeError):
                return default

        face_size_min = _parse_int_like(meta.get("face_size_min", "50"), 50)
        face_size_max = _parse_int_like(meta.get("face_size_max", "1000"), 1000)
        max_faces = _parse_int_like(
            meta.get("max_faces", "-1"), -1
        )  # -1 means no limit

        # Detect faces
        faces = self.model.detect_faces(
            payload,
            detection_confidence_threshold=detection_confidence_threshold,
            nms_threshold=nms_threshold,
            face_size_min=face_size_min,
            face_size_max=face_size_max,
        )

        # Limit faces if requested
        if 0 < max_faces < len(faces):
            faces = faces[:max_faces]

        # Extract embeddings for each face
        result_faces: list[Face] = []
        for face in faces:
            # Crop face from original image using bounding box
            face_crop = self.model.crop_face_from_image(payload, face.bbox)

            # Extract embedding with landmarks if available
            embedding = self.model.extract_embedding(
                cropped_face_array=face_crop, landmarks=face.landmarks
            )

            result_faces.append(
                Face(
                    bbox=[BboxItem(root=coord) for coord in face.bbox],
                    confidence=face.confidence,
                    landmarks=(
                        [coord for point in face.landmarks for coord in point]
                        if face.landmarks
                        else None
                    ),
                    embedding=embedding.tolist(),
                )
            )

        result = FaceV1(
            faces=result_faces,
            count=len(result_faces),
            model_id=self.model.get_info().model_id,
        ).model_dump()

        result_bytes = json.dumps(result).encode("utf-8")

        return (
            result_bytes,
            "application/json;schema=face_v1",
            {"face_count": str(len(result_faces))},
        )

    # -------- Helper Methods ----------

    def _assemble(
        self,
        correlation_id: str,
        request: pb.InferRequest,
        buffers: dict[str, bytearray],
    ) -> tuple[bytes, bool] | tuple[None, bool]:
        """
        Reassembles chunked request payloads. Returns (payload, ready) where ready indicates
        if all chunks for this correlation_id have been received.
        """
        total_chunks = request.total if request.total and request.total > 0 else 1
        seq_index = request.seq if request.seq >= 0 else 0

        if total_chunks == 1:
            # Treat missing chunk metadata as single-chunk payloads
            return request.payload, True

        # Multi-chunk request
        if correlation_id not in buffers:
            buffers[correlation_id] = bytearray()

        buffer = buffers[correlation_id]
        buffer.extend(request.payload)

        # Check if all chunks received
        if seq_index >= total_chunks - 1:
            payload = bytes(buffer)
            return payload, True

        return None, False
