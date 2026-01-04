"""
clip_service.py

A gRPC service for the general-purpose CLIP model, structured to mirror the
elegant design of the BioCLIP service. It uses the streaming Inference protocol
to expose tasks:
  - embed: Creates a vector embedding from a text string.
  - image_embed: Creates a vector embedding from an image.
  - classify: Classifies an image against the default ImageNet dataset (if dataset available).
  - classify_scene: Performs a high-level scene analysis on an image (if dataset available).
"""

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import grpc
from google.protobuf import empty_pb2
from lumen_resources import EmbeddingV1, LabelsV1
from lumen_resources.lumen_config import BackendSettings, Services
from lumen_resources.result_schemas.labels_v1 import Label
from typing_extensions import override

import lumen_clip.proto.ml_service_pb2 as pb
import lumen_clip.proto.ml_service_pb2_grpc as rpc
from lumen_clip.backends import create_backend
from lumen_clip.backends.base import RuntimeKind
from lumen_clip.registry import TaskRegistry
from lumen_clip.resources.loader import ModelResources, ResourceLoader

from ..resources import ResourceNotFoundError
from .clip_model import CLIPModelManager

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Returns the current time in milliseconds."""
    return int(time.time() * 1000)


class GeneralCLIPService(rpc.InferenceServicer):
    """
    Implements the streaming Inference service contract for a general-purpose
    OpenCLIP model, offering text embedding, ImageNet classification, and
    scene classification tasks.
    """

    def __init__(self, backend, resources: ModelResources) -> None:
        """
        Initialize CLIPService.

        Args:
            backend: Backend instance (TorchBackend or ONNXRTBackend)
            resources: ModelResources with model configuration and data
        """
        self.model = CLIPModelManager(backend=backend, resources=resources)
        self.is_initialized = False
        self.registry = TaskRegistry()
        self._setup_registry()

    @classmethod
    def from_config(
        cls,
        service_config: Services,
        cache_dir: Path,
    ):
        """
        Create GeneralCLIPService from configuration.

        Args:
            service_config: Services config from lumen_config (services.clip).
            cache_dir: Cache directory path.

        Returns:
            Initialized GeneralCLIPService instance.
        """

        # Extract model_config from service_config.models
        # Supports keys: "general", "clip", "general_clip"
        model_config = None
        for key in ["general", "clip", "general_clip"]:
            if key in service_config.models:
                model_config = service_config.models[key]
                break

        if model_config is None:
            raise ValueError(
                "No suitable model config found in service_config.models. "
                "Expected one of: 'general', 'clip', 'general_clip'"
            )

        # Get backend_settings from service_config
        backend_settings = service_config.backend_settings

        # Load resources using the validated model_config
        logger.info(f"Loading resources for General CLIP model: {model_config.model}")
        try:
            resources = ResourceLoader.load_model_resources(cache_dir, model_config)
        except Exception as e:
            logger.error(f"Failed to load resources for General CLIP model: {e}")
            raise ResourceNotFoundError(
                f"Failed to load resources for General CLIP model: {e}"
            ) from e

        # Create backend based on runtime using factory
        if backend_settings is None:
            backend_settings = BackendSettings(
                device="cpu", batch_size=1, onnx_providers=None
            )

        # Determine precision preference from ModelConfig
        # Only applies to Runtime.onnx and Runtime.rknn
        prefer_fp16 = False
        if model_config.precision and model_config.runtime.value in ["onnx", "rknn"]:
            prefer_fp16 = model_config.precision in ["fp16", "q4fp16"]

        # Use factory to create backend
        backend = create_backend(
            backend_settings,
            resources,
            RuntimeKind(model_config.runtime.value),
            prefer_fp16=prefer_fp16,
        )

        # Create service
        service = cls(backend, resources)

        # Log classification support status
        if not resources.has_classification_support():
            logger.warning(
                "CLIP service started without classification support "
                "(no ImageNet dataset found). Only embed tasks will be available."
            )

        return service

    def _setup_registry(self) -> None:
        """Set up the task registry with all available tasks."""
        self.registry.set_service_name("lumen-clip")

        # Always available tasks
        self.registry.register_task(
            name="clip_text_embed",
            handler=self._handle_embed,
            description="Create text embedding from input text",
            input_mimes=["application/json", "text/plain"],
            output_mime="application/json;schema=embedding_v1",
            metadata={},
        )

        self.registry.register_task(
            name="clip_image_embed",
            handler=self._handle_image_embed,
            description="Create image embedding from input image",
            input_mimes=["image/jpeg", "image/png", "image/webp"],
            output_mime="application/json;schema=embedding_v1",
            metadata={},
        )

        # Classification tasks (only if dataset available)
        if self.model.resources.has_classification_support():
            self.registry.register_task(
                name="clip_classify",
                handler=self._handle_classify,
                description="Classify image against ImageNet dataset",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mime="application/json;schema=labels_v1",
                metadata={},
            )

            self.registry.register_task(
                name="clip_scene_classify",
                handler=self._handle_classify_scene,
                description="Perform scene analysis on image",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mime="application/json;schema=labels_v1",
                metadata={},
            )
        else:
            logger.info("Classification tasks not registered (no dataset available)")

    def get_supported_tasks(self) -> list[str]:
        """Get list of supported task names for routing.

        Returns:
            List of task keys registered in the task registry.
        """
        return self.registry.list_task_names()

    def initialize(self) -> None:
        """Loads the model and prepares it for inference."""
        logger.info("Initializing CLIPModelManager...")
        self.model.initialize()
        self.is_initialized = True
        info = self.model.info()
        logger.info(
            "CLIP model ready: %s with %s (loaded in %.2fs)",
            info.model_name,
            info.backend_info,
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
                    yield pb.InferResponse(
                        correlation_id=cid,
                        is_final=True,
                        error=pb.Error(
                            code=pb.ERROR_CODE_INVALID_ARGUMENT,
                            message=str(e),
                        ),
                    )
                    continue

                # 3. Yield a successful response
                meta = dict(extra_meta or {})
                meta["lat_ms"] = str(_now_ms() - t0)

                yield pb.InferResponse(
                    correlation_id=cid,
                    is_final=True,
                    result=result_bytes,
                    result_mime=result_mime,
                    meta=meta,
                )

            except Exception as e:
                logger.exception(
                    "Error during inference for task '%s': %s", req.task, e
                )
                yield pb.InferResponse(
                    correlation_id=cid,
                    is_final=True,
                    error=pb.Error(code=pb.ERROR_CODE_INTERNAL, message=str(e)),
                )

    @override
    def GetCapabilities(self, request, context) -> pb.Capability:
        """Returns the capabilities of the service in a single response. [cite: 21]"""
        return self._build_capability()

    @override
    def StreamCapabilities(self, request, context) -> Iterable[pb.Capability]:
        """Streams the capabilities of the service."""
        yield self._build_capability()

    @override
    def Health(self, request, context):
        """A simple health check endpoint. [cite: 22]"""
        return empty_pb2.Empty()

    # -------- Task Handlers ----------

    def _handle_embed(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handles text embedding requests."""
        text = payload.decode("utf-8")
        vec = self.model.encode_text(text).tolist()

        info = self.model.info()
        model_id = f"{info.model_name}:{info.model_id}"

        resp_obj = EmbeddingV1(vector=vec, dim=len(vec), model_id=model_id).model_dump()
        return (
            json.dumps(resp_obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=embedding_v1",
            {"dim": str(len(vec))},
        )

    def _handle_classify(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handles ImageNet classification requests."""
        # Check if classification is supported
        if not self.model.supports_classification:
            raise RuntimeError(
                "Classification not supported: no ImageNet dataset loaded. "
                "Please ensure ImageNet_1k.npz exists in the model directory."
            )

        top_k = int(meta.get("topk", "5"))
        scores = self.model.classify_image(payload, top_k=top_k)

        info = self.model.info()
        model_id = info.model_id

        resp_obj = LabelsV1(
            labels=[Label(label=label, score=float(score)) for label, score in scores],
            model_id=model_id,
        ).model_dump()
        return (
            json.dumps(resp_obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=labels_v1",
            {"labels_count": str(len(scores))},
        )

    def _handle_classify_scene(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handles scene classification requests."""
        label, score = self.model.classify_scene(payload)

        info = self.model.info()
        model_id = info.model_id

        resp_obj = LabelsV1(
            labels=[Label(label=label, score=float(score))],
            model_id=model_id,
        ).model_dump()
        return (
            json.dumps(resp_obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=labels_v1",
            {"labels_count": "1"},
        )

    def _handle_image_embed(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handles image embedding requests."""
        vec = self.model.encode_image(payload).tolist()

        info = self.model.info()
        model_id = info.model_id

        resp_obj = EmbeddingV1(vector=vec, dim=len(vec), model_id=model_id).model_dump()
        return (
            json.dumps(resp_obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=embedding_v1",
            {"dim": str(len(vec))},
        )

    # -------- Helpers ----------

    def _assemble(
        self, cid: str, req: pb.InferRequest, buffers: dict[str, bytearray]
    ) -> tuple[bytes, bool]:
        """
        Reassembles chunked request payloads.

        Returns a tuple of (payload_bytes, ready). If a request is not chunked,
        it is returned immediately with ready=True. If chunked, data is buffered
        until the final chunk arrives.
        """
        # Default path for non-chunked requests
        if req.total <= 1:
            return bytes(req.payload), True

        # Append chunk to buffer
        buf = buffers.setdefault(cid, bytearray())
        buf.extend(req.payload)

        # Check if all chunks have arrived
        if req.total and (req.seq + 1 == req.total):
            data = bytes(buf)
            del buffers[cid]
            return data, True

        return b"", False  # Not ready yet

    def _build_capability(self) -> pb.Capability:
        """Constructs the capability message using the task registry."""
        info = self.model.info()

        # Use registry to build capability automatically
        extra_metadata = {
            "device": info.device,
            "embedding_dim": str(info.embedding_dim),
            "model_version": info.model_version,
            "supports_classification": str(info.supports_classification),
        }

        return self.registry.build_capability(
            service_name="lumen_clip",
            model_id=info.model_name,
            runtime=info.runtime,
            precisions=info.precisions,
            extra_metadata=extra_metadata,
        )
