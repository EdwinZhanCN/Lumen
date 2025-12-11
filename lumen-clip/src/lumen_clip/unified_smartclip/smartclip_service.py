"""
smartclip_service.py

A unified gRPC service for SmartCLIP that combines general CLIP and BioCLIP models.
It follows the same architectural pattern as other Lumen services and uses the
streaming Inference protocol to expose tasks:
  - smartclip_text_embed: Create text embeddings using SmartCLIP (intelligent model selection)
  - smartclip_image_embed: Create image embeddings using SmartCLIP (intelligent model selection)
  - smartclip_classify: Intelligent classification using the best available model
  - smartclip_scene_classify: Scene analysis using CLIP model
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import grpc
from google.protobuf import empty_pb2
from lumen_resources.lumen_config import BackendSettings, ModelConfig
from typing_extensions import override

import lumen_clip.proto.ml_service_pb2 as pb
import lumen_clip.proto.ml_service_pb2_grpc as rpc
from lumen_clip.backends import BaseClipBackend, ONNXRTBackend, TorchBackend
from lumen_clip.expert_bioclip.bioclip_model import BioCLIPModelManager
from lumen_clip.general_clip.clip_model import CLIPModelManager
from lumen_clip.registry import TaskRegistry
from lumen_clip.resources.loader import ModelResources, ResourceLoader

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Returns the current time in milliseconds."""
    return int(time.time() * 1000)


class SmartCLIPService(rpc.InferenceServicer):
    """
    Implements the streaming Inference service contract for SmartCLIP,
    offering intelligent model selection between CLIP and BioCLIP models.
    """

    def __init__(
        self,
        clip_backend: BaseClipBackend,
        clip_resources: ModelResources,
        bioclip_backend: BaseClipBackend,
        bioclip_resources: ModelResources,
    ) -> None:
        """
        Initialize SmartCLIPService with both CLIP and BioCLIP models.

        Args:
            clip_backend: Backend instance for CLIP model
            clip_resources: ModelResources for CLIP model
            bioclip_backend: Backend instance for BioCLIP model
            bioclip_resources: ModelResources for BioCLIP model
        """
        self.clip_model = CLIPModelManager(
            backend=clip_backend, resources=clip_resources
        )
        self.bioclip_model = BioCLIPModelManager(
            backend=bioclip_backend, resources=bioclip_resources
        )
        self.is_initialized = False
        self.registry = TaskRegistry()
        self._setup_registry()

    @classmethod
    def from_config(
        cls,
        clip_config: ModelConfig,
        bioclip_config: ModelConfig,
        cache_dir: Path,
        backend_settings: BackendSettings | None,
    ):
        """
        Create SmartCLIPService from configuration.

        Args:
            clip_config: Configuration for CLIP model
            bioclip_config: Configuration for BioCLIP model
            cache_dir: Cache directory path
            backend_settings: Backend settings

        Returns:
            Initialized SmartCLIPService instance
        """
        from lumen_clip.resources.exceptions import ConfigError

        # Load CLIP resources
        logger.info(f"Loading resources for CLIP model: {clip_config.model}")
        clip_resources = ResourceLoader.load_model_resources(cache_dir, clip_config)

        # Load BioCLIP resources
        logger.info(f"Loading resources for BioCLIP model: {bioclip_config.model}")
        bioclip_resources = ResourceLoader.load_model_resources(
            cache_dir, bioclip_config
        )

        # Handle optional backend_settings
        device_pref = "cpu"
        max_batch_size = 1
        providers_list = None

        if backend_settings:
            device_pref = backend_settings.device or "cpu"
            max_batch_size = backend_settings.batch_size or 8
            providers_list = backend_settings.onnx_providers

        # CLIP backend
        clip_runtime = clip_config.runtime.value
        if clip_runtime == "torch":
            clip_backend = TorchBackend(
                resources=clip_resources,
                device_preference=device_pref,
                max_batch_size=max_batch_size,
            )
        elif clip_runtime == "onnx":
            clip_backend = ONNXRTBackend(
                resources=clip_resources,
                providers=providers_list,
                device_preference=device_pref,
                max_batch_size=max_batch_size,
            )
        else:
            raise ConfigError(f"Unsupported CLIP runtime: {clip_runtime}")

        # BioCLIP backend
        bioclip_runtime = bioclip_config.runtime.value
        if bioclip_runtime == "torch":
            bioclip_backend = TorchBackend(
                resources=bioclip_resources,
                device_preference=device_pref,
                max_batch_size=max_batch_size,
            )
        elif bioclip_runtime == "onnx":
            bioclip_backend = ONNXRTBackend(
                resources=bioclip_resources,
                providers=providers_list,
                device_preference=device_pref,
                max_batch_size=max_batch_size,
            )
        else:
            raise ConfigError(f"Unsupported BioCLIP runtime: {bioclip_runtime}")

        # Create service
        service = cls(clip_backend, clip_resources, bioclip_backend, bioclip_resources)

        # Log classification support status
        if not clip_resources.has_classification_support():
            logger.info("CLIP classification not available (no ImageNet dataset)")
        if not bioclip_resources.has_classification_support():
            logger.info("BioCLIP classification not available (no TreeOfLife dataset)")

        return service

    def _setup_registry(self) -> None:
        """Set up the task registry with all available tasks."""
        self.registry.set_service_name("lumen-smartclip")

        # Text embedding tasks (intelligent model selection)
        self.registry.register_task(
            name="smartclip_text_embed",
            handler=self._handle_text_embed,
            description="Create text embedding using intelligent model selection",
            input_mimes=["application/json", "text/plain"],
            output_mime="application/json;schema=embedding_v1",
            metadata={"task_type": "text_embedding", "framework": "pytorch"},
        )

        # Image embedding tasks (intelligent model selection)
        self.registry.register_task(
            name="smartclip_image_embed",
            handler=self._handle_image_embed,
            description="Create image embedding using intelligent model selection",
            input_mimes=["image/jpeg", "image/png", "image/webp"],
            output_mime="application/json;schema=embedding_v1",
            metadata={},
        )

        # General classification tasks (using CLIP)
        if self.clip_model.resources.has_classification_support():
            self.registry.register_task(
                name="smartclip_classify",
                handler=self._handle_classify,
                description="Intelligent classification using best available model",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mime="application/json;schema=labels_v1",
                metadata={},
            )

            self.registry.register_task(
                name="smartclip_scene_classify",
                handler=self._handle_scene_classify,
                description="Scene analysis using CLIP model",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mime="application/json;schema=labels_v1",
                metadata={},
            )

        # BioCLIP-specific classification (if available)
        if self.bioclip_model.resources.has_classification_support():
            self.registry.register_task(
                name="smartclip_bioclassify",
                handler=self._handle_bioclassify,
                description="Biological classification using BioCLIP model",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mime="application/json;schema=labels_v1",
                metadata={},
            )

    def initialize(self) -> None:
        """Initialize both underlying model managers."""
        logger.info("Initializing CLIPModelManager...")
        self.clip_model.initialize()

        logger.info("Initializing BioCLIPModelManager...")
        self.bioclip_model.initialize()

        self.is_initialized = True

        clip_info = self.clip_model.info()
        bioclip_info = self.bioclip_model.info()
        logger.info(
            "SmartCLIP ready - CLIP: %s (%.2fs), BioCLIP: %s (%.2fs)",
            clip_info.model_name,
            clip_info.load_time,
            bioclip_info.model_name,
            bioclip_info.load_time,
        )

    # -------- gRPC Service Methods ----------

    @override
    def Infer(
        self, request_iterator: Iterable[pb.InferRequest], context: grpc.ServicerContext
    ):
        """
        Handles the bidirectional streaming inference RPC. Routes incoming requests
        to the appropriate task handler.
        """
        if not self.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Models not initialized")

        buffers: Dict[str, bytearray] = {}

        for req in request_iterator:
            cid = req.correlation_id or f"cid-{_now_ms()}"
            t0 = _now_ms()
            try:
                # Reassemble payload if chunked
                payload, ready = self._assemble(cid, req, buffers)
                if not ready:
                    continue

                # Route to handler using TaskRegistry
                try:
                    meta = dict(req.meta)
                    handler = self.registry.get_handler(req.task)
                    result_bytes, result_mime, extra_meta = handler(
                        payload, req.payload_mime, meta
                    )
                except ValueError as e:
                    yield pb.InferResponse(
                        correlation_id=cid,
                        is_final=True,
                        error=pb.Error(
                            code=pb.ERROR_CODE_INVALID_ARGUMENT,
                            message=str(e),
                        ),
                    )
                    continue

                # Stream successful response
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
        """Returns the capabilities of the service."""
        return self._build_capability()

    @override
    def StreamCapabilities(self, request, context) -> Iterable[pb.Capability]:
        """Streams the capabilities of the service."""
        yield self._build_capability()

    @override
    def Health(self, request, context):
        """Simple health check endpoint."""
        return empty_pb2.Empty()

    # -------- Task Handlers ----------

    def _handle_text_embed(
        self, payload: bytes, payload_mime: str, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
        """Handles text embedding with intelligent model selection."""
        if not payload_mime.startswith("text/"):
            raise ValueError(f"text_embed expects text/* payload, got {payload_mime!r}")

        text = payload.decode("utf-8")

        # Use CLIP for text embedding (generally better for general text)
        vec = self.clip_model.encode_text(text).tolist()

        info = self.clip_model.info()
        model_id = f"smartclip:{info.model_name}"

        obj = {"vector": vec, "dim": len(vec), "model_id": model_id}
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=embedding_v1",
            {"dim": str(len(vec))},
        )

    def _handle_image_embed(
        self, payload: bytes, payload_mime: str, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
        """Handles image embedding with intelligent model selection."""
        if not payload_mime.startswith("image/"):
            raise ValueError(
                f"image_embed expects image/* payload, got {payload_mime!r}"
            )

        # Use CLIP for general image embedding
        vec = self.clip_model.encode_image(payload).tolist()

        info = self.clip_model.info()
        model_id = f"smartclip:{info.model_name}"

        obj = {"vector": vec, "dim": len(vec), "model_id": model_id}
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=embedding_v1",
            {"dim": str(len(vec))},
        )

    def _handle_classify(
        self, payload: bytes, payload_mime: str, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
        """Handles intelligent image classification."""
        if not self.clip_model.supports_classification:
            raise RuntimeError(
                "Classification not supported: no ImageNet dataset loaded for CLIP model"
            )

        top_k = int(meta.get("topk", "5"))

        # Use CLIP for general classification
        scores = self.clip_model.classify_image(payload, top_k=top_k)

        info = self.clip_model.info()
        model_id = f"smartclip:{info.model_name}"

        obj = {
            "labels": [
                {"label": label, "score": float(score)} for label, score in scores
            ],
            "model_id": model_id,
        }
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=labels_v1",
            {"labels_count": str(len(scores))},
        )

    def _handle_scene_classify(
        self, payload: bytes, payload_mime: str, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
        """Handles scene classification using CLIP."""
        label, score = self.clip_model.classify_scene(payload)

        info = self.clip_model.info()
        model_id = f"smartclip:{info.model_name}"

        obj = {
            "labels": [{"label": label, "score": float(score)}],
            "model_id": model_id,
        }
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=labels_v1",
            {"labels_count": "1"},
        )

    def _handle_bioclassify(
        self, payload: bytes, payload_mime: str, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
        """Handles biological classification using BioCLIP."""
        if not self.bioclip_model.supports_classification:
            raise RuntimeError(
                "Bio classification not supported: no TreeOfLife dataset loaded for BioCLIP model"
            )

        namespace = meta.get("namespace", "bioatlas")
        if namespace != "bioatlas":
            raise ValueError(
                f"unsupported namespace {namespace!r}, expected 'bioatlas'"
            )

        topk = int(meta.get("topk", "5"))
        pairs = self.bioclip_model.classify_image(payload, top_k=topk)

        info = self.bioclip_model.info()
        model_id = f"smartclip:{info.model_name}"

        obj = {
            "labels": [{"label": name, "score": float(score)} for name, score in pairs],
            "model_id": model_id,
        }
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=labels_v1",
            {"labels_count": str(len(pairs))},
        )

    # -------- Helper Methods ----------

    def _assemble(
        self, cid: str, req: pb.InferRequest, buffers: Dict[str, bytearray]
    ) -> Tuple[bytes, bool]:
        """
        Reassembles chunked request payloads.
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

        return b"", False

    def _build_capability(self) -> pb.Capability:
        """Constructs the capability message using the task registry."""
        clip_info = self.clip_model.info()
        bioclip_info = self.bioclip_model.info()

        # Use registry to build capability automatically
        extra_metadata = {
            "device": clip_info.device,
            "clip_model": clip_info.model_name,
            "bioclip_model": bioclip_info.model_name,
            "clip_embedding_dim": str(clip_info.embedding_dim),
            "bioclip_embedding_dim": str(bioclip_info.embedding_dim),
            "supports_classification": str(
                clip_info.supports_classification
                or bioclip_info.supports_classification
            ),
        }

        return self.registry.build_capability(
            service_name="lumen_smartclip",
            model_id=f"{clip_info.model_name}+{bioclip_info.model_name}",
            runtime=clip_info.runtime,
            precisions=clip_info.precisions,
            extra_metadata=extra_metadata,
        )
