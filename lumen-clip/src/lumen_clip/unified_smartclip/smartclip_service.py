"""
unified_service.py

UnifiedCLIPService: A unified, high-performance gRPC service that combines the capabilities of
general CLIP and BioCLIP models.

Features:
- Exposes 7 tasks for classification and embedding (including image/text embeddings).
- Implements server-side request batching to maximize throughput.
- Uses backend-driven image batch embedding via backend.image_batch_to_vectors()
  (each backend may fall back to sequential processing if true batching is unavailable).
- No torchvision-based preprocessing here; image decoding/preprocessing is handled inside backends.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any, final
from typing_extensions import override

import grpc

from google.protobuf import empty_pb2
from lumen_clip.backends import TorchBackend, ONNXRTBackend, BaseClipBackend

# Import the service definition and model managers
import lumen_clip.proto.ml_service_pb2 as pb
import lumen_clip.proto.ml_service_pb2_grpc as rpc
from lumen_clip.expert_bioclip import BioCLIPModelManager
from lumen_clip.general_clip import CLIPModelManager
from lumen_clip.resources.loader import ModelResources, ResourceLoader
from lumen_resources.lumen_config import BackendSettings, ModelConfig

logger = logging.getLogger(__name__)

BATCH_SIZE = 8


class UnifiedCLIPService(rpc.InferenceServicer):
    """A unified gRPC service that intelligently routes requests to CLIP or BioCLIP models."""

    SERVICE_NAME = "clip-unified"

    def __init__(
        self,
        clip_backend: BaseClipBackend,
        bioclip_backend: BaseClipBackend,
        clip_resources: ModelResources,
        bioclip_resources: ModelResources,
    ) -> None:
        """
        Initialize UnifiedCLIPService.

        Args:
            clip_backend: Backend instance for CLIP model
            bioclip_backend: Backend instance for BioCLIP model
            clip_resources: ModelResources for CLIP model
            bioclip_resources: ModelResources for BioCLIP model
        """
        # Initialize model managers with the selected backends
        self.clip_resources = clip_resources
        self.bioclip_resources = bioclip_resources
        self.clip_model = CLIPModelManager(
            backend=clip_backend, resources=clip_resources
        )
        self.bioclip_model = BioCLIPModelManager(
            backend=bioclip_backend, resources=bioclip_resources
        )

        # Store backend references for device info queries if needed
        self._clip_backend = clip_backend
        self._bioclip_backend = bioclip_backend

    @classmethod
    def from_config(
        cls,
        general_model_config: ModelConfig,
        bioclip_model_config: ModelConfig,
        cache_dir: Path,
        backend_settings: BackendSettings | None,
    ) -> "UnifiedCLIPService":
        """
        Create UnifiedCLIPService from configurations for both models.

        Args:
            general_model_config: Configuration for the general-purpose CLIP model.
            bioclip_model_config: Configuration for the BioCLIP model.
            cache_dir: Cache directory path.

        Returns:
            Initialized UnifiedCLIPService instance.
        """
        from lumen_clip.resources.exceptions import ConfigError
        from lumen_clip.general_clip.clip_model import CLIPModelManager
        from lumen_clip.expert_bioclip.bioclip_model import BioCLIPModelManager

        # --- Helper function to create a backend from config ---
        def create_backend(resources: ModelResources, runtime: str):
            device_pref = getattr(backend_settings, "device", "cpu")
            max_batch_size = getattr(backend_settings, "batch_size", 8)
            if runtime == "torch":
                return TorchBackend(
                    resources=resources,
                    device_preference=device_pref,
                    max_batch_size=max_batch_size,
                )
            elif runtime == "onnx":
                providers_list = getattr(
                    backend_settings, "onnx_providers", ["CPUExecutionProvider"]
                )
                return ONNXRTBackend(
                    resources=resources,
                    providers=providers_list,
                    device_preference=device_pref,
                    max_batch_size=max_batch_size,
                )
            else:
                raise ConfigError(f"Unsupported runtime: {runtime}")

        # 1. Load resources and create backend for General CLIP
        logger.info(
            f"Loading resources for General CLIP model: {general_model_config.model}"
        )
        general_resources = ResourceLoader.load_model_resources(
            cache_dir, general_model_config
        )
        general_backend = create_backend(
            general_resources, general_model_config.runtime.value
        )

        # 2. Load resources and create backend for BioCLIP
        logger.info(
            f"Loading resources for BioCLIP model: {bioclip_model_config.model}"
        )
        bioclip_resources = ResourceLoader.load_model_resources(
            cache_dir, bioclip_model_config
        )
        bioclip_backend = create_backend(
            bioclip_resources, bioclip_model_config.runtime.value
        )

        # 3. Create the unified service instance
        service = cls(
            clip_backend=general_backend,
            bioclip_backend=bioclip_backend,
            clip_resources=general_resources,
            bioclip_resources=bioclip_resources,
        )

        logger.info("UnifiedCLIPService created with both General and BioCLIP models.")
        if not general_resources.has_classification_support():
            logger.warning("General CLIP is missing ImageNet dataset.")
        if not bioclip_resources.has_classification_support():
            logger.warning("BioCLIP is missing its dataset (e.g., TreeOfLife).")

        return service

    def initialize(self) -> None:
        """Initializes both underlying model managers."""
        logger.info("Initializing CLIPModelManager...")
        self.clip_model.initialize()
        logger.info("Initializing BioCLIPModelManager...")
        self.bioclip_model.initialize()
        logger.info("✅ All models initialized successfully.")

    @override
    def Infer(
        self, request_iterator: Iterable[pb.InferRequest], context: grpc.ServicerContext
    ) -> Iterable[pb.InferResponse]:
        """
        Handles bidirectional streaming inference with server-side batching.
        """
        if not self.clip_model.is_initialized or not self.bioclip_model.is_initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Models are not initialized."
            )
            return

        batch: list[pb.InferRequest] = []
        for req in request_iterator:
            batch.append(req)
            if len(batch) >= BATCH_SIZE:
                for response in self._process_batch(batch):
                    yield response
                batch.clear()

        # Process any remaining requests in the final, possibly smaller, batch
        if batch:
            for response in self._process_batch(batch):
                yield response

    def _process_batch(
        self, batch: list[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        """Processes a batch of requests, groups them by task, and yields responses."""
        t0 = time.time()
        logger.info(f"Processing a batch of {len(batch)} requests...")

        # Group requests by their task name
        tasks: dict[str, list[pb.InferRequest]] = defaultdict(list)
        for req in batch:
            tasks[req.task].append(req)

        # Process each task group
        for task_name, requests in tasks.items():
            handler = getattr(self, f"_handle_{task_name}", None)
            if handler:
                try:
                    for response in handler(requests):
                        yield response
                except Exception as e:
                    logger.exception(f"Error processing task '{task_name}': {e}")
                    # Yield an error response for all failed requests in this group
                    for req in requests:
                        yield pb.InferResponse(
                            correlation_id=req.correlation_id,
                            is_final=True,
                            error=pb.Error(code=pb.ERROR_CODE_INTERNAL, message=str(e)),
                        )
            else:
                logger.warning(f"Unknown task received: {task_name}")
                for req in requests:
                    yield pb.InferResponse(
                        correlation_id=req.correlation_id,
                        is_final=True,
                        error=pb.Error(
                            code=pb.ERROR_CODE_INVALID_ARGUMENT,
                            message=f"Unknown task: {task_name}",
                        ),
                    )

        processing_time = (time.time() - t0) * 1000
        logger.info(f"Batch processing finished in {processing_time:.2f} ms.")

    # --- Task Handlers (process lists of requests) ---

    def _handle_clip_classify(
        self, requests: list[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            top_k = int(req.meta.get("topk", "5"))
            scores = self.clip_model.classify_image(req.payload, top_k)
            yield self._build_label_response(
                req.correlation_id, scores, self.clip_model.info()
            )

    def _handle_bioclip_classify(
        self, requests: list[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            top_k = int(req.meta.get("topk", "3"))
            scores = self.bioclip_model.classify_image(req.payload, top_k)
            yield self._build_label_response(
                req.correlation_id, scores, self.bioclip_model.info()
            )

    def _handle_smart_classify(
        self, requests: list[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            # Check if both models support classification
            if not self.clip_model.supports_classification:
                yield pb.InferResponse(
                    correlation_id=req.correlation_id,
                    is_final=True,
                    error=pb.Error(
                        code=pb.ERROR_CODE_INVALID_ARGUMENT,
                        message="smart_classify requires CLIP classification support (no ImageNet dataset)",
                    ),
                )
                continue

            # 1) Scene classification using CLIP
            scene_label, scene_score = self.clip_model.classify_scene(req.payload)

            # Check if it's animal-like based on scene classification
            is_animal_like = (
                ("animal" in scene_label)
                or ("bird" in scene_label)
                or ("insect" in scene_label)
            )

            if not is_animal_like:
                yield self._build_label_response(
                    req.correlation_id,
                    [(scene_label, scene_score)],
                    self.clip_model.info(),
                    meta={"source": "scene_classification"},
                )
                continue

            # 2) Animal-like: classify with BioCLIP (if supported)
            if not self.bioclip_model.supports_classification:
                # Fallback to CLIP classification
                yield self._build_label_response(
                    req.correlation_id,
                    [(scene_label, scene_score)],
                    self.clip_model.info(),
                    meta={"source": "scene_classification_fallback"},
                )
                continue

            top_k = int(req.meta.get("topk", "3"))
            scores = self.bioclip_model.classify_image(req.payload, top_k)
            yield self._build_label_response(
                req.correlation_id,
                scores,
                self.bioclip_model.info(),
                meta={"source": "bioclip_classification"},
            )

    def _handle_clip_embed(
        self, requests: list[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            vec = self.clip_model.encode_text(req.payload.decode("utf-8")).tolist()
            yield self._build_embed_response(
                req.correlation_id, vec, self.clip_model.info()
            )

    def _handle_bioclip_embed(
        self, requests: list[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            vec = self.bioclip_model.encode_text(req.payload.decode("utf-8")).tolist()
            yield self._build_embed_response(
                req.correlation_id, vec, self.bioclip_model.info()
            )

    def _handle_clip_image_embed(
        self, requests: list[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        # Use backend batch embedding (falls back to sequential in backend if not supported)
        image_bytes = [req.payload for req in requests]
        vecs = self.clip_model.backend.image_batch_to_vectors(image_bytes)
        for i, req in enumerate(requests):
            vec = vecs[i].tolist()
            yield self._build_embed_response(
                req.correlation_id, vec, self.clip_model.info()
            )

    def _handle_bioclip_image_embed(
        self, requests: list[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        # Use backend batch embedding (falls back to sequential in backend if not supported)
        image_bytes = [req.payload for req in requests]
        vecs = self.bioclip_model.backend.image_batch_to_vectors(image_bytes)
        for i, req in enumerate(requests):
            vec = vecs[i].tolist()
            yield self._build_embed_response(
                req.correlation_id, vec, self.bioclip_model.info()
            )

    # --- Response Builders ---

    def _build_label_response(
        self,
        cid: str,
        scores: list[tuple[str, float]],
        model_info: Mapping[str, Any],
        meta: dict[str, str] | None = None,
    ) -> pb.InferResponse:
        model_id = f"{model_info.get('model_name', model_info.get('model_version'))}:{model_info.get('pretrained', '')}"
        obj = {
            "labels": [{"label": label, "score": score} for label, score in scores],
            "model_id": model_id.strip(":"),
        }
        response_meta = {"labels_count": str(len(scores))}
        if meta:
            response_meta.update(meta)
        return pb.InferResponse(
            correlation_id=cid,
            is_final=True,
            result=json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            result_mime="application/json;schema=labels_v1",
            meta=response_meta,
        )

    def _build_embed_response(
        self, cid: str, vec: list[float], model_info: Mapping[str, Any]
    ) -> pb.InferResponse:
        model_id = f"{model_info.get('model_name', model_info.get('model_version'))}:{model_info.get('pretrained', '')}"
        obj = {"vector": vec, "dim": len(vec), "model_id": model_id.strip(":")}
        return pb.InferResponse(
            correlation_id=cid,
            is_final=True,
            result=json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            result_mime="application/json;schema=embedding_v1",
            meta={"dim": str(len(vec))},
        )

    # --- Capabilities and Health ---

    @override
    def GetCapabilities(self, request, context) -> pb.Capability:
        return self._build_capability()

    @override
    def StreamCapabilities(self, request, context) -> Iterable[pb.Capability]:
        yield self._build_capability()

    @override
    def Health(self, request, context):
        return empty_pb2.Empty()

    def _build_capability(self) -> pb.Capability:
        """Constructs a unified capability message from both managed models."""
        general_res = self.clip_resources
        general_backend_info = self._clip_backend.get_info()
        bioclip_res = self.bioclip_resources
        bioclip_backend_info = self._bioclip_backend.get_info()

        tasks = [
            pb.IOTask(
                name="clip_embed",
                input_mimes=["text/plain;charset=utf-8"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
            pb.IOTask(
                name="clip_image_embed",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
            pb.IOTask(
                name="bioclip_embed",
                input_mimes=["text/plain;charset=utf-8"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
            pb.IOTask(
                name="bioclip_image_embed",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
        ]

        if self.clip_model.supports_classification:
            tasks.append(
                pb.IOTask(
                    name="clip_classify",
                    input_mimes=["image/jpeg", "image/png"],
                    output_mimes=["application/json;schema=labels_v1"],
                )
            )
            tasks.append(
                pb.IOTask(
                    name="clip_classify_scene",
                    input_mimes=["image/jpeg", "image/png"],
                    output_mimes=["application/json;schema=labels_v1"],
                )
            )

        if self.bioclip_model.supports_classification:
            tasks.append(
                pb.IOTask(
                    name="bioclip_classify",
                    input_mimes=["image/jpeg", "image/png"],
                    output_mimes=["application/json;schema=labels_v1"],
                )
            )

        if (
            self.clip_model.supports_classification
            and self.bioclip_model.supports_classification
        ):
            tasks.append(
                pb.IOTask(
                    name="smart_classify",
                    input_mimes=["image/jpeg", "image/png"],
                    output_mimes=["application/json;schema=labels_v1"],
                )
            )

        return pb.Capability(
            service_name="clip-unified",
            model_ids=[general_res.model_info.name, bioclip_res.model_info.name],
            runtime=general_res.runtime,
            max_concurrency=4,
            precisions=general_backend_info.precisions or ["unknown"],
            extra={
                "general_model_name": general_res.model_info.name,
                "general_model_version": general_res.model_info.version,
                "general_runtime": general_res.runtime,
                "general_device": general_backend_info.device or "unknown",
                "general_embedding_dim": str(general_res.get_embedding_dim()),
                "bioclip_model_name": bioclip_res.model_info.name,
                "bioclip_model_version": bioclip_res.model_info.version,
                "bioclip_runtime": bioclip_res.runtime,
                "bioclip_device": bioclip_backend_info.device or "unknown",
                "bioclip_embedding_dim": str(bioclip_res.get_embedding_dim()),
            },
            tasks=tasks,
        )
