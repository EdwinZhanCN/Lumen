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
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import grpc

from google.protobuf import empty_pb2
from backends import TorchBackend, ONNXRTBackend

# Import the service definition and model managers
import ml_service_pb2 as pb
import ml_service_pb2_grpc as rpc
from biological_atlas import BioCLIPModelManager
from image_classification import CLIPModelManager
from resources.loader import ModelResources, ResourceLoader

# --- Constants ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
logger = logging.getLogger(__name__)


class UnifiedCLIPService(rpc.InferenceServicer):
    """A unified gRPC service that intelligently routes requests to CLIP or BioCLIP models."""

    SERVICE_NAME = "clip-unified"

    def __init__(
        self,
        clip_backend,
        bioclip_backend,
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
    def from_config(cls, config: dict, cache_dir: Path):
        """
        Create UnifiedCLIPService from configuration.

        Args:
            config: Service configuration dict
            cache_dir: Cache directory path

        Returns:
            Initialized UnifiedCLIPService instance

        Raises:
            ResourceNotFoundError: If required resources are missing
            ConfigError: If configuration is invalid
        """
        from resources.exceptions import ResourceNotFoundError, ConfigError

        # Validate that both models are configured
        if "models" not in config:
            raise ConfigError("Unified service requires 'models' configuration")

        if "clip_default" not in config["models"]:
            raise ConfigError(
                "Unified service requires 'models.clip_default' configuration"
            )

        if "bioclip_default" not in config["models"]:
            raise ConfigError(
                "Unified service requires 'models.bioclip_default' configuration"
            )

        # Load CLIP resources
        clip_model_cfg = config["models"]["clip_default"]
        logger.info(f"Loading resources for CLIP model: {clip_model_cfg['model']}")
        clip_resources = ResourceLoader.load_model_resources(
            cache_dir=cache_dir,
            model_name=clip_model_cfg["model"],
            runtime=clip_model_cfg["runtime"],
        )

        # Load BioCLIP resources
        bioclip_model_cfg = config["models"]["bioclip_default"]
        bioclip_dataset = bioclip_model_cfg.get("dataset", "TreeOfLife-10M")
        logger.info(
            f"Loading resources for BioCLIP model: {bioclip_model_cfg['model']}"
        )
        bioclip_resources = ResourceLoader.load_model_resources(
            cache_dir=cache_dir,
            model_name=bioclip_model_cfg["model"],
            runtime=bioclip_model_cfg["runtime"],
            dataset=bioclip_dataset,
        )

        # Create backends based on runtime
        runtime = clip_model_cfg["runtime"]
        batch_size = int(config.get("env", {}).get("BATCH_SIZE", 8))
        device = config.get("env", {}).get("DEVICE")

        if runtime == "torch":
            clip_backend = TorchBackend(
                resources=clip_resources,
                device_preference=device,
                max_batch_size=batch_size,
            )
            bioclip_backend = TorchBackend(
                resources=bioclip_resources,
                device_preference=device,
                max_batch_size=batch_size,
            )
        elif runtime == "onnx":
            providers = config.get("env", {}).get(
                "ONNX_PROVIDERS", "CPUExecutionProvider"
            )
            providers_list = [p.strip() for p in providers.split(",")]
            clip_backend = ONNXRTBackend(
                resources=clip_resources,
                providers=providers_list,
                device_preference=device,
                max_batch_size=batch_size,
            )
            bioclip_backend = ONNXRTBackend(
                resources=bioclip_resources,
                providers=providers_list,
                device_preference=device,
                max_batch_size=batch_size,
            )
        else:
            raise ConfigError(f"Unsupported runtime: {runtime}")

        # Create service
        service = cls(clip_backend, bioclip_backend, clip_resources, bioclip_resources)

        # Log warnings if classification not supported
        if not clip_resources.has_classification_support():
            logger.warning(
                "Unified service: CLIP classification disabled (no ImageNet dataset)"
            )
        if not bioclip_resources.has_classification_support():
            logger.warning(
                f"Unified service: BioCLIP classification disabled (no {bioclip_dataset} dataset)"
            )

        return service

    def initialize(self) -> None:
        """Initializes both underlying model managers."""
        logger.info("Initializing CLIPModelManager...")
        self.clip_model.initialize()
        logger.info("Initializing BioCLIPModelManager...")
        self.bioclip_model.initialize()
        logger.info("✅ All models initialized successfully.")

    def Infer(
        self, request_iterator: Iterable[pb.InferRequest], context: grpc.ServicerContext
    ):
        """
        Handles bidirectional streaming inference with server-side batching.
        """
        if not self.clip_model.is_initialized or not self.bioclip_model.is_initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Models are not initialized."
            )
            return

        batch: List[pb.InferRequest] = []
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
        self, batch: List[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        """Processes a batch of requests, groups them by task, and yields responses."""
        t0 = time.time()
        logger.info(f"Processing a batch of {len(batch)} requests...")

        # Group requests by their task name
        tasks = defaultdict(list)
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
        self, requests: List[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            top_k = int(req.meta.get("topk", "5"))
            scores = self.clip_model.classify_image(req.payload, top_k)
            yield self._build_label_response(
                req.correlation_id, scores, self.clip_model.info()
            )

    def _handle_bioclip_classify(
        self, requests: List[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            top_k = int(req.meta.get("topk", "3"))
            scores = self.bioclip_model.classify_image(req.payload, top_k)
            yield self._build_label_response(
                req.correlation_id, scores, self.bioclip_model.info()
            )

    def _handle_smart_classify(
        self, requests: List[pb.InferRequest]
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
        self, requests: List[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            vec = self.clip_model.encode_text(req.payload.decode("utf-8")).tolist()
            yield self._build_embed_response(
                req.correlation_id, vec, self.clip_model.info()
            )

    def _handle_bioclip_embed(
        self, requests: List[pb.InferRequest]
    ) -> Iterable[pb.InferResponse]:
        for req in requests:
            vec = self.bioclip_model.encode_text(req.payload.decode("utf-8")).tolist()
            yield self._build_embed_response(
                req.correlation_id, vec, self.bioclip_model.info()
            )

    def _handle_clip_image_embed(
        self, requests: List[pb.InferRequest]
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
        self, requests: List[pb.InferRequest]
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
        scores: List[Tuple[str, float]],
        model_info: dict,
        meta: dict | None = None,
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
        self, cid: str, vec: list, model_info: dict
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

    def GetCapabilities(self, request, context) -> pb.Capability:
        return self._build_capability()

    def StreamCapabilities(self, request, context) -> Iterable[pb.Capability]:
        yield self._build_capability()

    def Health(self, request, context):
        return empty_pb2.Empty()

    def _build_capability(self) -> pb.Capability:
        # Base tasks (always available)
        tasks = [
            pb.IOTask(
                name="clip_embed",
                input_mimes=["text/plain"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
            pb.IOTask(
                name="bioclip_embed",
                input_mimes=["text/plain"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
            pb.IOTask(
                name="clip_image_embed",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
            pb.IOTask(
                name="bioclip_image_embed",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
        ]

        # Classification tasks (only if datasets are available)
        if self.clip_model.supports_classification:
            tasks.append(
                pb.IOTask(
                    name="clip_classify",
                    input_mimes=["image/jpeg", "image/png", "image/webp"],
                    output_mimes=["application/json;schema=labels_v1"],
                )
            )

        if self.bioclip_model.supports_classification:
            tasks.append(
                pb.IOTask(
                    name="bioclip_classify",
                    input_mimes=["image/jpeg", "image/png", "image/webp"],
                    output_mimes=["application/json;schema=labels_v1"],
                )
            )

        # Smart classify (only if CLIP classification is available)
        if self.clip_model.supports_classification:
            tasks.append(
                pb.IOTask(
                    name="smart_classify",
                    input_mimes=["image/jpeg", "image/png", "image/webp"],
                    output_mimes=["application/json;schema=labels_v1"],
                )
            )

        # Report effective runtime; if mixed backends are used, declare "mixed"
        clip_rt = self.clip_model.backend.get_info().runtime
        bio_rt = self.bioclip_model.backend.get_info().runtime
        effective_runtime = clip_rt if clip_rt == bio_rt else "mixed"

        return pb.Capability(
            service_name=self.SERVICE_NAME,
            model_ids=[self.clip_model.model_id, self.bioclip_model.model_id],
            runtime=effective_runtime,
            max_concurrency=16,
            tasks=tasks,
            extra={
                "batch_size": str(BATCH_SIZE),
                "runtime.clip": clip_rt,
                "runtime.bioclip": bio_rt,
                "clip_classification": str(self.clip_model.supports_classification),
                "bioclip_classification": str(
                    self.bioclip_model.supports_classification
                ),
            },
        )


# Backward compatibility alias
CLIPService = UnifiedCLIPService
