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
from pathlib import Path
from typing import Dict, Iterable, Tuple

import grpc
from google.protobuf import empty_pb2

import ml_service_pb2 as pb
import ml_service_pb2_grpc as rpc
from backends import TorchBackend, ONNXRTBackend
from resources.loader import ModelResources, ResourceLoader
from .clip_model import CLIPModelManager

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Returns the current time in milliseconds."""
    return int(time.time() * 1000)


class CLIPService(rpc.InferenceServicer):
    """
    Implements the streaming Inference service contract for a general-purpose
    OpenCLIP model, offering text embedding, ImageNet classification, and
    scene classification tasks.
    """

    SERVICE_NAME = "clip-general"

    def __init__(self, backend, resources: ModelResources) -> None:
        """
        Initialize CLIPService.

        Args:
            backend: Backend instance (TorchBackend or ONNXRTBackend)
            resources: ModelResources with model configuration and data
        """
        self.model = CLIPModelManager(backend=backend, resources=resources)
        self.is_initialized = False

    @classmethod
    def from_config(cls, config: dict, cache_dir: Path):
        """
        Create CLIPService from configuration.

        Args:
            config: Service configuration dict
            cache_dir: Cache directory path

        Returns:
            Initialized CLIPService instance

        Raises:
            ResourceNotFoundError: If required resources are missing
            ConfigError: If configuration is invalid
        """
        from resources.exceptions import ResourceNotFoundError, ConfigError

        # Get model configuration
        if "models" not in config or "default" not in config["models"]:
            raise ConfigError("CLIP service requires 'models.default' configuration")

        model_cfg = config["models"]["default"]

        # Load resources
        logger.info(f"Loading resources for CLIP model: {model_cfg['model']}")
        resources = ResourceLoader.load_model_resources(
            cache_dir=cache_dir,
            model_name=model_cfg["model"],
            runtime=model_cfg["runtime"],
        )

        # Create backend based on runtime
        runtime = model_cfg["runtime"]
        if runtime == "torch":
            backend = TorchBackend(
                resources=resources,
                device_preference=config.get("env", {}).get("DEVICE"),
                max_batch_size=int(config.get("env", {}).get("BATCH_SIZE", 8)),
            )
        elif runtime == "onnx":
            providers = config.get("env", {}).get(
                "ONNX_PROVIDERS", "CPUExecutionProvider"
            )
            providers_list = [p.strip() for p in providers.split(",")]
            backend = ONNXRTBackend(
                resources=resources,
                providers=providers_list,
                device_preference=config.get("env", {}).get("DEVICE"),
                max_batch_size=int(config.get("env", {}).get("BATCH_SIZE", 8)),
            )
        else:
            raise ConfigError(f"Unsupported runtime: {runtime}")

        # Create service
        service = cls(backend, resources)

        # Log classification support status
        if not resources.has_classification_support():
            logger.warning(
                "CLIP service started without classification support "
                "(no ImageNet dataset found). Only embed tasks will be available."
            )

        return service

    def initialize(self) -> None:
        """Loads the model and prepares it for inference."""
        logger.info("Initializing CLIPModelManager...")
        self.model.initialize()
        self.is_initialized = True
        info = self.model.info()
        logger.info(
            "CLIP model ready: %s (%s) on %s (loaded in %.2fs)",
            info.get("model_name"),
            info.get("pretrained"),
            info.get("device"),
            info.get("load_time_seconds"),
        )

    # -------- gRPC Service Methods ----------

    def Infer(
        self, request_iterator: Iterable[pb.InferRequest], context: grpc.ServicerContext
    ):
        """
        Handles the bidirectional streaming inference RPC. It routes incoming requests
        to the appropriate task handler.
        """
        if not self.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        buffers: Dict[str, bytearray] = {}  # Buffers for reassembling chunked requests

        for req in request_iterator:
            cid = req.correlation_id or f"cid-{_now_ms()}"
            t0 = _now_ms()

            try:
                # 1. Reassemble payload if it was sent in chunks
                payload, ready = self._assemble(cid, req, buffers)
                if not ready:
                    continue

                # 2. Route to the correct handler based on the task
                if req.task == "embed":
                    result_bytes, result_mime, extra_meta = self._handle_embed(
                        payload, dict(req.meta)
                    )
                elif req.task == "image_embed":
                    result_bytes, result_mime, extra_meta = self._handle_image_embed(
                        payload, dict(req.meta)
                    )
                elif req.task == "classify":
                    result_bytes, result_mime, extra_meta = self._handle_classify(
                        payload, dict(req.meta)
                    )
                elif req.task == "classify_scene":
                    result_bytes, result_mime, extra_meta = self._handle_classify_scene(
                        payload, dict(req.meta)
                    )
                else:
                    yield pb.InferResponse(
                        correlation_id=cid,
                        is_final=True,
                        error=pb.Error(
                            code=pb.ERROR_CODE_INVALID_ARGUMENT,
                            message=f"Unknown task: {req.task}",
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

    def GetCapabilities(self, request, context) -> pb.Capability:
        """Returns the capabilities of the service in a single response. [cite: 21]"""
        return self._build_capability()

    def StreamCapabilities(self, request, context) -> Iterable[pb.Capability]:
        """Streams the capabilities of the service."""
        yield self._build_capability()

    def Health(self, request, context):
        """A simple health check endpoint. [cite: 22]"""
        return empty_pb2.Empty()

    # -------- Task Handlers ----------

    def _handle_embed(
        self, payload: bytes, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
        """Handles text embedding requests."""
        text = payload.decode("utf-8")
        vec = self.model.encode_text(text).tolist()

        info = self.model.info()
        model_id = f"{info.get('model_name')}:{info.get('pretrained')}"

        obj = {"vector": vec, "dim": len(vec), "model_id": model_id}
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=embedding_v1",
            {"dim": str(len(vec))},
        )

    def _handle_classify(
        self, payload: bytes, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
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
        model_id = info.get("model_id", "unknown")

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

    def _handle_classify_scene(
        self, payload: bytes, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
        """Handles scene classification requests."""
        label, score = self.model.classify_scene(payload)

        info = self.model.info()
        model_id = info.get("model_id", "unknown")

        # The labels_v1 schema supports multiple labels, so we format the single result into a list
        obj = {
            "labels": [{"label": label, "score": float(score)}],
            "model_id": model_id,
        }
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=labels_v1",
            {"labels_count": "1"},
        )

    def _handle_image_embed(
        self, payload: bytes, meta: Dict[str, str]
    ) -> Tuple[bytes, str, Dict[str, str]]:
        """Handles image embedding requests."""
        vec = self.model.encode_image(payload).tolist()

        info = self.model.info()
        model_id = info.get("model_id", "unknown")

        obj = {"vector": vec, "dim": len(vec), "model_id": model_id}
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=embedding_v1",
            {"dim": str(len(vec))},
        )

    # -------- Helpers ----------

    def _assemble(
        self, cid: str, req: pb.InferRequest, buffers: Dict[str, bytearray]
    ) -> Tuple[bytes, bool]:
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
        """Constructs the capability message based on the model's current state."""
        info = self.model.info()
        backend_info_raw = info.get("backend_info", {})
        backend_info = backend_info_raw if isinstance(backend_info_raw, dict) else {}
        model_id_raw = info.get("model_id", "unknown")
        model_id = str(model_id_raw) if model_id_raw is not None else "unknown"

        # Base tasks (always available)
        tasks = [
            pb.IOTask(
                name="embed",
                input_mimes=["text/plain;charset=utf-8"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
            pb.IOTask(
                name="image_embed",
                input_mimes=["image/jpeg", "image/png", "image/webp"],
                output_mimes=["application/json;schema=embedding_v1"],
            ),
        ]

        # Classification tasks (only if dataset is available)
        if self.model.supports_classification:
            tasks.extend(
                [
                    pb.IOTask(
                        name="classify",
                        input_mimes=["image/jpeg", "image/png", "image/webp"],
                        output_mimes=["application/json;schema=labels_v1"],
                        limits={"topk_max": "50"},
                    ),
                    pb.IOTask(
                        name="classify_scene",
                        input_mimes=["image/jpeg", "image/png", "image/webp"],
                        output_mimes=["application/json;schema=labels_v1"],
                    ),
                ]
            )

        runtime = backend_info.get("runtime", "unknown")
        runtime_str = str(runtime) if runtime is not None else "unknown"

        precisions_raw = backend_info.get("precisions", ["fp32"])
        precisions = precisions_raw if isinstance(precisions_raw, list) else ["fp32"]

        device = backend_info.get("device", "unknown")
        device_str = str(device) if device is not None else "unknown"

        embedding_dim = backend_info.get("embedding_dim", "unknown")
        embedding_dim_str = (
            str(embedding_dim) if embedding_dim is not None else "unknown"
        )

        return pb.Capability(
            service_name=self.SERVICE_NAME,
            model_ids=[model_id],
            runtime=runtime_str,
            max_concurrency=4,
            precisions=precisions,
            extra={
                "device": device_str,
                "embedding_dim": embedding_dim_str,
                "supports_classification": str(self.model.supports_classification),
            },
            tasks=tasks,
        )
