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
from lumen_resources.lumen_config import BackendSettings, ModelConfig

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
    def from_config(
        cls,
        model_config: ModelConfig,
        cache_dir: Path,
        backend_settings: BackendSettings | None,
    ):
        """
        Create GeneralCLIPService from configuration.

        Args:
            model_config: The specific model's configuration from lumen_config.
            cache_dir: Cache directory path.

        Returns:
            Initialized GeneralCLIPService instance.
        """
        from resources.exceptions import ConfigError

        # Load resources using the validated model_config
        logger.info(f"Loading resources for General CLIP model: {model_config.model}")
        resources = ResourceLoader.load_model_resources(cache_dir, model_config)

        # Create backend based on runtime
        runtime = model_config.runtime.value
        device_pref = getattr(backend_settings, "device", "cpu")
        max_batch_size = getattr(backend_settings, "batch_size", 8)
        if runtime == "torch":
            backend = TorchBackend(
                resources=resources,
                device_preference=device_pref,
                max_batch_size=max_batch_size,
            )
        elif runtime == "onnx":
            providers_list = getattr(
                backend_settings, "onnx_providers", ["CPUExecutionProvider"]
            )
            backend = ONNXRTBackend(
                resources=resources,
                providers=providers_list,
                device_preference=device_pref,
                max_batch_size=max_batch_size,
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
            "CLIP model ready: %s with %s (loaded in %.2fs)",
            info.get("model_name"),
            info.get("backend_info", {}) or "unknown",
            info.get("load_time", 0.0),
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
        """Constructs the capability message from authoritative sources."""
        res = self.model.resources
        # CORRECT: Call .get_info() which returns a BackendInfo object.
        backend_info = self.model.backend.get_info()

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

        return pb.Capability(
            service_name=self.SERVICE_NAME,
            model_ids=[res.model_info.name],
            runtime=res.runtime,
            max_concurrency=4,
            # CORRECT: Access info via attributes, not dict keys.
            precisions=backend_info.precisions or ["unknown"],
            extra={
                "device": backend_info.device or "unknown",
                "embedding_dim": str(res.get_embedding_dim()),
                "model_version": res.model_info.version,
                "supports_classification": str(self.model.supports_classification),
            },
            tasks=tasks,
        )
