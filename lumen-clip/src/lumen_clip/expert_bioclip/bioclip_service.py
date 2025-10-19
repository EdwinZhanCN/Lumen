"""
BioCLIP gRPC service.

This module implements a gRPC Inference service for the BioCLIP model. It
provides streaming request/response handling and exposes the following tasks
via the Inference protocol:

- "embed": Create a text embedding from a UTF-8 string.
- "image_embed": Create an image embedding from binary image data.
- "classify": Classify an image using the TreeOfLife dataset (when available).

The service handles model initialization, optional fragment reassembly for
large payloads, error reporting, and capability advertising for ML serving
infrastructure integration.
"""

import json
import logging
import time
from pathlib import Path
from collections.abc import Iterable
from typing_extensions import override

import grpc
from google.protobuf import empty_pb2

import lumen_clip.proto.ml_service_pb2 as pb
import lumen_clip.proto.ml_service_pb2_grpc as rpc
from lumen_clip.backends import BaseClipBackend, TorchBackend, ONNXRTBackend
from lumen_clip.resources.loader import ModelResources, ResourceLoader
from .bioclip_model import BioCLIPModelManager
from lumen_resources.lumen_config import BackendSettings, ModelConfig

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


class BioCLIPService(rpc.InferenceServicer):
    """
    gRPC InferenceServicer for the BioCLIP model.

    This service implements the Inference protocol and exposes handlers for:
      - Text embedding (task="embed")
      - Image embedding (task="image_embed")
      - TreeOfLife classification (task="classify", requires meta.namespace="bioatlas")

    Attributes:
        MODEL_VERSION: Default model version string.
        model: `BioCLIPModelManager` instance used for inference.
        start_time: Timestamp when the service instance was created.
        is_initialized: Boolean indicating whether the model has been initialized.
    """

    MODEL_VERSION: str = "bioclip2"

    def __init__(self, backend: BaseClipBackend, resources: ModelResources) -> None:
        """
        Initialize BioCLIPService.

        Args:
            backend: Backend instance (TorchBackend or ONNXRTBackend)
            resources: ModelResources with model configuration and data
        """
        self.model: BioCLIPModelManager = BioCLIPModelManager(
            backend=backend, resources=resources
        )
        self.start_time: float = time.time()
        self.is_initialized: bool = False

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        cache_dir: Path,
        backend_settings: BackendSettings | None,
    ):
        """
        Create BioCLIPService from configuration.

        Args:
            model_config: ModelConfig instance
            cache_dir: Cache directory path
            backend_settings: BackendSettings instance or None

        Returns:
            Initialized BioCLIPService instance

        Raises:
            ResourceNotFoundError: If required resources are missing
            ConfigError: If configuration is invalid
        """
        from lumen_clip.resources.exceptions import ConfigError

        # Get dataset name if specified
        dataset = model_config.dataset

        if dataset is None:
            logger.warning("Dataset not specified classify feature will be disabled")

        # Load resources
        logger.info(f"Loading resources for BioCLIP model: {model_config.model}")
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
                "BioCLIP service started without classification support "
                + f"(no {dataset} dataset found). Only embed tasks will be available."
            )

        return service

    # -------- lifecycle ----------
    def initialize(self) -> None:
        logger.info("Initializing BioCLIP model...")
        self.model.initialize()
        self.is_initialized = True
        info = self.model.info()
        logger.info(
            "BioCLIP ready on %s (load %.2fs)",
            info.get("device"),
            info.get("load_time"),
        )

    # -------- Inference ----------
    @override
    def Infer(
        self, request_iterator: Iterable[pb.InferRequest], context: grpc.ServicerContext
    ):
        """
        Bidirectional streaming server implementation (synchronous style: generator yields responses on demand).
        Both embed and classify are one-request-one-response; supports reassembly if client uses seq/total for fragmentation.
        """
        if not self.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        buffers: dict[str, bytearray] = {}  # correlation_id -> buffer

        for req in request_iterator:
            cid = req.correlation_id or f"cid-{_now_ms()}"
            t0 = _now_ms()

            try:
                # --- Fragment reassembly (optional) ---
                payload, ready = self._assemble(cid, req, buffers)
                if not ready:
                    # Fragment not yet complete; wait for additional fragments
                    continue

                # --- Route to task handler ---
                if req.task == "embed":
                    result_bytes, result_mime, extra_meta = self._handle_embed(
                        req.payload_mime, payload, dict(req.meta)
                    )
                elif req.task == "classify":
                    # Check if classification is supported
                    if not self.model.supports_classification:
                        yield pb.InferResponse(
                            correlation_id=cid,
                            is_final=True,
                            error=pb.Error(
                                code=pb.ERROR_CODE_INVALID_ARGUMENT,
                                message="Classification not supported: no dataset loaded. "
                                + "Please ensure TreeOfLife dataset exists in the model directory.",
                            ),
                        )
                        continue
                    result_bytes, result_mime, extra_meta = self._handle_classify(
                        req.payload_mime, payload, dict(req.meta)
                    )
                elif req.task == "image_embed":
                    result_bytes, result_mime, extra_meta = self._handle_image_embed(
                        req.payload_mime, payload, dict(req.meta)
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

                # --- Successful response ---
                meta = dict(extra_meta or {})
                meta["lat_ms"] = str(_now_ms() - t0)

                yield pb.InferResponse(
                    correlation_id=cid,
                    is_final=True,
                    result=result_bytes,
                    result_mime=result_mime,  # e.g. application/json;schema=embedding_v1
                    meta=meta,
                    seq=0,
                    total=1,
                    offset=0,
                    result_schema="",  # May be left empty; use "embedding_v1" or "labels_v1" if needed
                )

            except grpc.RpcError:
                raise
            except Exception as e:
                logger.exception("Infer error: %s", e)
                yield pb.InferResponse(
                    correlation_id=cid,
                    is_final=True,
                    error=pb.Error(code=pb.ERROR_CODE_INTERNAL, message=str(e)),
                )

    # -------- Capabilities / Health ----------
    @override
    def GetCapabilities(self, request, context) -> pb.Capability:
        return self._build_capability()

    @override
    def StreamCapabilities(self, request, context):
        # Send a single capability message for this instance. If capabilities can
        # change at runtime, this could be adapted to stream updates periodically
        # or on demand.
        yield self._build_capability()

    @override
    def Health(self, request, context):
        # Extend to provide richer health information if needed. The current
        # protocol expects an empty response.
        return empty_pb2.Empty()

    # -------- IOTask：embed / classify ----------
    def _handle_embed(
        self, payload_mime: str, payload: bytes, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """
        Text embedding:
          - Expects payload_mime: "text/plain;charset=utf-8"
          - Output result_mime: "application/json;schema=embedding_v1"
          - Output JSON: {"vector":[...], "dim":768, "model_id":"bioclip2"}
        """
        if not payload_mime.startswith("text/"):
            raise ValueError(f"embed expects text/* payload, got {payload_mime!r}")
        text = payload.decode("utf-8")
        vec = self.model.encode_text(text).tolist()
        info = self.model.info()
        model_id = info.get("model_id", self.MODEL_VERSION)
        obj = {"vector": vec, "dim": len(vec), "model_id": model_id}
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=embedding_v1",
            {"dim": str(len(vec))},
        )

    def _handle_image_embed(
        self, payload_mime: str, payload: bytes, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """
        Image embedding:
          - Expects payload_mime: "image/jpeg" / "image/png" / "image/webp"
          - Output result_mime: "application/json;schema=embedding_v1"
          - Output JSON: {"vector":[...], "dim":512, "model_id":"bioclip2"}
        """
        if not payload_mime.startswith("image/"):
            raise ValueError(
                f"image_embed expects image/* payload, got {payload_mime!r}"
            )

        vec = self.model.encode_image(payload).tolist()
        info = self.model.info()
        model_id = info.get("model_id", self.MODEL_VERSION)
        obj = {"vector": vec, "dim": len(vec), "model_id": model_id}
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=embedding_v1",
            {"dim": str(len(vec))},
        )

    def _handle_classify(
        self, payload_mime: str, payload: bytes, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """
        TreeOfLife classification:
          - Expects payload_mime: "image/jpeg" / "image/png"
          - Expects meta.namespace="bioatlas"
          - Optional meta.topk (default 5)
          - Output result_mime: "application/json;schema=labels_v1"
          - Output JSON: {"labels":[{"label":"...","score":0.91},...], "model_id":"bioclip2"}
        """
        if not payload_mime.startswith("image/"):
            raise ValueError(f"classify expects image/* payload, got {payload_mime!r}")

        namespace = (meta or {}).get("namespace", "bioatlas")
        if namespace != "bioatlas":
            raise ValueError(
                f"unsupported namespace {namespace!r}, expected 'bioatlas'"
            )

        topk = int((meta or {}).get("topk", "5"))
        pairs = self.model.classify_image(
            payload, top_k=topk
        )  # list[tuple[str, float]]

        info = self.model.info()
        model_id = info.get("model_id", self.MODEL_VERSION)
        obj = {
            "labels": [{"label": name, "score": float(score)} for name, score in pairs],
            "model_id": model_id,
        }
        return (
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json;schema=labels_v1",
            {"labels_count": str(len(pairs))},
        )

    # -------- buffers / 分片重组 ----------
    def _assemble(
        self, cid: str, req: pb.InferRequest, buffers: dict[str, bytearray]
    ) -> tuple[bytes, bool]:
        """
        Return a tuple (payload_bytes, ready).

        - If the client does not use seq/total: ready is True immediately.
        - If seq/total are used: buffer fragments in buffers[cid] until all
          fragments are received (seq + 1 == total); then ready becomes True.
        """
        # No fragmentation (default path)
        if not req.total and not req.seq and not req.offset:
            return bytes(req.payload), True

        buf = buffers.setdefault(cid, bytearray())
        buf.extend(req.payload)
        if req.total and (req.seq + 1 == req.total):
            data = bytes(buf)
            del buffers[cid]
            return data, True
        return b"", False

    def _build_capability(self) -> pb.Capability:
        """Constructs the capability message from authoritative sources."""
        # Authoritative sources: model resources and the BackendInfo object.
        res = self.model.resources
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
            tasks.append(
                pb.IOTask(
                    name="classify",
                    input_mimes=["image/jpeg", "image/png", "image/webp"],
                    output_mimes=["application/json;schema=labels_v1"],
                    limits={"topk_max": "50"},
                )
            )

        return pb.Capability(
            service_name="clip-bioclip",
            model_ids=[res.model_info.name],
            runtime=res.runtime,
            max_concurrency=4,
            precisions=backend_info.precisions or ["unknown"],
            extra={
                "device": backend_info.device or "unknown",
                "embedding_dim": str(res.get_embedding_dim()),
                "model_version": res.model_info.version,
                "supports_classification": str(self.model.supports_classification),
            },
            tasks=tasks,
        )
