# clip_inference_service.py
import json
import logging
import time
from pathlib import Path
from collections.abc import Iterable
from typing_extensions import override

import grpc
from google.protobuf import empty_pb2

import ml_service_pb2 as pb
import ml_service_pb2_grpc as rpc
from backends import TorchBackend, ONNXRTBackend
from resources.loader import ModelResources, ResourceLoader
from .bioclip_model import BioCLIPModelManager
from lumen_resources.lumen_config import BackendSettings, ModelConfig
import os

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


class BioCLIPService(rpc.InferenceServicer):
    """
    使用新的 Inference 协议，统一承载两个服务：
      - 文本嵌入：task="embed"
      - BioAtlas 分类：task="classify", meta.namespace="bioatlas"
    """

    MODEL_VERSION: str = "bioclip2"

    def __init__(self, backend, resources: ModelResources) -> None:
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
            config: Service configuration dict
            cache_dir: Cache directory path

        Returns:
            Initialized BioCLIPService instance

        Raises:
            ResourceNotFoundError: If required resources are missing
            ConfigError: If configuration is invalid
        """
        from resources.exceptions import ConfigError

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
                f"BioCLIP service started without classification support "
                f"(no {dataset} dataset found). Only embed tasks will be available."
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
        双向流的服务端实现（同步 style：生成器按需 yield 响应）。
        embed / classify 都是一发一收；如客户端使用了分片 seq/total，这里也支持重组。
        """
        if not self.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        buffers: dict[str, bytearray] = {}  # correlation_id -> buffer

        for req in request_iterator:
            cid = req.correlation_id or f"cid-{_now_ms()}"
            t0 = _now_ms()

            try:
                # --- 分片重组（可选）---
                payload, ready = self._assemble(cid, req, buffers)
                if not ready:
                    # 分片尚未收齐，不返回响应，继续等下一个分片
                    continue

                # --- 路由任务 ---
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
                                "Please ensure TreeOfLife dataset exists in the model directory.",
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

                # --- 成功响应 ---
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
                    result_schema="",  # 可留空；有需要再填 "embedding_v1"/"labels_v1"
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
    def GetCapabilities(self, request, context) -> pb.Capability:
        return self._build_capability()

    @override
    def StreamCapabilities(self, request, context):
        # 单实例就发一条；如果后续有动态热更，可以定时/按需再发
        yield self._build_capability()

    @override
    def Health(self, request, context):
        # 需要更细的健康信息可以扩展；当前协议就是 Empty
        return empty_pb2.Empty()

    # -------- 具体任务：embed / classify ----------
    def _handle_embed(
        self, payload_mime: str, payload: bytes, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """
        文本嵌入：
          - 期望 payload_mime: "text/plain;charset=utf-8"
          - 输出 result_mime:  "application/json;schema=embedding_v1"
          - 输出 JSON: {"vector":[...], "dim":768, "model_id":"bioclip2"}
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
        BioAtlas 分类：
          - 期望 payload_mime: "image/jpeg" / "image/png"
          - 期望 meta.namespace="bioatlas"
          - 可选 meta.topk（默认 5）
          - 输出 result_mime: "application/json;schema=labels_v1"
          - 输出 JSON: {"labels":[{"label":"...","score":0.91},...], "model_id":"bioclip2"}
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
        返回 (payload_bytes, ready)
        - 若客户端不使用 seq/total：直接 ready=True
        - 若使用 seq/total：缓存到 buffers[cid]，直到收齐（seq+1==total）才 ready=True
        """
        # 无分片（默认路径）
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
