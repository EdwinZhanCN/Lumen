"""
OCR Service Implementation

This module implements the gRPC service for Lumen-OCR, handling requests,
managing model lifecycle, and formatting responses. It serves as the bridge
between the gRPC interface and the backend model logic.
"""

import json
import logging
import time
from typing import Any, Optional

import grpc
from google.protobuf import empty_pb2
from lumen_resources import OCRV1
from lumen_resources.lumen_config import LumenConfig

from lumen_ocr.proto import ml_service_pb2, ml_service_pb2_grpc

from ..general_ocr.ocr_model import OcrModelManager
from ..registry import TaskRegistry

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)


class GeneralOcrService(ml_service_pb2_grpc.InferenceServicer):
    """
    Implementation of the Inference gRPC definition for OCR.

    This service handles:
    - OCR Prediction requests (End-to-End) via Infer stream
    - Capability reporting
    - Health checks
    """

    def __init__(
        self,
        manager: OcrModelManager,
        service_name: str = "ocr",
    ):
        """
        Initialize the OCR service.

        Args:
            manager: Initialized OcrModelManager instance.
            service_name: Name of the service for logging/identification.
        """
        self.manager = manager
        self.service_name = service_name
        self._initialized = False
        self._registry = TaskRegistry()
        self._setup_task_registry()

    def _setup_task_registry(self):
        """Register supported tasks."""
        self._registry.set_service_name(self.service_name)

        self._registry.register_task(
            name="ocr",
            handler=self._handle_ocr,
            description="Optical Character Recognition",
            input_mimes=["image/jpeg", "image/png", "image/bmp", "image/webp"],
            output_mime="application/json;schema=ocr_v1",
        )

    @classmethod
    def from_config(
        cls,
        config: LumenConfig,
        cache_dir: str,
        device_preference: Optional[str] = None,
    ) -> "GeneralOcrService":
        """
        Create service instance from Lumen configuration.

        Args:
            config: Parsed LumenConfig object.
            cache_dir: Directory for model caching.
            device_preference: Optional device hint (e.g., "cuda", "cpu").

        Returns:
            GeneralOcrService: Configured service instance.

        Raises:
            ValueError: If OCR service or models are not configured.
        """
        if "ocr" not in config.services:
            raise ValueError("OCR service not configured in Lumen config")

        ocr_config = config.services["ocr"]

        # Select model configuration
        # Default to 'general' if available, otherwise take the first one
        model_key = "general"
        if model_key not in ocr_config.models:
            if not ocr_config.models:
                raise ValueError("No models configured for OCR service")
            model_key = next(iter(ocr_config.models.keys()))
            logger.info(f"Defaulting to model '{model_key}' for OCR service")

        model_config = ocr_config.models[model_key]
        providers = None
        prefer_fp16 = True
        if ocr_config.backend_settings:
            providers = ocr_config.backend_settings.onnx_providers
            prefer_fp16 = getattr(ocr_config.backend_settings, "prefer_fp16", True)

        manager = OcrModelManager(
            config=model_config,
            cache_dir=cache_dir,
            providers=providers,
            device_preference=device_preference,
            prefer_fp16=prefer_fp16,
        )

        return cls(manager=manager)

    def initialize(self) -> None:
        """Initialize the underlying model manager."""
        if not self._initialized:
            logger.info("Initializing OCR Service...")
            try:
                self.manager.initialize()
                self._initialized = True
                logger.info("OCR Service initialized successfully.")
            except Exception as e:
                logger.critical(f"Failed to initialize OCR Service: {e}")
                raise

    # -------------------------------------------------------------------------
    # gRPC Methods
    # -------------------------------------------------------------------------

    def Infer(self, request_iterator, context):
        """
        Bidirectional streaming inference.
        """
        for request in request_iterator:
            start_time = _now_ms()
            try:
                # 1. Lazy Initialization
                if not self._initialized:
                    self.initialize()

                # 2. Route to handler
                task_name = request.task or "ocr"  # Default to ocr if not specified
                handler = self._registry.get_handler(task_name)

                # Merge metadata
                meta = dict(context.invocation_metadata())
                if request.meta:
                    meta.update(request.meta)

                # 3. Execute Handler
                result_bytes, result_mime, result_meta = handler(
                    request.payload, request.payload_mime, meta
                )

                duration = _now_ms() - start_time
                result_meta["duration_ms"] = str(duration)

                yield ml_service_pb2.InferResponse(
                    correlation_id=request.correlation_id,
                    is_final=True,
                    result=result_bytes,
                    result_mime=result_mime,
                    result_schema="ocr_v1",  # Could be dynamic based on task
                    meta=result_meta,
                )

            except Exception as e:
                logger.exception("Error during OCR inference")
                yield ml_service_pb2.InferResponse(
                    correlation_id=request.correlation_id,
                    is_final=True,
                    error=ml_service_pb2.Error(
                        code=ml_service_pb2.ERROR_CODE_INTERNAL,
                        message=str(e),
                    ),
                )

    def GetCapabilities(self, request, context) -> ml_service_pb2.Capability:
        """Return information about the loaded model and backend."""
        try:
            if not self._initialized:
                self.initialize()

            info = self.manager.get_info()
            backend_info = info.backend_info

            # Flatten extra metadata for protobuf map<string, string>
            flat_extra = {}
            if backend_info and backend_info.extra:
                for k, v in backend_info.extra.items():
                    if isinstance(v, (dict, list)):
                        flat_extra[k] = json.dumps(v)
                    else:
                        flat_extra[k] = str(v)

            return self._registry.build_capability(
                service_name=self.service_name,
                model_id=info.model_id,
                runtime=backend_info.runtime if backend_info else "unknown",
                precisions=backend_info.precisions if backend_info else [],
                extra_metadata=flat_extra,
            )
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.Capability()

    def StreamCapabilities(self, request, context):
        """Stream capabilities (just yield one for now)."""
        yield self.GetCapabilities(request, context)

    def Health(self, request, context):
        """Health check."""
        if self._initialized:
            return empty_pb2.Empty()
        else:
            return empty_pb2.Empty()

    # -------------------------------------------------------------------------
    # Task Handlers
    # -------------------------------------------------------------------------

    def _handle_ocr(
        self, payload: bytes, mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handle standard OCR task."""
        # Standard thresholds
        det_thresh = self._parse_float(meta.get("detection_threshold"), 0.3)
        rec_thresh = self._parse_float(meta.get("recognition_threshold"), 0.5)
        use_angle_cls = meta.get("use_angle_cls", "false").lower() == "true"

        # Advanced DBNet parameters
        box_thresh = self._parse_float(meta.get("ocr.box_thresh"), 0.6)
        unclip_ratio = self._parse_float(meta.get("ocr.unclip_ratio"), 1.5)

        results = self.manager.predict(
            image_bytes=payload,
            det_threshold=det_thresh,
            rec_threshold=rec_thresh,
            use_angle_cls=use_angle_cls,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )

        result_json = self._assemble_response_json(results)
        return result_json, "application/json;schema=ocr_v1", {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _parse_float(self, value: Any, default: float) -> float:
        """Safely parse float from metadata."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _assemble_response_json(self, results: list) -> bytes:
        """Convert backend results to JSON bytes matching ocr_v1 schema."""
        items = []
        for res in results:
            # res.box is list of tuples [(x,y), ...]
            # Schema expects [[x,y], ...]
            box_list = [[int(pt[0]), int(pt[1])] for pt in res.box]

            items.append(
                {"box": box_list, "text": res.text, "confidence": res.confidence}
            )

        response_dict = OCRV1(
            items=items, count=len(items), model_id=self.manager.get_info().model_id
        ).model_dump()

        return json.dumps(response_dict).encode("utf-8")
