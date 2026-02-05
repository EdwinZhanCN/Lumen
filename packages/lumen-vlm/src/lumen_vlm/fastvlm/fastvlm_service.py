"""
FastVLM gRPC Service

This module provides the gRPC service implementation for FastVLM multimodal
understanding and generation. It follows the Lumen architecture pattern and
implements the Inference protocol for handling vision-language tasks.

The service supports both streaming and non-streaming generation, handling
multimodal inputs (images + text) and generating appropriate text responses.
It integrates with the TaskRegistry for capability reporting and task routing.
"""

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import grpc
from google.protobuf import empty_pb2
from lumen_resources import TextGenerationV1
from lumen_resources.lumen_config import BackendSettings, ModelConfig, Services
from typing_extensions import override

import lumen_vlm.proto.ml_service_pb2 as pb
import lumen_vlm.proto.ml_service_pb2_grpc as rpc
from lumen_vlm.backends.base import (
    BaseFastVLMBackend,
    ChatMessage,
    GenerationResult,
)
from lumen_vlm.registry import TaskRegistry
from lumen_vlm.resources.loader import ModelResources, ResourceLoader

from .fastvlm_model import FastVLMModelManager

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Returns the current time in milliseconds."""
    return int(time.time() * 1000)


class GeneralFastVLMService(rpc.InferenceServicer):
    """gRPC service for FastVLM multimodal understanding and generation.

    This service implements the Lumen Inference protocol to provide vision-language
    understanding and generation capabilities over gRPC. It supports both streaming
    and non-streaming generation for multimodal inputs.

    Supported tasks:
    - vlm_generate: Generate text from image+text input
    - vlm_generate_stream: Generate text with streaming output

    Service features:
    - Configurable generation parameters (temperature, top_p, max_tokens)
    - Automatic image preprocessing and format handling
    - Chat template support for conversation context
    - Error handling with detailed gRPC status codes
    - Performance timing and logging
    - Capability reporting for client discovery

    Attributes:
        SERVICE_NAME: Unique service identifier "vlm-fast" for discovery
        model: FastVLMModelManager instance providing core VLM capabilities
        is_initialized: Flag indicating whether the service is ready for inference
    """

    SERVICE_NAME = "vlm-fast"

    def __init__(self, backend: BaseFastVLMBackend, resources: ModelResources) -> None:
        """Initialize GeneralFastVLMService with backend and model resources.

        Args:
            backend: FastVLM backend instance that provides the actual inference
                capabilities for vision encoding and text generation.
            resources: ModelResources containing model metadata, file paths, and
                configuration information loaded from lumen-resources.

        Raises:
            ValueError: If backend or resources are None.
            TypeError: If backend doesn't implement BaseFastVLMBackend interface.

        Note:
            The constructor creates the FastVLMModelManager but doesn't initialize it.
            Call `initialize()` to load models before serving requests.
        """
        if backend is None:
            raise ValueError("Backend cannot be None")
        if resources is None:
            raise ValueError("Resources cannot be None")

        self.model = FastVLMModelManager(backend=backend, resources=resources)
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
        """Create GeneralFastVLMService from service configuration.

        This factory method creates a fully configured service instance by:
        1. Extracting model configuration from service_config.models
        2. Loading model resources using the validated configuration
        3. Selecting appropriate backend based on runtime settings
        4. Configuring backend with device and performance settings
        5. Creating and returning the initialized service

        Args:
            service_config: Services config from lumen_config (services.vlm).
            cache_dir: Directory path for model caching and temporary files.
                Models will be downloaded and stored here if not present.

        Returns:
            GeneralFastVLMService: Fully configured service instance ready for
                initialization. The caller must still call `initialize()` to load models.

        Raises:
            ConfigError: If configuration is invalid or missing required fields.
            ResourceNotFoundError: If model files cannot be found or downloaded.
            RuntimeNotSupportedError: If specified runtime is not available.
        """
        from ..resources.exceptions import ConfigError

        # Extract model_config from service_config.models
        # Supports keys: "general", "vlm", "fastvlm"
        model_config = None
        for key in ["general", "vlm", "fastvlm"]:
            if key in service_config.models:
                model_config = service_config.models[key]
                break

        if model_config is None:
            # Fall back to first available model
            if not service_config.models:
                raise ValueError("No models configured for VLM service")
            model_key = next(iter(service_config.models.keys()))
            logger.info(f"Using model '{model_key}' for VLM service")
            model_config = service_config.models[model_key]

        # Get backend_settings from service_config
        backend_settings = service_config.backend_settings

        # Load resources using the validated model_config
        logger.info(f"Loading resources for FastVLM model: {model_config.model}")
        resources = ResourceLoader.load_model_resource(cache_dir, model_config)

        # Create backend based on runtime
        runtime = model_config.runtime.value
        device_pref = (
            getattr(backend_settings, "device", "cpu") if backend_settings else "cpu"
        )
        max_new_tokens = (
            getattr(backend_settings, "max_new_tokens", 512)
            if backend_settings
            else 512
        )

        # Determine precision preference from ModelConfig
        # Only applies to Runtime.onnx and Runtime.rknn
        prefer_fp16 = False
        if model_config.precision and runtime in ["onnx", "rknn"]:
            prefer_fp16 = model_config.precision in ["fp16", "q4fp16"]

        providers = (
            getattr(backend_settings, "onnx_providers", None)
            if backend_settings
            else None
        )

        if runtime == "onnx":
            from lumen_vlm.backends.onnxrt_backend import FastVLMONNXBackend

            backend = FastVLMONNXBackend(
                resources=resources,
                device_preference=device_pref,
                providers=providers,
                max_new_tokens=max_new_tokens,
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
        self.registry.set_service_name("lumen-vlm")

        # Register VLM generation task
        self.registry.register_task(
            name="vlm_generate",
            handler=self._handle_generate,
            description="Generate text from image and text input",
            input_mimes=["image/jpeg", "image/png", "image/webp"],
            output_mime="application/json;schema=text_generation_v1",
            metadata={
                "supports_streaming": "true",
                "supports_chat_history": "true",
                "max_image_size": "5048x5048",
            },
        )

        # Register streaming VLM generation task
        self.registry.register_task(
            name="vlm_generate_stream",
            handler=self._handle_generate_stream,
            description="Generate text from image and text input with streaming",
            input_mimes=["image/jpeg", "image/png", "image/webp"],
            output_mime="application/json;schema=text_generation_v1",
            metadata={
                "supports_streaming": "true",
                "supports_chat_history": "true",
                "max_image_size": "5048x5048",
            },
        )

        logger.info(
            f"Task registry initialized with {len(self.registry.list_task_names())} tasks"
        )

    def initialize(self) -> None:
        """Initialize the model manager and prepare for inference."""
        logger.info("Initializing FastVLM Model Manager...")
        self.model.initialize()
        self.is_initialized = True
        info = self.model.get_info()
        logger.info(
            "FastVLM model ready: %s with %s (loaded in %.2fs)",
            info.model_name,
            info.backend_info.runtime if info.backend_info else "unknown",
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

        if backend_info is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Backend info not available"
            )
            # The abort call raises, but type checker doesn't know that
            raise RuntimeError("Backend info not available")

        # Use registry to build capability automatically
        extra_metadata = {
            "model_name": model_info.model_name,
            "model_id": model_info.model_id,
            "max_new_tokens": str(backend_info.max_new_tokens or 512),
            "max_context_length": str(backend_info.max_context_length or 2048),
            "supports_streaming": "true",
            "supports_multimodal": "true",
        }

        return self.registry.build_capability(
            service_name=self.SERVICE_NAME,
            model_id=model_info.model_id,
            runtime=backend_info.runtime,
            precisions=backend_info.precisions,
            extra_metadata=extra_metadata,
        )

    @override
    def StreamCapabilities(self, request, context) -> Iterable[pb.Capability]:
        """Streams the capabilities of the service."""
        if not self.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        # Get model info
        model_info = self.model.get_info()
        backend_info = model_info.backend_info

        if backend_info is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Backend info not available"
            )
            # The abort call raises, but type checker doesn't know that
            raise RuntimeError("Backend info not available")

        # Use registry to build capability automatically
        extra_metadata = {
            "model_name": model_info.model_name,
            "model_id": model_info.model_id,
            "max_new_tokens": str(backend_info.max_new_tokens or 512),
            "max_context_length": str(backend_info.max_context_length or 2048),
            "supports_streaming": "true",
            "supports_multimodal": "true",
        }

        yield self.registry.build_capability(
            service_name=self.SERVICE_NAME,
            model_id=model_info.model_id,
            runtime=backend_info.runtime,
            precisions=backend_info.precisions,
            extra_metadata=extra_metadata,
        )

    @override
    def Health(self, request, context):
        """Simple health check endpoint."""
        return empty_pb2.Empty()

    # -------- Task Handlers ----------

    def _handle_generate(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handle VLM generation task."""
        # Extract generation parameters from meta
        max_new_tokens = int(meta.get("max_new_tokens", "512"))
        temperature = float(meta.get("temperature", "0.0"))
        top_p = float(meta.get("top_p", "1.0"))
        repetition_penalty = float(meta.get("repetition_penalty", "1.0"))
        do_sample = meta.get("do_sample", "false").lower() == "true"
        add_generation_prompt = (
            meta.get("add_generation_prompt", "true").lower() == "true"
        )

        # Extract stop sequences
        stop_sequences = None
        if "stop_sequences" in meta:
            try:
                stop_sequences = json.loads(meta["stop_sequences"])
                if not isinstance(stop_sequences, list):
                    stop_sequences = None
            except json.JSONDecodeError:
                logger.warning("Invalid stop_sequences format, ignoring")

        # Extract messages
        messages = self._extract_messages_from_meta(meta)
        if not messages:
            raise ValueError("No messages provided in metadata")

        # Extract image from payload
        image_bytes = payload
        if not image_bytes:
            raise ValueError("No image data provided")

        # Generate response (non-streaming returns GenerationResult)
        gen_result = self.model.generate(
            messages=messages,
            image_bytes=image_bytes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            do_sample=do_sample,
            add_generation_prompt=add_generation_prompt,
            stream=False,
        )

        # Type assertion: with stream=False, we expect GenerationResult
        if not isinstance(gen_result, GenerationResult):
            raise RuntimeError("Expected GenerationResult for non-streaming generation")
        result: GenerationResult = gen_result

        # Create response
        response = TextGenerationV1(
            text=result.text,
            model_id=self.model.get_info().model_id,
            finish_reason=result.finish_reason,  # type: ignore[arg-type]
            generated_tokens=len(result.tokens),
            input_tokens=len(result.tokens) - len(result.tokens),
        )

        result_bytes = json.dumps(response.model_dump(mode="json")).encode("utf-8")

        return (
            result_bytes,
            "application/json;schema=text_generation_v1",
            {
                "generated_tokens": str(len(result.tokens)),
                "finish_reason": result.finish_reason,
            },
        )

    def _handle_generate_stream(
        self, payload: bytes, payload_mime: str, meta: dict[str, str]
    ) -> tuple[bytes, str, dict[str, str]]:
        """Handle streaming VLM generation task."""
        # Extract parameters (same as non-streaming)
        max_new_tokens = int(meta.get("max_new_tokens", "512"))
        temperature = float(meta.get("temperature", "0.0"))
        top_p = float(meta.get("top_p", "1.0"))
        repetition_penalty = float(meta.get("repetition_penalty", "1.0"))
        do_sample = meta.get("do_sample", "false").lower() == "true"
        add_generation_prompt = (
            meta.get("add_generation_prompt", "true").lower() == "true"
        )

        stop_sequences = None
        if "stop_sequences" in meta:
            try:
                stop_sequences = json.loads(meta["stop_sequences"])
                if not isinstance(stop_sequences, list):
                    stop_sequences = None
            except json.JSONDecodeError:
                logger.warning("Invalid stop_sequences format, ignoring")

        messages = self._extract_messages_from_meta(meta)
        if not messages:
            raise ValueError("No messages provided in metadata")

        image_bytes = payload
        if not image_bytes:
            raise ValueError("No image data provided")

        # Generate streaming response
        chunks = list(
            self.model.generate_stream(
                messages=messages,
                image_bytes=image_bytes,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
                do_sample=do_sample,
                add_generation_prompt=add_generation_prompt,
            )
        )

        # Combine chunks into final result for now
        # In a full implementation, this would be handled by the streaming gRPC protocol
        final_text = "".join(chunk.text for chunk in chunks if chunk.text)
        all_tokens = []
        for chunk in chunks:
            all_tokens.extend(chunk.tokens)

        finish_reason: str = "stop"
        if chunks and hasattr(chunks[-1], "metadata"):
            finish_reason = chunks[-1].metadata.get("reason", "stop")

        response = TextGenerationV1(
            text=final_text,
            model_id=self.model.get_info().model_id,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            generated_tokens=len(all_tokens),
            input_tokens=0,  # Would need to track this separately
        )

        result_bytes = json.dumps(response.model_dump(mode="json")).encode("utf-8")

        return (
            result_bytes,
            "application/json;schema=text_generation_v1",
            {
                "generated_tokens": str(len(all_tokens)),
                "finish_reason": finish_reason,
                "streaming_chunks": str(len(chunks)),
            },
        )

    # -------- Helper Methods ----------

    def _extract_messages_from_meta(self, meta: dict[str, str]) -> list[ChatMessage]:
        """Extract chat messages from metadata."""
        messages = []

        # Handle messages as JSON string
        if "messages" in meta:
            try:
                messages_data = json.loads(meta["messages"])
                if isinstance(messages_data, list):
                    for msg in messages_data:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            messages.append(
                                ChatMessage(role=msg["role"], content=msg["content"])
                            )
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Invalid messages format in meta")

        # Handle single message fallback
        if not messages and "prompt" in meta:
            messages.append(ChatMessage(role="user", content=meta["prompt"]))

        return messages

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
