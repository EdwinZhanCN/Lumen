"""
FastVLM Model Manager

This module provides the high-level manager for FastVLM models, handling
initialization, resource loading, and inference delegation to backends.
It follows the standard Lumen architecture for model management and implements
the business logic layer for multimodal vision-language understanding.

The Model Manager abstracts away the complexity of vision encoding, text
generation, and multimodal fusion, providing a clean interface for the Service
layer to work with.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from ..backends.backend_exceptions import (
    BackendNotInitializedError,
    InferenceError,
    InvalidInputError,
)
from ..backends.base import (
    BackendInfo,
    BaseFastVLMBackend,
    ChatMessage,
    GenerationChunk,
    GenerationResult,
)
from ..resources.loader import ModelResources

logger = logging.getLogger(__name__)


class ModelDataNotFoundError(Exception):
    """Raised when model-specific data (labels, prompts, etc.) cannot be found."""

    pass


class CacheCorruptionError(Exception):
    """Raised when cached data is corrupted or incompatible."""

    pass


@dataclass
class ModelInfo:
    """Type-safe model information for FastVLM models.

    Provides a consistent interface for accessing model metadata and status.
    Compatible with other Lumen services while supporting VLM-specific fields.
    """

    model_name: str
    model_id: str
    is_initialized: bool
    backend_info: Optional[BackendInfo] = None
    load_time: float = 0.0
    extra_metadata: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "is_initialized": self.is_initialized,
            "backend_info": self.backend_info.as_dict() if self.backend_info else None,
            "load_time": self.load_time,
            "extra_metadata": self.extra_metadata,
        }


class FastVLMModelManager:
    """
    High-level manager for FastVLM multimodal models.

    This class serves as the business logic layer in the Lumen architecture,
    managing the lifecycle of FastVLM models and providing a clean interface
    for multimodal inference tasks. It handles:

    - Model initialization and resource management
    - Vision-language preprocessing and postprocessing
    - Generation request orchestration
    - Error handling and status reporting
    - Backend abstraction and multiplexing

    The manager follows dependency injection patterns, receiving a Backend
    instance rather than creating its own, which enables easy testing and
    runtime flexibility.

    Attributes:
        _backend: The FastVLM backend implementation (ONNX, TensorRT, etc.)
        _resources: Model resources including metadata and file paths
        _initialized: Whether the model manager is ready for inference
        _info: Cached model information for API responses
    """

    def __init__(
        self,
        backend: BaseFastVLMBackend,
        resources: ModelResources,
    ) -> None:
        """Initialize FastVLM Model Manager with backend and resources.

        Args:
            backend: FastVLM backend instance that provides the actual inference
                capabilities for vision encoding and text generation.
            resources: ModelResources containing model metadata, file paths, and
                configuration information loaded from lumen-resources.

        Raises:
            ValueError: If backend or resources are None.
            TypeError: If backend doesn't implement BaseFastVLMBackend interface.

        Note:
            The constructor creates the manager but doesn't initialize the model.
            Call `initialize()` to load models before inference.
        """
        if backend is None:
            raise ValueError("Backend cannot be None")
        if resources is None:
            raise ValueError("Resources cannot be None")

        if not isinstance(backend, BaseFastVLMBackend):
            raise TypeError(
                f"Backend must implement BaseFastVLMBackend, got {type(backend)}"
            )

        self._backend = backend
        self._resources = resources
        self._initialized = False
        self._info: Optional[ModelInfo] = None
        self._init_start_time: Optional[float] = None

        logger.debug(
            "FastVLM Model Manager created for %s with %s backend",
            resources.model_name,
            type(backend).__name__,
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._resources.model_name

    @property
    def model_id(self) -> str:
        """Get the unique model identifier."""
        return self._resources.model_id

    @property
    def is_initialized(self) -> bool:
        """Check if the model manager is initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize the model and backend.

        Loads the model weights and prepares the backend for inference.
        This method must be called before any inference operations.

        Raises:
            ModelLoadingError: If model loading fails.
            BackendNotInitializedError: If backend initialization fails.
            ResourceNotFoundError: If required model files are missing.
        """
        if self._initialized:
            logger.warning("Model manager already initialized")
            return

        logger.info(f"Initializing FastVLM Model Manager for {self.model_name}")
        self._init_start_time = time.time()

        try:
            # Initialize the backend
            self._backend.initialize()

            # Create model info
            backend_info = self._backend.get_info()
            self._info = ModelInfo(
                model_name=self.model_name,
                model_id=self.model_id,
                is_initialized=True,
                backend_info=backend_info,
                load_time=time.time() - self._init_start_time,
                extra_metadata=getattr(
                    self._resources.model_info, "extra_metadata", None
                ),
            )

            self._initialized = True
            logger.info(
                f"✅ FastVLM Model Manager initialized successfully in {self._info.load_time:.2f}s"
            )

        except Exception as exc:
            logger.error(f"Failed to initialize FastVLM Model Manager: {exc}")
            self._initialized = False
            raise

    def generate(
        self,
        messages: Union[Sequence[ChatMessage], Sequence[Mapping[str, str]]],
        image_bytes: bytes,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[list[str]] = None,
        do_sample: bool = False,
        add_generation_prompt: bool = True,
        stream: bool = False,
        **extra_kwargs: Any,
    ) -> Union[GenerationResult, Iterable[GenerationChunk]]:
        """Generate text from multimodal input.

        Args:
            messages: List of chat messages (ChatMessage objects or dicts with 'role'/'content')
            image_bytes: Image data as bytes
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop_sequences: List of strings that stop generation when encountered
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            add_generation_prompt: Whether to add generation prompt to chat template
            stream: Whether to stream generation chunks
            **extra_kwargs: Additional backend-specific parameters

        Returns:
            Generation result (single or streaming chunks)

        Raises:
            InvalidInputError: If input validation fails
            InferenceError: If generation fails
            BackendNotInitializedError: If model not initialized
        """
        self._ensure_initialized()

        normalized_messages = self._normalize_messages(messages)
        self._validate_messages(normalized_messages)

        # Build generation request
        request = self._backend.build_generation_request(
            messages=normalized_messages,
            image_bytes=image_bytes,
            add_generation_prompt=add_generation_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            do_sample=do_sample,
            stream=stream,
            extra=extra_kwargs,
        )

        # Delegate to backend
        try:
            return self._backend.generate(request)
        except Exception as exc:
            logger.error(f"Generation failed: {exc}")
            raise InferenceError(f"Generation failed: {exc}") from exc

    def generate_stream(
        self,
        messages: Union[Sequence[ChatMessage], Sequence[Mapping[str, str]]],
        image_bytes: bytes,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[list[str]] = None,
        do_sample: bool = False,
        add_generation_prompt: bool = True,
        **extra_kwargs: Any,
    ) -> Iterable[GenerationChunk]:
        """Generate text from multimodal input with streaming.

        This is a convenience method that always forces streaming mode.
        See generate() for parameter documentation.

        Returns:
            Iterable of generation chunks
        """
        result = self.generate(
            messages=messages,
            image_bytes=image_bytes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            do_sample=do_sample,
            add_generation_prompt=add_generation_prompt,
            stream=True,
            **extra_kwargs,
        )

        if isinstance(result, Iterable):
            return result

        raise InferenceError("Backend returned a non-streaming result in stream mode")

    def get_info(self) -> ModelInfo:
        """Get model information and status.

        Returns:
            ModelInfo: Current model status and metadata

        Raises:
            BackendNotInitializedError: If model is not initialized
        """
        if not self._initialized or self._info is None:
            raise BackendNotInitializedError(
                "Model manager must be initialized before getting info"
            )
        return self._info

    def get_backend_info(self) -> BackendInfo:
        """Get backend runtime information.

        Returns:
            BackendInfo: Backend runtime details and capabilities
        """
        self._ensure_initialized()
        return self._backend.get_info()

    def _ensure_initialized(self) -> None:
        """Ensure the model manager is initialized."""
        if not self._initialized:
            raise BackendNotInitializedError(
                "Model manager must be initialized before inference. Call initialize() first."
            )

    def _validate_image_input(self, image_bytes: bytes) -> None:
        """Validate image input data."""
        if not image_bytes:
            raise InvalidInputError("Image bytes cannot be empty")

        if len(image_bytes) > 50 * 1024 * 1024:  # 50MB limit
            raise InvalidInputError("Image too large (max 50MB)")

    def _normalize_messages(
        self,
        messages: Union[Sequence[ChatMessage], Sequence[Mapping[str, str]]],
    ) -> list[ChatMessage]:
        """Normalize heterogeneous message payloads into ChatMessage objects."""
        if not messages:
            raise InvalidInputError("Messages cannot be empty")

        normalized: list[ChatMessage] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized.append(msg)
            elif isinstance(msg, Mapping):
                role = msg.get("role")
                content = msg.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    raise InvalidInputError(
                        "Dictionary messages must include string 'role' and 'content' keys"
                    )
                normalized.append(ChatMessage(role=role, content=content))
            else:
                raise InvalidInputError(
                    f"Unsupported message type: {type(msg).__name__}"
                )

        return normalized

    def _validate_messages(self, messages: list[ChatMessage]) -> None:
        """Validate chat messages input."""
        if not messages:
            raise InvalidInputError("Messages cannot be empty")

        # Basic validation - at least one user or system message
        valid_roles = {"user", "assistant", "system"}
        for msg in messages:
            if msg.role not in valid_roles:
                raise InvalidInputError(f"Invalid role: {msg.role}")
            if not msg.content or not msg.content.strip():
                raise InvalidInputError("Message content cannot be empty")

    def close(self) -> None:
        """Close the model manager and release resources.

        This method should be called when shutting down the service to
        properly release GPU memory and other resources.
        """
        if self._initialized:
            try:
                self._backend.close()
                logger.info("FastVLM Model Manager closed successfully")
            except Exception as exc:
                logger.warning(f"Error closing backend: {exc}")
            finally:
                self._initialized = False
                self._info = None

    def __del__(self) -> None:
        """Destructor to ensure resources are cleaned up."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
