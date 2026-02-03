"""
Runtime Model Information for Lumen-VLM

This module provides shared data structures for type-safe model information
management across VLM (Vision-Language Model) services, tracking runtime state
and performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .backends.base import BackendInfo

if TYPE_CHECKING:
    from .backends.base import BaseFastVLMBackend


@dataclass
class RuntimeModelInfo:
    """Runtime model information for loaded VLM models.

    This dataclass provides runtime state and metadata for vision-language models
    that have been loaded into memory, including backend configuration, initialization
    status, and performance metrics. This is distinct from repository ModelInfo which
    contains static model file metadata.

    Attributes:
        model_name: Human-readable model name (e.g., "Qwen2-VL-2B", "LLaVA-1.5-7B").
        model_id: Stable model identifier for caching/reference.
        is_initialized: Whether the model is fully initialized and ready.
        load_time: Time taken to load the model in seconds.
        backend_info: Backend runtime information (device, models, etc.).
        max_new_tokens: Maximum number of tokens that can be generated.
        max_context_length: Maximum context length supported by the model.
        vision_image_size: Image size expected by the vision encoder.
        vision_patch_size: Patch size used by the vision encoder.
        vocab_size: Vocabulary size of the language model.
        supports_streaming: Whether streaming generation is supported.
        supports_multimodal: Whether the model supports vision+text input.
        extra_metadata: Additional metadata for extensibility.
    """

    # Core model identification
    model_name: str
    model_id: str

    # Model state
    is_initialized: bool
    load_time: float | None = None

    # Backend information
    backend_info: BackendInfo | None = None

    # VLM-specific capabilities
    max_new_tokens: int | None = None
    max_context_length: int | None = None
    vision_image_size: int | None = None
    vision_patch_size: int | None = None
    vocab_size: int | None = None
    supports_streaming: bool = True
    supports_multimodal: bool = True

    # Additional metadata for extensibility
    extra_metadata: dict[str, Any] | None = None

    @property
    def runtime(self) -> str:
        """Get the runtime from backend info."""
        return self.backend_info.runtime if self.backend_info else "unknown"

    @property
    def device(self) -> str:
        """Get the device from backend info."""
        if self.backend_info and self.backend_info.device is not None:
            return self.backend_info.device
        return "unknown"

    @property
    def precisions(self) -> list[str]:
        """Get the precisions from backend info."""
        return self.backend_info.precisions if self.backend_info else ["unknown"]

    @property
    def model_version(self) -> str | None:
        """Get the model version from backend info."""
        return self.backend_info.version if self.backend_info else None

    @classmethod
    def from_backend(
        cls,
        model_name: str,
        model_id: str,
        backend: BaseFastVLMBackend,
        load_time: float,
        supports_streaming: bool = True,
        supports_multimodal: bool = True,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RuntimeModelInfo:
        """Create RuntimeModelInfo from a backend instance.

        This factory method simplifies creating RuntimeModelInfo by extracting
        information directly from the backend instance.

        Args:
            model_name: Human-readable model name (e.g., "Qwen2-VL-2B").
            model_id: Stable model identifier for caching/reference.
            backend: Backend instance to extract runtime info from.
            load_time: Time taken to load the model in seconds.
            supports_streaming: Whether streaming generation is supported.
            supports_multimodal: Whether vision+text input is supported.
            extra_metadata: Additional metadata dictionary.

        Returns:
            RuntimeModelInfo: Configured runtime model information instance.

        Example:
            ```python
            info = RuntimeModelInfo.from_backend(
                model_name="Qwen2-VL-2B",
                model_id="qwen2_vl_onnx",
                backend=self._backend,
                load_time=5.2,
                supports_streaming=True
            )
            ```
        """
        backend_info = backend.get_info()

        return cls(
            model_name=model_name,
            model_id=model_id,
            is_initialized=backend.is_initialized,
            load_time=load_time,
            backend_info=backend_info,
            max_new_tokens=backend_info.max_new_tokens,
            max_context_length=backend_info.max_context_length,
            vision_image_size=backend_info.vision_image_size,
            vision_patch_size=backend_info.vision_patch_size,
            vocab_size=backend_info.vocab_size,
            supports_streaming=supports_streaming,
            supports_multimodal=supports_multimodal,
            extra_metadata=extra_metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "is_initialized": self.is_initialized,
            "load_time": self.load_time,
            "runtime": self.runtime,
            "device": self.device,
            "precisions": self.precisions,
            "model_version": self.model_version,
            "max_new_tokens": self.max_new_tokens,
            "max_context_length": self.max_context_length,
            "vision_image_size": self.vision_image_size,
            "vision_patch_size": self.vision_patch_size,
            "vocab_size": self.vocab_size,
            "supports_streaming": self.supports_streaming,
            "supports_multimodal": self.supports_multimodal,
            "backend_info": self.backend_info.as_dict() if self.backend_info else None,
            "extra_metadata": self.extra_metadata,
        }

    def to_capability_metadata(self) -> dict[str, str]:
        """Generate metadata dictionary for gRPC Capability messages.

        Converts runtime model information into string key-value pairs suitable
        for inclusion in gRPC Capability.extra field, which requires all values
        to be strings.

        Returns:
            dict[str, str]: String-valued metadata dictionary for gRPC capabilities.

        Example:
            ```python
            capability = pb.Capability(
                service_name="vlm",
                model_ids=[info.model_id],
                runtime=info.runtime,
                extra=info.to_capability_metadata(),
                # ...
            )
            ```
        """
        metadata = {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "is_initialized": str(self.is_initialized),
            "runtime": self.runtime,
            "device": self.device,
            "precisions": ",".join(self.precisions),
            "supports_streaming": str(self.supports_streaming),
            "supports_multimodal": str(self.supports_multimodal),
        }

        # Add load time if available
        if self.load_time is not None:
            metadata["load_time"] = f"{self.load_time:.2f}"

        # Add model version if available
        if self.model_version:
            metadata["model_version"] = self.model_version

        # Add VLM-specific parameters if available
        if self.max_new_tokens is not None:
            metadata["max_new_tokens"] = str(self.max_new_tokens)
        if self.max_context_length is not None:
            metadata["max_context_length"] = str(self.max_context_length)
        if self.vision_image_size is not None:
            metadata["vision_image_size"] = str(self.vision_image_size)
        if self.vision_patch_size is not None:
            metadata["vision_patch_size"] = str(self.vision_patch_size)
        if self.vocab_size is not None:
            metadata["vocab_size"] = str(self.vocab_size)

        # Add extra metadata if present
        if self.extra_metadata:
            for key, value in self.extra_metadata.items():
                # Ensure all values are strings and prefix to avoid conflicts
                metadata[f"extra_{key}"] = str(value) if value is not None else ""

        return metadata


__all__ = [
    "RuntimeModelInfo",
]
