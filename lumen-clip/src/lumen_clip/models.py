"""
Common dataclasses and types for Lumen-CLIP models.

This module provides shared data structures for type-safe model information
management across all Lumen-CLIP services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .backends import BackendInfo

if TYPE_CHECKING:
    from .backends.base import BaseClipBackend


@dataclass
class RuntimeModelInfo:
    """Runtime model information for loaded CLIP models.

    This dataclass provides runtime state and metadata for models that have been
    loaded into memory, including backend configuration, initialization status,
    and performance metrics. This is distinct from repository ModelInfo which
    contains static model file metadata.
    """

    # Core model identification
    model_name: str
    model_id: str

    # Model capabilities
    supports_classification: bool
    is_initialized: bool

    # Performance metrics
    load_time: float | None = None  # Time taken to load the model in seconds

    # Dataset information
    num_labels: int | None = None  # Number of classes/species

    # Backend information
    backend_info: BackendInfo | None = None

    # Model-specific features
    scene_classification_available: bool = False

    # Additional metadata for extensibility
    extra_metadata: dict[str, Any] | None = None
    model_version: str | None = None  # For BioCLIP

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension from backend info."""
        if self.backend_info and self.backend_info.text_embedding_dim is not None:
            return self.backend_info.text_embedding_dim
        return 0

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

    @classmethod
    def from_backend(
        cls,
        model_name: str,
        model_id: str,
        backend: BaseClipBackend,
        load_time: float,
        supports_classification: bool = False,
        num_labels: int | None = None,
        scene_classification_available: bool = False,
        model_version: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RuntimeModelInfo:
        """Create RuntimeModelInfo from a backend instance.

        This factory method simplifies creating RuntimeModelInfo by extracting
        information directly from the backend instance.

        Args:
            model_name: Human-readable model name (e.g., "ViT-B-32").
            model_id: Stable model identifier for caching/reference.
            backend: Backend instance to extract runtime info from.
            load_time: Time taken to load the model in seconds.
            supports_classification: Whether the model supports classification.
            num_labels: Number of classification labels if applicable.
            scene_classification_available: Whether scene classification is available.
            model_version: Model version string (e.g., for BioCLIP).
            extra_metadata: Additional metadata dictionary.

        Returns:
            RuntimeModelInfo: Configured runtime model information instance.

        Example:
            ```python
            info = RuntimeModelInfo.from_backend(
                model_name="ViT-B-32",
                model_id="clip_onnx",
                backend=self._backend,
                load_time=2.5,
                supports_classification=True,
                num_labels=1000
            )
            ```
        """
        return cls(
            model_name=model_name,
            model_id=model_id,
            is_initialized=backend.is_initialized,
            load_time=load_time,
            backend_info=backend.get_info(),
            supports_classification=supports_classification,
            num_labels=num_labels,
            scene_classification_available=scene_classification_available,
            model_version=model_version,
            extra_metadata=extra_metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses and serialization.

        Returns:
            dict: Dictionary representation with all fields, suitable for
                JSON serialization or API responses.

        Example:
            ```python
            info_dict = runtime_info.as_dict()
            json.dumps(info_dict)  # Safe for JSON serialization
            ```
        """
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "is_initialized": self.is_initialized,
            "load_time": self.load_time,
            "supports_classification": self.supports_classification,
            "num_labels": self.num_labels,
            "scene_classification_available": self.scene_classification_available,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "runtime": self.runtime,
            "device": self.device,
            "precisions": self.precisions,
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
                service_name="clip",
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
            "load_time": f"{self.load_time:.2f}",
            "supports_classification": str(self.supports_classification),
            "embedding_dim": str(self.embedding_dim),
            "runtime": self.runtime,
            "device": self.device,
            "precisions": ",".join(self.precisions),
            "scene_classification_available": str(self.scene_classification_available),
        }

        # Add optional fields if present
        if self.num_labels is not None:
            metadata["num_labels"] = str(self.num_labels)
        if self.model_version:
            metadata["model_version"] = self.model_version

        # Add extra metadata if present
        if self.extra_metadata:
            for key, value in self.extra_metadata.items():
                # Ensure all values are strings and prefix to avoid conflicts
                metadata[f"extra_{key}"] = str(value) if value is not None else ""

        return metadata


__all__ = [
    "RuntimeModelInfo",
]
