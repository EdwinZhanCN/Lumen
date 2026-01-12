"""
Runtime Model Information for Lumen-Face

This module provides shared data structures for type-safe model information
management across face recognition services, tracking runtime state and
performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .backends.base import BackendInfo

if TYPE_CHECKING:
    from .backends.base import FaceRecognitionBackend


@dataclass
class RuntimeModelInfo:
    """Runtime model information for loaded face recognition models.

    This dataclass provides runtime state and metadata for face recognition models
    that have been loaded into memory, including backend configuration, initialization
    status, and performance metrics. This is distinct from repository ModelInfo which
    contains static model file metadata.

    Attributes:
        model_name: Human-readable model name (e.g., "buffalo_l", "antelopev2").
        model_id: Stable model identifier for caching/reference.
        is_initialized: Whether the model is fully initialized and ready.
        load_time: Time taken to load the model in seconds.
        backend_info: Backend runtime information (device, models, etc.).
        face_embedding_dim: Dimension of face embedding vectors (typically 512).
        supports_batch: Whether batch processing is supported.
        detection_model: Name of the detection model component.
        recognition_model: Name of the recognition model component.
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

    # Face recognition specific capabilities
    face_embedding_dim: int | None = None
    supports_batch: bool = False
    detection_model: str | None = None
    recognition_model: str | None = None

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
    def max_batch_size(self) -> int | None:
        """Get the max batch size from backend info."""
        return self.backend_info.max_batch_size if self.backend_info else None

    @property
    def embedding_dim(self) -> int:
        """Get the face embedding dimension."""
        if self.face_embedding_dim is not None:
            return self.face_embedding_dim
        if self.backend_info and self.backend_info.face_embedding_dim is not None:
            return self.backend_info.face_embedding_dim
        return 0

    @classmethod
    def from_backend(
        cls,
        model_name: str,
        model_id: str,
        backend: FaceRecognitionBackend,
        load_time: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RuntimeModelInfo:
        """Create RuntimeModelInfo from a backend instance.

        This factory method simplifies creating RuntimeModelInfo by extracting
        information directly from the backend instance.

        Args:
            model_name: Human-readable model name (e.g., "buffalo_l").
            model_id: Stable model identifier for caching/reference.
            backend: Backend instance to extract runtime info from.
            load_time: Time taken to load the model in seconds.
            extra_metadata: Additional metadata dictionary.

        Returns:
            RuntimeModelInfo: Configured runtime model information instance.

        Example:
            ```python
            info = RuntimeModelInfo.from_backend(
                model_name="buffalo_l",
                model_id="buffalo_l_onnx",
                backend=self._backend,
                load_time=2.3
            )
            ```
        """
        backend_info = backend.get_runtime_info()

        return cls(
            model_name=model_name,
            model_id=model_id,
            is_initialized=backend.is_initialized(),
            load_time=load_time,
            backend_info=backend_info,
            face_embedding_dim=backend_info.face_embedding_dim,
            supports_batch=backend_info.supports_image_batch,
            detection_model=backend_info.model_name,
            recognition_model=backend_info.pretrained,
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
            "runtime": self.runtime,
            "device": self.device,
            "precisions": self.precisions,
            "max_batch_size": self.max_batch_size,
            "face_embedding_dim": self.embedding_dim,
            "supports_batch": self.supports_batch,
            "detection_model": self.detection_model,
            "recognition_model": self.recognition_model,
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
                service_name="face",
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
            "face_embedding_dim": str(self.embedding_dim),
            "supports_batch": str(self.supports_batch),
        }

        # Add load time if available
        if self.load_time is not None:
            metadata["load_time"] = f"{self.load_time:.2f}"

        # Add max batch size if available
        if self.max_batch_size is not None:
            metadata["max_batch_size"] = str(self.max_batch_size)

        # Add model component names if available
        if self.detection_model:
            metadata["detection_model"] = self.detection_model
        if self.recognition_model:
            metadata["recognition_model"] = self.recognition_model

        # Add extra metadata if present
        if self.extra_metadata:
            for key, value in self.extra_metadata.items():
                # Ensure all values are strings and prefix to avoid conflicts
                metadata[f"extra_{key}"] = str(value) if value is not None else ""

        return metadata


__all__ = [
    "RuntimeModelInfo",
]
