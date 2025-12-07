"""
Common dataclasses and types for Lumen-CLIP models.

This module provides shared data structures for type-safe model information
management across all Lumen-CLIP services.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from .backends import BackendInfo


@dataclass
class ModelInfo:
    """Type-safe model information for all Lumen-CLIP models.

    This dataclass provides a consistent interface for accessing model metadata
    across different model types (CLIP, BioCLIP, etc.) with type safety.
    """
    # Core model identification
    model_name: str
    model_id: str

    # Model capabilities
    supports_classification: bool
    is_initialized: bool

    # Performance metrics
    load_time: float

    # Dataset information
    num_labels: Optional[int] = None  # Number of classes/species

    # Backend information
    backend_info: Optional[BackendInfo] = None

    # Model-specific features
    scene_classification_available: bool = False

    # Additional metadata for extensibility
    extra_metadata: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = None  # For BioCLIP

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension from backend info."""
        return self.backend_info.text_embedding_dim if self.backend_info else 0

    @property
    def runtime(self) -> str:
        """Get the runtime from backend info."""
        return self.backend_info.runtime if self.backend_info else "unknown"

    @property
    def device(self) -> str:
        """Get the device from backend info."""
        return self.backend_info.device if self.backend_info else "unknown"

    @property
    def precisions(self) -> list[str]:
        """Get the precisions from backend info."""
        return self.backend_info.precisions if self.backend_info else ["unknown"]