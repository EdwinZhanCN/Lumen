"""Lumen Resources - Unified Model Resource Management.

Configuration-driven tool for managing ML model resources with production-grade
YAML configuration, JSON Schema validation, and Pydantic models. Provides a
unified interface for downloading models from multiple platforms including
HuggingFace Hub and ModelScope.

This package offers:
- Configuration-driven YAML setup for ML model resources
- Multi-platform support (HuggingFace Hub, ModelScope)
- Runtime flexibility (ONNX, PyTorch, TensorFlow, RKNN)
- Production-grade validation using JSON Schema and Pydantic
- CLI interface for command-line operations
- Programmatic API for integration into other applications

Example:
    >>> from lumen_resources import load_and_validate_config, Downloader
    >>>
    >>> # Load and validate configuration
    >>> config = load_and_validate_config("config.yaml")
    >>>
    >>> # Download models
    >>> downloader = Downloader(config, verbose=True)
    >>> results = downloader.download_all()
    >>>
    >>> # Check results
    >>> for model_type, result in results.items():
    ...     if result.success:
    ...         print(f"Downloaded: {model_type} to {result.model_path}")
    ...     else:
    ...         print(f"Failed: {model_type} - {result.error}")

The package follows a layered architecture:
- Configuration layer: Pydantic models for type-safe config handling
- Validation layer: JSON Schema and Pydantic validation
- Platform layer: Unified interface for different model repositories
- Download layer: Efficient model downloading with validation
- CLI layer: User-friendly command-line interface
"""

from .lumen_config import LumenConfig, Runtime, Region
from .downloader import Downloader, DownloadResult
from .exceptions import (
    ResourceError,
    ConfigError,
    DownloadError,
    PlatformUnavailableError,
    ValidationError,
    ModelInfoError,
)
from .lumen_config_validator import load_and_validate_config

from .model_info import ModelInfo, Source, Runtimes, Metadata
from .model_info_validator import load_and_validate_model_info

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "LumenConfig",
    "Runtime",
    "Region",
    "load_and_validate_config",
    # Model Info
    "ModelInfo",
    "Source",
    "Runtimes",
    "Metadata",
    "load_and_validate_model_info",
    # Downloader
    "Downloader",
    "DownloadResult",
    # Exceptions
    "ResourceError",
    "ConfigError",
    "DownloadError",
    "PlatformUnavailableError",
    "ValidationError",
    "ModelInfoError",
]
