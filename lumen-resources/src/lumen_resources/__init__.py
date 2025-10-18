"""
Lumen Resources - Unified Model Resource Management

Configuration-driven tool for managing ML model resources with production-grade
YAML configuration, JSON Schema validation, and Pydantic models.

Usage:
    from lumen_resources.validator import load_and_validate_config
    from lumen_resources.downloader import Downloader

    # Load and validate configuration
    config = load_and_validate_config("config.yaml")

    # Download models
    downloader = Downloader(config, verbose=True)
    results = downloader.download_all()

    # Check results
    for model_type, result in results.items():
        if result.success:
            print(f"Downloaded: {model_type} to {result.model_path}")
        else:
            print(f"Failed: {model_type} - {result.error}")
"""

from .lumen_config import LumenServicesConfiguration, Runtime, Region
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
    "LumenServicesConfiguration",
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
