"""
Lumen Resources - Unified Model Resource Management

A simple, configuration-driven tool for downloading and managing ML model resources
from HuggingFace and ModelScope platforms.

Usage:
    from lumen_resources import ResourceConfig, Downloader

    config = ResourceConfig.from_json(Path("config.json"))
    downloader = Downloader(config)
    results = downloader.download_all()

    for model_type, result in results.items():
        if result.success:
            print(f"✅ {model_type}: {result.model_path}")
        else:
            print(f"❌ {model_type}: {result.error}")
"""

from .config import ResourceConfig, ModelConfig, RuntimeType, PlatformType
from .downloader import Downloader, DownloadResult
from .exceptions import (
    ResourceError,
    ConfigError,
    DownloadError,
    PlatformUnavailableError,
    ValidationError,
    ModelInfoError,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "ResourceConfig",
    "ModelConfig",
    "RuntimeType",
    "PlatformType",
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
