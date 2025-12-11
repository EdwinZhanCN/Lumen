"""
Lumen CLIP - Python Machine Learning Module for Lumilio Photos

A high-performance CLIP-based image understanding service providing:
- General CLIP embeddings and similarity
- BioCLIP for biological and scientific imagery
- Unified SmartCLIP for multi-modal analysis
- gRPC-based inference service with mDNS discovery

Key Features:
- Multiple backend support (CPU, CUDA, MPS, OpenVINO)
- Modular architecture with specialized model managers
- Automatic model downloading and management
- Type-safe configuration with Pydantic models
- High-performance gRPC streaming interface

Usage:
    >>> from lumen_clip.server import serve
    >>> serve(port=50051)  # Start gRPC service

    >>> from lumen_clip.general_clip import CLIPModelManager
    >>> manager = CLIPModelManager()
    >>> embeddings = manager.embed_images(images)

Service Components:
    - GeneralCLIPService: Standard CLIP embeddings
    - BioCLIPService: Biological image analysis
    - UnifiedCLIPService: Multi-modal intelligent analysis
    - ResourceLoader: Model loading and management

Configuration:
    Uses lumen-resources package for model configuration,
    downloading, and validation.

License: Part of Lumilio ecosystem
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lumen-clip")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

__author__ = "Edwin Zhan"
__email__ = "support@lumilio.org"

from .cli import main

__all__ = ["main"]
