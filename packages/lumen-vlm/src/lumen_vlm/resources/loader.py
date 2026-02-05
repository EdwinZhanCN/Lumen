"""
Resource loader for Lumen VLM models.

This module provides functionality to load and manage VLM model resources
including metadata, configurations, and model files. It integrates with
the lumen-resources package for standardized resource handling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from lumen_resources.lumen_config import ModelConfig

logger = logging.getLogger(__name__)


class ModelResources:
    """Container for VLM model resources and metadata.

    This class encapsulates all necessary resources for a VLM model including
    model files, configurations, and metadata. It provides a unified interface
    for accessing model information across different components.
    """

    def __init__(
        self,
        model_root_path: Path,
        model_info: Any,
        model_config: ModelConfig,
        tokenizer_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_root_path = model_root_path
        self.model_info = model_info
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config or {}

    @property
    def model_name(self) -> str:
        return self.model_info.name

    @property
    def model_id(self) -> str:
        return f"{self.model_name}_{self.model_config.runtime.value}"


class ResourceLoader:
    """Utility class for loading VLM model resources.

    Provides static methods for loading model resources from various sources
    including local files and remote repositories. Integrates with lumen-resources
    for standardized resource management.
    """

    @staticmethod
    def load_model_resource(
        cache_dir: Path, model_config: ModelConfig
    ) -> ModelResources:
        """Load model resources based on configuration.

        Args:
            cache_dir: Directory for model caching
            model_config: Model configuration containing name, runtime, etc.

        Returns:
            ModelResources: Loaded model resources

        Raises:
            ResourceNotFoundError: If model files cannot be found
            ConfigError: If configuration is invalid
        """
        from lumen_resources import ModelInfo  # Import here to avoid circular deps

        model_name = model_config.model
        model_path = cache_dir / "models" / model_name

        if not model_path.exists():
            from lumen_vlm.resources.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError(f"Model directory not found: {model_path}")

        # Load model info
        info_path = model_path / "model_info.json"
        if not info_path.exists():
            from lumen_vlm.resources.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError(f"Model info file not found: {info_path}")

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                model_info_data = json.load(f)
            model_info = ModelInfo(**model_info_data)
        except Exception as exc:
            from lumen_vlm.resources.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError(f"Failed to load model info: {exc}") from exc

        # Load tokenizer config if available
        tokenizer_config = {}
        tokenizer_path = model_path / "tokenizer_config.json"
        if tokenizer_path.exists():
            try:
                with open(tokenizer_path, "r", encoding="utf-8") as f:
                    tokenizer_config = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load tokenizer config: {exc}")

        logger.info(f"Loaded model resources for {model_name} from {model_path}")

        return ModelResources(
            model_root_path=model_path,
            model_info=model_info,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
        )

    @staticmethod
    def validate_model_resources(resources: ModelResources) -> bool:
        """Validate that all required model resources are present.

        Args:
            resources: Model resources to validate

        Returns:
            bool: True if all required resources are present
        """
        model_path = resources.model_root_path

        required_files = [
            "model_info.json",
            "tokenizer.json",
        ]

        for file_path in required_files:
            full_path = model_path / file_path
            if not full_path.exists():
                logger.warning(f"Missing required file: {full_path}")
                return False

        # Check for ONNX models - at least one precision variant must exist for each component
        onnx_path = model_path / "onnx"
        if onnx_path.exists():
            components = ["vision", "embed", "decoder"]
            for component in components:
                # Check if at least one precision variant exists
                fp16_path = onnx_path / f"{component}.fp16.onnx"
                fp32_path = onnx_path / f"{component}.fp32.onnx"
                fallback_path = onnx_path / f"{component}.onnx"

                if not (
                    fp16_path.exists() or fp32_path.exists() or fallback_path.exists()
                ):
                    logger.warning(
                        f"Missing ONNX model for component '{component}': "
                        f"expected one of {component}.fp16.onnx, {component}.fp32.onnx, or {component}.onnx"
                    )
                    return False

        return True
