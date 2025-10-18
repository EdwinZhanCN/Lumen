"""
loader.py

Resource loading for Lumen models.
This module acts as a low-level utility to load files from disk based on a
pre-validated ModelConfig intent, validated against the on-disk model_info.json manifest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

# Assuming these pydantic models are provided by the lumen_resources package
from lumen_resources.lumen_config import ModelConfig
from lumen_resources.model_info import ModelConfigurationSchema


from .exceptions import (
    ModelInfoError,
    ResourceNotFoundError,
    ResourceValidationError,
    RuntimeNotSupportedError,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResources:
    """
    Container for all resources associated with a model.

    Attributes:
        model_root_path: Path to the model's root directory (e.g., .../ViT-B-32/).
        runtime_files_path: Path to the directory containing runtime-specific model files.
        model_name: Name of the model.
        runtime: Runtime type (torch/onnx/rknn).
        model_info: Parsed model_info.json as a Pydantic object.
        config: Parsed config.json content.
        tokenizer_config: Parsed tokenizer.json content (optional).
        labels: Label names array (None if no dataset).
        label_embeddings: Pre-computed label embeddings (optional).
    """

    model_root_path: Path
    runtime_files_path: Path
    model_name: str
    runtime: str
    model_info: ModelConfigurationSchema
    config: dict[str, Any]
    tokenizer_config: dict[str, Any] | None
    labels: NDArray[np.object_] | None
    label_embeddings: NDArray[np.float32] | None

    def get_model_file(self, filename: str) -> Path:
        """Get a file path within the runtime-specific directory."""
        return self.runtime_files_path / filename

    def has_classification_support(self) -> bool:
        """Check if this model supports classification (has dataset)."""
        return self.labels is not None

    def get_embedding_dim(self) -> int:
        """Get embedding dimension from model_info."""
        return self.model_info.embedding_dim

    def get_image_size(self) -> tuple[int, int] | None:
        """Get image input size from config.json."""
        size = self.config.get("image_size")
        if not size and "vision_cfg" in self.config:
            size = self.config["vision_cfg"].get("image_size")

        if isinstance(size, (list, tuple)) and len(size) == 2:
            return tuple(size)
        if isinstance(size, int):
            return (size, size)
        return None


class ResourceLoader:
    """
    Loads and validates model resources from the standardized directory structure,
    based on the user's intent from ModelConfig.
    """

    @staticmethod
    def load_model_resources(
        cache_dir: str | Path, model_config: ModelConfig
    ) -> ModelResources:
        """
        Load all resources for a model based on configuration intent.

        Args:
            cache_dir: Root cache directory from global config.
            model_config: The specific model's configuration from lumen_config.

        Returns:
            ModelResources object with all loaded resources.
        """
        cache_dir = Path(cache_dir).expanduser().resolve()
        model_root_path = cache_dir / "models" / model_config.model

        logger.info(
            f"Loading resources for {model_config.model} (runtime: {model_config.runtime.value})"
        )

        # Step 1: Load the on-disk manifest (model_info.json)
        try:
            model_info = ModelConfigurationSchema.from_json_file(
                model_root_path / "model_info.json"
            )
        except FileNotFoundError:
            raise ResourceNotFoundError(
                f"model_info.json not found in {model_root_path}"
            )
        except ValidationError as e:
            raise ModelInfoError(f"model_info.json is invalid: {e}") from e

        # Step 2: Validate intent against the manifest and determine paths
        runtime_files_path = ResourceLoader._validate_intent_and_get_paths(
            model_root_path, model_config, model_info
        )

        # Step 3: Load remaining resources
        config = ResourceLoader._load_json(model_root_path / "config.json")
        tokenizer_config = ResourceLoader._load_tokenizer(model_root_path)
        labels, embeddings = ResourceLoader._load_dataset(
            model_root_path, model_info, model_config.dataset
        )

        resources = ModelResources(
            model_root_path=model_root_path,
            runtime_files_path=runtime_files_path,
            model_name=model_config.model,
            runtime=str(model_config.runtime.value),
            model_info=model_info,
            config=config,
            tokenizer_config=tokenizer_config,
            labels=labels,
            label_embeddings=embeddings,
        )

        ResourceLoader._log_resource_summary(resources)
        return resources

    @staticmethod
    def _validate_intent_and_get_paths(
        model_root: Path, config: ModelConfig, info: ModelConfigurationSchema
    ) -> Path:
        """
        Validates the user's ModelConfig intent against the model_info manifest.
        Returns the correct path to runtime files.
        """
        runtime = config.runtime.value
        if runtime not in info.runtimes:
            raise RuntimeNotSupportedError(
                f"Runtime '{runtime}' not listed in model_info.json. "
                f"Available: {list(info.runtimes.keys())}"
            )
        runtime_info = info.runtimes[runtime]
        if not runtime_info.available:
            raise RuntimeNotSupportedError(
                f"Runtime '{runtime}' is not available for this model."
            )

        # Specific validation for RKNN runtime
        if runtime == "rknn":
            if not config.rknn_device:
                raise ModelInfoError(
                    "rknn_device must be specified in config for rknn runtime."
                )
            if runtime_info.devices and config.rknn_device not in runtime_info.devices:
                raise RuntimeNotSupportedError(
                    f"Device '{config.rknn_device}' not supported for this model. "
                    f"Supported devices: {runtime_info.devices}"
                )
            runtime_path = model_root / runtime / config.rknn_device
        else:
            runtime_path = model_root / runtime

        if not runtime_path.exists():
            raise ResourceNotFoundError(f"Runtime directory not found: {runtime_path}")

        # Check for existence of required files for the runtime
        if runtime_info.files and isinstance(runtime_info.files, list):
            for file_path_str in runtime_info.files:
                full_path = model_root / file_path_str
                if not full_path.exists():
                    raise ResourceNotFoundError(
                        f"Required runtime file missing: {full_path}"
                    )
        return runtime_path

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Loads and parses a JSON file, raising specific errors."""
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ResourceNotFoundError(f"Required file not found: {path}")
        except json.JSONDecodeError as e:
            raise ResourceValidationError(f"Invalid JSON in {path.name}: {e}") from e
        except OSError as e:
            raise ResourceNotFoundError(f"Failed to load {path.name}: {e}") from e

    @staticmethod
    def _load_tokenizer(model_path: Path) -> dict[str, Any] | None:
        """Load tokenizer.json if it exists."""
        tokenizer_path = model_path / "tokenizer.json"
        if tokenizer_path.exists():
            try:
                tokenizer_config = ResourceLoader._load_json(tokenizer_path)
                logger.info("Loaded tokenizer.json")
                return tokenizer_config
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer.json: {e}. Fallback to SimpleTokenizer"
                )
        else:
            logger.info("No tokenizer.json found, will use SimpleTokenizer")
        return None

    @staticmethod
    def _load_dataset(
        model_path: Path, model_info: ModelConfigurationSchema, dataset: str | None
    ) -> tuple[NDArray[np.object_] | None, NDArray[np.float32] | None]:
        """Load dataset labels and embeddings if available."""
        if not model_info.datasets:
            logger.info("No datasets configured in model_info.json")
            return None, None

        dataset_name = dataset or "ImageNet_1k"  # Default for CLIP
        if model_info.model_type == "bioclip":
            dataset_name = dataset or "TreeOfLife-10M"

        dataset_filename = model_info.datasets.get(dataset_name)
        if not dataset_filename:
            logger.warning(f"Dataset '{dataset_name}' not found in model_info.json")
            return None, None

        dataset_path = model_path / dataset_filename
        if not dataset_path.exists():
            logger.warning(
                f"Dataset file not found: {dataset_path}. Classification disabled."
            )
            return None, None

        try:
            if dataset_path.suffix == ".npz":
                data = np.load(dataset_path, allow_pickle=True)
                labels, embeddings = data["labels"], data.get("embeddings")
                logger.info(
                    f"Loaded NPZ dataset '{dataset_name}': {len(labels)} classes"
                )
                return labels, embeddings
            elif dataset_path.suffix == ".npy":
                data = np.load(dataset_path, allow_pickle=True).item()
                labels = np.array(data["labels"], dtype=object)
                embeddings = data.get("embeddings")
                logger.info(
                    f"Loaded NPY dataset '{dataset_name}': {len(labels)} classes"
                )
                return labels, embeddings
            else:
                logger.warning(f"Unsupported dataset format: {dataset_path.suffix}")
                return None, None
        except Exception as e:
            logger.warning(
                f"Failed to load dataset '{dataset_name}': {e}. Disabling classification."
            )
            return None, None

    @staticmethod
    def _log_resource_summary(resources: ModelResources) -> None:
        """Log a summary of loaded resources."""
        logger.info(f"✅ Resources loaded for {resources.model_name}")
        logger.info(f"   Runtime: {resources.runtime}")
        logger.info(f"   Model type: {resources.model_info.model_type}")
        logger.info(f"   Embedding dim: {resources.get_embedding_dim()}")
        if image_size := resources.get_image_size():
            logger.info(f"   Image size: {image_size[0]}x{image_size[1]}")
        logger.info(
            f"   Tokenizer: {'custom (tokenizer.json)' if resources.tokenizer_config else 'SimpleTokenizer (fallback)'}"
        )
        if resources.has_classification_support():
            num_classes = len(resources.labels) if resources.labels is not None else 0
            logger.info(f"   Classification: ✅ enabled ({num_classes} classes)")
        else:
            logger.info("   Classification: ⚠️  disabled (no dataset)")
