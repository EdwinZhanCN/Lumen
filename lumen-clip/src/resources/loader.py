"""
loader.py

Resource loading and validation for lumen-clip models.
Handles loading model configurations, tokenizers, datasets, and validation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .exceptions import (
    DatasetNotFoundError,
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
        model_path: Path to the model directory
        model_name: Name of the model
        runtime: Runtime type (torch/onnx/rknn)
        model_info: Parsed model_info.json content
        config: Parsed config.json content
        tokenizer_config: Parsed tokenizer.json content (optional)
        labels: Label names array (None if no dataset)
        label_embeddings: Pre-computed label embeddings (optional)
    """

    model_path: Path
    model_name: str
    runtime: str

    # Metadata
    model_info: dict[str, Any]
    config: dict[str, Any]
    tokenizer_config: dict[str, Any] | None

    # Dataset (optional - None means classification not supported)
    labels: NDArray[np.object_] | None
    label_embeddings: NDArray[np.float32] | None

    def get_runtime_path(self) -> Path:
        """Get the runtime-specific directory path."""
        return self.model_path / self.runtime

    def get_model_file(self, filename: str) -> Path:
        """Get a file path within the runtime directory."""
        return self.get_runtime_path() / filename

    def has_classification_support(self) -> bool:
        """Check if this model supports classification (has dataset)."""
        return self.labels is not None

    def get_embedding_dim(self) -> int | None:
        """Get embedding dimension from config or infer from embeddings."""
        # Try to get from config first
        if "embed_dim" in self.config:
            return self.config["embed_dim"]
        if "embedding_dim" in self.config:
            return self.config["embedding_dim"]

        # Try to infer from label_embeddings
        if self.label_embeddings is not None:
            return self.label_embeddings.shape[1]

        return None

    def get_image_size(self) -> tuple[int, int] | None:
        """Get image input size from config."""
        if "image_size" in self.config:
            size = self.config["image_size"]
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return tuple(size)
            if isinstance(size, int):
                return (size, size)

        # Check vision config
        if "vision_cfg" in self.config:
            vision_cfg = self.config["vision_cfg"]
            if "image_size" in vision_cfg:
                size = vision_cfg["image_size"]
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    return tuple(size)
                if isinstance(size, int):
                    return (size, size)

        return None


class ResourceLoader:
    """
    Loads and validates model resources from the standardized directory structure.

    Directory structure:
        {cache_dir}/models/{model_name}/
        ├── model_info.json       (required)
        ├── config.json           (required)
        ├── tokenizer.json        (optional)
        ├── ImageNet_1k.npz       (optional - CLIP)
        ├── TreeOfLife-*.npy      (optional - BioCLIP)
        └── {runtime}/            (required)
            ├── torch/model.pt
            ├── onnx/vision.onnx + text.onnx
            └── rknn/{device}/vision.rknn + text.rknn
    """

    @staticmethod
    def load_model_resources(
        cache_dir: str | Path,
        model_name: str,
        runtime: str,
        dataset: str | None = None,
    ) -> ModelResources:
        """
        Load all resources for a model.

        Args:
            cache_dir: Root cache directory
            model_name: Name of the model
            runtime: Runtime type (torch/onnx/rknn)
            dataset: Dataset name for BioCLIP (e.g., "TreeOfLife-10M")

        Returns:
            ModelResources object with all loaded resources

        Raises:
            ResourceNotFoundError: If required files are missing
            RuntimeNotSupportedError: If runtime is not supported
            ModelInfoError: If model_info.json is invalid
        """
        cache_dir = Path(cache_dir).expanduser().resolve()
        model_path = cache_dir / "models" / model_name

        logger.info(f"Loading resources for {model_name} (runtime: {runtime})")

        # 1. Validate base structure
        ResourceLoader._validate_base_structure(model_path)

        # 2. Load model_info.json
        model_info = ResourceLoader._load_json(model_path / "model_info.json")
        ResourceLoader._validate_model_info(model_info)

        # 3. Validate runtime support
        ResourceLoader._validate_runtime_support(model_path, model_info, runtime)

        # 4. Load config.json
        config = ResourceLoader._load_json(model_path / "config.json")

        # 5. Load tokenizer.json (optional)
        tokenizer_config = ResourceLoader._load_tokenizer(model_path)

        # 6. Load dataset (optional - failure only logs warning)
        labels, embeddings = ResourceLoader._load_dataset(
            model_path, model_info, dataset
        )

        resources = ModelResources(
            model_path=model_path,
            model_name=model_name,
            runtime=runtime,
            model_info=model_info,
            config=config,
            tokenizer_config=tokenizer_config,
            labels=labels,
            label_embeddings=embeddings,
        )

        # Log resource summary
        ResourceLoader._log_resource_summary(resources)

        return resources

    @staticmethod
    def _validate_base_structure(model_path: Path) -> None:
        """Validate that required base files exist."""
        if not model_path.exists():
            raise ResourceNotFoundError(f"Model directory not found: {model_path}")

        required_files = ["model_info.json", "config.json"]
        for filename in required_files:
            file_path = model_path / filename
            if not file_path.exists():
                raise ResourceNotFoundError(
                    f"Required file missing: {filename} in {model_path}"
                )

    @staticmethod
    def _validate_model_info(model_info: dict) -> None:
        """Validate model_info.json structure."""
        required_fields = ["name", "model_type", "runtimes"]
        for field in required_fields:
            if field not in model_info:
                raise ModelInfoError(f"model_info.json missing required field: {field}")

        if not isinstance(model_info["runtimes"], dict):
            raise ModelInfoError("model_info.json: 'runtimes' must be a dict")

    @staticmethod
    def _validate_runtime_support(
        model_path: Path, model_info: dict, runtime: str
    ) -> None:
        """Validate that the runtime is supported and files exist."""
        runtimes = model_info.get("runtimes", {})

        if runtime not in runtimes:
            available = list(runtimes.keys())
            raise RuntimeNotSupportedError(
                f"Runtime '{runtime}' not supported. Available: {available}"
            )

        runtime_info = runtimes[runtime]
        if not runtime_info.get("available", False):
            raise RuntimeNotSupportedError(
                f"Runtime '{runtime}' is not available for this model"
            )

        # Verify runtime files exist
        required_files = runtime_info.get("files", [])
        for file_path_str in required_files:
            file_path = model_path / file_path_str
            if not file_path.exists():
                raise ResourceNotFoundError(
                    f"Required runtime file missing: {file_path_str}"
                )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load and parse a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ResourceValidationError(f"Invalid JSON in {path.name}: {e}") from e
        except Exception as e:
            raise ResourceNotFoundError(f"Failed to load {path.name}: {e}") from e

    @staticmethod
    def _load_tokenizer(model_path: Path) -> dict[str, Any] | None:
        """Load tokenizer.json if it exists (optional)."""
        tokenizer_path = model_path / "tokenizer.json"
        if tokenizer_path.exists():
            try:
                tokenizer_config = ResourceLoader._load_json(tokenizer_path)
                logger.info("Loaded tokenizer.json")
                return tokenizer_config
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer.json: {e}. "
                    f"Will fallback to SimpleTokenizer"
                )
                return None
        else:
            logger.info("No tokenizer.json found, will use SimpleTokenizer")
            return None

    @staticmethod
    def _load_dataset(
        model_path: Path, model_info: dict, dataset: str | None
    ) -> tuple[NDArray[np.object_] | None, NDArray[np.float32] | None]:
        """
        Load dataset (labels and embeddings) if available.

        Returns:
            Tuple of (labels, embeddings) or (None, None) if not found
        """
        model_type = model_info.get("model_type", "clip")
        datasets = model_info.get("datasets", {})

        if not datasets:
            logger.info("No datasets configured in model_info.json")
            return None, None

        # Determine which dataset file to load
        try:
            if model_type == "clip":
                # CLIP: ImageNet_1k.npz
                dataset_filename = datasets.get("ImageNet_1k")
                if not dataset_filename:
                    logger.warning("No ImageNet_1k dataset in model_info.json")
                    return None, None

                dataset_path = model_path / dataset_filename
                if not dataset_path.exists():
                    logger.warning(
                        f"Dataset file not found: {dataset_filename}. "
                        f"Classification will be disabled."
                    )
                    return None, None

                # Load .npz file
                data = np.load(dataset_path, allow_pickle=True)
                labels = data["labels"]
                embeddings = data.get("embeddings")  # Optional

                logger.info(f"Loaded ImageNet dataset: {len(labels)} classes")
                return labels, embeddings

            elif model_type == "bioclip":
                # BioCLIP: TreeOfLife-{dataset}.npy
                dataset_name = dataset or "TreeOfLife-10M"
                dataset_filename = datasets.get(dataset_name)

                if not dataset_filename:
                    logger.warning(
                        f"Dataset '{dataset_name}' not found in model_info.json"
                    )
                    return None, None

                dataset_path = model_path / dataset_filename
                if not dataset_path.exists():
                    logger.warning(
                        f"Dataset file not found: {dataset_filename}. "
                        f"Classification will be disabled."
                    )
                    return None, None

                # Load .npy file (expects dict with 'labels' and optional 'embeddings')
                data = np.load(dataset_path, allow_pickle=True).item()
                labels = np.array(data["labels"], dtype=object)
                embeddings = data.get("embeddings")

                logger.info(
                    f"Loaded BioCLIP dataset '{dataset_name}': {len(labels)} species"
                )
                return labels, embeddings

            else:
                logger.warning(f"Unknown model_type: {model_type}")
                return None, None

        except Exception as e:
            logger.warning(
                f"Failed to load dataset: {e}. Classification will be disabled."
            )
            return None, None

    @staticmethod
    def _log_resource_summary(resources: ModelResources) -> None:
        """Log a summary of loaded resources."""
        logger.info(f"✅ Resources loaded for {resources.model_name}")
        logger.info(f"   Runtime: {resources.runtime}")
        logger.info(
            f"   Model type: {resources.model_info.get('model_type', 'unknown')}"
        )

        # Log embedding dimension
        embed_dim = resources.get_embedding_dim()
        if embed_dim:
            logger.info(f"   Embedding dim: {embed_dim}")

        # Log image size
        image_size = resources.get_image_size()
        if image_size:
            logger.info(f"   Image size: {image_size[0]}x{image_size[1]}")

        # Log tokenizer
        if resources.tokenizer_config:
            logger.info(f"   Tokenizer: custom (tokenizer.json)")
        else:
            logger.info(f"   Tokenizer: SimpleTokenizer (fallback)")

        # Log classification support
        if resources.has_classification_support():
            assert (
                resources.labels is not None
            )  # has_classification_support ensures this
            num_classes = len(resources.labels)
            logger.info(f"   Classification: ✅ enabled ({num_classes} classes)")
        else:
            logger.info(f"   Classification: ⚠️  disabled (no dataset)")


def validate_single_service_config(config: dict) -> str:
    """
    Validate that only one service is enabled in the configuration.

    Args:
        config: Parsed YAML configuration

    Returns:
        Name of the enabled service

    Raises:
        ConfigError: If zero or multiple services are enabled
    """
    from .exceptions import ConfigError

    services = config.get("services", {})
    if not services:
        raise ConfigError("No services defined in configuration")

    enabled_services = [
        name for name, cfg in services.items() if cfg.get("enabled", False)
    ]

    if len(enabled_services) == 0:
        raise ConfigError(
            "No service enabled in configuration. "
            "Please set 'enabled: true' for exactly one service."
        )

    if len(enabled_services) > 1:
        raise ConfigError(
            f"Only one service can be enabled at a time. "
            f"Found {len(enabled_services)} enabled services: {', '.join(enabled_services)}. "
            f"Please disable all but one service."
        )

    return enabled_services[0]
