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
from typing import Any, Literal, cast

import lumen_resources
import numpy as np

# Assuming these pydantic models are provided by the lumen_resources package
from lumen_resources.lumen_config import ModelConfig
from lumen_resources.model_info import Datasets, ModelInfo
from numpy.typing import NDArray
from pydantic import ValidationError

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
    model_info: ModelInfo
    config: dict[str, Any]
    tokenizer_config: dict[str, Any] | None
    labels: NDArray[np.object_] | None
    label_embeddings: NDArray[np.float32] | None
    source_format: Literal["huggingface", "openclip", "modelscope", "custom"]
    source_repo: str

    def get_model_file(self, filename: str) -> Path:
        """Get a file path within the runtime-specific directory."""
        return self.runtime_files_path / filename

    def has_classification_support(self) -> bool:
        """Check if this model supports classification (has dataset)."""
        return self.labels is not None

    def get_embedding_dim(self) -> int:
        """Get embedding dimension from model_info."""
        embedding_dim = self.model_info.embedding_dim
        if embedding_dim is None:
            raise ModelInfoError("Cannot get embedding_dim from model_info.json.")
        return embedding_dim

    def get_image_size(self) -> tuple[int, int] | None:
        """Get image input size from config.json."""
        size = self.config.get("image_size", None)
        if size is None:
            vision_cfg = self.config.get("vision_cfg")
            if isinstance(vision_cfg, dict):
                size = vision_cfg.get("image_size", None)

        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                h = int(size[0])
                w = int(size[1])
                return h, w
            except (TypeError, ValueError):
                return None

        if isinstance(size, int):
            return size, size

        return None

    def get_normalization_stats(self) -> dict[str, list[float]]:
        """
        Get image normalization statistics from config or fall back to defaults based on model type.

        Returns:
            Dictionary with 'mean' and 'std' keys, each containing lists of 3 float values.
        """
        # Try to get from OpenCLIP preprocess_cfg first
        preprocess_cfg = self.config.get("preprocess_cfg", None)

        if isinstance(preprocess_cfg, dict):
            mean = preprocess_cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
            std = preprocess_cfg.get("std", [0.26862954, 0.26130258, 0.27577711])

            # Ensure they're lists of 3 floats
            if isinstance(mean, (list, tuple)) and len(mean) == 3:
                mean = [float(x) for x in mean]
            else:
                mean = [0.48145466, 0.4578275, 0.40821073]

            if isinstance(std, (list, tuple)) and len(std) == 3:
                std = [float(x) for x in std]
            else:
                std = [0.26862954, 0.26130258, 0.27577711]

            return {"mean": mean, "std": std}

        # Fall back to defaults based on model type/source
        model_name = self.model_name.lower()

        # ImageNet-based models use different stats
        if any(x in model_name for x in ["imagenet", "resnet", "vgg", "efficientnet"]):
            return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

        # Default to OpenAI CLIP stats for most CLIP variants
        return {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }


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
            model_info = lumen_resources.load_and_validate_model_info(
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

        # Step 3: Load config and tokenizer based on source format
        source_format = model_info.source.format.value
        if source_format == "openclip":
            # Load model config.json (required) and attempt to load tokenizer.json (optional)
            config = ResourceLoader._load_json(
                model_root_path / "open_clip_config.json"
            )
            tokenizer_config = ResourceLoader._load_for_openclip(model_root_path)
            if tokenizer_config is None:
                logger.info(
                    "No tokenizer config loaded for OpenCLIP model; using SimpleTokenizer fallback"
                )
        elif source_format == "huggingface":
            config = ResourceLoader._load_json(model_root_path / "config.json")
            tokenizer_config = ResourceLoader._load_for_huggingface(model_root_path)
        else:
            raise ModelInfoError(
                f"Unsupported source_format '{source_format}'. Must be 'openclip' or 'huggingface'."
            )

        # Step 4: Load dataset if specified
        labels, embeddings = ResourceLoader._load_dataset(
            model_root_path, model_info, model_config.dataset
        )

        source_repo = model_info.source.repo_id

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
            source_format=source_format,
            source_repo=source_repo,
        )

        ResourceLoader._log_resource_summary(resources)
        return resources

    @staticmethod
    def _validate_intent_and_get_paths(
        model_root: Path, config: ModelConfig, info: ModelInfo
    ) -> Path:
        """
        Validates the user's ModelConfig intent against the model_info manifest.
        Returns the correct path to runtime files.
        """
        runtime = config.runtime.value
        if runtime not in info.runtimes:
            raise RuntimeNotSupportedError(
                f"Runtime '{runtime}' not listed in model_info.json. "
                + f"Available: {list(info.runtimes.keys())}"
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
                    + f"Supported devices: {runtime_info.devices}"
                )
            runtime_path = model_root / runtime / config.rknn_device
        elif runtime == "torch":
            runtime_path = model_root
        else:
            runtime_path = model_root / runtime

        if not runtime_path.exists():
            raise ResourceNotFoundError(f"Runtime directory not found: {runtime_path}")

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
    def _load_for_openclip(model_path: Path) -> dict[str, Any] | None:
        """Load tokenizer for OpenCLIP models. Fallback to SimpleTokenizer if not found."""
        tokenizer_path = model_path / "tokenizer.json"
        if tokenizer_path.exists():
            try:
                tokenizer_config = ResourceLoader._load_json(tokenizer_path)
                logger.info("Loaded tokenizer.json for OpenCLIP model")
                return tokenizer_config
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer.json: {e}. Fallback to SimpleTokenizer"
                )
        else:
            logger.info(
                "No tokenizer.json found, will use SimpleTokenizer for OpenCLIP model"
            )
        return None

    @staticmethod
    def _load_for_huggingface(model_path: Path) -> dict[str, Any] | None:
        """Load tokenizer for HuggingFace models. tokenizer.json is required."""
        tokenizer_path = model_path / "tokenizer.json"
        if not tokenizer_path.exists():
            logger.info(
                "No tokenizer.json found, will use preprocessor.json for HuggingFace"
            )
            return None
        try:
            tokenizer_config = ResourceLoader._load_json(tokenizer_path)
            logger.info("Loaded tokenizer.json for HuggingFace model")
            return tokenizer_config
        except Exception as e:
            raise ResourceValidationError(
                f"Failed to load tokenizer.json for HuggingFace model: {e}"
            ) from e

    @staticmethod
    def _load_dataset(
        model_root_path: Path, model_info: ModelInfo, dataset: str | None
    ) -> tuple[NDArray[np.object_] | None, NDArray[np.float32] | None]:
        # 1. check if repo provides datasets
        if not model_info.datasets:
            logger.info(
                f"No datasets configured in model_info.json {model_info.datasets}"
            )
            return None, None

        # 2. determine dataset name to load
        dataset_name = dataset

        # 3. validate dataset name
        if not dataset_name or dataset_name not in model_info.datasets:
            logger.warning(
                f"Dataset '{dataset_name}' not in model_info.json. Disabling classification."
            )
            return None, None

        dataset_info: Datasets | None = model_info.datasets.get(dataset_name)
        if dataset_info is None:
            raise ResourceValidationError("Failed to get datasets information.")
        labels_path = model_root_path / dataset_info.labels
        embeddings_path = model_root_path / dataset_info.embeddings

        try:
            labels: NDArray[np.object_] | None = None
            embeddings: NDArray[np.float32] | None = None

            if isinstance(dataset_info, Datasets):
                if not labels_path.exists() or not embeddings_path.exists():
                    raise FileNotFoundError(
                        f"Missing dataset files: {labels_path} or {embeddings_path}"
                    )

                with open(labels_path, encoding="utf-8") as f:
                    labels = np.array(json.load(f), dtype=object)

                logger.info(
                    f"Loading embeddings for '{dataset_name}' with memory mapping..."
                )
                # FIX: Cast the result of np.load to the specific type we expect.
                raw_embeddings = np.load(embeddings_path, mmap_mode="r")
                embeddings = cast(NDArray[np.float32], raw_embeddings)

            if labels is not None:
                logger.info(f"Loaded dataset '{dataset_name}': {len(labels)} classes")

            return labels, embeddings

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
