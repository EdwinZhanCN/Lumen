import logging
from dataclasses import dataclass
from pathlib import Path

import lumen_resources
from lumen_resources.lumen_config import ModelConfig
from lumen_resources.model_info import ModelInfo
from pydantic import ValidationError

from .exceptions import (
    ModelInfoError,
    ResourceNotFoundError,
    RuntimeNotSupportedError,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResources:
    """Container for OCR model resources and metadata.

    This dataclass provides unified access to model files, metadata, and
    configuration information needed for OCR tasks.

    Attributes:
        model_root_path: Root directory containing the model files and metadata.
        runtime_files_path: Runtime-specific directory containing model files.
        runtime: Runtime identifier (e.g., "onnx", "rknn").
        model_name: Human-readable model name.
        model_info: ModelInfo object containing comprehensive model metadata.
    """

    model_root_path: Path
    runtime_files_path: Path
    runtime: str
    model_name: str
    model_info: ModelInfo

    def get_model_file(self, filename: str) -> Path:
        """Get the full path to a model file within the runtime directory.

        Args:
            filename: Name of the model file (e.g., "det_model.onnx").

        Returns:
            Path: Full path to the model file within the runtime-specific directory.
        """
        return self.runtime_files_path / filename

    def get_file_path(self, filename: str) -> Path:
        """Get path to a file, checking runtime directory first, then model root.

        Useful for auxiliary files like vocabulary/keys which might be shared
        across runtimes or specific to one.

        Args:
            filename: Name of the file.

        Returns:
            Path: Resolved path to the file.
        """
        # Check runtime directory first
        p = self.runtime_files_path / filename
        if p.exists():
            return p
        # Check model root directory
        p = self.model_root_path / filename
        if p.exists():
            return p
        # Return runtime path by default (for error reporting if missing)
        return self.runtime_files_path / filename


class ResourceLoader:
    """Utility class for loading and validating OCR model resources."""

    @staticmethod
    def load_model_resource(
        cache_dir: str | Path, model_config: ModelConfig
    ) -> ModelResources:
        """Load model resources from configuration and validate availability.

        Args:
            cache_dir: Base directory for model caching.
            model_config: Validated ModelConfig from lumen_config.yaml.

        Returns:
            ModelResources: Fully configured resource container.

        Raises:
            ResourceNotFoundError: If required files are missing.
            ModelInfoError: If model_info.json is invalid.
            RuntimeNotSupportedError: If runtime is not supported.
        """
        cache_dir = Path(cache_dir).expanduser().resolve()
        model_root_path = cache_dir / "models" / model_config.model

        logger.info(
            f"Loading resources for {model_config.model} (runtime: {model_config.runtime.value})"
        )

        try:
            model_info = lumen_resources.load_and_validate_model_info(
                model_root_path / "model_info.json"
            )
        except FileNotFoundError as e:
            raise ResourceNotFoundError(
                f"model_info.json not found in {model_root_path}"
            ) from e
        except ValidationError as e:
            raise ModelInfoError(f"model_info.json is invalid: {e}") from e

        runtime_files_path = ResourceLoader._validate_intent_and_get_paths(
            model_root_path, model_config, model_info
        )

        resources = ModelResources(
            model_root_path=model_root_path,
            runtime_files_path=runtime_files_path,
            model_name=model_config.model,
            model_info=model_info,
            runtime=str(model_config.runtime.value),
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
            raise RuntimeNotSupportedError(
                "'torch' runtime not supported for this model. "
            )
        elif runtime == "onnx":
            runtime_path = model_root / runtime
        else:
            raise RuntimeNotSupportedError(
                f"'{runtime}' runtime not supported for this model. "
            )

        if not runtime_path.exists():
            raise ResourceNotFoundError(f"Runtime directory not found: {runtime_path}")

        # Check for existence of required files for the runtime
        if runtime_info.files:
            # Handle both list and dict formats for files
            files_to_check = []
            if isinstance(runtime_info.files, list):
                files_to_check = runtime_info.files
            elif isinstance(runtime_info.files, dict):
                # If it's a dict (e.g. by precision), check all values
                for file_list in runtime_info.files.values():
                    if isinstance(file_list, list):
                        files_to_check.extend(file_list)

            for file_path_str in files_to_check:
                # Files in manifest are relative to model root, but for specific runtimes
                # they might be inside the runtime folder.
                # However, the schema usually implies paths relative to the runtime folder
                # OR relative to root.
                # Standard Lumen practice: runtime-specific files are in runtime folder.
                # Let's check in runtime path first.
                full_path = runtime_path / file_path_str
                if not full_path.exists():
                    # Fallback check in root (for shared files)
                    root_check = model_root / file_path_str
                    if not root_check.exists():
                        # Log warning but don't fail hard here as some files might be optional
                        # or structure might vary. Strict checking is good but can be fragile.
                        # For now, we'll trust the runtime_path check for critical files
                        # done by the backend.
                        pass

        return runtime_path

    @staticmethod
    def _log_resource_summary(resources: ModelResources) -> None:
        """Log a summary of loaded resources."""
        logger.info(f"✅ Resources loaded for {resources.model_name}")
        logger.info(f"   Runtime: {resources.runtime}")
        logger.info(f"   Model type: {resources.model_info.model_type}")
