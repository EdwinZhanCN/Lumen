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
    """Container for face model resources and metadata.

    This dataclass provides unified access to model files, metadata, and
    configuration information needed for face detection and recognition.
    It abstracts away the file system structure and provides convenient
    access to model-specific information.

    Attributes:
        model_root_path: Root directory containing the model files and metadata.
            Typically structured as: cache_dir/models/{model_name}/
        runtime_files_path: Runtime-specific directory containing model files.
            Path varies by runtime: model_root_path/{runtime}/ or model_root_path/{runtime}/{device}/
        runtime: Runtime identifier (e.g., "onnx", "rknn", "torch") indicating
            which inference engine should be used.
        model_name: Human-readable model name (e.g., "buffalo_l", "arcface")
            used for logging and identification.
        model_info: ModelInfo object containing comprehensive model metadata
            including capabilities, supported operations, and configuration.

    Example:
        ```python
        # Access model files
        detection_model = resources.get_model_file("detection.fp32.onnx")
        recognition_model = resources.get_model_file("recognition.fp32.onnx")

        # Get model information
        embedding_dim = resources.get_embedding_dim()  # typically 512
        model_type = resources.model_info.model_type   # "face_recognition"
        ```

    Note:
        This class is typically created by ResourceLoader.load_model_resource()
        and should not be instantiated directly in most cases.
    """

    model_root_path: Path
    runtime_files_path: Path
    runtime: str
    model_name: str
    model_info: ModelInfo

    def get_model_file(self, filename: str) -> Path:
        """Get the full path to a model file within the runtime directory.

        Args:
            filename: Name of the model file (e.g., "detection.fp32.onnx",
                "recognition.fp32.onnx", "model.json").

        Returns:
            Path: Full path to the model file within the runtime-specific directory.
                The path combines runtime_files_path with the provided filename.

        Raises:
            ValueError: If filename is empty or None.

        Example:
            ```python
            detection_path = resources.get_model_file("detection.fp32.onnx")
            # Returns: /cache/models/buffalo_l/onnx/detection.fp32.onnx
            ```
        """
        return self.runtime_files_path / filename

    def get_embedding_dim(self) -> int:
        """Get the face embedding dimension from model metadata.

        Returns:
            int: Dimension of the face embedding vectors produced by this model.
                Typical values are 128, 256, or 512 depending on the model architecture.
                Returns 0 if not specified in the model metadata.

        Note:
            This value is used to validate embedding outputs and configure
            downstream processing pipelines that depend on specific embedding dimensions.
        """
        return self.model_info.embedding_dim or 0


class ResourceLoader:
    """Utility class for loading and validating face model resources.

    This class provides static methods to load model resources from validated
    Lumen configurations. It handles the complex logic of runtime validation,
    file path resolution, and resource preparation required for face recognition
    models.

    Key responsibilities:
    - Load and validate model_info.json metadata
    - Validate runtime compatibility and availability
    - Resolve runtime-specific file paths
    - Validate required model files exist
    - Create ModelResources instances for backend use

    Supported runtimes:
    - onnx: ONNX Runtime models with .onnx files
    - rknn: Rockchip NPU models with device-specific directories
    - torch: PyTorch models with .pth/.pt files
    """

    @staticmethod
    def load_model_resource(
        cache_dir: str | Path, model_config: ModelConfig
    ) -> ModelResources:
        """Load model resources from configuration and validate availability.

        This is the primary entry point for loading face model resources. It validates
        the model configuration, loads metadata, verifies file availability, and creates
        a ModelResources instance ready for backend initialization.

        Args:
            cache_dir: Base directory for model caching. Models are stored in
                cache_dir/models/{model_name}/ structure.
            model_config: Validated ModelConfig from lumen_config.yaml containing
                model name, runtime settings, and configuration parameters.

        Returns:
            ModelResources: Fully configured resource container with validated paths
                and metadata. Ready to be passed to backend constructors.

        Raises:
            ResourceNotFoundError: If model_info.json or required model files are missing.
            ModelInfoError: If model_info.json is corrupted or invalid.
            RuntimeNotSupportedError: If specified runtime is not supported by the model.
            ConfigError: If runtime-specific configuration is invalid (e.g., missing rknn_device).

        Example:
            ```python
            # Load from validated config
            config = load_and_validate_config("config.yaml")
            model_config = config.services["face"].models["general"]

            # Create resources
            resources = ResourceLoader.load_model_resource(
                cache_dir="~/.cache/lumen",
                model_config=model_config
            )

            # Use with backend
            backend = ONNXRTBackend(resources=resources)
            ```

        Note:
            This method performs comprehensive validation and will raise detailed
            exceptions for any configuration or file availability issues.
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
            runtime_path = model_root
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
    def _log_resource_summary(resources: ModelResources) -> None:
        """Log a summary of loaded resources."""
        logger.info(f"✅ Resources loaded for {resources.model_name}")
        logger.info(f"   Runtime: {resources.runtime}")
        logger.info(f"   Model type: {resources.model_info.model_type}")
        logger.info(f"   Embedding dim: {resources.get_embedding_dim()}")
