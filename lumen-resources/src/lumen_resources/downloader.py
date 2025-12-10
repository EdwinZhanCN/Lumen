"""
Resource Downloader Manager

@requires: Valid LumenConfig and platform adapter
@returns: Download results with validation
@errors: DownloadError, ValidationError
"""

from dataclasses import dataclass
from pathlib import Path

from .exceptions import DownloadError, ModelInfoError, ValidationError
from .lumen_config import LumenConfig, ModelConfig, Region, Runtime
from .model_info import ModelInfo
from .model_info_validator import load_and_validate_model_info
from .platform import Platform, PlatformType


@dataclass
class DownloadResult:
    """Result of a single model download operation.

    Contains information about the download attempt including success status,
    file paths, missing files, and error messages if applicable.

    Attributes:
        model_type: Model type identifier (e.g., "clip:default").
        model_name: Name of the model repository.
        runtime: Runtime type used for the model.
        success: Whether the download was successful. Defaults to False.
        model_path: Local path where model was downloaded. None if failed.
        missing_files: List of required files that are missing. None if no missing files.
        error: Error message if download failed. None if successful.

    Example:
        >>> result = DownloadResult(
        ...     model_type="clip:default",
        ...     model_name="ViT-B-32",
        ...     runtime="torch",
        ...     success=True,
        ...     model_path=Path("/models/clip_vit_b32")
        ... )
        >>> print(result.success)
        True
    """

    model_type: str
    model_name: str
    runtime: str
    success: bool = False
    model_path: Path | None = None
    missing_files: list[str] | None = None
    error: str | None = None

    def __post_init__(self):
        """Initialize missing_files as empty list if None."""
        if self.missing_files is None:
            self.missing_files = []


class Downloader:
    """Main resource downloader for Lumen models.

    Handles downloading models from various platforms (Hugging Face, ModelScope)
    with support for different runtimes (torch, onnx, rknn) and validation
    of model integrity and metadata.

    Attributes:
        config: Lumen services configuration.
        verbose: Whether to print progress messages.
        platform: Platform adapter for downloading models.

    Example:
        >>> config = load_and_validate_config("config.yaml")
        >>> downloader = Downloader(config, verbose=True)
        >>> results = downloader.download_all()
        >>> for model_type, result in results.items():
        ...     print(f"{model_type}: {'✅' if result.success else '❌'}")
    """

    def __init__(self, config: LumenConfig, verbose: bool = True):
        """Initialize downloader with configuration.

        Args:
            config: Validated Lumen services configuration.
            verbose: Whether to print progress messages during download.

        Raises:
            ValidationError: If configuration is invalid.
            OSError: If cache directory cannot be created.
        """
        self.config: LumenConfig = config
        self.verbose: bool = verbose

        # Determine platform type and owner from region
        platform_type = (
            PlatformType.MODELSCOPE
            if config.metadata.region == Region.cn
            else PlatformType.HUGGINGFACE
        )
        platform_owner = (
            "LumilioPhotos" if config.metadata.region == Region.cn else "Lumilio-Photos"
        )

        self.platform: Platform = Platform(platform_type, platform_owner)

        # Ensure cache directory exists
        cache_dir = Path(config.metadata.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "models").mkdir(parents=True, exist_ok=True)

    def download_all(self, force: bool = False) -> dict[str, DownloadResult]:
        """Download all enabled models from all enabled services.

        Iterates through all enabled services and their model configurations,
        downloading each model with its required files and validating integrity.

        Args:
            force: Whether to force re-download even if models are already cached.

        Returns:
            Dictionary mapping model type identifiers ("service:alias") to DownloadResult objects.

        Example:
            >>> downloader = Downloader(config)
            >>> results = downloader.download_all(force=True)
            >>> for model_type, result in results.items():
            ...     if result.success:
            ...         print(f"✅ {model_type} -> {result.model_path}")
            ...     else:
            ...         print(f"❌ {model_type}: {result.error}")
        """
        results: dict[str, DownloadResult] = {}

        # Iterate through enabled services and their models
        for service_name, service_config in self.config.services.items():
            if not service_config.enabled:
                continue

            for alias, model_config in service_config.models.items():
                model_type = f"{service_name}:{alias}"

                if self.verbose:
                    print(f"\n📦 Processing {model_type.upper()}")
                    print(f"    Model: {model_config.model}")
                    print(f"    Runtime: {model_config.runtime.value}")

                result = self._download_model(model_type, model_config, force)
                results[model_type] = result

                # Print result
                if self.verbose:
                    if result.success:
                        print(f"✅ Download successful: {result.model_path}")
                        if result.missing_files:
                            print(
                                f"⚠️  Missing files: {', '.join(result.missing_files)}"
                            )
                        else:
                            print("✅ All files verified")
                    else:
                        print(f"❌ Download failed: {result.error}")

        return results

    def _get_runtime_patterns(self, runtime: Runtime) -> list[str]:
        """Get file patterns to download based on runtime.

        Determines which file patterns to include in downloads based on the
        model runtime. Always includes model_info.json and config files.

        Args:
            runtime: The model runtime (torch, onnx, rknn).

        Returns:
            List of file glob patterns for the download.

        Example:
            >>> patterns = downloader._get_runtime_patterns(Runtime.torch)
            >>> print("model_info.json" in patterns)
            True
        """
        patterns = [
            "model_info.json",
            "*config*",
        ]  # Always include model_info.json and config files.

        if runtime == Runtime.torch:
            patterns.extend(
                [
                    "*.bin",
                    "*.pt",
                    "*.pth",
                    "*.safetensors",
                    "pytorch_model*.bin",
                    "model.safetensors",
                    "*vocab*",
                    "*tokenizer*",
                    "special_tokens_map.json",
                ]
            )
        elif runtime == Runtime.onnx:
            patterns.extend(
                [
                    "*.onnx",
                    "*.ort",
                    "*vocab*",
                    "*tokenizer*",
                    "special_tokens_map.json",
                    "preprocessor_config.json",
                ]
            )
        elif runtime == Runtime.rknn:
            patterns.extend(
                [
                    "*.rknn",
                    "*vocab*",
                    "*tokenizer*",
                    "special_tokens_map.json",
                    "preprocessor_config.json",
                ]
            )

        return patterns

    def _download_model(
        self, model_type: str, model_config: ModelConfig, force: bool
    ) -> DownloadResult:
        """Download a single model with its runtime files.

        Handles the complete download process for a single model including
        runtime files, metadata validation, and integrity checks. Performs
        rollback on failure by cleaning up downloaded files.

        Args:
            model_type: Identifier for the model (e.g., "clip:default").
            model_config: Model configuration from LumenConfig.
            force: Whether to force re-download even if already cached.

        Returns:
            DownloadResult with success status, file paths, and error details.

        Raises:
            DownloadError: If platform download fails.
            ModelInfoError: If model_info.json is missing or invalid.
            ValidationError: If model configuration is not supported.
        """
        result = DownloadResult(
            model_type=model_type,
            model_name=model_config.model,
            runtime=model_config.runtime.value
            if hasattr(model_config.runtime, "value")
            else str(model_config.runtime),
        )

        try:
            # Phase 1: Prepare and download runtime + JSON only
            patterns = self._get_runtime_patterns(model_config.runtime)
            cache_dir = Path(self.config.metadata.cache_dir).expanduser()

            model_path = self.platform.download_model(
                repo_name=model_config.model,
                cache_dir=cache_dir,
                allow_patterns=patterns,
                force=force,
            )

            result.model_path = model_path

            # Load and validate model_info.json
            model_info = self._load_model_info(model_path)

            # Logical validation
            self._validate_model_config(model_info, model_config)

            # If dataset specified, download by relative paths from model_info.json
            if model_config.dataset and model_info.datasets:
                dataset_files = model_info.datasets.get(model_config.dataset)
                if dataset_files:
                    for file_rel in [dataset_files.labels, dataset_files.embeddings]:
                        dataset_path = model_path / file_rel
                        if not dataset_path.exists():
                            # Download only the dataset file by its relative path
                            try:
                                _ = self.platform.download_model(
                                    repo_name=model_config.model,
                                    cache_dir=cache_dir,
                                    allow_patterns=[file_rel],
                                    force=False,
                                )
                            except DownloadError as e:
                                raise DownloadError(
                                    f"Failed to download dataset file {file_rel}: {e}"
                                )

            # Final: File integrity validation
            missing = self._validate_files(model_path, model_info, model_config)
            result.missing_files = missing

            if missing:
                raise ValidationError(
                    f"Missing required files after download: {', '.join(missing)}"
                )

            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)

            # Rollback: cleanup model directory on failure
            if result.model_path and result.model_path.exists():
                if self.verbose:
                    print(f"   🔄 Rolling back: cleaning up {result.model_path}")
                cache_dir = Path(self.config.metadata.cache_dir).expanduser()
                self.platform.cleanup_model(model_config.model, cache_dir)

        return result

    def _load_model_info(self, model_path: Path) -> ModelInfo:
        """Load and parse model_info.json using validator.

        Loads the model_info.json file from the model directory and validates
        it against the ModelInfo schema to ensure metadata integrity.

        Args:
            model_path: Path to the downloaded model directory.

        Returns:
            Validated ModelInfo object containing model metadata.

        Raises:
            ModelInfoError: If model_info.json is missing or fails validation.

        Example:
            >>> model_info = downloader._load_model_info(Path("/models/clip_vit_b32"))
            >>> print(model_info.name)
            'ViT-B-32'
        """
        info_file = model_path / "model_info.json"

        if not info_file.exists():
            msg = (
                f"model_info.json not found in {model_path}. "
                "Repository must contain model metadata."
            )
            raise ModelInfoError(msg)

        try:
            return load_and_validate_model_info(info_file)
        except Exception as e:
            raise ModelInfoError(f"Failed to load/validate model_info.json: {e}")

    def _validate_model_config(
        self, model_info: ModelInfo, model_config: ModelConfig
    ) -> None:
        """Validate that model supports the requested configuration.

        Checks if the model metadata indicates support for the requested
        runtime, device (for RKNN), and dataset configurations.

        Args:
            model_info: Validated model metadata from model_info.json.
            model_config: Requested model configuration from LumenConfig.

        Raises:
            ValidationError: If the requested configuration is not supported
                by the model according to its metadata.

        Example:
            >>> downloader._validate_model_config(model_info, model_config)  # No exception
            >>> # If runtime is not supported, raises ValidationError
        """
        runtime = model_config.runtime
        rknn_device = model_config.rknn_device

        # Check if runtime is available
        runtime_config = model_info.runtimes.get(runtime.value)
        if not runtime_config or not runtime_config.available:
            raise ValidationError(
                f"Model {model_config.model} does not support runtime '{runtime.value}'"
            )

        # Special check for RKNN device support
        if runtime == Runtime.rknn and rknn_device:
            if not runtime_config.devices or rknn_device not in runtime_config.devices:
                msg = (
                    f"Model {model_config.model} does not support "
                    f"runtime '{runtime.value}' with device '{rknn_device}'"
                )
                raise ValidationError(msg)

        # Validate dataset if specified
        if model_config.dataset:
            if (
                not model_info.datasets
                or model_config.dataset not in model_info.datasets
            ):
                msg = (
                    f"Dataset '{model_config.dataset}' not available for "
                    f"model {model_config.model}"
                )
                raise ValidationError(msg)

    def _validate_files(
        self, model_path: Path, model_info: ModelInfo, model_config: ModelConfig
    ) -> list[str]:
        """Validate that required model files exist.

        Checks that all required files for the specified runtime and device
        are present in the downloaded model directory. Also validates dataset
        files if specified in the configuration.

        Args:
            model_path: Path to the downloaded model directory.
            model_info: Validated model metadata from model_info.json.
            model_config: Model configuration specifying runtime and dataset.

        Returns:
            List of missing file paths relative to model directory.
            Empty list if all required files are present.

        Example:
            >>> missing = downloader._validate_files(model_path, model_info, model_config)
            >>> if not missing:
            ...     print("All required files are present")
            ... else:
            ...     print(f"Missing files: {missing}")
        """
        missing: list[str] = []

        # Get runtime configuration
        runtime_config = model_info.runtimes.get(model_config.runtime.value)
        if not runtime_config or not runtime_config.files:
            return missing

        # Get required files based on runtime type
        if model_config.runtime == Runtime.rknn and model_config.rknn_device:
            # RKNN files are organized by device
            if isinstance(runtime_config.files, dict):
                required_files = runtime_config.files.get(model_config.rknn_device, [])
            else:
                required_files = []
        else:
            # Other runtimes have simple list
            if isinstance(runtime_config.files, list):
                required_files = runtime_config.files
            else:
                required_files = []

        # Check each required file
        for file_path in required_files:
            full_path = model_path / file_path
            if not full_path.exists():
                missing.append(file_path)

        # Check dataset files if specified
        if model_config.dataset and model_info.datasets:
            dataset_files = model_info.datasets.get(model_config.dataset)
            if dataset_files:
                for file_rel in [dataset_files.labels, dataset_files.embeddings]:
                    dataset_path = model_path / file_rel
                    if not dataset_path.exists():
                        missing.append(file_rel)

        return missing
