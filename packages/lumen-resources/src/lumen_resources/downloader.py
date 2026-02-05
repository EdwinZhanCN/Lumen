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
                    if model_config.precision:
                        print(f"    Precision: {model_config.precision}")

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

    def _get_runtime_patterns(
        self, runtime: Runtime, precision: str | None
    ) -> list[str]:
        """Get file patterns to download based on runtime and precision.

        Determines which file patterns to include in downloads based on the
        model runtime and precision configuration. Always includes model_info.json
        and config files.

        Args:
            runtime: The model runtime (torch, onnx, rknn).
            precision: The precision variant (fp32, fp16, int8, q4fp16, etc.).

        Returns:
            List of file glob patterns for the download.

        Example:
            >>> patterns = downloader._get_runtime_patterns(Runtime.onnx, "fp16")
            >>> print("*.fp16.onnx" in patterns)
            True
        """
        patterns = [
            "model_info.json",
            "*config*",
            "*.txt",
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
                    "*vocab*",
                    "*tokenizer*",
                    "special_tokens_map.json",
                    "preprocessor_config.json",
                ]
            )
            # Add precision-specific ONNX model files if precision is specified
            if precision:
                patterns.extend([f"*.{precision}.onnx"])
            else:
                # If no precision specified, include all common precisions
                patterns.extend(["*.fp32.onnx", "*.fp16.onnx", "*.int8.onnx"])
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
            # RKNN files are already quantized, precision field may indicate variant
            if precision:
                patterns.extend([f"*.{precision}.rknn"])
            else:
                patterns.extend(["*.rknn"])

        return patterns

    def _download_model(
        self, model_type: str, model_config: ModelConfig, force: bool
    ) -> DownloadResult:
        """Download a single model with its runtime files.

        Uses the precision specified in model_config to download the appropriate
        model variant. If no precision is specified for ONNX/RKNN runtimes, will
        download all available precision variants.

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
        # Get file patterns based on runtime and precision from ModelConfig
        patterns = self._get_runtime_patterns(
            model_config.runtime, model_config.precision
        )

        # Download with the determined patterns
        return self._download_model_with_patterns(
            model_type, model_config, force, patterns
        )

    def _download_model_with_patterns(
        self,
        model_type: str,
        model_config: ModelConfig,
        force: bool,
        patterns: list[str],
    ) -> DownloadResult:
        """Download a model with specific file patterns.

        This is the core download method that handles the actual downloading
        with a given set of file patterns.

        Args:
            model_type: Identifier for the model (e.g., "clip:default").
            model_config: Model configuration from LumenConfig.
            force: Whether to force re-download even if already cached.
            patterns: File patterns to include in the download.

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
                    for file_rel in [
                        dataset_files.labels,
                        dataset_files.embeddings,
                    ]:
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

        Args:
            model_path: Local path where model files are located.

        Returns:
            Parsed ModelInfo object.

        Raises:
            ModelInfoError: If model_info.json is missing or invalid.
        """
        info_path = model_path / "model_info.json"
        if not info_path.exists():
            raise ModelInfoError(f"Missing model_info.json in {model_path}")

        try:
            return load_and_validate_model_info(info_path)
        except Exception as e:
            raise ModelInfoError(f"Failed to load model_info.json: {e}")

    def _validate_model_config(
        self, model_info: ModelInfo, model_config: ModelConfig
    ) -> None:
        """Validate model configuration against model_info.json.

        Checks that the requested runtime and dataset are supported
        by the model metadata.

        Args:
            model_info: Parsed model information.
            model_config: Model configuration to validate.

        Raises:
            ValidationError: If configuration is not supported by the model.
        """
        # Validate runtime support
        runtime_str = (
            model_config.runtime.value
            if hasattr(model_config.runtime, "value")
            else str(model_config.runtime)
        )
        if runtime_str not in model_info.runtimes:
            raise ValidationError(
                f"Runtime {runtime_str} not supported by model {model_config.model}. "
                f"Supported runtimes: {', '.join(model_info.runtimes)}"
            )

        # Validate dataset if specified
        if model_config.dataset:
            if (
                not model_info.datasets
                or model_config.dataset not in model_info.datasets
            ):
                raise ValidationError(
                    f"Dataset {model_config.dataset} not supported by model {model_config.model}. "
                    f"Available datasets: {', '.join(model_info.datasets.keys() if model_info.datasets else [])}"
                )

        # Validate RKNN device if RKNN runtime
        if model_config.runtime == Runtime.rknn and not model_config.rknn_device:
            raise ValidationError(
                f"RKNN runtime requires rknn_device specification for model {model_config.model}"
            )

    def _validate_files(
        self,
        model_path: Path,
        model_info: ModelInfo,
        model_config: ModelConfig,
    ) -> list[str]:
        """Validate that all required files are present after download.

        Checks model files, tokenizer files, and dataset files against
        the model_info.json metadata based on the runtime configuration.

        Args:
            model_path: Local path where model files are located.
            model_info: Parsed model information.
            model_config: Model configuration to validate.

        Returns:
            List of missing file paths. Empty list if all files present.

        Raises:
            ValidationError: If critical files are missing.
        """
        missing: list[str] = []
        runtime_str = (
            model_config.runtime.value
            if hasattr(model_config.runtime, "value")
            else str(model_config.runtime)
        )

        # Check runtime files
        runtime_config = model_info.runtimes.get(runtime_str)
        if runtime_config and runtime_config.files:
            if isinstance(runtime_config.files, list):
                runtime_files = runtime_config.files

                # For ONNX runtime, filter by precision if specified
                if runtime_str == "onnx" and model_config.precision:
                    runtime_files = [
                        f
                        for f in runtime_files
                        if not f.endswith(
                            (".fp16.onnx", ".fp32.onnx", ".int8.onnx", ".q4fp16.onnx")
                        )
                        or f.endswith(f".{model_config.precision}.onnx")
                    ]
            elif isinstance(runtime_config.files, dict) and model_config.rknn_device:
                # RKNN files are organized by device
                runtime_files = runtime_config.files.get(model_config.rknn_device, [])
            else:
                runtime_files = []

            for file_name in runtime_files:
                if not (model_path / file_name).exists():
                    missing.append(file_name)

        # Check dataset files if specified
        if model_config.dataset and model_info.datasets:
            dataset_files = model_info.datasets.get(model_config.dataset)
            if dataset_files:
                for file_rel in [dataset_files.labels, dataset_files.embeddings]:
                    dataset_path = model_path / file_rel
                    if not dataset_path.exists():
                        missing.append(file_rel)

        return missing
