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
                prefer_fp16 = False
                if service_config.backend_settings:
                    prefer_fp16 = service_config.backend_settings.prefer_fp16 or False

                if self.verbose:
                    print(f"\n📦 Processing {model_type.upper()}")
                    print(f"    Model: {model_config.model}")
                    print(f"    Runtime: {model_config.runtime.value}")

                result = self._download_model(
                    model_type, model_config, force, prefer_fp16
                )
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

    def _get_runtime_patterns(self, runtime: Runtime, pref_fp16: bool) -> list[str]:
        """Get file patterns to download based on runtime.

        Determines which file patterns to include in downloads based on the
        model runtime. Always includes model_info.json and config files.

        Args:
            runtime: The model runtime (torch, onnx, rknn).
            pref_fp16: Whether to prefer FP16 models over FP32.

        Returns:
            List of file glob patterns for the download.

        Example:
            >>> patterns = downloader._get_runtime_patterns(Runtime.torch, False)
            >>> print("model_info.json" in patterns)
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
            # Only add one precision based on preference to save space
            if pref_fp16:
                patterns.extend(["*.fp16.onnx"])
            else:
                patterns.extend(["*.fp32.onnx"])
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
        self, model_type: str, model_config: ModelConfig, force: bool, pref_fp16: bool
    ) -> DownloadResult:
        """Download a single model with its runtime files using fallback strategy.

        First attempts to download with the preferred precision (FP16/FP32),
        and if that fails due to file mismatch, falls back to the other precision.
        This ensures model availability while minimizing storage usage.

        Args:
            model_type: Identifier for the model (e.g., "clip:default").
            model_config: Model configuration from LumenConfig.
            force: Whether to force re-download even if already cached.
            pref_fp16: Whether to prefer FP16 models over FP32.

        Returns:
            DownloadResult with success status, file paths, and error details.

        Raises:
            DownloadError: If platform download fails for both precisions.
            ModelInfoError: If model_info.json is missing or invalid.
            ValidationError: If model configuration is not supported.
        """
        # First attempt with preferred precision
        preferred_patterns = self._get_runtime_patterns(model_config.runtime, pref_fp16)
        fallback_patterns = self._get_runtime_patterns(
            model_config.runtime, not pref_fp16
        )

        # Try preferred precision first
        try:
            return self._download_model_with_patterns(
                model_type, model_config, force, preferred_patterns, pref_fp16
            )
        except DownloadError as e:
            # Check if this is a "no matching files" error that warrants fallback
            if (
                self._should_fallback_download(str(e))
                and model_config.runtime == Runtime.onnx
            ):
                precision = "FP16" if pref_fp16 else "FP32"
                fallback_precision = "FP32" if pref_fp16 else "FP16"
                if self.verbose:
                    print(
                        f"   ⚠️ {precision} model not found, trying {fallback_precision}"
                    )

                try:
                    return self._download_model_with_patterns(
                        model_type,
                        model_config,
                        force,
                        fallback_patterns,
                        not pref_fp16,
                    )
                except DownloadError as fallback_error:
                    # If fallback also fails, report both errors
                    return DownloadResult(
                        model_type=model_type,
                        model_name=model_config.model,
                        runtime=model_config.runtime.value
                        if hasattr(model_config.runtime, "value")
                        else str(model_config.runtime),
                        success=False,
                        error=f"Failed to download with {precision}: {e}. "
                        f"Fallback with {fallback_precision} also failed: {fallback_error}",
                    )

            # Non-fallbackable error or non-ONNX runtime, just report original error
            raise

    def _download_model_with_patterns(
        self,
        model_type: str,
        model_config: ModelConfig,
        force: bool,
        patterns: list[str],
        is_fp16: bool | None = None,
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
            missing = self._validate_files(
                model_path, model_info, model_config, is_fp16
            )
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

    def _should_fallback_download(self, error_message: str) -> bool:
        """Determine if a download error should trigger fallback to another precision.

        Args:
            error_message: The error message from the download attempt.

        Returns:
            True if the error suggests we should try the other precision, False otherwise.
        """
        # Common patterns that indicate file matching issues
        fallback_indicators = [
            "No matching files found",
            "No files matched the pattern",
            "Cannot find any files matching",
            "File pattern matched no files",
            "No such file or directory",  # Sometimes used for remote files
        ]

        error_lower = error_message.lower()
        return any(
            indicator.lower() in error_lower for indicator in fallback_indicators
        )

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
        is_fp16: bool | None = None,
    ) -> list[str]:
        """Validate that all required files are present after download.

        Checks model files, tokenizer files, and dataset files against
        the model_info.json metadata based on the actual precision
        downloaded.

        Args:
            model_path: Local path where model files are located.
            model_info: Parsed model information.
            model_config: Model configuration to validate.
            is_fp16: Whether FP16 files were downloaded (None for non-ONNX
                runtimes).

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
                if runtime_str == "onnx" and is_fp16 is not None:
                    precision_str = "fp16" if is_fp16 else "fp32"
                    runtime_files = [
                        f
                        for f in runtime_files
                        if not f.endswith((".fp16.onnx", ".fp32.onnx"))
                        or f.endswith(f".{precision_str}.onnx")
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
