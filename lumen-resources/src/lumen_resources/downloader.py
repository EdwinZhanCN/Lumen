"""
Resource Downloader Manager

@requires: Valid LumenServicesConfiguration and platform adapter
@returns: Download results with validation
@errors: DownloadError, ValidationError
"""

from dataclasses import dataclass
from pathlib import Path

from .exceptions import DownloadError, ModelInfoError, ValidationError
from .lumen_config import LumenServicesConfiguration, ModelConfig, Region, Runtime
from .model_info import ModelInfo
from .model_info_validator import load_and_validate_model_info
from .platform import Platform, PlatformType


@dataclass
class DownloadResult:
    """Result of a single model download operation."""

    model_type: str
    model_name: str
    runtime: str
    success: bool = False
    model_path: Path | None = None
    missing_files: list[str] | None = None
    error: str | None = None

    def __post_init__(self):
        if self.missing_files is None:
            self.missing_files = []


class Downloader:
    """
    Main resource downloader.

    Contract:
    @requires: Valid LumenServicesConfiguration
    @returns: Dictionary of DownloadResult per model type
    @errors: DownloadError for critical failures
    """

    def __init__(self, config: LumenServicesConfiguration, verbose: bool = True):
        """
        Initialize downloader with configuration.

        Args:
            config: Lumen services configuration
            verbose: Whether to print progress messages
        """
        self.config: LumenServicesConfiguration = config
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
        """
        Download all enabled models from all enabled services.

        Args:
            force: Force re-download even if cached

        Returns:
            Dictionary mapping "service:alias" to DownloadResult
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
        """
        Get file patterns to download based on runtime.

        Args:
            runtime: Runtime type enum

        Returns:
            List of glob patterns for files to download
        """
        patterns = ["*.json"]  # Always include JSON metadata files

        if runtime == Runtime.torch:
            patterns.extend(
                [
                    "*.bin",
                    "*.pt",
                    "*.pth",
                    "*.safetensors",
                    "pytorch_model*.bin",
                    "model.safetensors",
                ]
            )
        elif runtime == Runtime.onnx:
            patterns.extend(["*.onnx", "*.ort"])
        elif runtime == Runtime.rknn:
            patterns.extend(["*.rknn"])

        return patterns

    def _download_model(
        self, model_type: str, model_config: ModelConfig, force: bool
    ) -> DownloadResult:
        """
        Download a single model with its runtime files.

        Args:
            model_type: Identifier for the model (e.g., "clip:default")
            model_config: Model configuration from LumenServicesConfiguration
            force: Force re-download

        Returns:
            DownloadResult with success status and details
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

            # If dataset specified, download by relative path from model_info.json
            if model_config.dataset and model_info.datasets:
                dataset_rel = model_info.datasets.get(model_config.dataset)
                if dataset_rel:
                    dataset_path = model_path / dataset_rel
                    if not dataset_path.exists():
                        # Download only the dataset file by its relative path
                        try:
                            _ = self.platform.download_model(
                                repo_name=model_config.model,
                                cache_dir=cache_dir,
                                allow_patterns=[dataset_rel],
                                force=False,
                            )
                        except DownloadError as e:
                            raise DownloadError(f"Failed to download dataset: {e}")

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
        """
        Load and parse model_info.json using validator.

        Args:
            model_path: Path to model directory

        Returns:
            Validated ModelInfo object

        Raises:
            ModelInfoError: If file is missing or invalid
        """
        info_file = model_path / "model_info.json"

        if not info_file.exists():
            msg = (
                f"model_info.json not found in {model_path}. "
                "Repository must contain model metadata."
            )
            raise ModelInfoError(msg)

        try:
            return load_and_validate_model_info(info_file, strict=True)
        except Exception as e:
            raise ModelInfoError(f"Failed to load/validate model_info.json: {e}")

    def _validate_model_config(
        self, model_info: ModelInfo, model_config: ModelConfig
    ) -> None:
        """
        Validate that model supports the requested configuration.

        Args:
            model_info: Validated model metadata
            model_config: Requested configuration

        Raises:
            ValidationError: If configuration is not supported
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
        """
        Validate that required model files exist.

        Args:
            model_path: Path to model directory
            model_info: Validated model metadata
            model_config: Model configuration

        Returns:
            List of missing file paths (empty if all present)
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

        # Check dataset file if specified
        if model_config.dataset and model_info.datasets:
            dataset_file = model_info.datasets.get(model_config.dataset)
            if dataset_file:
                dataset_path = model_path / dataset_file
                if not dataset_path.exists():
                    missing.append(dataset_file)

        return missing
