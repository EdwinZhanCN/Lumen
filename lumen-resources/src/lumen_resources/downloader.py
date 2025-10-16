"""
Resource Downloader Manager

@requires: Valid ResourceConfig and platform adapter
@returns: Download results with validation
@errors: DownloadError, ValidationError
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import ResourceConfig, ModelConfig, RuntimeType
from .exceptions import DownloadError, ValidationError, ModelInfoError
from .platform import Platform


@dataclass
class DownloadResult:
    """Result of a single model download operation."""

    model_type: str
    model_config: ModelConfig
    success: bool = False
    model_path: Path | None = None
    missing_files: list[str] | None = None
    error: str | None = None

    def __post_init__(self):
        if self.missing_files is None:
            self.missing_files = []


class ModelInfo:
    """Parsed model_info.json data."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data.get("name", "")
        self.version = data.get("version", "")
        self.runtimes = data.get("runtimes", {})
        self.datasets = data.get("datasets", {})

    def validate_runtime(
        self, runtime: RuntimeType, rknn_device: Optional[str] = None
    ) -> bool:
        """
        Check if the requested runtime is supported.

        Args:
            runtime: Requested runtime type
            rknn_device: Required device for RKNN runtime

        Returns:
            True if runtime is supported and available
        """
        runtime_info = self.runtimes.get(runtime.value, {})

        if not runtime_info.get("available", False):
            return False

        # Special check for RKNN device support
        if runtime == RuntimeType.RKNN and rknn_device:
            supported_devices = runtime_info.get("devices", [])
            if rknn_device not in supported_devices:
                return False

        return True

    def get_required_files(
        self, runtime: RuntimeType, rknn_device: Optional[str] = None
    ) -> List[str]:
        """
        Get list of required files for a runtime.

        Args:
            runtime: Runtime type
            rknn_device: Device for RKNN runtime

        Returns:
            List of file paths that should exist
        """
        runtime_info = self.runtimes.get(runtime.value, {})

        if runtime == RuntimeType.RKNN and rknn_device:
            # RKNN files are organized by device
            files_dict = runtime_info.get("files", {})
            return files_dict.get(rknn_device, [])
        else:
            return runtime_info.get("files", [])

    def get_dataset_file(self, dataset_name: str) -> Optional[str]:
        """
        Get dataset filename if it exists.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset filename or None
        """
        return self.datasets.get(dataset_name)


class Downloader:
    """
    Main resource downloader.

    Contract:
    @requires: Valid ResourceConfig
    @returns: Dictionary of DownloadResult per model type
    @errors: DownloadError for critical failures
    """

    def __init__(self, config: ResourceConfig, verbose: bool = True):
        """
        Initialize downloader with configuration.

        Args:
            config: Resource configuration
            verbose: Whether to print progress messages
        """
        self.config = config
        self.verbose = verbose
        self.platform = Platform(config.platform_type, config.platform_owner)

        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.config.cache_dir / "models").mkdir(parents=True, exist_ok=True)

    def download_all(self, force: bool = False) -> Dict[str, DownloadResult]:
        """
        Download all enabled models.

        Args:
            force: Force re-download even if cached

        Returns:
            Dictionary mapping model type to DownloadResult
        """
        results = {}

        for model_type, model_config in self.config.models.items():
            if self.verbose:
                print(f"\n📦 Processing {model_type.upper()}")
                print(f"   Model: {model_config.model}")
                print(f"   Runtime: {model_config.runtime.value}")

            result = self._download_model(model_type, model_config, force)
            results[model_type] = result

            if result.success:
                if self.verbose:
                    print(f"   ✅ Download successful: {result.model_path}")
                    if result.missing_files:
                        print(f"   ⚠️  Missing files: {', '.join(result.missing_files)}")
                    else:
                        print(f"   ✅ All files verified")
            else:
                if self.verbose:
                    print(f"   ❌ Download failed: {result.error}")

        return results

    def _download_model(
        self, model_type: str, model_config: ModelConfig, force: bool
    ) -> DownloadResult:
        """
        Download and validate a single model.

        Args:
            model_type: Type of model (clip, face, ocr, bioclip)
            model_config: Model configuration
            force: Force re-download

        Returns:
            DownloadResult with success status and details
        """
        result = DownloadResult(model_type=model_type, model_config=model_config)

        try:
            # Phase 1: Prepare and download runtime + JSON only
            patterns = model_config.get_runtime_patterns()

            model_path = self.platform.download_model(
                repo_name=model_config.model,
                cache_dir=self.config.cache_dir,
                allow_patterns=patterns,
                force=force,
            )

            result.model_path = model_path

            # Load and validate model_info.json
            model_info = self._load_model_info(model_path)

            # Logical validation
            self._validate_model_config(model_info, model_config)

            # Phase 2: If dataset specified, download by relative path from model_info.json
            if model_config.dataset:
                dataset_rel = model_info.get_dataset_file(model_config.dataset)
                if dataset_rel:
                    dataset_path = model_path / dataset_rel
                    if not dataset_path.exists():
                        # Download only the dataset file by its relative path
                        self.platform.download_model(
                            repo_name=model_config.model,
                            cache_dir=self.config.cache_dir,
                            allow_patterns=[dataset_rel],
                            force=False,
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
                self.platform.cleanup_model(model_config.model, self.config.cache_dir)

        return result

    def _load_model_info(self, model_path: Path) -> ModelInfo:
        """
        Load and parse model_info.json.

        Args:
            model_path: Path to model directory

        Returns:
            Parsed ModelInfo object

        Raises:
            ModelInfoError: If file is missing or invalid
        """
        info_file = model_path / "model_info.json"

        if not info_file.exists():
            raise ModelInfoError(
                f"model_info.json not found in {model_path}. "
                "Repository must contain model metadata."
            )

        try:
            with open(info_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ModelInfo(data)
        except json.JSONDecodeError as e:
            raise ModelInfoError(f"Invalid model_info.json format: {e}")
        except Exception as e:
            raise ModelInfoError(f"Failed to load model_info.json: {e}")

    def _validate_model_config(
        self, model_info: ModelInfo, model_config: ModelConfig
    ) -> None:
        """
        Validate that model supports the requested configuration.

        Args:
            model_info: Parsed model metadata
            model_config: Requested configuration

        Raises:
            ValidationError: If configuration is not supported
        """
        runtime = model_config.runtime
        rknn_device = model_config.rknn_device

        if not model_info.validate_runtime(runtime, rknn_device):
            if runtime == RuntimeType.RKNN and rknn_device:
                raise ValidationError(
                    f"Model {model_config.model} does not support "
                    f"runtime '{runtime.value}' with device '{rknn_device}'"
                )
            else:
                raise ValidationError(
                    f"Model {model_config.model} does not support "
                    f"runtime '{runtime.value}'"
                )

        # Validate dataset if specified
        if model_config.dataset:
            dataset_file = model_info.get_dataset_file(model_config.dataset)
            if not dataset_file:
                raise ValidationError(
                    f"Dataset '{model_config.dataset}' not available for "
                    f"model {model_config.model}"
                )

    def _validate_files(
        self, model_path: Path, model_info: ModelInfo, model_config: ModelConfig
    ) -> List[str]:
        """
        Validate that all required files exist.

        Args:
            model_path: Path to model directory
            model_info: Parsed model metadata
            model_config: Model configuration

        Returns:
            List of missing file paths (empty if all present)
        """
        missing = []

        # Get required files from model_info
        required_files = model_info.get_required_files(
            model_config.runtime, model_config.rknn_device
        )

        for file_path in required_files:
            full_path = model_path / file_path
            if not full_path.exists():
                missing.append(file_path)

        # Check dataset file if specified
        if model_config.dataset:
            dataset_file = model_info.get_dataset_file(model_config.dataset)
            if dataset_file:
                dataset_path = model_path / dataset_file
                if not dataset_path.exists():
                    missing.append(dataset_file)

        return missing
