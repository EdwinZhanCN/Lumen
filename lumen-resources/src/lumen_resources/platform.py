"""
Platform Adapter for Model Repository Access

This module provides a unified interface for downloading models from HuggingFace Hub
and ModelScope Hub with efficient file filtering capabilities.

Features:
- Unified API for both HuggingFace and ModelScope platforms
- File pattern filtering during download (not post-download)
- Automatic cache management and file organization
- Force download and cache invalidation support
- Supports two-phase dataset downloads used by the Downloader:
  1) First pass downloads runtime-specific files plus JSON metadata (model_info.json).
  2) Second pass optionally fetches dataset files using the exact relative path from
     model_info.json's "datasets" mapping via allow_patterns=[relative_path].

@requires: Platform-specific SDK installed (huggingface_hub or modelscope)
@returns: Downloaded model files in local cache with filtering applied
@errors: DownloadError, PlatformUnavailableError
"""

import shutil
from enum import Enum
from pathlib import Path
from types import ModuleType

from .exceptions import DownloadError, PlatformUnavailableError


class PlatformType(str, Enum):
    """Supported model repository platforms.

    Defines the platforms that can be used for downloading models.
    Each platform has its own SDK and API requirements.

    Attributes:
        HUGGINGFACE: Hugging Face Hub platform.
        MODELSCOPE: ModelScope Hub platform.

    Example:
        >>> platform_type = PlatformType.HUGGINGFACE
        >>> print(platform_type.value)
        'huggingface'
    """

    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"


class Platform:
    """Unified platform adapter for HuggingFace and ModelScope.

    Provides a consistent interface for downloading models from different
    repositories while handling platform-specific requirements and optimizations.
    Supports efficient file filtering during download to minimize bandwidth usage.

    Attributes:
        platform_type: The type of platform (HUGGINGFACE or MODELSCOPE).
        owner: Organization/owner name for model repositories.

    Example:
        >>> platform = Platform(PlatformType.HUGGINGFACE, "openai")
        >>> model_path = platform.download_model(
        ...     repo_name="clip-vit-base-patch32",
        ...     cache_dir=Path("/cache"),
        ...     allow_patterns=["*.json", "*.pt"]
        ... )
    """

    def __init__(self, platform_type: PlatformType, owner: str):
        """Initialize platform adapter.

        Args:
            platform_type: Type of platform (HUGGINGFACE or MODELSCOPE).
            owner: Organization/owner name on the platform.

        Raises:
            PlatformUnavailableError: If required SDK is not installed.
        """
        self.platform_type: PlatformType = platform_type
        self.owner: str = owner
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if the required platform SDK is available.

        Validates that the appropriate SDK (huggingface_hub or modelscope)
        is installed and imports the necessary functions for the platform.

        Raises:
            PlatformUnavailableError: If the required SDK is not installed.

        Example:
            >>> platform = Platform(PlatformType.HUGGINGFACE, "owner")
            >>> # If huggingface_hub is not installed, raises PlatformUnavailableError
        """
        if self.platform_type == PlatformType.HUGGINGFACE:
            try:
                import huggingface_hub

                self._hf_hub: ModuleType = huggingface_hub
            except ImportError:
                raise PlatformUnavailableError(
                    "HuggingFace Hub SDK not available. "
                    + "Install with: pip install huggingface_hub"
                )
        elif self.platform_type == PlatformType.MODELSCOPE:
            try:
                from modelscope.hub.snapshot_download import snapshot_download

                self._ms_snapshot_download = snapshot_download
            except ImportError:
                raise PlatformUnavailableError(
                    "ModelScope SDK not available. Install with: pip install modelscope"
                )

    def download_model(
        self,
        repo_name: str,
        cache_dir: Path,
        allow_patterns: list[str],
        force: bool = False,
    ) -> Path:
        """Download model files from the platform with efficient filtering.

        Downloads model files using pattern-based filtering to minimize bandwidth
        usage. Supports both HuggingFace and ModelScope platforms with their
        respective SDKs while providing a unified interface.

        Args:
            repo_name: Repository name (without owner prefix).
            cache_dir: Local cache directory for storing downloaded models.
            allow_patterns: List of glob patterns for files to download.
                Examples: ['*.json', '*.bin', 'tokenizer/*', 'model_info.json'].
            force: Force re-download even if cached.
                - HuggingFace: Uses native force_download parameter.
                - ModelScope: Clears cache directory before download.

        Returns:
            Path to the downloaded model directory.

        Raises:
            DownloadError: If download fails for any reason.

        Example:
            >>> platform = Platform(PlatformType.HUGGINGFACE, "openai")
            >>> model_path = platform.download_model(
            ...     repo_name="clip-vit-base-patch32",
            ...     cache_dir=Path("/cache"),
            ...     allow_patterns=["*.json", "*.pt"],
            ...     force=True
            ... )
            >>> print(model_path.name)
            'clip-vit-base-patch32'
        """
        repo_id = f"{self.owner}/{repo_name}"
        target_dir = cache_dir / "models" / repo_name

        try:
            if self.platform_type == PlatformType.HUGGINGFACE:
                return self._download_from_huggingface(
                    repo_id, target_dir, allow_patterns, force
                )
            elif self.platform_type == PlatformType.MODELSCOPE:
                return self._download_from_modelscope(
                    repo_id, target_dir, allow_patterns, force
                )
            else:
                raise DownloadError(f"Unsupported platform type: {self.platform_type}")
        except Exception as e:
            raise DownloadError(f"Failed to download {repo_id}: {e}") from e

    def _download_from_huggingface(
        self,
        repo_id: str,
        cache_dir: Path,
        allow_patterns: list[str],
        force: bool,
    ) -> Path:
        """Download from HuggingFace Hub.

        Uses the huggingface_hub library to download model files with
        pattern-based filtering and optional force re-download.

        Args:
            repo_id: Full repository ID (owner/repo).
            cache_dir: Local cache directory for storing files.
            allow_patterns: File patterns to download.
            force: Whether to force re-download ignoring cache.

        Returns:
            Path to the downloaded model directory.

        Example:
            >>> platform = Platform(PlatformType.HUGGINGFACE, "owner")
            >>> path = platform._download_from_huggingface(
            ...     "owner/repo", Path("/cache"), ["*.json"], False
            ... )
        """
        _ = self._hf_hub.snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_dir=cache_dir,
            local_files_only=False,
            force_download=force,
        )

        return cache_dir

    def _download_from_modelscope(
        self,
        repo_id: str,
        cache_dir: Path,
        allow_patterns: list[str],
        force: bool,
    ) -> Path:
        """Download from ModelScope Hub.

        Uses the ModelScope SDK to download model files with pattern-based filtering.
        Implements force download by clearing the cache directory before download.

        Args:
            repo_id: Full repository ID (owner/repo).
            cache_dir: Local cache directory for storing files.
            allow_patterns: File patterns to download.
            force: Force re-download by clearing cache first.

        Returns:
            Path to the downloaded model directory.

        Example:
            >>> platform = Platform(PlatformType.MODELSCOPE, "owner")
            >>> path = platform._download_from_modelscope(
            ...     "owner/repo", Path("/cache"), ["*.json"], False
            ... )
        """

        # Handle force download by clearing ModelScope cache
        if force:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        # ModelScope supports allow_patterns parameter (HuggingFace compatible)
        _ = self._ms_snapshot_download(
            model_id=repo_id,
            local_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            local_files_only=False,
        )

        return cache_dir

    def cleanup_model(self, repo_name: str, cache_dir: Path) -> None:
        """Remove a model directory from cache.

        Used for cleanup when download/validation fails or for manual cache management.
        Removes the entire model directory including all downloaded files.

        Args:
            repo_name: Repository name (without owner prefix).
            cache_dir: Base cache directory containing models.

        Example:
            >>> platform = Platform(PlatformType.HUGGINGFACE, "owner")
            >>> platform.cleanup_model("model-name", Path("/cache"))
            >>> # Model directory removed if it existed
        """
        target_dir = cache_dir / "models" / repo_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
