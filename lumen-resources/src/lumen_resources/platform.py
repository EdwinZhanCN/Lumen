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
  1) First pass downloads runtime-specific files plus JSON metadata (to obtain model_info.json).
  2) Second pass optionally fetches dataset files using the exact relative path from
     model_info.json's "datasets" mapping via allow_patterns=[relative_path].

@requires: Platform-specific SDK installed (huggingface_hub or modelscope)
@returns: Downloaded model files in local cache with filtering applied
@errors: DownloadError, PlatformUnavailableError
"""

import shutil
from enum import Enum
from pathlib import Path
from typing import Optional

from .exceptions import DownloadError, PlatformUnavailableError


class PlatformType(str, Enum):
    """Supported platforms."""

    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"


class Platform:
    """
    Unified platform adapter for HuggingFace and ModelScope.

    Contract:
    @requires: Platform SDK properly installed and configured
    @returns: Path to downloaded model directory
    @errors: DownloadError, PlatformUnavailableError
    """

    def __init__(self, platform_type: PlatformType, owner: str):
        """
        Initialize platform adapter.

        Args:
            platform_type: Type of platform (HUGGINGFACE or MODELSCOPE)
            owner: Organization/owner name on the platform

        Raises:
            PlatformUnavailableError: If required SDK is not installed
        """
        self.platform_type = platform_type
        self.owner = owner
        self._check_availability()

    def _check_availability(self) -> None:
        """
        Check if the required platform SDK is available.

        Raises:
            PlatformUnavailableError: If SDK is not installed
        """
        if self.platform_type == PlatformType.HUGGINGFACE:
            try:
                import huggingface_hub

                self._hf_hub = huggingface_hub
            except ImportError:
                raise PlatformUnavailableError(
                    "HuggingFace Hub SDK not available. "
                    "Install with: pip install huggingface_hub"
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
        """
        Download model files from the platform with efficient filtering.

        Both HuggingFace and ModelScope now support pattern-based filtering
        during download, eliminating the need to download unwanted files.

        Args:
            repo_name: Repository name (without owner prefix)
            cache_dir: Local cache directory
            allow_patterns: List of glob patterns for files to download
                          (e.g., ['*.json', '*.bin', 'tokenizer/*'])
            force: Force re-download even if cached
                  - HuggingFace: Uses native force_download parameter
                  - ModelScope: Clears cache directory before download

        Returns:
            Path to the downloaded model directory

        Raises:
            DownloadError: If download fails
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
        """
        Download from HuggingFace Hub.

        Args:
            repo_id: Full repository ID (owner/repo)
            cache_dir: Local cache directory
            allow_patterns: File patterns to download
            force: Force re-download

        Returns:
            Path to downloaded model directory
        """
        snapshot_path = self._hf_hub.snapshot_download(
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
        """
        Download from ModelScope Hub.

        Args:
            repo_id: Full repository ID (owner/repo)
            cache_dir: Local cache directory
            allow_patterns: File patterns to download
            force: Force re-download by clearing cache first

        Returns:
            Path to downloaded model directory
        """

        # Handle force download by clearing ModelScope cache
        if force:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        # ModelScope supports allow_patterns parameter (HuggingFace compatible)
        snapshot_path = self._ms_snapshot_download(
            model_id=repo_id,
            local_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            local_files_only=False,
        )

        return cache_dir

    def cleanup_model(self, repo_name: str, cache_dir: Path) -> None:
        """
        Remove a model directory from cache.

        Used for rollback when download/validation fails.

        Args:
            repo_name: Repository name
            cache_dir: Cache directory
        """
        target_dir = cache_dir / "models" / repo_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
