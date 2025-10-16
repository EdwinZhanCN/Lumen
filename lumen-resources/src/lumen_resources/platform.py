"""
Platform Adapter for Model Repository Access

This module provides a unified interface for downloading models from HuggingFace Hub
and ModelScope Hub with efficient file filtering capabilities.

Features:
- Unified API for both HuggingFace and ModelScope platforms
- File pattern filtering during download (not post-download)
- Automatic cache management and file organization
- Force download and cache invalidation support

@requires: Platform-specific SDK installed (huggingface_hub or modelscope)
@returns: Downloaded model files in local cache with filtering applied
@errors: DownloadError, PlatformUnavailableError
"""

import shutil
from pathlib import Path
from typing import Optional

from .config import PlatformType
from .exceptions import DownloadError, PlatformUnavailableError


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

        try:
            if self.platform_type == PlatformType.HUGGINGFACE:
                return self._download_from_huggingface(
                    repo_id, cache_dir, allow_patterns, force
                )
            elif self.platform_type == PlatformType.MODELSCOPE:
                return self._download_from_modelscope(
                    repo_id, cache_dir, allow_patterns, force
                )
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
        # HuggingFace uses its own cache structure, we'll organize it later
        snapshot_path = self._hf_hub.snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_dir / ".cache" / "huggingface"),
            local_files_only=False,
            force_download=force,
        )

        # Organize into our standard structure
        repo_name = repo_id.split("/")[-1]
        target_dir = cache_dir / "models" / repo_name

        # Create symlinks or copy files to maintain structure
        self._organize_files(Path(snapshot_path), target_dir)

        return target_dir

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
        ms_cache_dir = cache_dir / ".cache" / "modelscope"

        # Handle force download by clearing ModelScope cache
        if force:
            model_cache_path = ms_cache_dir / repo_id
            if model_cache_path.exists():
                shutil.rmtree(model_cache_path)

        # ModelScope supports allow_patterns parameter (HuggingFace compatible)
        snapshot_path = self._ms_snapshot_download(
            model_id=repo_id,
            cache_dir=str(ms_cache_dir),
            allow_patterns=allow_patterns,
            local_files_only=False,
        )

        # Organize into our standard structure
        repo_name = repo_id.split("/")[-1]
        target_dir = cache_dir / "models" / repo_name

        # Organize files into our standard structure (filtering already done by ModelScope)
        self._organize_files(Path(snapshot_path), target_dir)

        return target_dir

    def _organize_files(
        self,
        source_dir: Path,
        target_dir: Path,
        patterns: Optional[list[str]] = None,
    ) -> None:
        """
        Organize downloaded files into standard structure.

        Args:
            source_dir: Source directory from platform SDK
            target_dir: Target directory in our standard structure
            patterns: Optional list of patterns to filter files (only used for HuggingFace post-processing)
        """
        import fnmatch

        target_dir.mkdir(parents=True, exist_ok=True)

        for source_file in source_dir.rglob("*"):
            if source_file.is_file():
                # Get relative path from source
                rel_path = source_file.relative_to(source_dir)

                # If patterns specified, check if file matches (primarily for HuggingFace)
                # ModelScope filtering is done during download via allow_patterns
                if patterns:
                    matches = any(
                        fnmatch.fnmatch(str(rel_path), pattern) for pattern in patterns
                    )
                    if not matches:
                        continue

                # Create target path
                target_file = target_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)

                # Copy file if it doesn't exist or is different
                if (
                    not target_file.exists()
                    or source_file.stat().st_mtime > target_file.stat().st_mtime
                ):
                    shutil.copy2(source_file, target_file)

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
