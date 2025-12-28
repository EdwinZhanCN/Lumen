"""Package resolver for Lumen installations.

This module provides utilities for resolving package URLs from GitHub Releases,
selecting appropriate mirrors based on region, and building pip installation commands.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Callable

from lumen_resources.lumen_config import LumenConfig, Region

from ..core.config import DeviceConfig

logger = logging.getLogger(__name__)


class MirrorSelector:
    """Selects mirror URLs based on region."""

    GITHUB_MIRROR_CN = "https://gh-proxy.org/https://github.com"
    PYPI_MIRROR_CN = "https://mirrors.aliyun.com/pypi/simple/"

    def get_github_urls(self, base_url: str, region: Region) -> list[str]:
        """Get GitHub URLs with mirror fallback.

        Args:
            base_url: Original GitHub URL (e.g., https://github.com/...)
            region: Region.cn or Region.other

        Returns:
            List of URLs to try (mirror first if cn, then original)
        """
        urls = []
        if region == Region.cn:
            # Apply ghproxy mirror
            mirror_url = base_url.replace("https://github.com", self.GITHUB_MIRROR_CN)
            urls.append(mirror_url)
        urls.append(base_url)
        return urls

    def get_pypi_indexes(self, region: Region) -> list[str]:
        """Get PyPI index URLs with mirror fallback.

        Args:
            region: Region.cn or Region.other

        Returns:
            List of index URLs (mirror first if cn, then original)
        """
        indexes = []
        if region == Region.cn:
            indexes.append(self.PYPI_MIRROR_CN)
        # Always include PyPI official as fallback
        indexes.append("https://pypi.org/simple/")
        return indexes


class GitHubPackageResolver:
    """Resolves package download URLs from GitHub Releases."""

    REPO_OWNER = "EdwinZhanCN"
    REPO_NAME = "Lumen"
    API_BASE = "https://api.github.com"

    def __init__(self, region: Region):
        """Initialize resolver.

        Args:
            region: Region for mirror selection
        """
        self.region = region
        self.mirror_selector = MirrorSelector()

    def get_latest_release(self) -> str:
        """Get latest release tag from GitHub API.

        Returns:
            Release tag (e.g., "v0.1.0")

        Raises:
            Exception: If API call fails
        """
        url = (
            f"{self.API_BASE}/repos/{self.REPO_OWNER}/{self.REPO_NAME}/releases/latest"
        )

        logger.debug(f"[GitHubPackageResolver] Fetching latest release from {url}")

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                import json

                data = json.loads(response.read().decode())
                tag = data.get("tag_name")
                if not tag:
                    raise Exception("No tag_name found in release")
                logger.info(f"[GitHubPackageResolver] Latest release: {tag}")
                return tag
        except Exception as e:
            logger.error(f"[GitHubPackageResolver] Failed to fetch release: {e}")
            raise Exception(f"Failed to fetch latest release: {e}")

    def resolve_package_url(self, package_name: str) -> tuple[str, str]:
        """Resolve package wheel download URL.

        Args:
            package_name: Package name (e.g., "lumen_ocr")

        Returns:
            Tuple of (download_url, version)

        Raises:
            Exception: If wheel file not found
        """
        # Get latest release tag
        tag = self.get_latest_release()

        # We need to get the actual wheel filename from the release assets
        api_url = f"{self.API_BASE}/repos/{self.REPO_OWNER}/{self.REPO_NAME}/releases/tags/{tag}"

        logger.debug(f"[GitHubPackageResolver] Fetching release assets from {api_url}")

        try:
            with urllib.request.urlopen(api_url, timeout=30) as response:
                import json

                data = json.loads(response.read().decode())
                assets = data.get("assets", [])

                # Find matching wheel
                for asset in assets:
                    name = asset.get("name", "")
                    if name.startswith(package_name) and name.endswith(
                        "-py3-none-any.whl"
                    ):
                        download_url = asset.get("browser_download_url")
                        logger.info(f"[GitHubPackageResolver] Found wheel: {name}")
                        return download_url, tag

                raise Exception(f"Wheel file not found for {package_name}")

        except Exception as e:
            logger.error(f"[GitHubPackageResolver] Failed to resolve package URL: {e}")
            raise Exception(f"Failed to resolve package URL for {package_name}: {e}")

    def download_wheel(
        self,
        url: str,
        dest_dir: Path,
        log_callback: Callable[[str], None] | None = None,
    ) -> Path:
        """Download wheel file to destination directory.

        Args:
            url: Download URL
            dest_dir: Destination directory
            log_callback: Optional callback for log messages

        Returns:
            Path to downloaded wheel file

        Raises:
            Exception: If download fails
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = url.split("/")[-1]
        dest_path = dest_dir / filename

        # Get URLs with mirror fallback
        urls = self.mirror_selector.get_github_urls(url, self.region)

        # Try each URL
        for download_url in urls:
            try:
                if log_callback:
                    log_callback(f"Downloading {filename}...")

                logger.debug(f"[GitHubPackageResolver] Downloading from {download_url}")

                def download_progress(block_num, block_size, total_size):
                    if log_callback and total_size > 0:
                        progress = (block_num * block_size / total_size) * 100
                        if progress % 10 < 5:  # Log every ~10%
                            log_callback(f"Downloading {filename}: {progress:.0f}%")

                urllib.request.urlretrieve(download_url, dest_path, download_progress)

                logger.info(f"[GitHubPackageResolver] Downloaded: {dest_path}")
                if log_callback:
                    log_callback(f"Downloaded {filename}")

                return dest_path

            except Exception as e:
                logger.warning(
                    f"[GitHubPackageResolver] Failed to download from {download_url}: {e}"
                )
                if log_callback:
                    log_callback(f"Failed: {e}, trying next source...")
                continue

        raise Exception(f"Failed to download wheel from all sources: {filename}")


class LumenPackageResolver:
    """Resolves Lumen package names and installation commands."""

    @staticmethod
    def resolve_packages(lumen_config: LumenConfig) -> list[str]:
        """Extract package names from LumenConfig.

        Args:
            lumen_config: Lumen configuration

        Returns:
            List of package names (e.g., ["lumen_ocr", "lumen_clip"])
        """
        packages = []

        # Get deployment services
        deployment = lumen_config.deployment

        if hasattr(deployment, "services") and deployment.services:
            for service in deployment.services:
                if hasattr(service, "root"):
                    root = service.root
                    package_name = f"lumen_{root}"
                    packages.append(package_name)

        logger.info(f"[LumenPackageResolver] Resolved packages: {packages}")
        return list(set(packages))  # Remove duplicates

    @staticmethod
    def build_pip_install_args(
        packages: list[str],
        device_config: DeviceConfig,
        region: Region,
        wheel_paths: dict[str, Path] | None = None,
    ) -> list[str]:
        """Build pip installation command arguments.

        Args:
            packages: List of package names (e.g., ["lumen_ocr", "lumen_clip"])
            device_config: Device configuration with dependency metadata
            region: Region for mirror selection
            wheel_paths: Dictionary mapping package names to wheel file paths

        Returns:
            List of pip command arguments

        Example:
            ["install", "--index-url", "...", "--extra-index-url", "...",
             "/path/to/lumen_ocr-0.4.1-py3-none-any.whl[apple]", "--no-cache-dir"]
        """
        args = ["install"]

        # Add index URLs
        mirror_selector = MirrorSelector()
        indexes = mirror_selector.get_pypi_indexes(region)

        # Add --index-url (first mirror)
        if indexes:
            args.extend(["--index-url", indexes[0]])

        # Add custom extra index URLs (for CUDA, etc.)
        # NOTE: Don't add PyPI as extra-index-url automatically to preserve mirror speed
        if device_config.dependency_metadata:
            meta = device_config.dependency_metadata

            # Add custom extra index URLs
            if meta.extra_index_url:
                for url in meta.extra_index_url:
                    args.extend(["--extra-index-url", url])

        # Add wheel file paths with extras
        if wheel_paths:
            for pkg in packages:
                if pkg in wheel_paths:
                    wheel_path = str(wheel_paths[pkg])

                    # Add extras if specified
                    if (
                        device_config.dependency_metadata
                        and device_config.dependency_metadata.extra_deps
                    ):
                        extra_deps = ",".join(
                            device_config.dependency_metadata.extra_deps
                        )
                        wheel_path = f"{wheel_path}[{extra_deps}]"

                    args.append(wheel_path)

        # Add additional install args
        if (
            device_config.dependency_metadata
            and device_config.dependency_metadata.install_args
        ):
            args.extend(device_config.dependency_metadata.install_args)

        # Add --no-cache-dir to avoid cache issues
        args.append("--no-cache-dir")

        logger.debug(f"[LumenPackageResolver] Pip args: {args}")
        return args
