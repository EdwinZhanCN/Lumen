"""Lumen package installer for installing Lumen packages from GitHub Releases."""

import logging
import subprocess
from pathlib import Path
from typing import Callable

from lumen_resources.lumen_config import Region

from ...core.config import DeviceConfig
from ..package_resolver import GitHubPackageResolver, LumenPackageResolver
from .env_manager import PythonEnvManager

logger = logging.getLogger(__name__)


class LumenPackageInstaller:
    """Installs Lumen packages from GitHub Releases."""

    def __init__(
        self,
        env_manager: PythonEnvManager,
        github_resolver: GitHubPackageResolver,
        log_callback: Callable[[str], None] | None = None,
    ):
        """Initialize package installer.

        Args:
            env_manager: Python environment manager
            github_resolver: GitHub package resolver
            log_callback: Optional callback for log messages
        """
        self.env_manager = env_manager
        self.github_resolver = github_resolver
        self.log_callback = log_callback or (lambda msg: None)

        logger.debug("[LumenPackageInstaller] Initialized")

    def install_packages(
        self,
        packages: list[str],
        device_config: DeviceConfig,
        region: Region,
    ) -> None:
        """Install Lumen packages from GitHub Releases.

        Args:
            packages: List of package names (e.g., ["lumen_ocr", "lumen_clip"])
            device_config: Device configuration with dependency metadata
            region: Region for mirror selection

        Raises:
            Exception: If installation fails
        """
        logger.info(f"[LumenPackageInstaller] Installing packages: {packages}")
        self.log_callback(f"Installing {len(packages)} Lumen package(s)...")

        # Create wheel download directory
        cache_dir = self.env_manager.cache_dir
        wheel_dir = cache_dir / "wheels"
        wheel_dir.mkdir(parents=True, exist_ok=True)

        # Download all wheels first and collect paths
        self.log_callback("Downloading package wheels...")
        wheel_paths = {}  # {package_name: wheel_path}
        for package in packages:
            wheel_path = self._download_package_wheel(package, wheel_dir)
            wheel_paths[package] = wheel_path

        # Build pip install command
        self.log_callback("Building installation command...")
        pip_args = LumenPackageResolver.build_pip_install_args(
            packages=packages,
            device_config=device_config,
            region=region,
            wheel_paths=wheel_paths,
        )

        # Run pip install
        self.log_callback("Running pip install...")
        self._run_pip_install(pip_args)

        self.log_callback("All packages installed successfully")
        logger.info("[LumenPackageInstaller] All packages installed")

    def _download_package_wheel(self, package: str, wheel_dir: Path) -> Path:
        """Download package wheel from GitHub Releases.

        Args:
            package: Package name (e.g., "lumen_ocr")
            wheel_dir: Directory to save wheel files

        Returns:
            Path to downloaded wheel

        Raises:
            Exception: If download fails
        """
        logger.debug(f"[LumenPackageInstaller] Downloading wheel for {package}")
        self.log_callback(f"Downloading {package}...")

        try:
            url, version = self.github_resolver.resolve_package_url(package)
            logger.debug(f"[LumenPackageInstaller] Resolved {package} {version}: {url}")

            wheel_path = self.github_resolver.download_wheel(
                url, wheel_dir, self.log_callback
            )

            logger.info(f"[LumenPackageInstaller] Downloaded {package}: {wheel_path}")
            return wheel_path

        except Exception as e:
            logger.error(f"[LumenPackageInstaller] Failed to download {package}: {e}")
            self.log_callback(f"Failed to download {package}: {e}")
            raise Exception(f"Failed to download {package}: {e}")

    def _run_pip_install(self, pip_args: list[str]) -> None:
        """Run pip install command.

        Args:
            pip_args: Pip command arguments

        Raises:
            Exception: If installation fails
        """
        cmd_str = "pip " + " ".join(pip_args[:3]) + " ..."
        logger.debug(f"[LumenPackageInstaller] Running: {cmd_str}")
        self.log_callback("Running: pip install ...")

        try:
            result = self.env_manager.run_pip(*pip_args)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                logger.error(f"[LumenPackageInstaller] Pip install failed: {error_msg}")
                self.log_callback(f"Installation failed:\n{error_msg}")
                raise Exception(f"Pip install failed: {error_msg}")

            # Log output
            if result.stdout:
                logger.debug(f"[LumenPackageInstaller] Pip output:\n{result.stdout}")

            logger.info("[LumenPackageInstaller] Pip install successful")

        except subprocess.TimeoutExpired:
            logger.error("[LumenPackageInstaller] Pip install timed out")
            self.log_callback("Installation timed out")
            raise Exception("Pip install timed out")
