"""
Core installer for managing environment and driver installations.

This module provides installation functionality including:
- Micromamba installation
- Python environment setup via micromamba
- Driver installation via micromamba
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import yaml
from lumen_resources import LumenConfig
from lumen_resources.lumen_config import Region

from lumen_app.core.config import DeviceConfig
from lumen_app.utils.env_checker import DependencyInstaller
from lumen_app.utils.installation import MicromambaInstaller, PythonEnvManager
from lumen_app.utils.logger import get_logger
from lumen_app.utils.package_resolver import (
    GitHubPackageResolver,
    LumenPackageResolver,
)

logger = get_logger("lumen.core.installer")


class CoreInstaller:
    """Core installer that manages micromamba and environment setup."""

    def __init__(
        self,
        cache_dir: str | Path,
        env_name: str = "lumen_env",
        mamba_configs_dir: str | Path | None = None,
        micromamba_target: str = "micromamba",
        region: Region = Region.other,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.env_name = env_name
        self.mamba_configs_dir = Path(mamba_configs_dir) if mamba_configs_dir else None
        self.micromamba_target = micromamba_target
        self.region = region

    @property
    def micromamba_exe(self) -> str:
        installer = MicromambaInstaller(self.cache_dir)
        return str(installer.get_executable())

    @property
    def root_prefix(self) -> str:
        return str(self.cache_dir / self.micromamba_target)

    def install_micromamba(self, dry_run: bool = False) -> tuple[bool, str]:
        """Install micromamba into cache_dir."""
        logger.info("Installing micromamba...")
        try:
            installer = MicromambaInstaller(self.cache_dir)
            exe_path = installer.install(dry_run=dry_run)
            return True, f"Micromamba installed successfully at {exe_path}"
        except Exception as e:
            logger.error("Failed to install micromamba: %s", e)
            return False, f"Failed to install micromamba: {e}"

    def create_environment(
        self,
        config_filename: str = "default.yaml",
        dry_run: bool = False,
    ) -> tuple[bool, str]:
        """Create Python environment using micromamba and a config file."""
        logger.info(
            "Creating environment '%s' with config '%s'",
            self.env_name,
            config_filename,
        )

        if dry_run:
            logger.info("Dry run mode - skipping execution")
            return True, f"Would create environment {self.env_name}"

        try:
            # Extract yaml_config name from filename (e.g., "default.yaml" -> "default")
            yaml_config = config_filename.replace(".yaml", "")

            env_manager = PythonEnvManager(
                cache_dir=self.cache_dir,
                micromamba_exe=self.micromamba_exe,
            )
            env_path = env_manager.create_env(yaml_config=yaml_config)
            logger.info(
                "Successfully created environment %s at %s", self.env_name, env_path
            )
            return True, f"Successfully created environment {self.env_name}"

        except Exception as e:
            logger.error("Environment creation error: %s: %s", type(e).__name__, e)
            return False, f"Environment creation error: {str(e)}"

    def install_lumen_packages(
        self,
        lumen_config: LumenConfig,
        device_config: DeviceConfig,
        quiet: bool = True,
    ) -> tuple[bool, str]:
        """Install Lumen packages derived from LumenConfig.

        Downloads wheels from GitHub Releases and installs them with proper
        dependencies based on device configuration.

        Args:
            lumen_config: Lumen configuration with deployment services
            device_config: Device configuration with dependency metadata
            quiet: Whether to suppress pip output

        Returns:
            Tuple of (success, message)
        """
        logger.info("Installing Lumen packages from GitHub Releases")

        # Check environment exists
        env_manager = PythonEnvManager(
            cache_dir=self.cache_dir,
            micromamba_exe=self.micromamba_exe,
        )

        if not env_manager.env_exists():
            return False, f"Environment not found at {env_manager.get_env_path()}"

        try:
            # Resolve package names from config
            package_list = LumenPackageResolver.resolve_packages(lumen_config)

            if not package_list:
                logger.info("No packages to install.")
                return True, "No packages to install"

            logger.info("Packages to install: %s", ", ".join(package_list))

            # Create wheel download directory
            wheel_dir = self.cache_dir / "wheels"
            wheel_dir.mkdir(parents=True, exist_ok=True)

            # Initialize GitHub resolver
            github_resolver = GitHubPackageResolver(region=self.region)

            # Download all wheels first
            wheel_paths = {}
            for package in package_list:
                logger.info("Downloading wheel for %s...", package)
                try:
                    url, version = github_resolver.resolve_package_url(package)
                    wheel_path = github_resolver.download_wheel(url, wheel_dir)
                    wheel_paths[package] = wheel_path
                    logger.info("Downloaded %s version %s", package, version)
                except Exception as e:
                    logger.error("Failed to download %s: %s", package, e)
                    return False, f"Failed to download {package}: {e}"

            # Build pip install command with device-specific extras
            pip_args = LumenPackageResolver.build_pip_install_args(
                packages=package_list,
                device_config=device_config,
                region=self.region,
                wheel_paths=wheel_paths,
            )

            if quiet:
                pip_args.extend(["--quiet", "--no-warn-script-location"])

            # Run pip install
            logger.info("Running pip install with device-specific extras")
            result = env_manager.run_pip(*pip_args)

            if result.returncode == 0:
                logger.info("All packages installed successfully")
                return True, "All packages installed successfully"
            else:
                error_msg = result.stderr or result.stdout
                logger.error("Package installation failed: %s", error_msg)
                return False, f"Package installation failed: {error_msg}"

        except subprocess.TimeoutExpired:
            logger.error("Package installation timed out")
            return False, "Package installation timed out"
        except Exception as e:
            logger.error("Package installation error: %s", e)
            return False, f"Package installation error: {e}"

    def verify_installation(self) -> tuple[bool, str]:
        """Verify micromamba and environment are installed."""
        try:
            # Check micromamba
            installer = MicromambaInstaller(self.cache_dir)
            check_result = installer.check()

            if check_result.status.value != "installed":
                return False, "Micromamba not found"

            # Check environment
            env_manager = PythonEnvManager(
                cache_dir=self.cache_dir,
                micromamba_exe=self.micromamba_exe,
            )

            if not env_manager.env_exists():
                return False, "Python environment not found"

            # Optional: check uv availability
            env_path = env_manager.get_env_path()
            cmd = [
                str(self.micromamba_exe),
                "run",
                "-p",
                str(env_path),
                "uv",
                "--version",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info("uv installed: %s", result.stdout.strip())
            except subprocess.TimeoutExpired:
                logger.warning("uv check timed out")

            return True, "Installation verified"
        except Exception as e:
            logger.error("Verification error: %s", e)
            return False, f"Verification error: {e}"

    def save_config(
        self, lumen_config: LumenConfig, config_filename: str = "lumen-config.yaml"
    ) -> tuple[bool, str]:
        """Persist LumenConfig to disk."""
        config_path = self.cache_dir / config_filename

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            config_dict = lumen_config.model_dump(mode="json")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            return True, f"Config saved: {config_path}"
        except Exception as e:
            logger.error("Failed to save config: %s", e)
            return False, f"Failed to save config: {e}"

    def install_drivers(
        self, driver_names: Iterable[str], dry_run: bool = False
    ) -> list[tuple[str, bool, str]]:
        """Install driver packages using micromamba.

        Returns:
            List of (driver_name, success, message)
        """
        installer = DependencyInstaller(
            mamba_configs_dir=self.mamba_configs_dir,
            micromamba_path=self.micromamba_exe,
            root_prefix=self.root_prefix,
        )

        results: list[tuple[str, bool, str]] = []
        for name in driver_names:
            logger.info("Installing driver '%s'...", name)
            success, message = installer.install_driver(
                driver_name=name,
                env_name=self.env_name,
                dry_run=dry_run,
            )
            results.append((name, success, message))

        return results
