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

from lumen_app.utils.env_checker import DependencyInstaller, MicromambaChecker
from lumen_app.utils.logger import get_logger

logger = get_logger("lumen.core.installer")


class CoreInstaller:
    """Core installer that manages micromamba and environment setup."""

    def __init__(
        self,
        cache_dir: str | Path,
        env_name: str = "lumen_env",
        mamba_configs_dir: str | Path | None = None,
        micromamba_target: str = "micromamba",
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.env_name = env_name
        self.mamba_configs_dir = Path(mamba_configs_dir) if mamba_configs_dir else None
        self.micromamba_target = micromamba_target

    @property
    def micromamba_exe(self) -> str:
        return MicromambaChecker.get_executable_path(
            self.cache_dir, target_name=self.micromamba_target
        )

    @property
    def root_prefix(self) -> str:
        return str(self.cache_dir / self.micromamba_target)

    def install_micromamba(self, dry_run: bool = False) -> tuple[bool, str]:
        """Install micromamba into cache_dir."""
        logger.info("Installing micromamba...")
        return MicromambaChecker.install_micromamba(
            cache_dir=self.cache_dir,
            target_name=self.micromamba_target,
            dry_run=dry_run,
        )

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

        configs_dir = self.mamba_configs_dir
        if configs_dir is None:
            configs_dir = Path(__file__).resolve().parent.parent / "utils" / "mamba"

        config_file = Path(configs_dir) / config_filename
        logger.debug("Environment config file path: %s", config_file)

        if not config_file.exists():
            logger.error("Config file not found: %s", config_file)
            return False, f"Config file not found: {config_file}"

        cmd = [
            self.micromamba_exe,
            "create",
            "-y",
            "-n",
            self.env_name,
            "-f",
            str(config_file),
        ]
        if self.root_prefix:
            cmd.extend(["--root-prefix", self.root_prefix])

        logger.debug("Environment command: %s", " ".join(cmd))

        if dry_run:
            logger.info("Dry run mode - skipping execution")
            return True, f"Would run: {' '.join(cmd)}"

        try:
            logger.info("Executing environment creation (timeout=600s)")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                logger.info("Successfully created environment %s", self.env_name)
                return True, f"Successfully created environment {self.env_name}"

            logger.error(
                "Environment creation failed: returncode=%s, stderr=%s",
                result.returncode,
                result.stderr,
            )
            return False, f"Environment creation failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            logger.error("Environment creation timed out after 600s")
            return False, "Environment creation timed out"
        except FileNotFoundError:
            logger.error("micromamba not found at %s", self.micromamba_exe)
            return False, f"micromamba not found at {self.micromamba_exe}"
        except Exception as e:
            logger.error("Environment creation error: %s: %s", type(e).__name__, e)
            return False, f"Environment creation error: {str(e)}"

    def install_lumen_packages(
        self, lumen_config: LumenConfig, quiet: bool = True
    ) -> tuple[bool, str]:
        """Install Lumen packages derived from LumenConfig."""
        env_path = self.cache_dir / self.micromamba_target / "envs" / self.env_name

        packages: list[str] = []
        deployment = getattr(lumen_config, "deployment", None)
        services = getattr(deployment, "services", None) if deployment else None
        if services:
            for service in services:
                root = getattr(service, "root", None)
                if root:
                    packages.append(f"lumen-{root}")

        package_list = list(dict.fromkeys(packages))

        if not package_list:
            logger.info("No packages to install.")
            return True, "No packages to install"

        if not env_path.exists():
            return False, f"Environment not found at {env_path}"

        base_cmd = [
            self.micromamba_exe,
            "run",
            "-p",
            str(env_path),
            "pip",
            "install",
        ]

        cmd = base_cmd + package_list
        if quiet:
            cmd += ["--quiet", "--no-warn-script-location"]

        logger.info("Installing packages: %s", ", ".join(package_list))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return True, "All packages installed successfully"

            logger.warning(
                "Batch installation failed, trying one by one: %s", result.stderr
            )
            for package in package_list:
                single_cmd = base_cmd + [package]
                if quiet:
                    single_cmd.append("--quiet")

                single_result = subprocess.run(
                    single_cmd, capture_output=True, text=True, timeout=180
                )
                if single_result.returncode != 0:
                    return False, f"Failed to install {package}: {single_result.stderr}"

            return True, "All packages installed successfully (single installs)"
        except subprocess.TimeoutExpired:
            return False, "Package installation timed out"

    def verify_installation(self) -> tuple[bool, str]:
        """Verify micromamba and environment are installed."""
        micromamba_exe = Path(self.micromamba_exe)
        env_path = self.cache_dir / self.micromamba_target / "envs" / self.env_name

        if not micromamba_exe.exists():
            return False, "Micromamba not found"

        if not env_path.exists():
            return False, "Python environment not found"

        # Optional: check uv availability
        cmd = [
            str(micromamba_exe),
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
