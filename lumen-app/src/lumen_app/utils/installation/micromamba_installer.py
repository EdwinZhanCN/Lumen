"""Micromamba installer for managing micromamba installation and checking.

This module provides functionality to:
1. Check if micromamba is installed and accessible
2. Download and install micromamba with region-based mirror support
3. Get the path to micromamba executable
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lumen_resources.lumen_config import Region

logger = logging.getLogger(__name__)


class MicromambaStatus(Enum):
    """Micromamba installation status."""

    INSTALLED = "installed"
    NOT_INSTALLED = "not_installed"
    INCOMPATIBLE = "incompatible"


@dataclass
class MicromambaCheckResult:
    """Result of micromamba availability check.

    Attributes:
        status: Installation status
        version: Version string if installed
        executable_path: Path to micromamba executable
        details: Additional details
    """

    status: MicromambaStatus
    version: str | None = None
    executable_path: str | None = None
    details: str = ""


class MirrorSelector:
    """Selects mirror URLs based on region for micromamba downloads."""

    GITHUB_MIRROR_CN = "https://gh-proxy.org/https://github.com"

    def get_micromamba_urls(self, base_url: str, region: Region) -> list[str]:
        """Get micromamba download URLs with mirror fallback.

        Args:
            base_url: Original GitHub URL
            region: Region.cn or Region.other

        Returns:
            List of URLs to try (mirror first if cn, then original)
        """
        urls = []
        if region == Region.cn:
            # Apply ghproxy mirror for Chinese users
            mirror_url = base_url.replace("https://github.com", self.GITHUB_MIRROR_CN)
            urls.append(mirror_url)
        urls.append(base_url)
        return urls


class MicromambaInstaller:
    """Manages micromamba installation and retrieval.

    This class provides a complete solution for:
    - Checking if micromamba is installed
    - Downloading and installing micromamba with mirror support
    - Getting the executable path
    """

    def __init__(self, cache_dir: Path | str, region: Region = Region.other):
        """Initialize installer.

        Args:
            cache_dir: Cache directory for micromamba installation
            region: Region for mirror selection
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.target_name = "micromamba"
        self.region = region
        self.mirror_selector = MirrorSelector()

        logger.debug(
            f"[MicromambaInstaller] Initialized: cache_dir={self.cache_dir}, region={region.value}"
        )

    def check(self, custom_path: str | None = None) -> MicromambaCheckResult:
        """Check if micromamba is installed and accessible.

        Args:
            custom_path: Optional path to micromamba executable.
                        If None, checks PATH and install directory.

        Returns:
            MicromambaCheckResult with status and details
        """
        logger.debug("[MicromambaInstaller] Checking micromamba availability")

        # Determine executable path
        if custom_path:
            exe_path = Path(custom_path)
            logger.debug(f"[MicromambaInstaller] Using custom path: {custom_path}")
        else:
            # First check if it's in our install directory
            exe_path = self.get_executable()
            if not exe_path.exists():
                # Fall back to PATH
                exe_path = Path("micromamba")
                logger.debug("[MicromambaInstaller] Checking PATH for micromamba")

        # Try to run micromamba --version
        try:
            result = subprocess.run(
                [str(exe_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                version = result.stdout.strip().split()[-1]
                logger.info(
                    f"[MicromambaInstaller] Micromamba found: version {version}"
                )
                return MicromambaCheckResult(
                    status=MicromambaStatus.INSTALLED,
                    version=version,
                    executable_path=str(exe_path),
                    details=f"version {version}",
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"[MicromambaInstaller] Check failed: {e}")

        logger.info("[MicromambaInstaller] Micromamba not available")
        return MicromambaCheckResult(
            status=MicromambaStatus.NOT_INSTALLED,
            details=f"micromamba not found at {exe_path}",
        )

    def install(self, dry_run: bool = False) -> Path:
        """Download and install micromamba to cache_dir.

        Args:
            dry_run: If True, only print commands without executing

        Returns:
            Path to micromamba executable

        Raises:
            Exception: If installation fails
        """
        logger.info(f"[MicromambaInstaller] Installing micromamba (dry_run={dry_run})")

        install_dir = self.cache_dir / self.target_name
        install_dir.mkdir(parents=True, exist_ok=True)

        # Determine executable path based on platform
        exe_path = self._get_executable_path(install_dir)

        # Check if already installed AND executable
        if exe_path.exists():
            # Verify it's executable
            if platform.system() != "Windows" and not os.access(exe_path, os.X_OK):
                logger.warning(
                    f"[MicromambaInstaller] File exists but not executable: {exe_path}"
                )
                logger.info("[MicromambaInstaller] Fixing permissions...")
                try:
                    subprocess.run(
                        ["chmod", "+x", str(exe_path)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    logger.info(f"[MicromambaInstaller] Fixed permissions: {exe_path}")
                except Exception as e:
                    logger.warning(
                        f"[MicromambaInstaller] Failed to fix permissions: {e}"
                    )
                    # Fall through to re-download

            # Test if it works
            try:
                result = subprocess.run(
                    [str(exe_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    logger.info(
                        f"[MicromambaInstaller] Already installed and working: {exe_path}"
                    )
                    return exe_path
                else:
                    logger.warning(
                        f"[MicromambaInstaller] File exists but not working: {result.stderr}"
                    )
            except Exception as e:
                logger.warning(
                    f"[MicromambaInstaller] Executable test failed: {e}, re-downloading..."
                )
                # Remove broken file and re-download
                exe_path.unlink()

        # Build installation command based on platform
        system = platform.system()

        if system == "Windows":
            success, message = self._install_windows(install_dir, dry_run)
        else:
            success, message = self._install_unix(install_dir, exe_path, dry_run)

        if not success:
            raise Exception(f"Failed to install micromamba: {message}")

        # Verify installation
        if not exe_path.exists():
            raise Exception(
                f"Installation succeeded but executable not found: {exe_path}"
            )

        logger.info(f"[MicromambaInstaller] Successfully installed: {exe_path}")
        return exe_path

    def ensure_installed(self) -> Path:
        """Ensure micromamba is installed, installing if necessary.

        Returns:
            Path to micromamba executable

        Raises:
            Exception: If installation check or install fails
        """
        result = self.check()

        if result.status == MicromambaStatus.INSTALLED:
            logger.debug("[MicromambaInstaller] Already installed, skipping")
            if result.executable_path is None:
                raise Exception("Executable path is None despite being installed")
            return Path(result.executable_path)

        logger.info("[MicromambaInstaller] Not installed, installing now")
        return self.install()

    def get_executable(self) -> Path:
        """Get micromamba executable path in install directory.

        Returns:
            Path to micromamba executable
        """
        install_dir = self.cache_dir / self.target_name
        return self._get_executable_path(install_dir)

    def _get_executable_path(self, install_dir: Path) -> Path:
        """Get executable path for current platform.

        Args:
            install_dir: Installation directory

        Returns:
            Path to micromamba executable
        """
        if platform.system() == "Windows":
            return install_dir / "bin" / "micromamba.exe"
        else:
            return install_dir / "bin" / "micromamba"

    def _install_windows(self, install_dir: Path, dry_run: bool) -> tuple[bool, str]:
        """Install micromamba on Windows using direct binary download.

        Args:
            install_dir: Installation directory
            dry_run: If True, only print command without executing

        Returns:
            Tuple of (success, message)
        """
        logger.debug("[MicromambaInstaller] Installing on Windows")

        # Get executable path
        exe_path = self._get_executable_path(install_dir)
        bin_dir = exe_path.parent
        bin_dir.mkdir(parents=True, exist_ok=True)

        # Build download URL (Windows is always win-64)
        base_url = "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64"

        # Get URLs with mirror fallback
        download_urls = self.mirror_selector.get_micromamba_urls(base_url, self.region)

        logger.debug(f"[MicromambaInstaller] Platform: win-64, URLs: {download_urls}")

        # Try each URL
        for download_url in download_urls:
            logger.debug(f"[MicromambaInstaller] Trying URL: {download_url}")

            # Method 1: Try using curl.exe (available on Windows 10+)
            download_cmd = ["curl", "-fsSL", download_url, "-o", str(exe_path)]

            if dry_run:
                logger.info(
                    f"[MicromambaInstaller] Would run: {' '.join(download_cmd)}"
                )
                return True, f"Would download from {download_url}"

            try:
                # Try curl first
                logger.info("[MicromambaInstaller] Downloading micromamba with curl...")
                download_result = subprocess.run(
                    download_cmd, capture_output=True, text=True, timeout=120
                )

                if download_result.returncode != 0:
                    # Fallback to PowerShell
                    logger.debug(
                        "[MicromambaInstaller] curl failed, trying PowerShell..."
                    )
                    ps_cmd = [
                        "powershell",
                        "-Command",
                        f"Invoke-WebRequest -Uri '{download_url}' -OutFile '{exe_path}' -UseBasicParsing",
                    ]
                    download_result = subprocess.run(
                        ps_cmd, capture_output=True, text=True, timeout=120
                    )

                    if download_result.returncode != 0:
                        logger.warning(
                            f"[MicromambaInstaller] PowerShell download failed: {download_result.stderr}"
                        )
                        continue

                # Configure conda-forge channels
                self._configure_channels(exe_path)

                logger.info(f"[MicromambaInstaller] Successfully installed: {exe_path}")
                return True, f"Installed to {exe_path}"

            except subprocess.TimeoutExpired:
                logger.warning("[MicromambaInstaller] Download timed out")
                continue
            except Exception as e:
                logger.warning(
                    f"[MicromambaInstaller] Install error: {type(e).__name__}: {e}"
                )
                continue

        return False, "Failed to download from all sources"

    def _install_unix(
        self, install_dir: Path, exe_path: Path, dry_run: bool
    ) -> tuple[bool, str]:
        """Install micromamba on Unix (Linux/macOS).

        Args:
            install_dir: Installation directory
            exe_path: Expected executable path
            dry_run: If True, only print command without executing

        Returns:
            Tuple of (success, message)
        """
        logger.debug("[MicromambaInstaller] Installing on Unix")

        # Detect platform and architecture
        platform_name, arch = self._detect_platform()

        # Build download URL
        base_url = f"https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-{platform_name}-{arch}"

        # Get URLs with mirror fallback
        download_urls = self.mirror_selector.get_micromamba_urls(base_url, self.region)

        logger.debug(
            f"[MicromambaInstaller] Platform: {platform_name}-{arch}, URLs: {download_urls}"
        )

        # Create bin directory
        bin_dir = install_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        # Try each URL
        for download_url in download_urls:
            logger.debug(f"[MicromambaInstaller] Trying URL: {download_url}")

            download_cmd = ["curl", "-fsSL", download_url, "-o", str(exe_path)]
            chmod_cmd = ["chmod", "+x", str(exe_path)]

            if dry_run:
                logger.info(
                    f"[MicromambaInstaller] Would run: {' '.join(download_cmd)}"
                )
                return True, f"Would download from {download_url}"

            try:
                # Download
                logger.info("[MicromambaInstaller] Downloading micromamba...")
                download_result = subprocess.run(
                    download_cmd, capture_output=True, text=True, timeout=120
                )

                if download_result.returncode != 0:
                    logger.warning(
                        f"[MicromambaInstaller] Download failed: {download_result.stderr}"
                    )
                    continue

                # Make executable
                subprocess.run(chmod_cmd, capture_output=True, text=True, timeout=10)

                # Configure conda-forge channels
                self._configure_channels(exe_path)

                logger.info(f"[MicromambaInstaller] Successfully installed: {exe_path}")
                return True, f"Installed to {exe_path}"

            except subprocess.TimeoutExpired:
                logger.warning("[MicromambaInstaller] Download timed out")
                continue
            except Exception as e:
                logger.warning(
                    f"[MicromambaInstaller] Install error: {type(e).__name__}: {e}"
                )
                continue

        return False, "Failed to download from all sources"

    def _detect_platform(self) -> tuple[str, str]:
        """Detect platform and architecture for micromamba download.

        Returns:
            Tuple of (platform_name, arch)
            - platform_name: "linux" or "osx"
            - arch: "64", "aarch64", "arm64", or "ppc64le"

        Raises:
            Exception: If platform detection fails
        """
        try:
            # Get OS platform
            uname_result = subprocess.run(
                ["uname", "-s"], capture_output=True, text=True, timeout=5
            )
            uname_s = uname_result.stdout.strip()

            if uname_s == "Linux":
                platform_name = "linux"
            elif uname_s == "Darwin":
                platform_name = "osx"
            else:
                raise Exception(f"Unsupported platform: {uname_s}")

            # Get architecture
            uname_result = subprocess.run(
                ["uname", "-m"], capture_output=True, text=True, timeout=5
            )
            uname_m = uname_result.stdout.strip()

            # Map architecture names (same logic as install.sh)
            if uname_m in ("aarch64", "ppc64le", "arm64"):
                arch = uname_m
            else:
                arch = "64"

            logger.debug(f"[MicromambaInstaller] Detected: {platform_name}-{arch}")
            return platform_name, arch

        except Exception as e:
            logger.error(f"[MicromambaInstaller] Platform detection failed: {e}")
            raise Exception(f"Failed to detect platform: {e}")

    def _configure_channels(self, exe_path: Path) -> None:
        """Configure conda-forge channels for micromamba.

        Args:
            exe_path: Path to micromamba executable
        """
        logger.debug("[MicromambaInstaller] Configuring conda-forge channels")

        config_commands = [
            [str(exe_path), "config", "append", "channels", "conda-forge"],
            [str(exe_path), "config", "append", "channels", "nodefaults"],
            [str(exe_path), "config", "set", "channel_priority", "strict"],
        ]

        for cmd in config_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    logger.warning(
                        f"[MicromambaInstaller] Config failed: {result.stderr}"
                    )
            except Exception as e:
                logger.warning(f"[MicromambaInstaller] Config error: {e}")
