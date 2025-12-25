"""
Environment checker for Lumen AI services.

Provides driver validation for different hardware platforms.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lumen_resources.lumen_config import Runtime

from lumen_app.core.config import DeviceConfig
from lumen_app.utils.preset_registry import PresetRegistry

logger = logging.getLogger("EnvChecker")


class DriverStatus(Enum):
    """Driver availability status."""

    AVAILABLE = "available"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"


@dataclass
class DriverCheckResult:
    """Result of driver availability check."""

    name: str
    status: DriverStatus
    details: str = ""
    installable_via_mamba: bool = False
    mamba_config_path: str | None = None


@dataclass
class EnvironmentReport:
    """Complete environment status report."""

    preset_name: str
    drivers: list[DriverCheckResult]
    ready: bool
    missing_installable: list[str] = field(default_factory=list)


class MicromambaChecker:
    """Checks micromamba availability and provides installation functionality."""

    @staticmethod
    def check_micromamba(micromamba_path: str | None = None) -> DriverCheckResult:
        """
        Check if micromamba is installed and accessible.

        Args:
            micromamba_path: Optional path to micromamba executable.
                            If None, checks PATH. If provided, uses this path.

        Returns:
            DriverCheckResult with micromamba status
        """
        # Determine micromamba executable path
        if micromamba_path:
            exe_path = Path(micromamba_path)
        else:
            # Check in PATH
            exe_path = Path("micromamba")

        # Try to run micromamba --version
        try:
            result = subprocess.run(
                [str(exe_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # Extract version
                version = result.stdout.strip().split()[-1]
                return DriverCheckResult(
                    name="micromamba",
                    status=DriverStatus.AVAILABLE,
                    details=f"version {version}",
                    installable_via_mamba=False,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return DriverCheckResult(
            name="micromamba",
            status=DriverStatus.MISSING,
            details=f"micromamba not found at {exe_path}",
            installable_via_mamba=False,
        )

    @staticmethod
    def install_micromamba(
        cache_dir: str | Path, target_name: str = "micromamba", dry_run: bool = False
    ) -> tuple[bool, str]:
        """
        Download and install micromamba to cache_dir.

        Args:
            cache_dir: Directory to install micromamba (e.g., ~/.lumen/micromamba)
            target_name: Subdirectory name for micromamba installation
            dry_run: If True, only print the command without executing

        Returns:
            Tuple of (success: bool, message: str)
        """
        cache_dir = Path(cache_dir)
        install_dir = cache_dir / target_name
        install_dir.mkdir(parents=True, exist_ok=True)

        system = platform.system()
        micromamba_exe = install_dir / (
            "micromamba.exe" if system == "Windows" else "micromamba"
        )

        # Check if already installed
        if micromamba_exe.exists():
            return True, f"micromamba already exists at {micromamba_exe}"

        # Build installation command
        if system == "Windows":
            # Windows: download installer and run with prefix
            install_script = install_dir / "install.ps1"
            cmd = [
                "powershell",
                "-Command",
                f"Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -OutFile {install_script}; "
                f"& {install_script} -prefix {install_dir} -batch",
            ]
        else:
            # Unix: download and run install script
            install_script = install_dir / "install.sh"
            cmd = [
                "bash",
                "-c",
                f"curl -Ls https://micro.mamba.pm/install.sh > {install_script} && "
                f"bash {install_script} -p {install_dir} -b",
            ]

        if dry_run:
            return True, f"Would run: {' '.join(cmd)}"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode == 0 and micromamba_exe.exists():
                return True, f"Successfully installed micromamba to {micromamba_exe}"
            else:
                return False, f"Installation failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, f"Installation error: {str(e)}"

    @staticmethod
    def get_executable_path(
        cache_dir: str | Path, target_name: str = "micromamba"
    ) -> str:
        """
        Get the path to micromamba executable in cache_dir.

        Args:
            cache_dir: Base cache directory
            target_name: Subdirectory name for micromamba installation

        Returns:
            Path to micromamba executable
        """
        install_dir = Path(cache_dir) / target_name

        if platform.system() == "Windows":
            # Windows: bin/micromamba.exe
            exe_path = install_dir / "bin" / "micromamba.exe"
        else:
            # Unix: bin/micromamba
            exe_path = install_dir / "bin" / "micromamba"

        return str(exe_path)


class DriverChecker:
    """Checks driver availability for different hardware platforms."""

    @staticmethod
    def check_nvidia_gpu() -> DriverCheckResult:
        """Check NVIDIA GPU/CUDA driver by running nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Parse GPU info from output
                lines = result.stdout.split("\n")
                details = "NVIDIA GPU detected"

                for line in lines:
                    if "CUDA Version" in line:
                        details = line.strip()
                        break

                return DriverCheckResult(
                    name="CUDA",
                    status=DriverStatus.AVAILABLE,
                    details=details,
                    installable_via_mamba=True,
                    mamba_config_path="cuda.yaml",
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"nvidia-smi check failed: {e}")

        return DriverCheckResult(
            name="CUDA",
            status=DriverStatus.MISSING,
            details="nvidia-smi command not found or failed",
            installable_via_mamba=True,
            mamba_config_path="cuda.yaml",
        )

    @staticmethod
    def check_amd_ryzen_ai_npu() -> DriverCheckResult:
        """Check AMD Ryzen AI NPU driver on Windows."""
        if platform.system() != "Windows":
            return DriverCheckResult(
                name="AMD Ryzen AI NPU",
                status=DriverStatus.MISSING,
                details="Only available on Windows",
                installable_via_mamba=False,
            )

        # Check if amdipu.dll exists
        dll_path = Path(r"C:\Windows\System32\amdipu.dll")
        if not dll_path.exists():
            return DriverCheckResult(
                name="AMD Ryzen AI NPU",
                status=DriverStatus.MISSING,
                details="amdipu.dll not found in C:\\Windows\\System32\\",
                installable_via_mamba=False,
            )

        # Check if amdipu service is running
        try:
            result = subprocess.run(
                ["sc", "query", "amdipu"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and "RUNNING" in result.stdout:
                return DriverCheckResult(
                    name="AMD Ryzen AI NPU",
                    status=DriverStatus.AVAILABLE,
                    details="amdipu service is running",
                    installable_via_mamba=False,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"amdipu service check failed: {e}")

        return DriverCheckResult(
            name="AMD Ryzen AI NPU",
            status=DriverStatus.MISSING,
            details="amdipu service is not running",
            installable_via_mamba=False,
        )

    @staticmethod
    def check_intel_gpu_openvino() -> DriverCheckResult:
        """Check Intel GPU / OpenVINO compatibility."""
        is_intel_cpu = False

        # Check CPU vendor based on platform
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    is_intel_cpu = "GenuineIntel" in cpuinfo

            elif platform.system() == "Darwin":
                # macOS - check sysctl
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                is_intel_cpu = "Intel" in result.stdout

            elif platform.system() == "Windows":
                # Windows - use wmic to check CPU manufacturer
                result = subprocess.run(
                    ["wmic", "cpu", "get", "manufacturer"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    # Output format: "Manufacturer\n GenuineIntel \n\n"
                    is_intel_cpu = (
                        "GenuineIntel" in result.stdout or "Intel" in result.stdout
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"CPU vendor check failed: {e}")

        # If not Intel CPU, return INCOMPATIBLE
        if not is_intel_cpu:
            return DriverCheckResult(
                name="OpenVINO",
                status=DriverStatus.INCOMPATIBLE,
                details="CPU vendor is not Intel",
                installable_via_mamba=True,
                mamba_config_path="openvino.yaml",
            )

        # Intel CPU detected - OpenVINO can be installed
        # Note: We don't check if openvino package is installed since
        # the runtime environment may not have it yet
        return DriverCheckResult(
            name="OpenVINO",
            status=DriverStatus.MISSING,
            details="Compatible Intel CPU detected, openvino package not installed",
            installable_via_mamba=True,
            mamba_config_path="openvino.yaml",
        )

    @staticmethod
    def check_apple_silicon() -> DriverCheckResult:
        """Check Apple Silicon CoreML support."""
        if platform.system() != "Darwin":
            return DriverCheckResult(
                name="CoreML",
                status=DriverStatus.INCOMPATIBLE,
                details="Only available on macOS (Darwin)",
                installable_via_mamba=False,
            )

        if platform.machine() != "arm64":
            return DriverCheckResult(
                name="CoreML",
                status=DriverStatus.INCOMPATIBLE,
                details=f"Architecture is {platform.machine()}, not arm64",
                installable_via_mamba=False,
            )

        # Check for Apple chip
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and "Apple" in result.stdout:
                return DriverCheckResult(
                    name="CoreML",
                    status=DriverStatus.AVAILABLE,
                    details=result.stdout.strip(),
                    installable_via_mamba=False,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"Apple Silicon check failed: {e}")

        return DriverCheckResult(
            name="CoreML",
            status=DriverStatus.MISSING,
            details="Not an Apple Silicon chip",
            installable_via_mamba=False,
        )

    @staticmethod
    def check_rockchip_rknn() -> DriverCheckResult:
        """Check Rockchip RKNN device node on Linux."""
        if platform.system() != "Linux":
            return DriverCheckResult(
                name="RKNN",
                status=DriverStatus.INCOMPATIBLE,
                details="Only available on Linux",
                installable_via_mamba=False,
            )

        # Check for /dev/rknpu device file
        rknpu_paths = [
            Path("/dev/rknpu"),
            Path("/dev/rknpu0"),
            Path("/dev/rp1"),
        ]

        for device_path in rknpu_paths:
            if device_path.exists():
                return DriverCheckResult(
                    name="RKNN",
                    status=DriverStatus.AVAILABLE,
                    details=f"Device found: {device_path}",
                    installable_via_mamba=False,
                )

        # Check for Rockchip in cpuinfo as fallback
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
                if "rockchip" in cpuinfo or "rk35" in cpuinfo:
                    return DriverCheckResult(
                        name="RKNN",
                        status=DriverStatus.MISSING,
                        details="Rockchip CPU detected but device node not found",
                        installable_via_mamba=False,
                    )
        except OSError:
            pass

        return DriverCheckResult(
            name="RKNN",
            status=DriverStatus.MISSING,
            details="RKNN device node not found",
            installable_via_mamba=False,
        )

    @staticmethod
    def check_amd_gpu_directml() -> DriverCheckResult:
        """Check AMD GPU DirectML support on Windows."""
        if platform.system() != "Windows":
            return DriverCheckResult(
                name="DirectML",
                status=DriverStatus.INCOMPATIBLE,
                details="Only available on Windows",
                installable_via_mamba=False,
            )

        # Check GPU name using wmic
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                output = result.stdout.lower()
                if "radeon" in output or "amd" in output:
                    return DriverCheckResult(
                        name="DirectML",
                        status=DriverStatus.AVAILABLE,
                        details="AMD Radeon GPU detected",
                        installable_via_mamba=False,
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"AMD GPU check failed: {e}")

        return DriverCheckResult(
            name="DirectML",
            status=DriverStatus.MISSING,
            details="No AMD Radeon GPU found",
            installable_via_mamba=False,
        )

    @staticmethod
    def check_for_preset(preset_name: str) -> list[DriverCheckResult]:
        """
        Check drivers required for a specific DeviceConfig preset.

        Args:
            preset_name: Name of the preset method (e.g., "nvidia_gpu", "apple_silicon")

        Returns:
            List of DriverCheckResult for required drivers
        """
        results = []

        # Validate preset exists
        if not PresetRegistry.preset_exists(preset_name):
            logger.warning(f"Unknown preset '{preset_name}', returning empty results")
            return results

        # Create config from preset to determine required checks
        try:
            config = PresetRegistry.create_config(preset_name)
            # Delegate to device_config check
            return DriverChecker.check_for_device_config(config)
        except Exception as e:
            logger.error(f"Failed to check preset '{preset_name}': {e}")
            return results

    @staticmethod
    def check_for_device_config(device_config: DeviceConfig) -> list[DriverCheckResult]:
        """
        Check drivers required for a DeviceConfig instance.

        Args:
            device_config: DeviceConfig instance

        Returns:
            List of DriverCheckResult for required drivers
        """
        results = []
        providers = device_config.onnx_providers or []
        runtime = device_config.runtime

        # Check based on onnx providers
        for provider in providers:
            provider_name = (
                provider
                if isinstance(provider, str)
                else (provider[0] if provider else "")
            )

            if "CUDA" in provider_name or "TensorRT" in provider_name:
                results.append(DriverChecker.check_nvidia_gpu())
            elif "CoreML" in provider_name:
                results.append(DriverChecker.check_apple_silicon())
            elif "OpenVINO" in provider_name:
                results.append(DriverChecker.check_intel_gpu_openvino())
            elif "DML" in provider_name:
                results.append(DriverChecker.check_amd_gpu_directml())

        # Check RKNN runtime
        if runtime == Runtime.rknn:
            results.append(DriverChecker.check_rockchip_rknn())

        # Remove duplicates by name
        seen = {}
        for result in results:
            if result.name not in seen:
                seen[result.name] = result

        return list(seen.values())


class DependencyInstaller:
    """Installs missing dependencies using micromamba."""

    def __init__(
        self,
        mamba_configs_dir: str | Path | None = None,
        micromamba_path: str | None = None,
    ):
        """
        Initialize installer with mamba config directory and micromamba path.

        Args:
            mamba_configs_dir: Path to directory containing *.yaml files.
                             Defaults to lumen_app/utils/mamba
            micromamba_path: Optional path to micromamba executable.
                            If None, uses 'micromamba' from PATH
        """
        if mamba_configs_dir is None:
            current_file = Path(__file__)
            mamba_configs_dir = current_file.parent / "mamba"

        self.configs_dir = Path(mamba_configs_dir)
        self.micromamba_path = micromamba_path or "micromamba"

    def install_driver(
        self, driver_name: str, env_name: str = "lumen_env", dry_run: bool = False
    ) -> tuple[bool, str]:
        """
        Install driver using micromamba.

        Args:
            driver_name: Name of driver (e.g., "cuda", "openvino")
            env_name: Name of the conda environment to create/update
            dry_run: If True, only print the command without executing

        Returns:
            Tuple of (success: bool, message: str)
        """
        # Map driver names to config files
        config_map = {
            "cuda": "cuda.yaml",
            "openvino": "openvino.yaml",
            "tensorrt": "tensorrt.yaml",
        }

        config_filename = config_map.get(driver_name)
        if not config_filename:
            return False, f"No mamba config available for {driver_name}"

        config_file = self.configs_dir / config_filename
        if not config_file.exists():
            return False, f"Config file not found: {config_file}"

        # Build micromamba command with specified path
        cmd = [
            self.micromamba_path,
            "install",
            "-y",
            "-n",
            env_name,
            "-f",
            str(config_file),
        ]

        if dry_run:
            return True, f"Would run: {' '.join(cmd)}"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
            )

            if result.returncode == 0:
                return True, f"Successfully installed {driver_name}"
            else:
                return False, f"Installation failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except FileNotFoundError:
            return False, f"micromamba not found at {self.micromamba_path}"
        except Exception as e:
            return False, f"Installation error: {str(e)}"

    def get_install_command(self, driver_name: str, env_name: str = "lumen_env") -> str:
        """Get the installation command for manual execution."""
        config_map = {
            "cuda": "cuda.yaml",
            "openvino": "openvino.yaml",
            "tensorrt": "tensorrt.yaml",
        }

        config_filename = config_map.get(driver_name)
        if not config_filename:
            return f"# No mamba config available for {driver_name}"

        config_file = self.configs_dir / config_filename
        if not config_file.exists():
            return f"# Config file not found: {config_file}"

        return f"micromamba install -y -n {env_name} -f {config_file}"


class EnvironmentChecker:
    """Main entry point for environment checking."""

    @staticmethod
    def check_preset(preset_name: str) -> EnvironmentReport:
        """
        Check environment for a specific preset.

        Args:
            preset_name: Name of the DeviceConfig preset method

        Returns:
            EnvironmentReport with full status
        """
        drivers = DriverChecker.check_for_preset(preset_name)

        all_available = all(d.status == DriverStatus.AVAILABLE for d in drivers)
        missing_installable = [
            d.name.lower()
            for d in drivers
            if d.status == DriverStatus.MISSING and d.installable_via_mamba
        ]

        return EnvironmentReport(
            preset_name=preset_name,
            drivers=drivers,
            ready=all_available,
            missing_installable=missing_installable,
        )

    @staticmethod
    def check_device_config(device_config: DeviceConfig) -> EnvironmentReport:
        """
        Check environment for a DeviceConfig instance.

        Args:
            device_config: DeviceConfig to check

        Returns:
            EnvironmentReport with full status
        """
        drivers = DriverChecker.check_for_device_config(device_config)

        all_available = all(d.status == DriverStatus.AVAILABLE for d in drivers)
        missing_installable = [
            d.name.lower()
            for d in drivers
            if d.status == DriverStatus.MISSING and d.installable_via_mamba
        ]

        # Determine preset name from config using PresetRegistry
        preset_name = "custom"
        for name in PresetRegistry.get_preset_names():
            preset_config = PresetRegistry.create_config(name)
            if (
                preset_config.runtime == device_config.runtime
                and preset_config.onnx_providers == device_config.onnx_providers
            ):
                preset_name = name
                break

        return EnvironmentReport(
            preset_name=preset_name,
            drivers=drivers,
            ready=all_available,
            missing_installable=missing_installable,
        )
