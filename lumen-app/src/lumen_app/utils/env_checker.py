"""
Environment checker for Lumen AI services.

Provides driver validation for different hardware platforms.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lumen_resources.lumen_config import Runtime

from lumen_app.core.config import DeviceConfig
from lumen_app.utils.logger import get_logger
from lumen_app.utils.preset_registry import PresetRegistry

logger = get_logger("lumen.env_checker")


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


class DriverChecker:
    """Checks driver availability for different hardware platforms."""

    @staticmethod
    def check_nvidia_gpu() -> DriverCheckResult:
        """Check NVIDIA GPU/CUDA driver by running nvidia-smi."""
        logger.info("[DriverChecker] Checking NVIDIA GPU/CUDA driver")

        try:
            logger.debug("[DriverChecker] Executing: nvidia-smi")
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

                logger.info(f"[DriverChecker] NVIDIA GPU detected: {details}")
                return DriverCheckResult(
                    name="CUDA",
                    status=DriverStatus.AVAILABLE,
                    details=details,
                    installable_via_mamba=True,
                    mamba_config_path="cuda.yaml",
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(
                f"[DriverChecker] nvidia-smi check failed: {type(e).__name__}: {e}"
            )

        logger.info("[DriverChecker] NVIDIA GPU not detected")
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
        logger.info("[DriverChecker] Checking AMD Ryzen AI NPU")

        if platform.system() != "Windows":
            logger.debug(
                "[DriverChecker] Not Windows platform, marking as incompatible"
            )
            return DriverCheckResult(
                name="AMD Ryzen AI NPU",
                status=DriverStatus.MISSING,
                details="Only available on Windows",
                installable_via_mamba=False,
            )

        # Check if amdipu.dll exists
        dll_path = Path(r"C:\Windows\System32\amdipu.dll")
        logger.debug(f"[DriverChecker] Checking for {dll_path}")

        if not dll_path.exists():
            logger.debug("[DriverChecker] amdipu.dll not found")
            return DriverCheckResult(
                name="AMD Ryzen AI NPU",
                status=DriverStatus.MISSING,
                details="amdipu.dll not found in C:\\Windows\\System32\\",
                installable_via_mamba=False,
            )

        # Check if amdipu service is running
        try:
            logger.debug("[DriverChecker] Executing: sc query amdipu")
            result = subprocess.run(
                ["sc", "query", "amdipu"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and "RUNNING" in result.stdout:
                logger.info("[DriverChecker] AMD Ryzen AI NPU detected")
                return DriverCheckResult(
                    name="AMD Ryzen AI NPU",
                    status=DriverStatus.AVAILABLE,
                    details="amdipu service is running",
                    installable_via_mamba=False,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(
                f"[DriverChecker] amdipu service check failed: {type(e).__name__}: {e}"
            )

        logger.info("[DriverChecker] AMD Ryzen AI NPU not available")
        return DriverCheckResult(
            name="AMD Ryzen AI NPU",
            status=DriverStatus.MISSING,
            details="amdipu service is not running",
            installable_via_mamba=False,
        )

    @staticmethod
    def check_intel_gpu_openvino() -> DriverCheckResult:
        """Check Intel GPU / OpenVINO compatibility."""
        logger.info("[DriverChecker] Checking Intel GPU / OpenVINO")
        is_intel_cpu = False

        # Check CPU vendor based on platform
        try:
            system = platform.system()
            logger.debug(f"[DriverChecker] Platform: {system}")

            if system == "Linux":
                logger.debug("[DriverChecker] Reading /proc/cpuinfo")
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    is_intel_cpu = "GenuineIntel" in cpuinfo
                    logger.debug(f"[DriverChecker] Intel CPU detected: {is_intel_cpu}")

            elif system == "Darwin":
                # macOS - check sysctl
                logger.debug(
                    "[DriverChecker] Executing: sysctl -n machdep.cpu.brand_string"
                )
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                is_intel_cpu = "Intel" in result.stdout
                logger.debug(
                    f"[DriverChecker] CPU: {result.stdout.strip()}, Intel: {is_intel_cpu}"
                )

            elif system == "Windows":
                # Windows - use wmic to check CPU manufacturer
                logger.debug("[DriverChecker] Executing: wmic cpu get manufacturer")
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
                    logger.debug(f"[DriverChecker] Intel CPU detected: {is_intel_cpu}")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(
                f"[DriverChecker] CPU vendor check failed: {type(e).__name__}: {e}"
            )

        # If not Intel CPU, return INCOMPATIBLE
        if not is_intel_cpu:
            logger.info("[DriverChecker] Not Intel CPU - OpenVINO incompatible")
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
        logger.info("[DriverChecker] Intel CPU detected - OpenVINO installable")
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
        logger.info("[DriverChecker] Checking Apple Silicon CoreML")

        system = platform.system()
        machine = platform.machine()
        logger.debug(f"[DriverChecker] Platform: {system}, Architecture: {machine}")

        if system != "Darwin":
            logger.debug("[DriverChecker] Not macOS platform")
            return DriverCheckResult(
                name="CoreML",
                status=DriverStatus.INCOMPATIBLE,
                details="Only available on macOS (Darwin)",
                installable_via_mamba=False,
            )

        if machine != "arm64":
            logger.debug(f"[DriverChecker] Not arm64 architecture: {machine}")
            return DriverCheckResult(
                name="CoreML",
                status=DriverStatus.INCOMPATIBLE,
                details=f"Architecture is {machine}, not arm64",
                installable_via_mamba=False,
            )

        # Check for Apple chip
        try:
            logger.debug(
                "[DriverChecker] Executing: sysctl -n machdep.cpu.brand_string"
            )
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and "Apple" in result.stdout:
                logger.info(
                    f"[DriverChecker] Apple Silicon detected: {result.stdout.strip()}"
                )
                return DriverCheckResult(
                    name="CoreML",
                    status=DriverStatus.AVAILABLE,
                    details=result.stdout.strip(),
                    installable_via_mamba=False,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(
                f"[DriverChecker] Apple Silicon check failed: {type(e).__name__}: {e}"
            )

        logger.info("[DriverChecker] Not an Apple Silicon chip")
        return DriverCheckResult(
            name="CoreML",
            status=DriverStatus.MISSING,
            details="Not an Apple Silicon chip",
            installable_via_mamba=False,
        )

    @staticmethod
    def check_rockchip_rknn() -> DriverCheckResult:
        """Check Rockchip RKNN device node on Linux."""
        logger.info("[DriverChecker] Checking Rockchip RKNN")

        if platform.system() != "Linux":
            logger.debug("[DriverChecker] Not Linux platform")
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

        logger.debug(
            f"[DriverChecker] Checking for RKNN device nodes: {[str(p) for p in rknpu_paths]}"
        )
        for device_path in rknpu_paths:
            if device_path.exists():
                logger.info(f"[DriverChecker] RKNN device found: {device_path}")
                return DriverCheckResult(
                    name="RKNN",
                    status=DriverStatus.AVAILABLE,
                    details=f"Device found: {device_path}",
                    installable_via_mamba=False,
                )

        # Check for Rockchip in cpuinfo as fallback
        try:
            logger.debug("[DriverChecker] Reading /proc/cpuinfo for Rockchip CPU")
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
                if "rockchip" in cpuinfo or "rk35" in cpuinfo:
                    logger.info(
                        "[DriverChecker] Rockchip CPU detected but no device node"
                    )
                    return DriverCheckResult(
                        name="RKNN",
                        status=DriverStatus.MISSING,
                        details="Rockchip CPU detected but device node not found",
                        installable_via_mamba=False,
                    )
        except OSError as e:
            logger.debug(
                f"[DriverChecker] Failed to read cpuinfo: {type(e).__name__}: {e}"
            )

        logger.info("[DriverChecker] RKNN not detected")
        return DriverCheckResult(
            name="RKNN",
            status=DriverStatus.MISSING,
            details="RKNN device node not found",
            installable_via_mamba=False,
        )

    @staticmethod
    def check_amd_gpu_directml() -> DriverCheckResult:
        """Check AMD GPU DirectML support on Windows."""
        logger.info("[DriverChecker] Checking AMD GPU DirectML")

        if platform.system() != "Windows":
            logger.debug("[DriverChecker] Not Windows platform")
            return DriverCheckResult(
                name="DirectML",
                status=DriverStatus.INCOMPATIBLE,
                details="Only available on Windows",
                installable_via_mamba=False,
            )

        # Check GPU name using wmic
        try:
            logger.debug(
                "[DriverChecker] Executing: wmic path win32_videocontroller get name"
            )
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                output = result.stdout.lower()
                logger.debug(f"[DriverChecker] GPU detection output: {output[:100]}...")
                if "radeon" in output or "amd" in output:
                    logger.info("[DriverChecker] AMD Radeon GPU detected")
                    return DriverCheckResult(
                        name="DirectML",
                        status=DriverStatus.AVAILABLE,
                        details="AMD Radeon GPU detected",
                        installable_via_mamba=False,
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(
                f"[DriverChecker] AMD GPU check failed: {type(e).__name__}: {e}"
            )

        logger.info("[DriverChecker] No AMD GPU detected")
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
        logger.info(f"[DriverChecker] Checking drivers for preset: {preset_name}")
        results = []

        # Validate preset exists
        if not PresetRegistry.preset_exists(preset_name):
            logger.warning(
                f"[DriverChecker] Unknown preset '{preset_name}', returning empty results"
            )
            return results

        # Create config from preset to determine required checks
        try:
            logger.debug(f"[DriverChecker] Creating config from preset '{preset_name}'")
            config = PresetRegistry.create_config(preset_name)
            logger.debug(
                f"[DriverChecker] Config created: runtime={config.runtime}, providers={config.onnx_providers}"
            )
            # Delegate to device_config check
            results = DriverChecker.check_for_device_config(config)
            logger.info(
                f"[DriverChecker] Completed preset check, found {len(results)} driver(s)"
            )
            return results
        except Exception as e:
            logger.error(
                f"[DriverChecker] Failed to check preset '{preset_name}': {type(e).__name__}: {e}"
            )
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
        logger.info("[DriverChecker] Checking drivers for device config")
        results = []
        providers = device_config.onnx_providers or []
        runtime = device_config.runtime

        logger.debug(f"[DriverChecker] Config runtime={runtime}, providers={providers}")

        # Check based on onnx providers
        for provider in providers:
            provider_name = (
                provider
                if isinstance(provider, str)
                else (provider[0] if provider else "")
            )

            logger.debug(f"[DriverChecker] Checking provider: {provider_name}")

            if "CUDA" in provider_name or "TensorRT" in provider_name:
                logger.debug("[DriverChecker] Triggering NVIDIA GPU check")
                results.append(DriverChecker.check_nvidia_gpu())
            elif "CoreML" in provider_name:
                logger.debug("[DriverChecker] Triggering Apple Silicon check")
                results.append(DriverChecker.check_apple_silicon())
            elif "OpenVINO" in provider_name:
                logger.debug("[DriverChecker] Triggering Intel GPU/OpenVINO check")
                results.append(DriverChecker.check_intel_gpu_openvino())
            elif "DML" in provider_name:
                logger.debug("[DriverChecker] Triggering AMD GPU DirectML check")
                results.append(DriverChecker.check_amd_gpu_directml())

        # Check RKNN runtime
        if runtime == Runtime.rknn:
            logger.debug("[DriverChecker] Triggering RKNN check")
            results.append(DriverChecker.check_rockchip_rknn())

        # Remove duplicates by name
        seen = {}
        for result in results:
            if result.name not in seen:
                seen[result.name] = result

        unique_results = list(seen.values())
        logger.info(
            f"[DriverChecker] Device config check completed: {len(unique_results)} unique driver(s)"
        )
        return unique_results


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

        logger.info(
            f"[DependencyInstaller] Initialized with configs_dir={self.configs_dir}, micromamba_path={self.micromamba_path}"
        )

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
        logger.info(
            f"[DependencyInstaller] Installing driver '{driver_name}' (dry_run={dry_run})"
        )

        # Map driver names to config files
        config_map = {
            "cuda": "cuda.yaml",
            "openvino": "openvino.yaml",
            "tensorrt": "tensorrt.yaml",
        }

        config_filename = config_map.get(driver_name)
        if not config_filename:
            logger.warning(
                f"[DependencyInstaller] No mamba config available for {driver_name}"
            )
            return False, f"No mamba config available for {driver_name}"

        config_file = self.configs_dir / config_filename
        logger.debug(f"[DependencyInstaller] Config file path: {config_file}")

        if not config_file.exists():
            logger.error(f"[DependencyInstaller] Config file not found: {config_file}")
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

        logger.debug(f"[DependencyInstaller] Installation command: {' '.join(cmd)}")

        if dry_run:
            logger.info("[DependencyInstaller] Dry run mode - skipping execution")
            return True, f"Would run: {' '.join(cmd)}"

        try:
            logger.info("[DependencyInstaller] Executing installation (timeout=600s)")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
            )

            if result.returncode == 0:
                logger.info(
                    f"[DependencyInstaller] Successfully installed {driver_name}"
                )
                return True, f"Successfully installed {driver_name}"
            else:
                logger.error(
                    f"[DependencyInstaller] Installation failed: returncode={result.returncode}, stderr={result.stderr}"
                )
                return False, f"Installation failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            logger.error("[DependencyInstaller] Installation timed out after 600s")
            return False, "Installation timed out"
        except FileNotFoundError:
            logger.error(
                f"[DependencyInstaller] micromamba not found at {self.micromamba_path}"
            )
            return False, f"micromamba not found at {self.micromamba_path}"
        except Exception as e:
            logger.error(
                f"[DependencyInstaller] Installation error: {type(e).__name__}: {e}"
            )
            return False, f"Installation error: {str(e)}"

    def get_install_command(self, driver_name: str, env_name: str = "lumen_env") -> str:
        """Get the installation command for manual execution."""
        logger.debug(
            f"[DependencyInstaller] Getting install command for '{driver_name}'"
        )

        config_map = {
            "cuda": "cuda.yaml",
            "openvino": "openvino.yaml",
            "tensorrt": "tensorrt.yaml",
        }

        config_filename = config_map.get(driver_name)
        if not config_filename:
            logger.warning(
                f"[DependencyInstaller] No mamba config available for {driver_name}"
            )
            return f"# No mamba config available for {driver_name}"

        config_file = self.configs_dir / config_filename
        if not config_file.exists():
            logger.error(f"[DependencyInstaller] Config file not found: {config_file}")
            return f"# Config file not found: {config_file}"

        cmd = f"micromamba install -y -n {env_name} -f {config_file}"
        logger.debug(f"[DependencyInstaller] Generated command: {cmd}")
        return cmd


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
        logger.info(
            f"[EnvironmentChecker] Checking environment for preset: {preset_name}"
        )
        drivers = DriverChecker.check_for_preset(preset_name)

        all_available = all(d.status == DriverStatus.AVAILABLE for d in drivers)
        missing_installable = [
            d.name.lower()
            for d in drivers
            if d.status == DriverStatus.MISSING and d.installable_via_mamba
        ]

        logger.info(
            f"[EnvironmentChecker] Preset check result: ready={all_available}, drivers={len(drivers)}, missing_installable={len(missing_installable)}"
        )
        for driver in drivers:
            logger.debug(
                f"[EnvironmentChecker]   - {driver.name}: {driver.status.value} ({driver.details})"
            )

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
        logger.info("[EnvironmentChecker] Checking environment for device config")
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
                logger.debug(f"[EnvironmentChecker] Matched preset: {name}")
                break

        logger.info(
            f"[EnvironmentChecker] Device config check result: preset={preset_name}, ready={all_available}, drivers={len(drivers)}, missing_installable={len(missing_installable)}"
        )
        for driver in drivers:
            logger.debug(
                f"[EnvironmentChecker]   - {driver.name}: {driver.status.value} ({driver.details})"
            )

        return EnvironmentReport(
            preset_name=preset_name,
            drivers=drivers,
            ready=all_available,
            missing_installable=missing_installable,
        )
