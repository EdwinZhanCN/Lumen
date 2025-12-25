"""
Tests for EnvironmentChecker utility.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lumen_app.core.config import DeviceConfig
from lumen_app.utils.env_checker import (
    DependencyInstaller,
    DriverChecker,
    DriverStatus,
    EnvironmentChecker,
    EnvironmentReport,
    MicromambaChecker,
)

# =============================================================================
# DriverChecker Tests
# =============================================================================


def test_check_nvidia_gpu_available():
    """Test NVIDIA GPU check when nvidia-smi succeeds."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "CUDA Version: 12.6"
        mock_run.return_value = mock_result

        result = DriverChecker.check_nvidia_gpu()

        assert result.status == DriverStatus.AVAILABLE
        assert result.name == "CUDA"
        assert result.installable_via_mamba is True


def test_check_nvidia_gpu_missing():
    """Test NVIDIA GPU check when nvidia-smi fails."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        result = DriverChecker.check_nvidia_gpu()

        assert result.status == DriverStatus.MISSING
        assert result.name == "CUDA"


def test_check_amd_ryzen_ai_npu_not_windows():
    """Test AMD Ryzen AI NPU check on non-Windows."""
    with patch("platform.system", return_value="Linux"):
        result = DriverChecker.check_amd_ryzen_ai_npu()

        assert result.status == DriverStatus.MISSING
        assert "Only available on Windows" in result.details


def test_check_amd_ryzen_ai_npu_dll_missing():
    """Test AMD Ryzen AI NPU check when DLL is missing."""
    with patch("platform.system", return_value="Windows"):
        with patch("pathlib.Path.exists", return_value=False):
            result = DriverChecker.check_amd_ryzen_ai_npu()

            assert result.status == DriverStatus.MISSING
            assert "amdipu.dll not found" in result.details


def test_check_amd_ryzen_ai_npu_service_running():
    """Test AMD Ryzen AI NPU check when service is running."""
    with patch("platform.system", return_value="Windows"):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "STATE: RUNNING"
                mock_run.return_value = mock_result

                result = DriverChecker.check_amd_ryzen_ai_npu()

                assert result.status == DriverStatus.AVAILABLE
                assert "amdipu service is running" in result.details


# =============================================================================
# MicromambaChecker Tests
# =============================================================================


def test_check_micromamba_available():
    """Test micromamba check when installed."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "micromamba 1.5.3\n"
        mock_run.return_value = mock_result

        result = MicromambaChecker.check_micromamba()

        assert result.status == DriverStatus.AVAILABLE
        assert result.name == "micromamba"
        assert "1.5.3" in result.details


def test_check_micromamba_missing():
    """Test micromamba check when not installed."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        result = MicromambaChecker.check_micromamba()

        assert result.status == DriverStatus.MISSING
        assert result.name == "micromamba"
        assert "not found" in result.details


def test_micromamba_install_to_cache_dir():
    """Test micromamba installation to cache directory."""
    with patch("subprocess.run") as mock_run:
        # Mock successful installation
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with patch("pathlib.Path.exists", return_value=False):
            with patch("pathlib.Path.mkdir"):  # Mock mkdir to avoid creating dirs
                success, message = MicromambaChecker.install_micromamba(
                    cache_dir="/test/cache", dry_run=True
                )

            assert success is True
            assert "Would run" in message


def test_micromamba_get_executable_path():
    """Test getting micromamba executable path."""
    with patch("platform.system", return_value="Darwin"):
        path = MicromambaChecker.get_executable_path("/test/cache")
        assert "bin/micromamba" in path

    with patch("platform.system", return_value="Windows"):
        path = MicromambaChecker.get_executable_path("/test/cache")
        assert "bin/micromamba.exe" in path


# =============================================================================
# DriverChecker Tests
# =============================================================================


def test_check_intel_gpu_openvino_genuine_intel():
    """Test Intel GPU/OpenVINO check."""
    result = DriverChecker.check_intel_gpu_openvino()

    # Should return OpenVINO check result
    assert result.name == "OpenVINO"
    # Status depends on platform (INCOMPATIBLE on Apple Silicon, MISSING on Intel without package, etc.)
    assert isinstance(result.status, DriverStatus)


def test_check_intel_gpu_openvino_not_intel():
    """Test Intel GPU/OpenVINO with non-Intel CPU."""
    with patch("platform.system", return_value="Linux"):
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = MagicMock()
            mock_open.return_value.__exit__ = MagicMock()
            mock_open.return_value.read.return_value = "vendor_id\t: AuthenticAMD\n"

            result = DriverChecker.check_intel_gpu_openvino()

            assert result.status == DriverStatus.INCOMPATIBLE
            assert "not Intel" in result.details


def test_check_apple_silicon_not_macos():
    """Test Apple Silicon check on non-macOS."""
    with patch("platform.system", return_value="Linux"):
        result = DriverChecker.check_apple_silicon()

        assert result.status == DriverStatus.INCOMPATIBLE
        assert "Only available on macOS" in result.details


def test_check_apple_silicon_not_arm64():
    """Test Apple Silicon check on x86_64 macOS."""
    with patch("platform.system", return_value="Darwin"):
        with patch("platform.machine", return_value="x86_64"):
            result = DriverChecker.check_apple_silicon()

            assert result.status == DriverStatus.INCOMPATIBLE
            assert "not arm64" in result.details


def test_check_apple_silicon_available():
    """Test Apple Silicon check on M1/M2/M3."""
    with patch("platform.system", return_value="Darwin"):
        with patch("platform.machine", return_value="arm64"):
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "Apple M2"
                mock_run.return_value = mock_result

                result = DriverChecker.check_apple_silicon()

                assert result.status == DriverStatus.AVAILABLE
                assert "Apple" in result.details


def test_check_rockchip_rknn_not_linux():
    """Test RKNN check on non-Linux."""
    with patch("platform.system", return_value="Windows"):
        result = DriverChecker.check_rockchip_rknn()

        assert result.status == DriverStatus.INCOMPATIBLE
        assert "Only available on Linux" in result.details


def test_check_rockchip_rknn_device_found():
    """Test RKNN check when device node exists."""
    with patch("platform.system", return_value="Linux"):
        # Mock Path.exists to return True for /dev/rknpu
        original_exists = Path.exists

        def mock_exists(self):
            return str(self) == "/dev/rknpu" or original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            result = DriverChecker.check_rockchip_rknn()

            assert result.status == DriverStatus.AVAILABLE
            assert "Device found" in result.details


def test_check_rockchip_rknn_missing():
    """Test RKNN check when device not found."""
    with patch("platform.system", return_value="Linux"):
        with patch("pathlib.Path.exists", return_value=False):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__ = MagicMock()
                mock_open.return_value.__exit__ = MagicMock()
                mock_open.return_value.read.return_value = "vendor_id\t: GenuineIntel\n"

                result = DriverChecker.check_rockchip_rknn()

                assert result.status == DriverStatus.MISSING


def test_check_amd_gpu_directml_not_windows():
    """Test AMD GPU DirectML check on non-Windows."""
    with patch("platform.system", return_value="Linux"):
        result = DriverChecker.check_amd_gpu_directml()

        assert result.status == DriverStatus.INCOMPATIBLE
        assert "Only available on Windows" in result.details


def test_check_amd_gpu_directml_radeon_found():
    """Test AMD GPU DirectML check when Radeon GPU found."""
    with patch("platform.system", return_value="Windows"):
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "AMD Radeon RX 7900 XTX"
            mock_run.return_value = mock_result

            result = DriverChecker.check_amd_gpu_directml()

            assert result.status == DriverStatus.AVAILABLE
            assert "Radeon" in result.details


def test_check_for_preset_nvidia():
    """Test checking drivers for NVIDIA GPU preset."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        results = DriverChecker.check_for_preset("nvidia_gpu")

        assert len(results) == 1
        assert results[0].name == "CUDA"


def test_check_for_preset_cpu():
    """Test checking drivers for CPU preset (no special drivers)."""
    results = DriverChecker.check_for_preset("cpu")

    assert len(results) == 0


def test_check_for_preset_apple_silicon():
    """Test checking drivers for Apple Silicon preset."""
    with patch("platform.system", return_value="Darwin"):
        with patch("platform.machine", return_value="arm64"):
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "Apple M2"
                mock_run.return_value = mock_result

                results = DriverChecker.check_for_preset("apple_silicon")

                assert len(results) == 1
                assert results[0].name == "CoreML"


def test_check_for_device_config_cpu():
    """Test checking drivers for CPU DeviceConfig."""
    config = DeviceConfig.cpu()
    results = DriverChecker.check_for_device_config(config)

    assert len(results) == 0


def test_check_for_device_config_nvidia():
    """Test checking drivers for NVIDIA DeviceConfig."""
    config = DeviceConfig.nvidia_gpu()

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        results = DriverChecker.check_for_device_config(config)

        assert len(results) >= 1
        assert any(r.name == "CUDA" for r in results)


# =============================================================================
# DependencyInstaller Tests
# =============================================================================


def test_installer_init_default_path():
    """Test installer initialization with default path."""
    installer = DependencyInstaller()

    assert installer.configs_dir is not None
    assert installer.configs_dir.name == "mamba"


def test_installer_get_install_command_cuda():
    """Test getting install command for CUDA."""
    with patch("pathlib.Path.exists", return_value=True):
        installer = DependencyInstaller(mamba_configs_dir="/fake/path")

        cmd = installer.get_install_command("cuda")

        assert "micromamba" in cmd
        assert "cuda" in cmd.lower()


def test_installer_install_dry_run():
    """Test installer dry run mode."""
    with patch("pathlib.Path.exists", return_value=True):
        installer = DependencyInstaller(mamba_configs_dir="/fake/path")

        success, message = installer.install_driver("cuda", dry_run=True)

        assert success is True
        assert "Would run" in message


def test_installer_install_unsupported_driver():
    """Test installer with unsupported driver."""
    installer = DependencyInstaller()

    success, message = installer.install_driver("unsupported_driver")

    assert success is False
    assert "No mamba config available" in message


# =============================================================================
# EnvironmentChecker Tests
# =============================================================================


def test_check_preset_nvidia_missing():
    """Test environment check for NVIDIA preset with missing driver."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        report = EnvironmentChecker.check_preset("nvidia_gpu")

        assert isinstance(report, EnvironmentReport)
        assert report.preset_name == "nvidia_gpu"
        assert report.ready is False
        # missing_installable now contains lowercase names
        assert "cuda" in report.missing_installable


def test_check_preset_cpu_ready():
    """Test environment check for CPU preset (always ready)."""
    report = EnvironmentChecker.check_preset("cpu")

    assert isinstance(report, EnvironmentReport)
    assert report.preset_name == "cpu"
    assert report.ready is True
    assert len(report.drivers) == 0


def test_check_device_config_cpu():
    """Test environment check for CPU DeviceConfig."""
    config = DeviceConfig.cpu()
    report = EnvironmentChecker.check_device_config(config)

    assert isinstance(report, EnvironmentReport)
    assert report.ready is True
    assert len(report.drivers) == 0


def test_check_device_config_apple_silicon():
    """Test environment check for Apple Silicon DeviceConfig."""
    config = DeviceConfig.apple_silicon()

    with patch("platform.system", return_value="Darwin"):
        with patch("platform.machine", return_value="arm64"):
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "Apple M2"
                mock_run.return_value = mock_result

                report = EnvironmentChecker.check_device_config(config)

                assert isinstance(report, EnvironmentReport)
                assert report.ready is True


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_workflow_check_and_install():
    """Test full workflow: check preset, get install commands."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        # 1. Check preset
        report = EnvironmentChecker.check_preset("nvidia_gpu")

        assert report.ready is False
        assert len(report.missing_installable) > 0

        # 2. Get install command (driver names are uppercase in result)
        installer = DependencyInstaller()
        for driver_name in report.missing_installable:
            # Convert to lowercase for installer
            cmd = installer.get_install_command(driver_name.lower())
            # Should have valid command or error message
            assert isinstance(cmd, str)
            assert len(cmd) > 0


def test_environment_report_all_available():
    """Test EnvironmentReport when all drivers available."""
    report = EnvironmentReport(preset_name="cpu", drivers=[], ready=True)

    assert report.ready is True


def test_environment_report_missing_drivers():
    """Test EnvironmentReport with missing drivers."""
    from lumen_app.utils.env_checker import DriverCheckResult

    report = EnvironmentReport(
        preset_name="nvidia_gpu",
        drivers=[
            DriverCheckResult(
                name="CUDA",
                status=DriverStatus.MISSING,
                installable_via_mamba=True,
                mamba_config_path="cuda.yaml",
            )
        ],
        ready=False,
        missing_installable=["CUDA"],
    )

    assert report.ready is False
    assert "CUDA" in report.missing_installable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
