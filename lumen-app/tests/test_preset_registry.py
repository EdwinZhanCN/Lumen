"""Tests for hardware preset platform support rules."""

from lumen_app.utils.preset_registry import PresetRegistry


def test_gpu_presets_are_not_supported_on_macos():
    """Intel/NVIDIA presets should not be offered on macOS."""
    assert not PresetRegistry.is_supported_on_current_platform(
        "intel_gpu", system_name="Darwin"
    )
    assert not PresetRegistry.is_supported_on_current_platform(
        "nvidia_gpu", system_name="Darwin"
    )
    assert not PresetRegistry.is_supported_on_current_platform(
        "nvidia_gpu_high", system_name="Darwin"
    )


def test_platform_specific_presets_keep_expected_supported_systems():
    """Platform-specific presets should still match their intended OS."""
    assert PresetRegistry.is_supported_on_current_platform(
        "apple_silicon", system_name="Darwin"
    )
    assert PresetRegistry.is_supported_on_current_platform(
        "intel_gpu", system_name="Linux"
    )
    assert PresetRegistry.is_supported_on_current_platform(
        "nvidia_gpu", system_name="Windows"
    )
    assert PresetRegistry.is_supported_on_current_platform(
        "cpu", system_name="Darwin"
    )


def test_rockchip_preset_is_not_exposed():
    """Rockchip preset should stay disabled in lumen-app."""
    assert not PresetRegistry.preset_exists("rockchip")
    assert "rockchip" not in PresetRegistry.get_preset_names()
