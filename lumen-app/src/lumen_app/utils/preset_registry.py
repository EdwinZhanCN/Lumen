"""Explicit preset registry for Lumen device configurations."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Callable

from lumen_app.services.config import DeviceConfig
from lumen_app.utils.logger import get_logger

logger = get_logger("lumen.preset_registry")


@dataclass(frozen=True)
class PresetInfo:
    """Information about a device preset."""

    name: str
    description: str
    factory: Callable[..., DeviceConfig]
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    requires_drivers: bool = True
    priority: int = 50  # Lower number = higher priority (0-100)
    aliases: tuple[str, ...] = ()
    supported_systems: tuple[str, ...] | None = None

    def create_config(self, **overrides: Any) -> DeviceConfig:
        """Create a DeviceConfig for this preset."""
        kwargs = {**self.default_kwargs, **overrides}
        return self.factory(**kwargs)

    @property
    def factory_method(self) -> Callable[[], DeviceConfig]:
        """Backwards-compatible no-arg factory."""
        return lambda: self.create_config()


class PresetRegistry:
    """Registry of known DeviceConfig presets."""

    _presets: dict[str, PresetInfo] | None = None
    _aliases: dict[str, str] | None = None

    @classmethod
    def _build_registry(cls):
        """Build preset and alias mappings once."""
        if cls._presets is not None and cls._aliases is not None:
            return

        preset_definitions = [
            PresetInfo(
                name="nvidia_gpu_high",
                description="Preset for high RAM (>= 12GB) Nvidia GPUs",
                factory=DeviceConfig.nvidia_gpu_high,
                priority=5,
            ),
            PresetInfo(
                name="nvidia_gpu",
                description="Preset for low RAM (< 12GB) Nvidia GPUs",
                factory=DeviceConfig.nvidia_gpu,
                priority=10,
            ),
            PresetInfo(
                name="nvidia_jetson_high",
                description="Preset for high RAM (>= 12GB) Nvidia Jetson Devices",
                factory=DeviceConfig.nvidia_jetson_high,
                priority=12,
                supported_systems=("Linux",),
            ),
            PresetInfo(
                name="nvidia_jetson",
                description="Preset for low RAM (< 12GB) Nvidia Jetson Devices",
                factory=DeviceConfig.nvidia_jetson,
                priority=15,
                supported_systems=("Linux",),
            ),
            PresetInfo(
                name="apple_silicon",
                description="Preset for Apple Silicon",
                factory=DeviceConfig.apple_silicon,
                priority=20,
                supported_systems=("Darwin",),
            ),
            PresetInfo(
                name="rockchip",
                description="Preset for Rockchip NPU (RK3588)",
                factory=DeviceConfig.rockchip,
                default_kwargs={"rknn_device": "rk3588"},
                priority=25,
                aliases=("rockchip_rk3588",),
                supported_systems=("Linux",),
            ),
            PresetInfo(
                name="intel_gpu",
                description="Preset for Intel iGPU or Arc GPU",
                factory=DeviceConfig.intel_gpu,
                priority=30,
            ),
            PresetInfo(
                name="amd_gpu_win",
                description="Preset for AMD Ryzen GPUs",
                factory=DeviceConfig.amd_gpu_win,
                priority=35,
                supported_systems=("Windows",),
            ),
            PresetInfo(
                name="amd_npu",
                description="Preset for AMD Ryzen NPUs",
                factory=DeviceConfig.amd_npu,
                priority=40,
                supported_systems=("Windows",),
            ),
            PresetInfo(
                name="cpu",
                description="Preset General CPUs",
                factory=DeviceConfig.cpu,
                requires_drivers=False,
                priority=100,
            ),
        ]

        presets: dict[str, PresetInfo] = {}
        aliases: dict[str, str] = {}

        for preset in preset_definitions:
            try:
                # Validate preset factory and default kwargs are callable.
                preset.create_config()
            except Exception as e:
                logger.warning("Skipping preset %s: %s", preset.name, e)
                continue

            presets[preset.name] = preset
            for alias in preset.aliases:
                aliases[alias] = preset.name

        cls._presets = presets
        cls._aliases = aliases
        logger.info("Loaded %d device presets", len(presets))

    @classmethod
    def _canonical_name(cls, preset_name: str) -> str:
        cls._build_registry()
        assert cls._aliases is not None
        return cls._aliases.get(preset_name, preset_name)

    @classmethod
    def get_all_presets(cls) -> dict[str, PresetInfo]:
        """Get all available canonical presets."""
        cls._build_registry()
        assert cls._presets is not None
        return cls._presets.copy()

    @classmethod
    def get_preset(cls, name: str) -> PresetInfo | None:
        """Get a specific preset by name (alias supported)."""
        cls._build_registry()
        assert cls._presets is not None
        return cls._presets.get(cls._canonical_name(name))

    @classmethod
    def preset_exists(cls, name: str) -> bool:
        """Check if a preset exists (alias supported)."""
        return cls.get_preset(name) is not None

    @classmethod
    def get_preset_names(cls) -> list[str]:
        """Get canonical preset names sorted by priority."""
        presets = cls.get_all_presets()
        ordered = sorted(presets.values(), key=lambda p: p.priority)
        return [preset.name for preset in ordered]

    @classmethod
    def get_detection_order(cls) -> list[str]:
        """Get supported preset names in detection order."""
        names = cls.get_preset_names()
        return [
            name
            for name in names
            if cls.is_supported_on_current_platform(name, platform.system())
        ]

    @classmethod
    def is_supported_on_current_platform(
        cls, preset_name: str, system_name: str | None = None
    ) -> bool:
        """Check if preset is supported on current OS."""
        preset = cls.get_preset(preset_name)
        if preset is None:
            return False
        if not preset.supported_systems:
            return True
        current_system = system_name or platform.system()
        return current_system in preset.supported_systems

    @classmethod
    def create_config(cls, preset_name: str, **overrides: Any) -> DeviceConfig:
        """Create a DeviceConfig from preset name."""
        preset = cls.get_preset(preset_name)
        if preset is None:
            available = ", ".join(cls.get_preset_names())
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        return preset.create_config(**overrides)

    @classmethod
    def get_driver_requirements(cls, preset_name: str) -> list[str]:
        """Get required driver checks for a preset."""
        preset = cls.get_preset(preset_name)
        if preset is None or not preset.requires_drivers:
            return []

        config = preset.create_config()

        requirements: list[str] = []
        providers = config.onnx_providers or []

        for provider in providers:
            provider_name = (
                provider
                if isinstance(provider, str)
                else (provider[0] if provider else "")
            )

            if "CUDA" in provider_name or "TensorRT" in provider_name:
                requirements.append("cuda")
            elif "CoreML" in provider_name:
                requirements.append("coreml")
            elif "OpenVINO" in provider_name:
                requirements.append("openvino")
            elif "DML" in provider_name:
                requirements.append("directml")

        if config.runtime.value == "rknn":
            requirements.append("rknn")

        return sorted(set(requirements))
