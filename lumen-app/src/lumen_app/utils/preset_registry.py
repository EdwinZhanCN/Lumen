"""
Preset registry for Lumen device configurations.

Provides dynamic discovery of available presets from DeviceConfig,
eliminating hardcoded preset identifiers.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Callable

from lumen_app.core.config import DeviceConfig

logger = logging.getLogger("PresetRegistry")


@dataclass
class PresetInfo:
    """Information about a device preset."""

    name: str
    description: str
    factory_method: Callable[[], DeviceConfig]
    requires_drivers: bool = True


class PresetRegistry:
    """
    Dynamic registry of DeviceConfig presets.

    Automatically discovers all factory methods (classmethods) from DeviceConfig
    that return DeviceConfig instances, eliminating the need for hardcoded preset names.
    """

    _presets: dict[str, PresetInfo] | None = None

    @classmethod
    def _discover_presets(cls) -> dict[str, PresetInfo]:
        """Discover all preset factory methods from DeviceConfig."""
        if cls._presets is not None:
            return cls._presets

        presets = {}

        # Scan all members of DeviceConfig (including bound methods)
        for name in dir(DeviceConfig):
            # Skip private attributes
            if name.startswith("_"):
                continue

            # Get the member
            try:
                member = getattr(DeviceConfig, name)
            except AttributeError:
                continue

            # Skip non-methods and dataclass fields
            if not callable(member):
                continue

            # Check if it's a classmethod descriptor
            is_classmethod = False
            try:
                raw_attr = inspect.getattr_static(DeviceConfig, name)
                is_classmethod = isinstance(raw_attr, classmethod)
            except AttributeError:
                pass

            if not is_classmethod:
                continue

            # Try to call it and see if it returns a DeviceConfig
            try:
                result = member()
                if isinstance(result, DeviceConfig):
                    # Get description from DeviceConfig.description or docstring
                    if hasattr(result, "description") and result.description:
                        description = result.description
                    else:
                        docstring = inspect.getdoc(member) or ""
                        description = docstring.split("\n")[0] if docstring else name

                    presets[name] = PresetInfo(
                        name=name,
                        description=description,
                        factory_method=member,
                        requires_drivers=name
                        != "cpu",  # CPU preset requires no special drivers
                    )
                    logger.debug(f"Discovered preset: {name}")
            except Exception as e:
                logger.debug(f"Skipping {name}: {e}")
                continue

        cls._presets = presets
        logger.info(f"Discovered {len(presets)} device presets")
        return presets

    @classmethod
    def get_all_presets(cls) -> dict[str, PresetInfo]:
        """Get all available presets."""
        return cls._discover_presets().copy()

    @classmethod
    def get_preset(cls, name: str) -> PresetInfo | None:
        """Get a specific preset by name."""
        presets = cls._discover_presets()
        return presets.get(name)

    @classmethod
    def preset_exists(cls, name: str) -> bool:
        """Check if a preset exists."""
        return name in cls._discover_presets()

    @classmethod
    def get_preset_names(cls) -> list[str]:
        """Get list of all preset names."""
        return list(cls._discover_presets().keys())

    @classmethod
    def create_config(cls, preset_name: str) -> DeviceConfig:
        """
        Create a DeviceConfig from preset name.

        Args:
            preset_name: Name of the preset (e.g., "nvidia_gpu", "apple_silicon")

        Returns:
            DeviceConfig instance

        Raises:
            ValueError: If preset doesn't exist
        """
        preset = cls.get_preset(preset_name)
        if preset is None:
            available = ", ".join(cls.get_preset_names())
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        return preset.factory_method()

    @classmethod
    def get_driver_requirements(cls, preset_name: str) -> list[str]:
        """
        Get required driver checks for a preset.

        Args:
            preset_name: Name of the preset

        Returns:
            List of driver names to check
        """
        preset = cls.get_preset(preset_name)
        if preset is None or not preset.requires_drivers:
            return []

        config = preset.factory_method()

        # Map runtime/providers to required drivers
        requirements = []
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

        # RKNN runtime
        if config.runtime.value == "rknn":
            requirements.append("rknn")

        return list(set(requirements))  # Remove duplicates
