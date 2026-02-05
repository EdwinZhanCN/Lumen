"""Hardware detection and driver API endpoints."""

from __future__ import annotations

import platform
from typing import Literal

from fastapi import APIRouter, HTTPException

from lumen_app.schemas.hardware import (
    DriverCheckResponse,
    HardwareInfoResponse,
    HardwarePresetResponse,
)
from lumen_app.utils.env_checker import EnvironmentChecker
from lumen_app.utils.logger import get_logger
from lumen_app.utils.preset_registry import PresetRegistry

logger = get_logger("lumen.web.api.hardware")
router = APIRouter()


def _map_driver_status(status) -> Literal["available", "missing", "incompatible"]:
    """Map internal DriverStatus enum to API status string."""
    from lumen_app.utils.env_checker import DriverStatus as InternalDriverStatus

    if status == InternalDriverStatus.AVAILABLE:
        return "available"
    elif status == InternalDriverStatus.INCOMPATIBLE:
        return "incompatible"
    else:
        return "missing"


@router.get("/info", response_model=HardwareInfoResponse)
async def get_hardware_info():
    """Get comprehensive hardware information."""
    logger.info("Getting hardware info")

    # Get system info
    info = HardwareInfoResponse(
        platform=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
    )

    # Get all presets
    presets = []
    for name in PresetRegistry.get_preset_names():
        preset_info = PresetRegistry.get_preset(name)
        if preset_info:
            try:
                device_config = preset_info.factory_method()
                preset = HardwarePresetResponse(
                    name=name,
                    description=preset_info.description,
                    runtime=device_config.runtime.value,
                    providers=[
                        p if isinstance(p, str) else p[0]
                        for p in (device_config.onnx_providers or [])
                    ],
                    requires_drivers=preset_info.requires_drivers,
                )
                presets.append(preset)
            except Exception as e:
                logger.warning(f"Failed to process preset {name}: {e}")

    info.presets = presets

    # Detect recommended preset using priority-based detection order
    detection_order = PresetRegistry.get_detection_order()

    detected_drivers = []
    for preset_name in detection_order:
        if PresetRegistry.preset_exists(preset_name):
            try:
                report = EnvironmentChecker.check_preset(preset_name)
                if report.ready:
                    info.recommended_preset = preset_name
                    info.drivers = [
                        DriverCheckResponse(
                            name=d.name,
                            status=_map_driver_status(d.status),
                            details=d.details,
                            installable_via_mamba=d.installable_via_mamba,
                            mamba_config_path=d.mamba_config_path,
                        )
                        for d in report.drivers
                    ]
                    info.all_drivers_available = True
                    break
                elif detected_drivers:
                    # Collect driver info from first preset with available info
                    continue
            except Exception as e:
                logger.warning(f"Failed to check preset {preset_name}: {e}")

    # If no recommended preset found, default to CPU
    if not info.recommended_preset:
        info.recommended_preset = "cpu"
        info.all_drivers_available = True  # CPU doesn't need drivers

    return info


@router.get("/presets", response_model=list[HardwarePresetResponse])
async def list_hardware_presets():
    """List all available hardware presets."""
    logger.info("Listing hardware presets")

    presets = []
    for name in PresetRegistry.get_preset_names():
        preset_info = PresetRegistry.get_preset(name)
        if preset_info:
            try:
                device_config = preset_info.factory_method()
                preset = HardwarePresetResponse(
                    name=name,
                    description=preset_info.description,
                    runtime=device_config.runtime.value,
                    providers=[
                        p if isinstance(p, str) else p[0]
                        for p in (device_config.onnx_providers or [])
                    ],
                    requires_drivers=preset_info.requires_drivers,
                )
                presets.append(preset)
            except Exception as e:
                logger.warning(f"Failed to process preset {name}: {e}")

    return presets


@router.get("/presets/{preset_name}/check", response_model=list[DriverCheckResponse])
async def check_preset_drivers(preset_name: str):
    """Check driver status for a specific preset."""
    logger.info(f"Checking drivers for preset: {preset_name}")

    if not PresetRegistry.preset_exists(preset_name):
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

    try:
        report = EnvironmentChecker.check_preset(preset_name)
        return [
            DriverCheckResponse(
                name=d.name,
                status=_map_driver_status(d.status),
                details=d.details,
                installable_via_mamba=d.installable_via_mamba,
                mamba_config_path=d.mamba_config_path,
            )
            for d in report.drivers
        ]
    except Exception as e:
        logger.error(f"Failed to check preset {preset_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check drivers: {str(e)}"
        )


@router.post("/detect", response_model=dict)
async def detect_hardware():
    """Detect available hardware and recommended preset."""
    logger.info("Detecting hardware")

    # Check each preset in priority order
    detection_order = PresetRegistry.get_detection_order()

    detected = []
    recommended = None

    for preset_name in detection_order:
        if not PresetRegistry.preset_exists(preset_name):
            continue

        try:
            report = EnvironmentChecker.check_preset(preset_name)
            status = {
                "preset": preset_name,
                "ready": report.ready,
                "drivers": [
                    {
                        "name": d.name,
                        "status": _map_driver_status(d.status),
                        "installable": d.installable_via_mamba,
                    }
                    for d in report.drivers
                ],
            }
            detected.append(status)

            if report.ready and not recommended:
                recommended = preset_name
        except Exception as e:
            logger.warning(f"Failed to check {preset_name}: {e}")

    return {
        "recommended_preset": recommended or "cpu",
        "detailed_status": detected,
    }
