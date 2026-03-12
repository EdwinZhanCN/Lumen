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


def _build_driver_response_list(report) -> list[DriverCheckResponse]:
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


def _resolve_availability(
    driver_results: list[DriverCheckResponse], ready: bool
) -> Literal["ready", "missing_drivers", "incompatible"]:
    if ready:
        return "ready"
    if any(driver.status == "incompatible" for driver in driver_results):
        return "incompatible"
    return "missing_drivers"


def _build_preset_response(
    preset_name: str, check_environment: bool = False
) -> HardwarePresetResponse | None:
    """Build hardware preset response, optionally with environment status."""
    preset_info = PresetRegistry.get_preset(preset_name)
    if not preset_info:
        return None

    try:
        device_config = preset_info.create_config()
    except Exception as e:
        logger.warning("Failed to create config for preset %s: %s", preset_name, e)
        return None

    response = HardwarePresetResponse(
        name=preset_info.name,
        description=preset_info.description,
        runtime=device_config.runtime.value,
        providers=[
            provider if isinstance(provider, str) else provider[0]
            for provider in (device_config.onnx_providers or [])
        ],
        requires_drivers=preset_info.requires_drivers,
        supported_on_current_platform=PresetRegistry.is_supported_on_current_platform(
            preset_info.name
        ),
        supported_systems=list(preset_info.supported_systems or ()),
    )

    if not check_environment:
        return response

    response.environment_checked = True
    if not response.supported_on_current_platform:
        response.availability = "incompatible"
        response.ready = False
        response.drivers = [
            DriverCheckResponse(
                name="Platform",
                status="incompatible",
                details=(
                    "Preset only supports: "
                    f"{', '.join(response.supported_systems) or 'unknown'}"
                ),
                installable_via_mamba=False,
            )
        ]
        return response

    report = EnvironmentChecker.check_preset(preset_info.name)
    response.ready = report.ready
    response.drivers = _build_driver_response_list(report)
    response.missing_installable = report.missing_installable
    response.availability = _resolve_availability(response.drivers, response.ready)
    return response


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

    presets: list[HardwarePresetResponse] = []
    preset_map: dict[str, HardwarePresetResponse] = {}
    for preset_name in PresetRegistry.get_preset_names():
        preset = _build_preset_response(preset_name, check_environment=True)
        if preset is None:
            continue
        presets.append(preset)
        preset_map[preset.name] = preset

    info.presets = presets

    for preset_name in PresetRegistry.get_detection_order():
        preset = preset_map.get(preset_name)
        if preset and preset.ready:
            info.recommended_preset = preset.name
            break

    if not info.recommended_preset and "cpu" in preset_map:
        info.recommended_preset = "cpu"

    selected = (
        preset_map.get(info.recommended_preset) if info.recommended_preset else None
    )
    if selected:
        info.drivers = selected.drivers
        info.all_drivers_available = selected.ready
        info.missing_installable = selected.missing_installable

    return info


@router.get("/presets", response_model=list[HardwarePresetResponse])
async def list_hardware_presets():
    """List all available hardware presets."""
    logger.info("Listing hardware presets")

    presets: list[HardwarePresetResponse] = []
    for preset_name in PresetRegistry.get_preset_names():
        preset = _build_preset_response(preset_name, check_environment=False)
        if preset is not None:
            presets.append(preset)

    return presets


@router.get("/presets/{preset_name}/check", response_model=list[DriverCheckResponse])
async def check_preset_drivers(preset_name: str):
    """Check driver status for a specific preset."""
    logger.info(f"Checking drivers for preset: {preset_name}")

    if not PresetRegistry.preset_exists(preset_name):
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

    try:
        preset = _build_preset_response(preset_name, check_environment=True)
        if preset is None:
            raise HTTPException(
                status_code=500, detail=f"Failed to check drivers for {preset_name}"
            )
        return preset.drivers
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
            preset = _build_preset_response(preset_name, check_environment=True)
            if preset is None:
                continue
            status = {
                "preset": preset.name,
                "ready": preset.ready,
                "availability": preset.availability,
                "drivers": [
                    {
                        "name": driver.name,
                        "status": driver.status,
                        "installable": driver.installable_via_mamba,
                    }
                    for driver in preset.drivers
                ],
                "missing_installable": preset.missing_installable,
            }
            detected.append(status)

            if preset.ready and not recommended:
                recommended = preset.name
        except Exception as e:
            logger.warning(f"Failed to check {preset_name}: {e}")

    return {
        "recommended_preset": recommended or "cpu",
        "detailed_status": detected,
    }
