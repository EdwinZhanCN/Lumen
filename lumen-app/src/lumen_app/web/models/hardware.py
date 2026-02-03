"""Hardware detection and driver response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DriverCheckResponse(BaseModel):
    """Driver check result for API responses.

    Note: This is different from the internal DriverStatus enum in env_checker.py
    """

    name: str
    status: Literal["available", "missing", "incompatible"] = "missing"
    details: str = ""
    installable_via_mamba: bool = False
    mamba_config_path: str | None = None


class HardwarePresetResponse(BaseModel):
    """Hardware preset information for API responses."""

    name: str
    description: str
    requires_drivers: bool = True
    runtime: str
    providers: list[str] = Field(default_factory=list)


class HardwareInfoResponse(BaseModel):
    """Complete hardware detection report for API responses."""

    # System information
    platform: str
    machine: str
    processor: str
    python_version: str

    # Detected hardware presets
    presets: list[HardwarePresetResponse] = Field(default_factory=list)
    recommended_preset: str | None = None

    # Driver status for recommended preset
    drivers: list[DriverCheckResponse] = Field(default_factory=list)
    all_drivers_available: bool = False
    missing_installable: list[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "platform": "Linux",
                "machine": "x86_64",
                "processor": "x86_64",
                "python_version": "3.11.0",
                "presets": [],
                "recommended_preset": "nvidia_gpu",
                "drivers": [],
                "all_drivers_available": False,
                "missing_installable": [],
            }
        }
