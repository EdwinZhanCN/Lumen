"""Hardware detection and driver models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DriverStatus(BaseModel):
    """Driver check result."""

    name: str
    status: Literal["available", "missing", "incompatible"] = "missing"
    details: str = ""
    installable_via_mamba: bool = False
    mamba_config_path: str | None = None


class HardwarePreset(BaseModel):
    """Hardware preset information."""

    name: str
    description: str
    requires_drivers: bool = True
    runtime: str
    providers: list[str] = Field(default_factory=list)


class HardwareInfo(BaseModel):
    """Complete hardware detection report."""

    platform: str
    machine: str
    processor: str
    python_version: str

    # Detected hardware
    presets: list[HardwarePreset] = Field(default_factory=list)
    recommended_preset: str | None = None

    # Driver status
    drivers: list[DriverStatus] = Field(default_factory=list)
    all_drivers_available: bool = False
    missing_installable: list[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "platform": "Linux",
                "machine": "x86_64",
                "processor": "x86_64",
                "python_version": "3.11.0",
                "recommended_preset": "nvidia_gpu",
                "all_drivers_available": False,
            }
        }
