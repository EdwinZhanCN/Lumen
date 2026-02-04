"""Installation models - simplified one-click setup."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Service component status."""

    micromamba: bool = False
    environment: bool = False
    config: bool = False
    drivers: bool = False


class CheckInstallationPathResponse(BaseModel):
    """Response for checking installation path."""

    has_existing_service: bool = False
    service_status: ServiceStatus = Field(default_factory=ServiceStatus)
    ready_to_start: bool = False
    recommended_action: Literal["start_existing", "configure_new", "repair"] = (
        "configure_new"
    )
    message: str = ""


class InstallSetupRequest(BaseModel):
    """Request to start a complete installation setup.

    This will automatically install all required components for the selected preset:
    - micromamba (if not present)
    - conda environment (if not exists)
    - required drivers for the preset
    """

    preset: str
    cache_dir: str = "~/.lumen"
    environment_name: str = "lumen_env"
    force_reinstall: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "preset": "nvidia_gpu",
                "cache_dir": "~/.lumen",
                "environment_name": "lumen_env",
                "force_reinstall": False,
            }
        }


class InstallStep(BaseModel):
    """A single step in the installation process."""

    name: str
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    progress: int = Field(0, ge=0, le=100)
    message: str = ""
    started_at: float | None = None
    completed_at: float | None = None


class InstallTaskResponse(BaseModel):
    """Installation task status and progress."""

    task_id: str
    preset: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    progress: int = Field(0, ge=0, le=100)
    current_step: str = ""
    steps: list[InstallStep] = Field(default_factory=list)
    created_at: float
    updated_at: float
    completed_at: float | None = None
    error: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc-123",
                "preset": "nvidia_gpu",
                "status": "running",
                "progress": 45,
                "current_step": "Installing CUDA drivers",
                "steps": [
                    {
                        "name": "Check micromamba",
                        "status": "completed",
                        "progress": 100,
                        "message": "micromamba already installed",
                    },
                    {
                        "name": "Create environment",
                        "status": "running",
                        "progress": 60,
                        "message": "Creating lumen_env...",
                    },
                ],
                "created_at": 1234567890.0,
                "updated_at": 1234567895.0,
                "completed_at": None,
                "error": None,
            }
        }


class InstallTaskListResponse(BaseModel):
    """List of installation tasks."""

    tasks: list[InstallTaskResponse]
    total: int


class InstallStatusResponse(BaseModel):
    """Current installation status of the system."""

    micromamba_installed: bool
    micromamba_path: str | None = None
    environment_exists: bool
    environment_name: str | None = None
    environment_path: str | None = None
    drivers_checked: bool = False
    drivers: dict[str, str] = Field(default_factory=dict)  # driver_name -> status
    ready_for_preset: str | None = None
    missing_components: list[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "micromamba_installed": True,
                "micromamba_path": "/usr/local/bin/micromamba",
                "environment_exists": True,
                "environment_name": "lumen_env",
                "environment_path": "~/.lumen/envs/lumen_env",
                "drivers_checked": True,
                "drivers": {
                    "cuda": "available",
                    "cudnn": "missing",
                },
                "ready_for_preset": "nvidia_gpu",
                "missing_components": ["cudnn"],
            }
        }


class InstallLogsResponse(BaseModel):
    """Installation task logs."""

    task_id: str
    logs: list[str] = Field(default_factory=list)
    total_lines: int = 0
