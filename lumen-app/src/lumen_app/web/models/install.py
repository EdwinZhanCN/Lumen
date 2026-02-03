"""Installation task models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InstallRequest(BaseModel):
    """Request to start an installation task."""

    task_type: Literal["micromamba", "environment", "drivers", "packages"]
    options: dict = Field(default_factory=dict)
    # For driver installation
    drivers: list[str] = Field(default_factory=list)
    # For package installation
    packages: list[str] = Field(default_factory=list)
    environment: str = "lumen_env"

    class Config:
        json_schema_extra = {
            "example": {
                "task_type": "drivers",
                "drivers": ["cuda", "openvino"],
                "environment": "lumen_env",
            }
        }


class InstallStatus(BaseModel):
    """Installation task status."""

    task_id: str
    task_type: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: int = Field(0, ge=0, le=100)
    message: str = ""
    created_at: float = 0
    updated_at: float | None = None
    completed_at: float | None = None
    error_details: str | None = None


class InstallTask(BaseModel):
    """Detailed installation task information."""

    task_id: str
    task_type: str
    status: str
    progress: int
    message: str
    logs: list[str] = Field(default_factory=list)
    created_at: float
    updated_at: float | None = None


class InstallListResponse(BaseModel):
    """List of installation tasks."""

    tasks: list[InstallStatus]
    total: int
