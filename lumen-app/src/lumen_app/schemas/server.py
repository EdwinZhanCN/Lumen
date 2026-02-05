"""Server management models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server configuration."""

    port: int = 50051
    host: str = "0.0.0.0"
    enable_mdns: bool = True
    service_name: str = "lumen-ai"
    config_path: str | None = None
    environment: str = "lumen_env"

    class Config:
        json_schema_extra = {
            "example": {
                "port": 50051,
                "host": "0.0.0.0",
                "enable_mdns": True,
                "service_name": "lumen-ai",
                "config_path": "/path/to/config.yaml",
                "environment": "lumen_env",
            }
        }


class ServerStatus(BaseModel):
    """Server status information."""

    running: bool = False
    pid: int | None = None
    port: int = 50051
    host: str = "0.0.0.0"
    uptime_seconds: float | None = None
    service_name: str = "lumen-ai"
    config_path: str | None = None
    environment: str = "lumen_env"

    # Health status
    health: Literal["healthy", "unhealthy", "unknown"] = "unknown"
    last_error: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "running": True,
                "pid": 12345,
                "port": 50051,
                "host": "0.0.0.0",
                "uptime_seconds": 3600.0,
                "service_name": "lumen-ai",
                "health": "healthy",
            }
        }


class ServerLogs(BaseModel):
    """Server logs."""

    logs: list[str] = Field(default_factory=list)
    total_lines: int = 0
    new_lines: int = 0


class ServerStartRequest(BaseModel):
    """Request to start the server."""

    config_path: str | None = None
    port: int | None = None
    host: str | None = None
    environment: str = "lumen_env"


class ServerStopRequest(BaseModel):
    """Request to stop the server."""

    force: bool = False
    timeout: int = 30


class ServerRestartRequest(BaseModel):
    """Request to restart the server."""

    config_path: str | None = None
    port: int | None = None
    host: str | None = None
    environment: str = "lumen_env"
    force: bool = False
    timeout: int = 30
