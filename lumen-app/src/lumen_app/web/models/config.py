"""Configuration models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConfigPreset(BaseModel):
    """Configuration preset for a specific hardware."""

    name: str
    description: str
    runtime: Literal["onnx", "rknn", "pytorch"]
    onnx_providers: list[str | dict[str, Any]] = Field(default_factory=list)
    batch_size: int | None = None
    precision: str | None = None
    requires_drivers: bool = True


class ServiceConfig(BaseModel):
    """Individual service configuration."""

    name: str
    enabled: bool = True
    package: str
    model: str
    batch_size: int = 1
    precision: str = "fp16"


class ServerSettings(BaseModel):
    """Server configuration settings."""

    port: int = 50051
    host: str = "0.0.0.0"
    enable_mdns: bool = True
    service_name: str = "lumen-ai"


class ConfigRequest(BaseModel):
    """Request to generate configuration."""

    preset: str
    region: str = "en"
    cache_dir: str = "~/.lumen"
    port: int = 50051
    service_name: str = "lumen-ai"
    selected_services: list[str] = Field(default_factory=lambda: ["ocr"])
    clip_model: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "preset": "nvidia_gpu",
                "region": "en",
                "cache_dir": "~/.lumen",
                "port": 50051,
                "selected_services": ["ocr", "clip"],
            }
        }


class ConfigResponse(BaseModel):
    """Generated configuration response."""

    success: bool
    preset: str
    config_path: str | None = None
    config_content: dict | None = None
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
