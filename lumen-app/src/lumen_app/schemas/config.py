"""Configuration models."""

from __future__ import annotations

from typing import Literal, Union

from lumen_resources.lumen_config import Region
from pydantic import BaseModel, Field


class ConfigRequest(BaseModel):
    """Request to generate configuration.

    This matches the Config class constructor parameters:
    - cache_dir: str
    - device_config: DeviceConfig (created from preset)
    - region: Region
    - service_name: str
    - port: int | None

    Plus config generation method selection:
    - config_type: Literal["minimal", "light_weight", "basic", "brave"]
    - clip_model: Optional clip model for light_weight and basic configs
    """

    # Config constructor parameters
    cache_dir: str = "~/.lumen"
    preset: str  # Used to create device_config via PresetRegistry
    region: Region = Region.other
    service_name: str = "lumen-ai"
    port: int | None = 50051

    # Config generation method selection
    config_type: Literal["minimal", "light_weight", "basic", "brave"] = "minimal"
    clip_model: Union[
        Literal["MobileCLIP2-S2", "CN-CLIP_ViT-B-16"],  # For light_weight
        Literal["MobileCLIP2-S4", "CN-CLIP_ViT-L-14"],  # For basic
        None,
    ] = None

    class Config:
        json_schema_extra = {
            "example": {
                "cache_dir": "~/.lumen",
                "preset": "nvidia_gpu",
                "region": "other",
                "service_name": "lumen-ai",
                "port": 50051,
                "config_type": "light_weight",
                "clip_model": "MobileCLIP2-S2",
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
