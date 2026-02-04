"""Configuration API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from lumen_resources.exceptions import ConfigError
from lumen_resources.lumen_config_validator import load_and_validate_config

from lumen_app.core.config import Config
from lumen_app.core.installer import CoreInstaller
from lumen_app.utils.logger import get_logger
from lumen_app.utils.preset_registry import PresetRegistry
from lumen_app.web.core.state import app_state
from lumen_app.web.models.config import (
    ConfigRequest,
    ConfigResponse,
)

logger = get_logger("lumen.web.api.config")
router = APIRouter()


@router.post("/generate", response_model=ConfigResponse)
async def generate_config(request: ConfigRequest):
    """Generate a Lumen configuration from preset and options."""
    logger.info(f"Generating config for preset: {request.preset}")

    # Validate preset
    if not PresetRegistry.preset_exists(request.preset):
        return ConfigResponse(
            success=False,
            preset=request.preset,
            message=f"Unknown preset: {request.preset}",
        )

    try:
        # Create device config from preset
        device_config = PresetRegistry.create_config(request.preset)

        # Create Config instance
        config = Config(
            cache_dir=request.cache_dir,
            device_config=device_config,
            region=request.region,
            service_name=request.service_name,
            port=request.port,
        )

        # Generate config based on config_type
        if request.config_type == "minimal":
            lumen_config = config.minimal()
        elif request.config_type == "light_weight":
            # Ensure clip_model is one of the valid types for light_weight
            light_clip_model: Literal["MobileCLIP2-S2", "CN-CLIP_ViT-B-16"] = (
                request.clip_model  # type: ignore
                if request.clip_model in ["MobileCLIP2-S2", "CN-CLIP_ViT-B-16"]
                else "MobileCLIP2-S2"
            )
            lumen_config = config.light_weight(clip_model=light_clip_model)
        elif request.config_type == "basic":
            # Ensure clip_model is one of the valid types for basic
            basic_clip_model: Literal["MobileCLIP2-S4", "CN-CLIP_ViT-L-14"] = (
                request.clip_model  # type: ignore
                if request.clip_model in ["MobileCLIP2-S4", "CN-CLIP_ViT-L-14"]
                else "MobileCLIP2-S4"
            )
            lumen_config = config.basic(clip_model=basic_clip_model)
        elif request.config_type == "brave":
            lumen_config = config.brave()
        else:
            raise ValueError(f"Unknown config_type: {request.config_type}")

        # Save config to file
        config_dict = lumen_config.model_dump(mode="json")
        installer = CoreInstaller(cache_dir=request.cache_dir)
        success, message = installer.save_config(lumen_config)

        if not success:
            raise ValueError(message)

        config_path = str(Path(request.cache_dir).expanduser() / "lumen-config.yaml")

        # Update app state
        app_state.set_config(config, device_config)

        return ConfigResponse(
            success=True,
            preset=request.preset,
            config_path=str(config_path),
            config_content=config_dict,
            message=f"Configuration generated successfully at {config_path}",
        )

    except Exception as e:
        logger.error(f"Failed to generate config: {e}", exc_info=True)
        return ConfigResponse(
            success=False,
            preset=request.preset,
            message=f"Failed to generate configuration: {str(e)}",
        )


@router.get("/current")
async def get_current_config():
    """Get the currently loaded configuration."""
    lumen_config = app_state.get_lumen_config()

    if not lumen_config:
        return {"loaded": False, "message": "No configuration loaded"}

    # Extract first enabled service's backend settings for device info
    device_info = None
    for service_config in lumen_config.services.values():
        if service_config.enabled and service_config.backend_settings:
            backend = service_config.backend_settings
            # Get runtime from first model
            runtime = "onnx"
            for model_cfg in service_config.models.values():
                runtime = model_cfg.runtime.value
                break

            device_info = {
                "runtime": runtime,
                "batch_size": backend.batch_size or 1,
                "precision": "fp32",
                "rknn_device": None,
                "onnx_providers": backend.onnx_providers or [],
            }
            break

    return {
        "loaded": True,
        "cache_dir": lumen_config.metadata.cache_dir,
        "region": lumen_config.metadata.region.value,
        "port": lumen_config.server.port,
        "service_name": lumen_config.server.mdns.service_name
        if lumen_config.server.mdns
        else "lumen-server",
        "device": device_info,
    }


@router.post("/validate")
async def validate_config(config: dict):
    """Validate a configuration."""
    # TODO: Implement validation logic
    return {"valid": True, "errors": [], "warnings": []}


@router.post("/validate-path")
async def validate_path(request: dict):
    """Validate installation path for permissions and disk space."""
    path_str = request.get("path", "")

    if not path_str:
        return {
            "valid": False,
            "exists": False,
            "writable": False,
            "error": "路径不能为空",
        }

    try:
        import os
        import shutil

        # Expand user home directory
        path = Path(path_str).expanduser()

        # Check if path exists
        exists = path.exists()

        # Check if writable
        writable = False
        error = None

        if exists:
            # Check if we can write to existing directory
            writable = os.access(path, os.W_OK)
            if not writable:
                error = f"没有写入权限: {path}"
        else:
            # Check if we can create the directory
            parent = path.parent
            if parent.exists():
                writable = os.access(parent, os.W_OK)
                if not writable:
                    error = f"没有创建目录的权限: {parent}"
            else:
                error = f"父目录不存在: {parent}"

        # Check disk space (only if writable)
        free_space_gb = 0
        if writable:
            stat = shutil.disk_usage(path if exists else path.parent)
            free_space_gb = stat.free / (1024**3)  # Convert to GB

        return {
            "valid": writable and free_space_gb >= 10,
            "exists": exists,
            "writable": writable,
            "free_space_gb": round(free_space_gb, 2),
            "error": error,
            "warning": "磁盘空间不足 (建议至少 10GB)"
            if writable and free_space_gb < 10
            else None,
        }

    except Exception as e:
        logger.error(f"Failed to validate path: {e}")
        return {
            "valid": False,
            "exists": False,
            "writable": False,
            "error": f"路径验证失败: {str(e)}",
        }


@router.post("/load")
async def load_config(config_path: str):
    """Load and validate a configuration from file, then set it in app_state."""
    try:
        path = Path(config_path).expanduser().resolve()

        if not path.exists():
            raise HTTPException(
                status_code=404, detail=f"Config file not found: {config_path}"
            )

        # Validate and load using lumen-resources
        try:
            lumen_config = load_and_validate_config(path)
        except ConfigError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid configuration: {str(e)}"
            )

        # Store LumenConfig directly in app_state
        app_state.set_lumen_config(lumen_config)

        logger.info(f"Configuration loaded successfully from {path}")

        return {
            "loaded": True,
            "config_path": str(path),
            "cache_dir": lumen_config.metadata.cache_dir,
            "region": lumen_config.metadata.region.value,
            "port": lumen_config.server.port,
            "service_name": lumen_config.server.mdns.service_name
            if lumen_config.server.mdns
            else "lumen-server",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")


@router.get("/yaml")
async def get_config_yaml():
    """Get the current configuration as raw YAML string."""
    lumen_config = app_state.get_lumen_config()

    if not lumen_config:
        raise HTTPException(status_code=404, detail="No configuration loaded")

    try:
        import yaml

        # Convert Pydantic model to dict and then to YAML
        config_dict = lumen_config.model_dump(mode="json")
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        return {
            "loaded": True,
            "yaml": yaml_str,
            "cache_dir": lumen_config.metadata.cache_dir,
        }

    except Exception as e:
        logger.error(f"Failed to serialize config to YAML: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to serialize config: {str(e)}"
        )
