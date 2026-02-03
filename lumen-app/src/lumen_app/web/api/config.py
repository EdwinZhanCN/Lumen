"""Configuration API endpoints."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from lumen_app.core.config import Config
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

        # Generate config based on selected services
        if len(request.selected_services) == 1 and "ocr" in request.selected_services:
            # Minimal config with just OCR
            lumen_config = config.minimal()
        elif set(request.selected_services) <= {"ocr", "clip", "face"}:
            # Light weight config
            clip_model = request.clip_model or "MobileCLIP2-S2"
            lumen_config = config.light_weight(clip_model=clip_model)
        else:
            # Basic config with all services
            clip_model = request.clip_model or "MobileCLIP2-S4"
            lumen_config = config.basic(clip_model=clip_model)

        # Save config to file
        config_path = Path(request.cache_dir).expanduser() / "lumen-config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        config_dict = lumen_config.model_dump()
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

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
    config, device_config = app_state.get_config()

    if not config or not device_config:
        return {"loaded": False, "message": "No configuration loaded"}

    return {
        "loaded": True,
        "cache_dir": config.cache_dir,
        "region": config.region,
        "port": config.port,
        "service_name": config.service_name,
        "device": {
            "runtime": device_config.runtime.value,
            "batch_size": device_config.batch_size,
            "precision": device_config.precision,
            "rknn_device": device_config.rknn_device,
            "onnx_providers": [
                p if isinstance(p, str) else p[0]
                for p in (device_config.onnx_providers or [])
            ],
        },
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
            "error": "路径不能为空"
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
            "warning": "磁盘空间不足 (建议至少 10GB)" if writable and free_space_gb < 10 else None
        }

    except Exception as e:
        logger.error(f"Failed to validate path: {e}")
        return {
            "valid": False,
            "exists": False,
            "writable": False,
            "error": f"路径验证失败: {str(e)}"
        }


@router.post("/load")
async def load_config(config_path: str):
    """Load a configuration from file."""
    try:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise HTTPException(
                status_code=404, detail=f"Config file not found: {config_path}"
            )

        with open(path) as f:
            config_content = yaml.safe_load(f)

        # TODO: Parse and validate the config, then set app_state

        return {"loaded": True, "config_path": str(path), "config": config_content}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")
