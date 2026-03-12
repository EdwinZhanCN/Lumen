"""Installation API endpoints - simplified one-click setup."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from lumen_app.schemas.install import (
    CheckInstallationPathResponse,
    InstallLogsResponse,
    InstallSetupRequest,
    InstallStatusResponse,
    InstallTaskListResponse,
    InstallTaskResponse,
    ServiceStatus,
)
from lumen_app.services.install_orchestrator import InstallOrchestrator
from lumen_app.services.state import app_state
from lumen_app.utils.installation import MicromambaInstaller, MicromambaStatus
from lumen_app.utils.logger import get_logger
from lumen_app.utils.preset_registry import PresetRegistry

logger = get_logger("lumen.web.api.install")
router = APIRouter()
install_orchestrator = InstallOrchestrator(
    task_repository=app_state.install_task_repository
)


def _detect_environment(cache_dir: Path) -> tuple[bool, str | None, str | None]:
    """Detect available micromamba environment under cache_dir."""
    envs_dir = cache_dir / "micromamba" / "envs"
    if not envs_dir.exists() or not envs_dir.is_dir():
        return False, None, None

    try:
        env_names = sorted([item.name for item in envs_dir.iterdir() if item.is_dir()])
    except OSError:
        return False, None, None

    if not env_names:
        return False, None, None

    preferred_name = "lumen_env" if "lumen_env" in env_names else env_names[0]
    preferred_path = str((envs_dir / preferred_name).resolve())
    return True, preferred_name, preferred_path


def _check_installation_components(cache_dir: Path) -> dict[str, bool]:
    """Check which installation components exist at the given path."""
    components = {
        "micromamba": False,
        "environment": False,
        "config": False,
        "drivers": False,
    }

    micromamba_installer = MicromambaInstaller(cache_dir)
    micromamba_result = micromamba_installer.check()
    components["micromamba"] = micromamba_result.status == MicromambaStatus.INSTALLED

    envs_dir = cache_dir / "micromamba" / "envs"
    if envs_dir.exists():
        components["environment"] = (
            any(
                (envs_dir / d).is_dir()
                for d in os.listdir(envs_dir)
                if (envs_dir / d).is_dir()
            )
            if os.path.exists(envs_dir)
            else False
        )

    config_path = cache_dir / "lumen-config.yaml"
    components["config"] = config_path.exists()

    # Placeholder: assume drivers are okay if runtime environment exists.
    components["drivers"] = components["micromamba"] and components["environment"]

    return components


@router.get("/check-path", response_model=CheckInstallationPathResponse)
async def check_installation_path(path: str) -> CheckInstallationPathResponse:
    """Check if an existing installation exists at the given path."""
    if not path or not path.strip():
        return CheckInstallationPathResponse(
            has_existing_service=False,
            ready_to_start=False,
            recommended_action="configure_new",
            message="路径为空",
        )

    try:
        cache_dir = Path(path).expanduser().resolve()

        if not str(cache_dir).startswith(("/", "~")) and os.name != "nt":
            if not cache_dir.is_absolute():
                return CheckInstallationPathResponse(
                    has_existing_service=False,
                    ready_to_start=False,
                    recommended_action="configure_new",
                    message="请使用绝对路径",
                )

        components = _check_installation_components(cache_dir)
        has_existing_service = components["micromamba"] and components["environment"]
        ready_to_start = (
            components["micromamba"]
            and components["environment"]
            and components["config"]
        )

        if ready_to_start:
            recommended_action = "start_existing"
            message = "检测到完整现有安装，可以直接启动服务"
        elif has_existing_service:
            recommended_action = "configure_new"
            message = "检测到部分安装，建议重新配置"
        else:
            recommended_action = "configure_new"
            message = "未检测到现有安装，将进行全新配置"

        return CheckInstallationPathResponse(
            has_existing_service=has_existing_service,
            service_status=ServiceStatus(**components),
            ready_to_start=ready_to_start,
            recommended_action=recommended_action,
            message=message,
        )
    except Exception as e:
        logger.error("Error checking installation path: %s", e, exc_info=True)
        return CheckInstallationPathResponse(
            has_existing_service=False,
            ready_to_start=False,
            recommended_action="configure_new",
            message=f"检查路径时出错: {str(e)}",
        )


@router.get("/status", response_model=InstallStatusResponse)
async def get_install_status(cache_dir: str = "~/.lumen"):
    """Get current installation status of the system."""
    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    logger.info("Checking installation status for cache_dir=%s", resolved_cache_dir)

    micromamba_installer = MicromambaInstaller(resolved_cache_dir)
    micromamba_result = micromamba_installer.check()
    micromamba_installed = micromamba_result.status == MicromambaStatus.INSTALLED
    micromamba_path = (
        micromamba_result.executable_path if micromamba_installed else None
    )

    environment_exists, environment_name, environment_path = _detect_environment(
        resolved_cache_dir
    )

    drivers_checked = False
    drivers = {}

    config_exists = (resolved_cache_dir / "lumen-config.yaml").exists()
    missing_components = []
    if not micromamba_installed:
        missing_components.append("micromamba")
    if not environment_exists:
        missing_components.append("environment")
    if not config_exists:
        missing_components.append("config")

    return InstallStatusResponse(
        micromamba_installed=micromamba_installed,
        micromamba_path=micromamba_path,
        environment_exists=environment_exists,
        environment_name=environment_name,
        environment_path=environment_path,
        drivers_checked=drivers_checked,
        drivers=drivers,
        ready_for_preset=None if missing_components else "cpu",
        missing_components=missing_components,
    )


@router.post("/setup", response_model=InstallTaskResponse)
async def start_installation(
    request: InstallSetupRequest, background_tasks: BackgroundTasks
):
    """Start a complete installation setup for the selected preset."""
    logger.info("Starting installation setup for preset: %s", request.preset)

    if not PresetRegistry.preset_exists(request.preset):
        raise HTTPException(status_code=400, detail=f"Unknown preset: {request.preset}")

    config_path = Path(request.cache_dir).expanduser() / "lumen-config.yaml"
    if not config_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Config file not found: {config_path}. 请先在上一步生成配置。",
        )

    task = await install_orchestrator.create_install_task(request)
    background_tasks.add_task(
        install_orchestrator.run_installation, task.task_id, request
    )
    return task


@router.get("/tasks", response_model=InstallTaskListResponse)
async def list_install_tasks():
    """List all installation tasks."""
    return await install_orchestrator.list_install_tasks()


@router.get("/tasks/{task_id}", response_model=InstallTaskResponse)
async def get_install_task(task_id: str):
    """Get installation task status and progress."""
    task = await install_orchestrator.get_install_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return task


@router.get("/tasks/{task_id}/logs", response_model=InstallLogsResponse)
async def get_install_logs(task_id: str, tail: int = 100):
    """Get installation task logs."""
    task = await install_orchestrator.get_install_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return await install_orchestrator.get_install_logs(task_id, tail=tail)
