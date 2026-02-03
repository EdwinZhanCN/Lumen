"""Installation task API endpoints."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException

from lumen_app.utils.logger import get_logger
from lumen_app.web.core.state import app_state
from lumen_app.web.models.install import (
    InstallListResponse,
    InstallRequest,
    InstallStatus,
    InstallTask,
)

logger = get_logger("lumen.web.api.install")
router = APIRouter()


async def _run_installation_task(task_id: str, request: InstallRequest):
    """Background task for installation."""
    logger.info(f"Starting installation task {task_id} of type {request.task_type}")

    try:
        await app_state.update_task(task_id, status="running", progress=0)

        if request.task_type == "micromamba":
            await _install_micromamba(task_id, request)
        elif request.task_type == "environment":
            await _install_environment(task_id, request)
        elif request.task_type == "drivers":
            await _install_drivers(task_id, request)
        elif request.task_type == "packages":
            await _install_packages(task_id, request)
        else:
            raise ValueError(f"Unknown task type: {request.task_type}")

        await app_state.update_task(
            task_id, status="completed", progress=100, message="Installation completed"
        )
        logger.info(f"Installation task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Installation task {task_id} failed: {e}", exc_info=True)
        await app_state.update_task(
            task_id, status="failed", message=str(e), error_details=str(e)
        )


async def _install_micromamba(task_id: str, request: InstallRequest):
    """Install micromamba."""
    from lumen_app.utils.env_checker import MicromambaChecker

    cache_dir = request.options.get("cache_dir", "~/.lumen")

    await app_state.update_task(
        task_id, progress=10, message="Downloading micromamba..."
    )

    success, message = MicromambaChecker.install_micromamba(
        cache_dir=cache_dir, dry_run=request.options.get("dry_run", False)
    )

    if not success:
        raise RuntimeError(f"Failed to install micromamba: {message}")

    await app_state.update_task(
        task_id, progress=90, message="Finalizing installation..."
    )


async def _install_environment(task_id: str, request: InstallRequest):
    """Create conda environment."""
    await app_state.update_task(task_id, progress=20, message="Creating environment...")
    # TODO: Implement environment creation
    await asyncio.sleep(1)  # Placeholder


async def _install_drivers(task_id: str, request: InstallRequest):
    """Install drivers."""
    from lumen_app.utils.env_checker import DependencyInstaller

    drivers = request.drivers
    if not drivers:
        raise ValueError("No drivers specified for installation")

    await app_state.update_task(
        task_id, progress=10, message=f"Installing drivers: {', '.join(drivers)}..."
    )

    # TODO: Get actual micromamba path from config
    installer = DependencyInstaller()

    for i, driver in enumerate(drivers):
        progress = 10 + int((i / len(drivers)) * 80)
        await app_state.update_task(
            task_id, progress=progress, message=f"Installing {driver}..."
        )

        success, message = installer.install_driver(
            driver_name=driver,
            env_name=request.environment,
            dry_run=request.options.get("dry_run", False),
        )

        if not success:
            raise RuntimeError(f"Failed to install driver {driver}: {message}")

    await app_state.update_task(
        task_id, progress=95, message="Finalizing driver installation..."
    )


async def _install_packages(task_id: str, request: InstallRequest):
    """Install packages."""
    await app_state.update_task(task_id, progress=20, message="Installing packages...")
    # TODO: Implement package installation
    await asyncio.sleep(1)  # Placeholder


@router.post("/tasks", response_model=InstallStatus)
async def create_install_task(
    request: InstallRequest, background_tasks: BackgroundTasks
):
    """Create a new installation task."""
    logger.info(f"Creating install task: {request.task_type}")

    # Create task
    task = await app_state.create_task(request.task_type)

    # Start background task
    background_tasks.add_task(_run_installation_task, task.id, request)

    return InstallStatus(
        task_id=task.id,
        task_type=task.type,
        status=task.status,
        progress=task.progress,
        message=task.message,
        created_at=task.created_at,
    )


@router.get("/tasks/{task_id}", response_model=InstallStatus)
async def get_task_status(task_id: str):
    """Get installation task status."""
    task = await app_state.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return InstallStatus(
        task_id=task.id,
        task_type=task.type,
        status=task.status,
        progress=task.progress,
        message=task.message,
        created_at=task.created_at,
        updated_at=task.created_at,  # TODO: Track actual update time
        completed_at=task.created_at
        if task.status in ("completed", "failed")
        else None,
        error_details=None,  # TODO: Track error details
    )


@router.get("/tasks", response_model=InstallListResponse)
async def list_tasks():
    """List all installation tasks."""
    tasks = await app_state.get_all_tasks()

    return InstallListResponse(
        tasks=[
            InstallStatus(
                task_id=task.id,
                task_type=task.type,
                status=task.status,
                progress=task.progress,
                message=task.message,
                created_at=task.created_at,
            )
            for task in tasks
        ],
        total=len(tasks),
    )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running installation task."""
    task = await app_state.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.status not in ("pending", "running"):
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel task with status: {task.status}"
        )

    await app_state.update_task(
        task_id, status="cancelled", message="Task cancelled by user"
    )

    return {"success": True, "message": f"Task {task_id} cancelled"}
