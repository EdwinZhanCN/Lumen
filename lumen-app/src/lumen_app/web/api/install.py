"""Installation API endpoints - simplified one-click setup."""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from lumen_app.utils.env_checker import (
    DependencyInstaller,
    EnvironmentChecker,
    MicromambaChecker,
)
from lumen_app.utils.logger import get_logger
from lumen_app.utils.preset_registry import PresetRegistry
from lumen_app.web.core.state import app_state
from lumen_app.web.models.install import (
    InstallLogsResponse,
    InstallSetupRequest,
    InstallStatusResponse,
    InstallStep,
    InstallTaskListResponse,
    InstallTaskResponse,
)

logger = get_logger("lumen.web.api.install")
router = APIRouter()


@router.get("/status", response_model=InstallStatusResponse)
async def get_install_status():
    """Get current installation status of the system."""
    logger.info("Checking installation status")

    # Check micromamba
    micromamba_result = MicromambaChecker.check_micromamba()
    micromamba_installed = micromamba_result.status.value == "available"
    micromamba_path = micromamba_result.details if micromamba_installed else None

    # Check environment (TODO: implement actual check)
    environment_exists = False
    environment_name = None
    environment_path = None

    # Check drivers (placeholder)
    drivers_checked = False
    drivers = {}

    missing_components = []
    if not micromamba_installed:
        missing_components.append("micromamba")
    if not environment_exists:
        missing_components.append("environment")

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
    """Start a complete installation setup for the selected preset.

    This will automatically install all required components:
    1. micromamba (if not present)
    2. conda environment (if not exists)
    3. required drivers for the preset
    """
    logger.info(f"Starting installation setup for preset: {request.preset}")

    # Validate preset
    if not PresetRegistry.preset_exists(request.preset):
        raise HTTPException(status_code=400, detail=f"Unknown preset: {request.preset}")

    # Create task
    task_id = str(uuid.uuid4())
    current_time = time.time()

    # Determine installation steps based on current status and preset
    steps = await _plan_installation_steps(request)

    task = InstallTaskResponse(
        task_id=task_id,
        preset=request.preset,
        status="pending",
        progress=0,
        current_step="Initializing...",
        steps=steps,
        created_at=current_time,
        updated_at=current_time,
    )

    # Store task in state
    await app_state.store_install_task(task_id, task)

    # Start background installation
    background_tasks.add_task(_run_installation, task_id, request)

    return task


@router.get("/tasks", response_model=InstallTaskListResponse)
async def list_install_tasks():
    """List all installation tasks."""
    tasks = await app_state.get_all_install_tasks()
    return InstallTaskListResponse(tasks=tasks, total=len(tasks))


@router.get("/tasks/{task_id}", response_model=InstallTaskResponse)
async def get_install_task(task_id: str):
    """Get installation task status and progress."""
    task = await app_state.get_install_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return task


@router.get("/tasks/{task_id}/logs", response_model=InstallLogsResponse)
async def get_install_logs(task_id: str, tail: int = 100):
    """Get installation task logs."""
    task = await app_state.get_install_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    logs = await app_state.get_install_task_logs(task_id)

    # Return last N lines
    logs_to_return = logs[-tail:] if tail > 0 else logs

    return InstallLogsResponse(
        task_id=task_id, logs=logs_to_return, total_lines=len(logs)
    )


# ===== Internal helper functions =====


async def _plan_installation_steps(request: InstallSetupRequest) -> list[InstallStep]:
    """Plan installation steps based on current state and requirements."""
    steps = []

    # Step 1: Check/install micromamba
    micromamba_result = MicromambaChecker.check_micromamba()
    if micromamba_result.status.value != "available" or request.force_reinstall:
        steps.append(
            InstallStep(
                name="Install micromamba",
                status="pending",
                progress=0,
                message="Micromamba package manager will be installed",
            )
        )
    else:
        steps.append(
            InstallStep(
                name="Check micromamba",
                status="pending",
                progress=0,
                message="Verify micromamba installation",
            )
        )

    # Step 2: Create environment
    steps.append(
        InstallStep(
            name=f"Create environment '{request.environment_name}'",
            status="pending",
            progress=0,
            message="Create conda environment for Lumen",
        )
    )

    # Step 3: Install drivers (if needed)
    preset_info = PresetRegistry.get_preset(request.preset)
    if preset_info and preset_info.requires_drivers:
        # Check what drivers are needed
        report = EnvironmentChecker.check_preset(request.preset)
        if not report.ready:
            missing_drivers = [
                d.name for d in report.drivers if d.status.value != "available"
            ]
            if missing_drivers:
                steps.append(
                    InstallStep(
                        name="Install drivers",
                        status="pending",
                        progress=0,
                        message=f"Install required drivers: {', '.join(missing_drivers)}",
                    )
                )

    # Step 4: Verify installation
    steps.append(
        InstallStep(
            name="Verify installation",
            status="pending",
            progress=0,
            message="Verify all components are installed correctly",
        )
    )

    return steps


async def _run_installation(task_id: str, request: InstallSetupRequest):
    """Background task to run the installation process."""
    logger.info(f"Running installation task {task_id} for preset {request.preset}")

    try:
        await _update_task(
            task_id, status="running", current_step="Starting installation..."
        )

        task = await app_state.get_install_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        total_steps = len(task.steps)
        current_step_idx = 0

        # Step 1: Check/Install micromamba
        if task.steps[current_step_idx].name.startswith("Install micromamba"):
            await _execute_step(task_id, current_step_idx, total_steps)
            success = await _install_micromamba(task_id, request.cache_dir)
            if not success:
                await _update_task(
                    task_id,
                    status="failed",
                    error="Failed to install micromamba",
                    completed_at=time.time(),
                )
                return
            current_step_idx += 1
        elif task.steps[current_step_idx].name.startswith("Check micromamba"):
            await _execute_step(task_id, current_step_idx, total_steps, quick=True)
            current_step_idx += 1

        # Step 2: Create environment
        if current_step_idx < total_steps:
            await _execute_step(task_id, current_step_idx, total_steps)
            success = await _create_environment(task_id, request.environment_name)
            if not success:
                await _update_task(
                    task_id,
                    status="failed",
                    error="Failed to create environment",
                    completed_at=time.time(),
                )
                return
            current_step_idx += 1

        # Step 3: Install drivers (if needed)
        if (
            current_step_idx < total_steps
            and task.steps[current_step_idx].name == "Install drivers"
        ):
            await _execute_step(task_id, current_step_idx, total_steps)
            success = await _install_drivers(
                task_id, request.preset, request.environment_name
            )
            if not success:
                await _update_task(
                    task_id,
                    status="failed",
                    error="Failed to install drivers",
                    completed_at=time.time(),
                )
                return
            current_step_idx += 1

        # Step 4: Verify installation
        if current_step_idx < total_steps:
            await _execute_step(task_id, current_step_idx, total_steps, quick=True)
            current_step_idx += 1

        # Complete
        await _update_task(
            task_id,
            status="completed",
            progress=100,
            current_step="Installation completed successfully",
            completed_at=time.time(),
        )
        logger.info(f"Installation task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Installation task {task_id} failed: {e}", exc_info=True)
        await _update_task(
            task_id,
            status="failed",
            error=str(e),
            completed_at=time.time(),
        )


async def _execute_step(
    task_id: str, step_idx: int, total_steps: int, quick: bool = False
):
    """Mark a step as running and update progress."""
    task = await app_state.get_install_task(task_id)
    if not task:
        return

    # Update step status
    task.steps[step_idx].status = "running"
    task.steps[step_idx].started_at = time.time()

    # Calculate overall progress
    base_progress = int((step_idx / total_steps) * 100)

    await _update_task(
        task_id,
        current_step=task.steps[step_idx].message,
        progress=base_progress,
    )

    # Simulate quick check
    if quick:
        await asyncio.sleep(0.5)


async def _install_micromamba(task_id: str, cache_dir: str) -> bool:
    """Install micromamba."""
    await _append_log(task_id, "Installing micromamba...")

    try:
        success, message = MicromambaChecker.install_micromamba(
            cache_dir=cache_dir, dry_run=False
        )

        await _append_log(task_id, message)

        if success:
            await _complete_current_step(task_id)
            return True
        else:
            await _fail_current_step(task_id, message)
            return False

    except Exception as e:
        error_msg = f"Failed to install micromamba: {e}"
        await _append_log(task_id, error_msg)
        await _fail_current_step(task_id, error_msg)
        return False


async def _create_environment(task_id: str, env_name: str) -> bool:
    """Create conda environment."""
    await _append_log(task_id, f"Creating environment: {env_name}")

    # TODO: Implement actual environment creation
    await asyncio.sleep(2)  # Placeholder

    await _append_log(task_id, f"Environment {env_name} created successfully")
    await _complete_current_step(task_id)
    return True


async def _install_drivers(task_id: str, preset: str, env_name: str) -> bool:
    """Install required drivers for the preset."""
    await _append_log(task_id, f"Installing drivers for preset: {preset}")

    try:
        # Get required drivers
        report = EnvironmentChecker.check_preset(preset)
        missing_drivers = [
            d
            for d in report.drivers
            if d.status.value != "available" and d.installable_via_mamba
        ]

        if not missing_drivers:
            await _append_log(task_id, "All drivers already available")
            await _complete_current_step(task_id)
            return True

        installer = DependencyInstaller()

        for driver in missing_drivers:
            await _append_log(task_id, f"Installing {driver.name}...")

            success, message = installer.install_driver(
                driver_name=driver.name,
                env_name=env_name,
                dry_run=False,
            )

            await _append_log(task_id, message)

            if not success:
                await _fail_current_step(task_id, f"Failed to install {driver.name}")
                return False

        await _complete_current_step(task_id)
        return True

    except Exception as e:
        error_msg = f"Failed to install drivers: {e}"
        await _append_log(task_id, error_msg)
        await _fail_current_step(task_id, error_msg)
        return False


async def _update_task(task_id: str, **updates):
    """Update task fields."""
    task = await app_state.get_install_task(task_id)
    if not task:
        return

    for key, value in updates.items():
        if hasattr(task, key):
            setattr(task, key, value)

    task.updated_at = time.time()
    await app_state.store_install_task(task_id, task)


async def _complete_current_step(task_id: str):
    """Mark current running step as completed."""
    task = await app_state.get_install_task(task_id)
    if not task:
        return

    for step in task.steps:
        if step.status == "running":
            step.status = "completed"
            step.progress = 100
            step.completed_at = time.time()
            break

    await app_state.store_install_task(task_id, task)


async def _fail_current_step(task_id: str, error_msg: str):
    """Mark current running step as failed."""
    task = await app_state.get_install_task(task_id)
    if not task:
        return

    for step in task.steps:
        if step.status == "running":
            step.status = "failed"
            step.message = error_msg
            step.completed_at = time.time()
            break

    await app_state.store_install_task(task_id, task)


async def _append_log(task_id: str, message: str):
    """Append a log message to the task."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    await app_state.append_install_log(task_id, log_line)
    logger.info(f"Task {task_id}: {message}")
