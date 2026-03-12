"""Installation orchestration service for Lumen App."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from lumen_resources import LumenConfig, load_and_validate_config

from lumen_app.schemas.install import (
    InstallLogsResponse,
    InstallSetupRequest,
    InstallStep,
    InstallTaskListResponse,
    InstallTaskResponse,
)
from lumen_app.services.install_task_repository import InstallTaskRepository
from lumen_app.services.installer import CoreInstaller
from lumen_app.utils.env_checker import (
    DependencyInstaller,
    EnvironmentChecker,
)
from lumen_app.utils.installation import MicromambaInstaller, MicromambaStatus
from lumen_app.utils.logger import get_logger
from lumen_app.utils.preset_registry import PresetRegistry

logger = get_logger("lumen.install.orchestrator")


class InstallOrchestrator:
    """Coordinate installation task planning, execution, and progress reporting."""

    STEP_INSTALL_MICROMAMBA = "install_micromamba"
    STEP_CHECK_MICROMAMBA = "check_micromamba"
    STEP_CREATE_ENVIRONMENT = "create_environment"
    STEP_INSTALL_DRIVERS = "install_drivers"
    STEP_INSTALL_PACKAGES = "install_lumen_packages"
    STEP_VERIFY_INSTALLATION = "verify_installation"

    def __init__(self, task_repository: InstallTaskRepository):
        self.task_repository = task_repository

    async def create_install_task(
        self, request: InstallSetupRequest
    ) -> InstallTaskResponse:
        """Create and persist a new installation task."""
        task_id = str(uuid.uuid4())
        current_time = time.time()

        steps = await self._plan_installation_steps(request)
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

        await self.task_repository.store_task(task_id, task)
        return task

    async def list_install_tasks(self) -> InstallTaskListResponse:
        """List all installation tasks."""
        tasks = await self.task_repository.list_tasks()
        return InstallTaskListResponse(tasks=tasks, total=len(tasks))

    async def get_install_task(self, task_id: str) -> InstallTaskResponse | None:
        """Get installation task by ID."""
        task = await self.task_repository.get_task(task_id)
        return task if isinstance(task, InstallTaskResponse) else None

    async def get_install_logs(
        self, task_id: str, tail: int = 100
    ) -> InstallLogsResponse:
        """Get installation logs for a task."""
        logs = await self.task_repository.get_logs(task_id)
        logs_to_return = logs[-tail:] if tail > 0 else logs
        return InstallLogsResponse(
            task_id=task_id, logs=logs_to_return, total_lines=len(logs)
        )

    async def run_installation(self, task_id: str, request: InstallSetupRequest):
        """Execute installation task in background."""
        logger.info(
            "Running installation task %s for preset %s", task_id, request.preset
        )

        try:
            await self._update_task(
                task_id, status="running", current_step="Starting installation..."
            )

            task = await self.get_install_task(task_id)
            if not task:
                logger.error("Task %s not found", task_id)
                return

            total_steps = len(task.steps)
            current_step_idx = 0
            resolved_cache_dir = str(Path(request.cache_dir).expanduser())

            # Step 1: Check/Install micromamba
            if task.steps[current_step_idx].step_id == self.STEP_INSTALL_MICROMAMBA:
                await self._execute_step(task_id, current_step_idx, total_steps)
                success = await self._install_micromamba(task_id, resolved_cache_dir)
                if not success:
                    await self._update_task(
                        task_id,
                        status="failed",
                        error="Failed to install micromamba",
                        completed_at=time.time(),
                    )
                    return
                current_step_idx += 1
            elif task.steps[current_step_idx].step_id == self.STEP_CHECK_MICROMAMBA:
                await self._execute_step(
                    task_id, current_step_idx, total_steps, quick=True
                )
                current_step_idx += 1

            # Step 2: Create environment
            if (
                current_step_idx < total_steps
                and task.steps[current_step_idx].step_id == self.STEP_CREATE_ENVIRONMENT
            ):
                await self._execute_step(task_id, current_step_idx, total_steps)
                success = await self._create_environment(
                    task_id, request.environment_name, resolved_cache_dir
                )
                if not success:
                    await self._update_task(
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
                and task.steps[current_step_idx].step_id == self.STEP_INSTALL_DRIVERS
            ):
                await self._execute_step(task_id, current_step_idx, total_steps)
                success = await self._install_drivers(
                    task_id,
                    request.preset,
                    request.environment_name,
                    resolved_cache_dir,
                )
                if not success:
                    await self._update_task(
                        task_id,
                        status="failed",
                        error="Failed to install drivers",
                        completed_at=time.time(),
                    )
                    return
                current_step_idx += 1

            # Step 4: Install Lumen packages
            if (
                current_step_idx < total_steps
                and task.steps[current_step_idx].step_id == self.STEP_INSTALL_PACKAGES
            ):
                await self._execute_step(task_id, current_step_idx, total_steps)
                success = await self._install_lumen_packages(
                    task_id, resolved_cache_dir, request.preset
                )
                if not success:
                    await self._update_task(
                        task_id,
                        status="failed",
                        error="Failed to install Lumen packages",
                        completed_at=time.time(),
                    )
                    return
                current_step_idx += 1

            # Step 5: Verify installation
            if (
                current_step_idx < total_steps
                and task.steps[current_step_idx].step_id
                == self.STEP_VERIFY_INSTALLATION
            ):
                await self._execute_step(task_id, current_step_idx, total_steps)
                success = await self._verify_installation(
                    task_id, resolved_cache_dir, request.environment_name
                )
                if not success:
                    await self._update_task(
                        task_id,
                        status="failed",
                        error="Failed to verify installation",
                        completed_at=time.time(),
                    )
                    return
                current_step_idx += 1

            await self._update_task(
                task_id,
                status="completed",
                progress=100,
                current_step="Installation completed successfully",
                completed_at=time.time(),
            )
            logger.info("Installation task %s completed successfully", task_id)

        except Exception as e:
            logger.error("Installation task %s failed: %s", task_id, e, exc_info=True)
            await self._update_task(
                task_id,
                status="failed",
                error=str(e),
                completed_at=time.time(),
            )

    async def _plan_installation_steps(
        self, request: InstallSetupRequest
    ) -> list[InstallStep]:
        """Plan installation steps based on current state and requirements."""
        steps = []

        resolved_cache_dir = Path(request.cache_dir).expanduser()
        micromamba_installer = MicromambaInstaller(resolved_cache_dir)
        micromamba_result = micromamba_installer.check()
        if (
            micromamba_result.status != MicromambaStatus.INSTALLED
            or request.force_reinstall
        ):
            steps.append(
                InstallStep(
                    step_id=self.STEP_INSTALL_MICROMAMBA,
                    name="Install micromamba",
                    status="pending",
                    progress=0,
                    message="Micromamba package manager will be installed",
                )
            )
        else:
            steps.append(
                InstallStep(
                    step_id=self.STEP_CHECK_MICROMAMBA,
                    name="Check micromamba",
                    status="pending",
                    progress=0,
                    message="Verify micromamba installation",
                )
            )

        steps.append(
            InstallStep(
                step_id=self.STEP_CREATE_ENVIRONMENT,
                name=f"Create environment '{request.environment_name}'",
                status="pending",
                progress=0,
                message="Create conda environment for Lumen",
            )
        )

        preset_info = PresetRegistry.get_preset(request.preset)
        if preset_info and preset_info.requires_drivers:
            report = EnvironmentChecker.check_preset(request.preset)
            if not report.ready:
                missing_drivers = [
                    d.name for d in report.drivers if d.status.value != "available"
                ]
                if missing_drivers:
                    steps.append(
                        InstallStep(
                            step_id=self.STEP_INSTALL_DRIVERS,
                            name="Install drivers",
                            status="pending",
                            progress=0,
                            message=f"Install required drivers: {', '.join(missing_drivers)}",
                        )
                    )

        steps.append(
            InstallStep(
                step_id=self.STEP_INSTALL_PACKAGES,
                name="Install Lumen packages",
                status="pending",
                progress=0,
                message="Install required Lumen packages",
            )
        )
        steps.append(
            InstallStep(
                step_id=self.STEP_VERIFY_INSTALLATION,
                name="Verify installation",
                status="pending",
                progress=0,
                message="Verify all components are installed correctly",
            )
        )

        return steps

    async def _execute_step(
        self, task_id: str, step_idx: int, total_steps: int, quick: bool = False
    ):
        """Mark step as running and update progress."""
        task = await self.get_install_task(task_id)
        if not task:
            return

        task.steps[step_idx].status = "running"
        task.steps[step_idx].started_at = time.time()

        base_progress = int((step_idx / total_steps) * 100)
        await self._update_task(
            task_id,
            current_step=task.steps[step_idx].message,
            progress=base_progress,
        )

        if quick:
            await asyncio.sleep(0.5)
            await self._complete_current_step(task_id)

    async def _install_micromamba(self, task_id: str, cache_dir: str) -> bool:
        """Install micromamba."""
        await self._append_log(task_id, "Installing micromamba...")

        try:
            micromamba_installer = MicromambaInstaller(cache_dir)
            exe_path = micromamba_installer.install(dry_run=False)

            await self._append_log(
                task_id, f"Micromamba installed successfully at {exe_path}"
            )
            await self._complete_current_step(task_id)
            return True
        except Exception as e:
            error_msg = f"Failed to install micromamba: {e}"
            await self._append_log(task_id, error_msg)
            await self._fail_current_step(task_id, error_msg)
            return False

    async def _create_environment(
        self, task_id: str, env_name: str, cache_dir: str
    ) -> bool:
        """Create conda environment."""
        await self._append_log(task_id, f"Creating environment: {env_name}")

        try:
            installer = CoreInstaller(cache_dir=cache_dir, env_name=env_name)
            success, message = installer.create_environment(
                config_filename="default.yaml",
                dry_run=False,
            )
            await self._append_log(task_id, message)

            if success:
                await self._complete_current_step(task_id)
                return True

            await self._fail_current_step(task_id, message)
            return False
        except Exception as e:
            error_msg = f"Failed to create environment: {e}"
            await self._append_log(task_id, error_msg)
            await self._fail_current_step(task_id, error_msg)
            return False

    async def _install_drivers(
        self, task_id: str, preset: str, env_name: str, cache_dir: str
    ) -> bool:
        """Install required drivers for preset."""
        await self._append_log(task_id, f"Installing drivers for preset: {preset}")

        try:
            report = EnvironmentChecker.check_preset(preset)
            missing_drivers = [
                d
                for d in report.drivers
                if d.status.value != "available" and d.installable_via_mamba
            ]

            if not missing_drivers:
                await self._append_log(task_id, "All drivers already available")
                await self._complete_current_step(task_id)
                return True

            micromamba_installer = MicromambaInstaller(cache_dir)
            micromamba_path = str(micromamba_installer.get_executable())
            root_prefix = str(Path(cache_dir).expanduser() / "micromamba")
            installer = DependencyInstaller(
                micromamba_path=micromamba_path,
                root_prefix=root_prefix,
            )

            for driver in missing_drivers:
                await self._append_log(task_id, f"Installing {driver.name}...")
                success, message = installer.install_driver(
                    driver_name=driver.name,
                    env_name=env_name,
                    dry_run=False,
                )
                await self._append_log(task_id, message)

                if not success:
                    await self._fail_current_step(
                        task_id, f"Failed to install {driver.name}"
                    )
                    return False

            await self._complete_current_step(task_id)
            return True
        except Exception as e:
            error_msg = f"Failed to install drivers: {e}"
            await self._append_log(task_id, error_msg)
            await self._fail_current_step(task_id, error_msg)
            return False

    async def _install_lumen_packages(
        self, task_id: str, cache_dir: str, preset: str
    ) -> bool:
        """Install Lumen packages derived from lumen-config.yaml."""
        await self._append_log(
            task_id, "Installing Lumen packages from GitHub Releases..."
        )

        try:
            config_path = Path(cache_dir).expanduser() / "lumen-config.yaml"
            if not config_path.exists():
                message = f"Config file not found: {config_path}"
                await self._append_log(task_id, message)
                await self._fail_current_step(task_id, message)
                return False

            lumen_config: LumenConfig = load_and_validate_config(str(config_path))

            device_config = PresetRegistry.create_config(preset)
            await self._append_log(task_id, f"Using preset: {preset}")

            region = lumen_config.metadata.region
            await self._append_log(task_id, f"Using region: {region.value}")

            installer = CoreInstaller(cache_dir=cache_dir, region=region)
            success, message = installer.install_lumen_packages(
                lumen_config, device_config
            )
            await self._append_log(task_id, message)

            if success:
                await self._complete_current_step(task_id)
                return True

            await self._fail_current_step(task_id, message)
            return False
        except Exception as e:
            error_msg = f"Failed to install Lumen packages: {e}"
            await self._append_log(task_id, error_msg)
            await self._fail_current_step(task_id, error_msg)
            return False

    async def _verify_installation(
        self, task_id: str, cache_dir: str, env_name: str
    ) -> bool:
        """Verify installation status."""
        await self._append_log(task_id, "Verifying installation...")

        try:
            installer = CoreInstaller(cache_dir=cache_dir, env_name=env_name)
            success, message = installer.verify_installation()
            await self._append_log(task_id, message)

            if success:
                await self._complete_current_step(task_id)
                return True

            await self._fail_current_step(task_id, message)
            return False
        except Exception as e:
            error_msg = f"Failed to verify installation: {e}"
            await self._append_log(task_id, error_msg)
            await self._fail_current_step(task_id, error_msg)
            return False

    async def _update_task(self, task_id: str, **updates):
        """Update installation task fields."""
        task = await self.get_install_task(task_id)
        if not task:
            return

        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        task.updated_at = time.time()
        await self.task_repository.store_task(task_id, task)

    async def _complete_current_step(self, task_id: str):
        """Mark current running step as completed."""
        task = await self.get_install_task(task_id)
        if not task:
            return

        for step in task.steps:
            if step.status == "running":
                step.status = "completed"
                step.progress = 100
                step.completed_at = time.time()
                break

        await self.task_repository.store_task(task_id, task)

    async def _fail_current_step(self, task_id: str, error_msg: str):
        """Mark current running step as failed."""
        task = await self.get_install_task(task_id)
        if not task:
            return

        for step in task.steps:
            if step.status == "running":
                step.status = "failed"
                step.message = error_msg
                step.completed_at = time.time()
                break

        await self.task_repository.store_task(task_id, task)

    async def _append_log(self, task_id: str, message: str):
        """Append log message to installation task."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        await self.task_repository.append_log(task_id, log_line)
        logger.info("Task %s: %s", task_id, message)
