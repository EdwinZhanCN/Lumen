"""Installation orchestration service for Lumen App."""

from __future__ import annotations

import asyncio
import shutil
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
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._task_requests: dict[str, InstallSetupRequest] = {}

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

        self._task_requests[task_id] = request
        self._cancel_events[task_id] = asyncio.Event()
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

    async def cancel_installation(self, task_id: str) -> InstallTaskResponse | None:
        """Request cancellation for an installation task."""
        task = await self.get_install_task(task_id)
        if not task:
            return None

        if task.status in ("completed", "failed", "cancelled"):
            return task

        cancel_event = self._cancel_events.setdefault(task_id, asyncio.Event())
        cancel_event.set()
        await self._append_log(task_id, "Cancellation requested by user.")

        request = self._task_requests.get(task_id)
        if task.status == "pending" and request is not None:
            await self._handle_cancellation(
                task_id, Path(request.cache_dir).expanduser().resolve()
            )
        else:
            await self._update_task(task_id, current_step="Cancelling installation...")

        return await self.get_install_task(task_id)

    async def run_installation(self, task_id: str, request: InstallSetupRequest):
        """Execute installation task in background."""
        logger.info(
            "Running installation task %s for preset %s", task_id, request.preset
        )

        resolved_cache_dir = Path(request.cache_dir).expanduser().resolve()
        self._task_requests[task_id] = request
        self._cancel_events.setdefault(task_id, asyncio.Event())

        try:
            if await self._handle_cancellation_if_requested(task_id, resolved_cache_dir):
                return

            await self._update_task(
                task_id, status="running", current_step="Starting installation..."
            )

            task = await self.get_install_task(task_id)
            if not task:
                logger.error("Task %s not found", task_id)
                return

            total_steps = len(task.steps)
            current_step_idx = 0

            # Step 1: Check/Install micromamba
            if task.steps[current_step_idx].step_id == self.STEP_INSTALL_MICROMAMBA:
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                await self._execute_step(task_id, current_step_idx, total_steps)
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                success = await self._install_micromamba(
                    task_id, str(resolved_cache_dir)
                )
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
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
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                await self._execute_step(
                    task_id, current_step_idx, total_steps, quick=True
                )
                current_step_idx += 1

            # Step 2: Create environment
            if (
                current_step_idx < total_steps
                and task.steps[current_step_idx].step_id == self.STEP_CREATE_ENVIRONMENT
            ):
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                await self._execute_step(task_id, current_step_idx, total_steps)
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                success = await self._create_environment(
                    task_id, request.environment_name, str(resolved_cache_dir)
                )
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
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
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                await self._execute_step(task_id, current_step_idx, total_steps)
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                success = await self._install_drivers(
                    task_id,
                    request.preset,
                    request.environment_name,
                    str(resolved_cache_dir),
                )
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
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
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                await self._execute_step(task_id, current_step_idx, total_steps)
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                success = await self._install_lumen_packages(
                    task_id, str(resolved_cache_dir), request.preset
                )
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
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
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                await self._execute_step(task_id, current_step_idx, total_steps)
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                success = await self._verify_installation(
                    task_id, str(resolved_cache_dir), request.environment_name
                )
                if await self._handle_cancellation_if_requested(
                    task_id, resolved_cache_dir
                ):
                    return
                if not success:
                    await self._update_task(
                        task_id,
                        status="failed",
                        error="Failed to verify installation",
                        completed_at=time.time(),
                    )
                    return
                current_step_idx += 1

            if await self._handle_cancellation_if_requested(task_id, resolved_cache_dir):
                return

            await self._update_task(
                task_id,
                status="completed",
                progress=100,
                current_step="Installation completed successfully",
                completed_at=time.time(),
            )
            logger.info("Installation task %s completed successfully", task_id)

        except Exception as e:
            if await self._handle_cancellation_if_requested(task_id, resolved_cache_dir):
                return
            logger.error("Installation task %s failed: %s", task_id, e, exc_info=True)
            await self._update_task(
                task_id,
                status="failed",
                error=str(e),
                completed_at=time.time(),
            )
        finally:
            self._cancel_events.pop(task_id, None)
            self._task_requests.pop(task_id, None)

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
        task.steps[step_idx].progress = 5

        await self._update_task(
            task_id,
            current_step=task.steps[step_idx].message,
            progress=self._calculate_task_progress(task.steps),
        )

        if quick:
            await asyncio.sleep(0.5)
            await self._complete_current_step(task_id)

    async def _install_micromamba(self, task_id: str, cache_dir: str) -> bool:
        """Install micromamba."""
        await self._append_log(task_id, "Installing micromamba...")
        await self._update_running_step(
            task_id,
            progress=20,
            message="Downloading and installing micromamba...",
        )

        try:
            installer = CoreInstaller(cache_dir=cache_dir)
            success, message = await asyncio.to_thread(
                installer.install_micromamba, False
            )

            await self._append_log(task_id, message)
            if success:
                await self._complete_current_step(task_id)
                return True

            await self._fail_current_step(task_id, message)
            return False
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
        await self._update_running_step(
            task_id,
            progress=20,
            message=f"Creating environment: {env_name}",
        )

        try:
            installer = CoreInstaller(cache_dir=cache_dir, env_name=env_name)
            success, message = await asyncio.to_thread(
                installer.create_environment,
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
        await self._update_running_step(
            task_id,
            progress=15,
            message=f"Checking required drivers for preset: {preset}",
        )

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

            total_drivers = len(missing_drivers)
            for idx, driver in enumerate(missing_drivers, start=1):
                await self._update_running_step(
                    task_id,
                    progress=20 + int(((idx - 1) / max(total_drivers, 1)) * 60),
                    message=f"Installing driver {idx}/{total_drivers}: {driver.name}",
                )
                await self._append_log(task_id, f"Installing {driver.name}...")
                success, message = await asyncio.to_thread(
                    installer.install_driver,
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
            await self._update_running_step(
                task_id,
                progress=10,
                message="Preparing package installation plan...",
            )

            installer = CoreInstaller(cache_dir=cache_dir, region=region)
            log_callback = self._create_threadsafe_log_callback(task_id)
            progress_callback = self._create_threadsafe_progress_callback(task_id)
            success, message = await asyncio.to_thread(
                installer.install_lumen_packages,
                lumen_config,
                device_config,
                True,
                log_callback,
                progress_callback,
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
        await self._update_running_step(
            task_id,
            progress=25,
            message="Verifying installation status...",
        )

        try:
            installer = CoreInstaller(cache_dir=cache_dir, env_name=env_name)
            success, message = await asyncio.to_thread(installer.verify_installation)
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

    def _calculate_task_progress(self, steps: list[InstallStep]) -> int:
        """Calculate overall task progress from individual step progress."""
        if not steps:
            return 0
        return int(sum(step.progress for step in steps) / len(steps))

    async def _update_running_step(
        self,
        task_id: str,
        *,
        progress: int | None = None,
        message: str | None = None,
    ) -> None:
        """Update the currently running step and sync aggregate task progress."""
        task = await self.get_install_task(task_id)
        if not task:
            return

        for step in task.steps:
            if step.status == "running":
                if progress is not None:
                    step.progress = progress
                if message is not None:
                    step.message = message
                break
        else:
            return

        task.progress = self._calculate_task_progress(task.steps)
        if message is not None:
            task.current_step = message
        task.updated_at = time.time()
        await self.task_repository.store_task(task_id, task)

    def _create_threadsafe_log_callback(self, task_id: str):
        """Create a sync log callback that can safely be used from worker threads."""
        loop = asyncio.get_running_loop()

        def callback(message: str) -> None:
            asyncio.run_coroutine_threadsafe(self._append_log(task_id, message), loop)

        return callback

    def _create_threadsafe_progress_callback(self, task_id: str):
        """Create a sync progress callback that can safely be used from worker threads."""
        loop = asyncio.get_running_loop()

        def callback(progress: int, message: str | None = None) -> None:
            asyncio.run_coroutine_threadsafe(
                self._update_running_step(task_id, progress=progress, message=message),
                loop,
            )

        return callback

    async def _handle_cancellation_if_requested(
        self, task_id: str, cache_dir: Path
    ) -> bool:
        """Cancel and clean installation state if the task has been cancelled."""
        cancel_event = self._cancel_events.get(task_id)
        if cancel_event is None or not cancel_event.is_set():
            return False

        task = await self.get_install_task(task_id)
        if task and task.status == "cancelled":
            return True

        await self._handle_cancellation(task_id, cache_dir)
        return True

    async def _handle_cancellation(self, task_id: str, cache_dir: Path) -> None:
        """Finalize task cancellation and clear installation artifacts."""
        await self._append_log(
            task_id, "Cancelling installation and clearing cache directory..."
        )
        cleanup_error = self._clear_cache_dir(cache_dir)

        task = await self.get_install_task(task_id)
        if not task:
            return

        for step in task.steps:
            if step.status == "running":
                step.status = "cancelled"
                step.message = "Cancelled by user"
                step.completed_at = time.time()
            elif step.status == "pending":
                step.status = "cancelled"
                step.message = "Cancelled before execution"
                step.completed_at = time.time()
            step.progress = 0

        task.status = "cancelled"
        task.progress = 0
        task.completed_at = time.time()
        task.error = cleanup_error
        if cleanup_error:
            task.current_step = "Installation cancelled, but cache cleanup failed"
            await self._append_log(task_id, cleanup_error)
        else:
            task.current_step = "Installation cancelled and cache directory cleared"
            await self._append_log(task_id, "Cache directory cleared successfully.")

        task.updated_at = time.time()
        await self.task_repository.store_task(task_id, task)

    def _clear_cache_dir(self, cache_dir: Path) -> str | None:
        """Delete all files under cache_dir while preserving the root directory."""
        resolved = cache_dir.expanduser().resolve()
        home_dir = Path.home().resolve()

        if resolved in (Path("/"), home_dir):
            return f"Refusing to clear unsafe cache directory: {resolved}"

        try:
            resolved.mkdir(parents=True, exist_ok=True)
            for child in resolved.iterdir():
                if child.is_symlink() or child.is_file():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child)
            return None
        except Exception as e:
            return f"Failed to clear cache directory {resolved}: {e}"

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

        task.progress = self._calculate_task_progress(task.steps)
        task.updated_at = time.time()
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
                step.progress = 100
                step.completed_at = time.time()
                break

        task.progress = self._calculate_task_progress(task.steps)
        task.current_step = error_msg
        task.updated_at = time.time()
        await self.task_repository.store_task(task_id, task)

    async def _append_log(self, task_id: str, message: str):
        """Append log message to installation task."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        await self.task_repository.append_log(task_id, log_line)
        logger.info("Task %s: %s", task_id, message)
