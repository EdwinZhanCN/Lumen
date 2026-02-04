"""Application state management for Lumen Web API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from lumen_resources.lumen_config import LumenConfig

from lumen_app.core.config import Config, DeviceConfig
from lumen_app.utils.env_checker import EnvironmentReport
from lumen_app.utils.logger import get_logger
from lumen_app.web.core.server_manager import ServerManager

# Optional import - AppService may not be available if gRPC is not installed
try:
    from lumen_app.core.service import AppService
except ImportError:
    AppService = None  # type: ignore[misc,assignment]

logger = get_logger("lumen.web.state")


@dataclass
class ServerStatus:
    """Server process status."""

    running: bool = False
    pid: int | None = None
    port: int = 50051
    config_path: str | None = None
    logs: list[dict] = field(default_factory=list)


@dataclass
class InstallationTask:
    """Installation task status."""

    id: str
    type: str  # "micromamba", "environment", "drivers"
    status: str  # "pending", "running", "completed", "failed"
    progress: int = 0
    message: str = ""
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class AppState:
    """Global application state."""

    def __init__(self):
        self._initialized = False
        self.current_config: Config | None = None
        self.device_config: DeviceConfig | None = None
        self.lumen_config: LumenConfig | None = None
        self.config_path: str | None = None
        self.cache_dir: str | None = None
        self.environment_report: EnvironmentReport | None = None
        self.server_status = ServerStatus()
        self.app_service: AppService | None = None

        # Server manager for gRPC ML server
        self.server_manager = ServerManager()

        # Installation tasks (old format - to be deprecated)
        self._tasks: dict[str, InstallationTask] = {}
        self._task_lock = asyncio.Lock()

        # New installation tasks
        self._install_tasks: dict[str, Any] = {}  # task_id -> InstallTaskResponse
        self._install_logs: dict[str, list[str]] = {}  # task_id -> logs
        self._install_lock = asyncio.Lock()

        # Log subscribers
        self._log_queues: list[asyncio.Queue] = []
        self._log_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize application state."""
        if self._initialized:
            return

        logger.info("Initializing application state")

        # Load default configuration
        try:
            # Start with CPU preset as default
            self.device_config = DeviceConfig.cpu()
            logger.info("Loaded default CPU configuration")
        except Exception as e:
            logger.error(f"Failed to load default configuration: {e}")

        self._initialized = True
        logger.info("Application state initialized")

    async def cleanup(self):
        """Cleanup application state."""
        logger.info("Cleaning up application state")

        # Stop server if running
        if self.server_status.running:
            await self.stop_server()

        # Clear log queues
        async with self._log_lock:
            for queue in self._log_queues:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            self._log_queues.clear()

        self._initialized = False
        logger.info("Application state cleaned up")

    # Configuration methods
    def set_config(self, config: Config, device_config: DeviceConfig):
        """Set current configuration."""
        self.current_config = config
        self.device_config = device_config
        logger.info(f"Configuration updated: {device_config}")

    def get_config(self) -> tuple[Config | None, DeviceConfig | None]:
        """Get current configuration."""
        return self.current_config, self.device_config

    def set_lumen_config(self, config: LumenConfig, config_path: str | None = None):
        """Set LumenConfig directly."""
        self.lumen_config = config
        self.config_path = config_path
        self.cache_dir = config.metadata.cache_dir
        self.server_manager.update_cache_dir(config.metadata.cache_dir)
        logger.info(f"LumenConfig loaded: {config.metadata.cache_dir}")

    def get_lumen_config(self) -> LumenConfig | None:
        """Get current LumenConfig."""
        return self.lumen_config

    # Task management
    async def create_task(self, task_type: str) -> InstallationTask:
        """Create a new installation task."""
        import uuid

        task_id = str(uuid.uuid4())
        task = InstallationTask(
            id=task_id,
            type=task_type,
            status="pending",
        )

        async with self._task_lock:
            self._tasks[task_id] = task

        logger.info(f"Created task {task_id} of type {task_type}")
        return task

    async def update_task(
        self,
        task_id: str,
        status: str | None = None,
        progress: int | None = None,
        message: str | None = None,
    ):
        """Update task status."""
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                if status:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if message:
                    task.message = message

    async def get_task(self, task_id: str) -> InstallationTask | None:
        """Get task by ID."""
        async with self._task_lock:
            return self._tasks.get(task_id)

    async def get_all_tasks(self) -> list[InstallationTask]:
        """Get all tasks."""
        async with self._task_lock:
            return list(self._tasks.values())

    # New install task management
    async def store_install_task(self, task_id: str, task: Any):
        """Store or update an installation task."""
        async with self._install_lock:
            self._install_tasks[task_id] = task
            if task_id not in self._install_logs:
                self._install_logs[task_id] = []

    async def get_install_task(self, task_id: str) -> Any | None:
        """Get installation task by ID."""
        async with self._install_lock:
            return self._install_tasks.get(task_id)

    async def get_all_install_tasks(self) -> list[Any]:
        """Get all installation tasks."""
        async with self._install_lock:
            return list(self._install_tasks.values())

    async def append_install_log(self, task_id: str, log_line: str):
        """Append a log line to installation task."""
        async with self._install_lock:
            if task_id not in self._install_logs:
                self._install_logs[task_id] = []
            self._install_logs[task_id].append(log_line)

    async def get_install_task_logs(self, task_id: str) -> list[str]:
        """Get installation task logs."""
        async with self._install_lock:
            return self._install_logs.get(task_id, [])

    # Log streaming
    async def subscribe_logs(self) -> asyncio.Queue:
        """Subscribe to log stream."""
        queue = asyncio.Queue(maxsize=1000)
        async with self._log_lock:
            self._log_queues.append(queue)
        logger.debug(f"New log subscriber, total: {len(self._log_queues)}")
        return queue

    async def unsubscribe_logs(self, queue: asyncio.Queue):
        """Unsubscribe from log stream."""
        async with self._log_lock:
            if queue in self._log_queues:
                self._log_queues.remove(queue)
        logger.debug(f"Log subscriber removed, remaining: {len(self._log_queues)}")

    async def broadcast_log(self, log_entry: dict):
        """Broadcast log to all subscribers."""
        async with self._log_lock:
            dead_queues = []
            for queue in self._log_queues:
                try:
                    queue.put_nowait(log_entry)
                except asyncio.QueueFull:
                    # Remove oldest log if queue is full
                    try:
                        queue.get_nowait()
                        queue.put_nowait(log_entry)
                    except asyncio.QueueEmpty:
                        pass
                except Exception:
                    dead_queues.append(queue)

            # Remove dead queues
            for queue in dead_queues:
                if queue in self._log_queues:
                    self._log_queues.remove(queue)

    # Server management
    async def start_server(self, config_path: str | None = None) -> bool:
        """Start the ML server."""
        if self.server_status.running:
            logger.warning("Server is already running")
            return False

        logger.info("Starting ML server")
        # TODO: Implement actual server startup logic
        self.server_status.running = True
        self.server_status.pid = 0  # Placeholder
        return True

    async def stop_server(self) -> bool:
        """Stop the ML server."""
        if not self.server_status.running:
            logger.warning("Server is not running")
            return False

        logger.info("Stopping ML server")
        # TODO: Implement actual server shutdown logic
        self.server_status.running = False
        self.server_status.pid = None
        return True


# Global application state instance
app_state = AppState()
