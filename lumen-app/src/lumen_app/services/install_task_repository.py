"""Repository abstraction for installation tasks and logs."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol


class InstallTaskRepository(Protocol):
    """Abstract repository contract for installation tasks."""

    async def store_task(self, task_id: str, task: Any) -> None:
        """Store or update installation task."""

    async def get_task(self, task_id: str) -> Any | None:
        """Get installation task by ID."""

    async def list_tasks(self) -> list[Any]:
        """List all installation tasks."""

    async def append_log(self, task_id: str, log_line: str) -> None:
        """Append a log line for the task."""

    async def get_logs(self, task_id: str) -> list[str]:
        """Get all logs for the task."""


class InMemoryInstallTaskRepository:
    """In-memory implementation of installation task repository."""

    def __init__(self):
        self._tasks: dict[str, Any] = {}
        self._logs: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def store_task(self, task_id: str, task: Any) -> None:
        """Store or update installation task."""
        async with self._lock:
            self._tasks[task_id] = task
            if task_id not in self._logs:
                self._logs[task_id] = []

    async def get_task(self, task_id: str) -> Any | None:
        """Get installation task by ID."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def list_tasks(self) -> list[Any]:
        """List all installation tasks."""
        async with self._lock:
            return list(self._tasks.values())

    async def append_log(self, task_id: str, log_line: str) -> None:
        """Append a log line for the task."""
        async with self._lock:
            if task_id not in self._logs:
                self._logs[task_id] = []
            self._logs[task_id].append(log_line)

    async def get_logs(self, task_id: str) -> list[str]:
        """Get all logs for the task."""
        async with self._lock:
            return self._logs.get(task_id, [])
