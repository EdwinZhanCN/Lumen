"""
Task Registry for Lumen Face Services.

Provides a centralized registry for managing face service tasks with
automatic capability generation and handler routing. Follows the same
pattern as lumen-clip for consistency across Lumen services.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import lumen_ocr.proto.ml_service_pb2 as pb

logger = logging.getLogger(__name__)


@dataclass
class TaskDefinition:
    """Definition of a face service task with metadata."""

    name: str
    handler: Callable[
        [bytes, str, dict[str, str]], tuple[bytes, str, dict[str, str]]
    ]  # def handler(payload, payload_mime, meta) -> (result, result_mime, meta)
    description: str
    input_mimes: list[str] = field(default_factory=list)
    output_mime: str = "application/json"
    metadata: dict[str, str] = field(default_factory=dict)

    def to_io_task(self) -> pb.IOTask:
        """Convert to protobuf IOTask for capability reporting."""
        return pb.IOTask(
            name=self.name,
            input_mimes=self.input_mimes,
            output_mimes=[self.output_mime],
            limits={
                "max_payload_size": str(50 * 1024 * 1024),  # 50MB
                "max_concurrency": "1",
                **{k: str(v) for k, v in self.metadata.items()},
            },
        )


class TaskRegistry:
    """Centralized registry for face service tasks."""

    def __init__(self):
        # [{name, definition}, ...]
        self._tasks: dict[str, TaskDefinition] = {}
        self._service_name: str = "unknown"

    def register_task(
        self,
        name: str,
        handler: Callable[
            [bytes, str, dict[str, str]], tuple[bytes, str, dict[str, str]]
        ],
        description: str,
        input_mimes: list[str] | None = None,
        output_mime: str = "application/json",
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Register a task with the registry."""
        if name in self._tasks:
            logger.warning(f"Task '{name}' already registered, overwriting")

        task_def = TaskDefinition(
            name=name,
            handler=handler,
            description=description,
            input_mimes=input_mimes or [],
            output_mime=output_mime,
            metadata=metadata or {},
        )

        self._tasks[name] = task_def
        logger.debug(f"Registered task: {name} - {description}")

    def set_service_name(self, service_name: str) -> None:
        """Set the service name for capability reporting."""
        self._service_name = service_name

    def get_handler(
        self, task_name: str
    ) -> Callable[[bytes, str, dict[str, str]], tuple[bytes, str, dict[str, str]]]:
        """Get the handler for a task."""
        if task_name not in self._tasks:
            available_tasks = list(self._tasks.keys())
            raise ValueError(
                f"Task '{task_name}' not found. Available tasks: {available_tasks}"
            )
        return self._tasks[task_name].handler

    def get_task_definition(self, task_name: str) -> TaskDefinition:
        """Get the complete task definition."""
        if task_name not in self._tasks:
            available_tasks = list(self._tasks.keys())
            raise ValueError(
                f"Task '{task_name}' not found. Available tasks: {available_tasks}"
            )
        return self._tasks[task_name]

    def list_task_names(self) -> list[str]:
        """Get list of all registered task names."""
        return list(self._tasks.keys())

    def list_task_definitions(self) -> list[TaskDefinition]:
        """Get list of all task definitions."""
        return list(self._tasks.values())

    def get_all_tasks(self) -> list[pb.IOTask]:
        """Get all tasks as protobuf IOTask objects for capabilities."""
        return [task.to_io_task() for task in self._tasks.values()]

    def build_capability(
        self,
        service_name: str,
        model_id: str,
        runtime: str,
        precisions: list[str],
        extra_metadata: dict[str, str] | None = None,
    ) -> pb.Capability:
        """Build capability protobuf using registered tasks."""
        # Ensure all extra_metadata values are strings and handle None values
        safe_extra = {}
        if extra_metadata:
            for key, value in extra_metadata.items():
                safe_extra[key] = str(value) if value is not None else ""

        return pb.Capability(
            service_name=service_name,
            model_ids=[model_id],
            runtime=runtime,
            max_concurrency=1,
            precisions=precisions,
            extra=safe_extra,
            tasks=self.get_all_tasks(),
            protocol_version="1.0",
        )
