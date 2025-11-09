"""
Task Registry for Lumen Face Service.

This module provides a centralized task definition and routing system that serves
as the single source of truth for all IOTasks provided by this node.

Key Features:
- Single source of truth for task definitions
- Automatic capability generation
- Dynamic request routing based on task names
- Extensible task registration system
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import lumen_face.proto.ml_service_pb2 as pb

logger = logging.getLogger(__name__)


@dataclass
class TaskDefinition:
    """Definition of an ML task with its handler and metadata.

    This class encapsulates all information about a task including:
    - The unique task name used for routing
    - The handler function that processes the task
    - Human-readable description
    - Supported input/output MIME types
    - Additional metadata for capability reporting
    """
    name: str
    handler: Callable
    description: str
    input_mimes: List[str]
    output_mime: str
    metadata: Dict[str, str]

    def to_protobuf(self) -> pb.IOTask:
        """Convert this task definition to protobuf IOTask.

        Returns:
            pb.IOTask: Protobuf representation of this task for capability reporting
        """
        return pb.IOTask(
            name=self.name,
            description=self.description,
            input_mimes=self.input_mimes,
            output_mime=self.output_mime,
            limits=self.metadata
        )


class TaskRegistry:
    """Single source of truth for all IOTasks provided by this node.

    This registry centralizes task definitions and provides:
    - Task discovery and capability reporting
    - Dynamic request routing based on task names
    - Automatic generation of IOTask protobufs
    - Extensible task registration system

    Usage:
        registry = TaskRegistry()
        registry.register_task("detect", handler_func, ...)

        # Get all tasks for capability reporting
        tasks = registry.get_all_tasks()

        # Route a request
        handler = registry.get_handler("lumen_face_detect")
        result = handler(payload, meta)
    """

    def __init__(self):
        """Initialize an empty task registry."""
        self._tasks: Dict[str, TaskDefinition] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def register_task(
        self,
        task_id: str,
        name: str,
        handler: Callable,
        description: str,
        input_mimes: List[str],
        output_mime: str,
        **metadata
    ) -> None:
        """Register a new task in the registry.

        Args:
            task_id: Internal identifier for the task (e.g., "detect")
            name: External task name used for routing (e.g., "lumen_face_detect")
            handler: Function that handles the task execution
            description: Human-readable description of the task
            input_mimes: List of supported input MIME types
            output_mime: Output MIME type
            **metadata: Additional task metadata

        Raises:
            ValueError: If task_id or name is already registered
        """
        if task_id in self._tasks:
            raise ValueError(f"Task ID '{task_id}' already registered")

        # Check for duplicate task names
        for existing_task in self._tasks.values():
            if existing_task.name == name:
                raise ValueError(f"Task name '{name}' already registered for task '{task_id}'")

        task_def = TaskDefinition(
            name=name,
            handler=handler,
            description=description,
            input_mimes=input_mimes,
            output_mime=output_mime,
            metadata=metadata
        )

        self._tasks[task_id] = task_def
        self._logger.info(f"Registered task '{task_id}' with name '{name}'")

    def get_handler(self, task_name: str) -> Callable:
        """Get the handler function for a given task name.

        Args:
            task_name: The external task name (e.g., "lumen_face_detect")

        Returns:
            Callable: The handler function for this task

        Raises:
            ValueError: If task_name is not found in registry
        """
        for task_def in self._tasks.values():
            if task_def.name == task_name:
                return task_def.handler

        available_names = [task.name for task in self._tasks.values()]
        raise ValueError(
            f"Task '{task_name}' not found. Available tasks: {available_names}"
        )

    def get_all_tasks(self) -> List[pb.IOTask]:
        """Get all tasks as protobuf IOTask objects.

        This method is used to generate the capability information that
        gets reported to the Lumen Hub during node discovery.

        Returns:
            List[pb.IOTask]: All registered tasks as protobuf objects
        """
        return [task.to_protobuf() for task in self._tasks.values()]

    def get_task_info(self) -> Dict[str, Any]:
        """Get a summary of all registered tasks.

        Returns:
            Dict containing task information for debugging and monitoring
        """
        return {
            task_id: {
                "name": task.name,
                "description": task.description,
                "input_mimes": task.input_mimes,
                "output_mime": task.output_mime,
                "metadata": task.metadata
            }
            for task_id, task in self._tasks.items()
        }

    def list_task_names(self) -> List[str]:
        """Get a list of all registered task names.

        Returns:
            List[str]: All external task names
        """
        return [task.name for task in self._tasks.values()]

    def task_exists(self, task_name: str) -> bool:
        """Check if a task name exists in the registry.

        Args:
            task_name: The task name to check

        Returns:
            bool: True if task exists, False otherwise
        """
        return any(task.name == task_name for task in self._tasks.values())