"""Exceptions for the LumNN orchestration layer."""

from __future__ import annotations


class LumNNError(Exception):
    """Base exception for all LumNN framework errors."""


class InvalidMLPacketError(LumNNError):
    """Raised when an MLPacket is malformed or incompatible with the API contract."""


class InvalidPipelineGraphError(LumNNError):
    """Raised when the declared pipeline graph is invalid or cannot be compiled."""


class PipelineNotCompiledError(LumNNError):
    """Raised when dispatch is called before a pipeline has been compiled."""


class NodeExecutionError(LumNNError):
    """Raised when a node fails during pipeline execution."""

    def __init__(self, node_name: str, message: str) -> None:
        super().__init__(f"Node '{node_name}' failed: {message}")
        self.node_name = node_name
        self.message = message


class DeviceResolutionError(LumNNError):
    """Raised when a device hint cannot be normalized into a valid target."""
