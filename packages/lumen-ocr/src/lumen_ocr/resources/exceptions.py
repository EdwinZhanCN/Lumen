"""
Exceptions for Lumen-OCR resource management.
"""


class ResourceError(Exception):
    """Base exception for resource-related errors."""

    pass


class ResourceNotFoundError(ResourceError):
    """Raised when a requested resource (file or directory) is not found."""

    pass


class ModelInfoError(ResourceError):
    """Raised when model metadata is invalid or corrupted."""

    pass


class RuntimeNotSupportedError(ResourceError):
    """Raised when the requested runtime is not supported by the model."""

    pass
