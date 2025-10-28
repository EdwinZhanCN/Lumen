"""
backend_exceptions.py

Exception classes for the resources module.
Defines all error types that can occur during resource loading and validation.
"""


class ResourceError(Exception):
    """Base exception for all resource-related errors."""

    pass


class ResourceNotFoundError(ResourceError):
    """Raised when a required resource file cannot be found."""

    pass


class ResourceValidationError(ResourceError):
    """Raised when resource validation fails."""

    pass


class ConfigError(ResourceError):
    """Raised when configuration is invalid."""

    pass


class RuntimeNotSupportedError(ResourceError):
    """Raised when the requested runtime is not supported by the model."""

    pass


class DatasetNotFoundError(ResourceError):
    """Raised when dataset file is not found (can be caught for graceful degradation)."""

    pass


class ModelInfoError(ResourceError):
    """Raised when model_info.json is invalid or missing required fields."""

    pass


class TokenizerError(ResourceError):
    """Raised when tokenizer loading fails."""

    pass
