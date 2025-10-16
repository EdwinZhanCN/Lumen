"""
Resource Manager Exception Definitions

Following Lumen's contract: each layer defines its own error types.
"""


class ResourceError(Exception):
    """Base exception for all resource management operations."""

    pass


class ConfigError(ResourceError):
    """
    Raised when configuration is invalid or malformed.

    @context: Configuration parsing and validation
    """

    pass


class DownloadError(ResourceError):
    """
    Raised when resource download fails.

    @context: Platform adapter operations
    """

    pass


class PlatformUnavailableError(ResourceError):
    """
    Raised when requested platform or its dependencies are not available.

    @context: Platform adapter initialization
    """

    pass


class ValidationError(ResourceError):
    """
    Raised when model validation fails.

    @context: Model integrity checks
    """

    pass


class ModelInfoError(ResourceError):
    """
    Raised when model_info.json is missing or invalid.

    @context: Model metadata parsing
    """

    pass
