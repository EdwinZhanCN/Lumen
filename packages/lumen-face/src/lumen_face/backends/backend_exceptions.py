"""
Backend Exception Definitions

Following Lumen's development protocol.
"""


class BackendError(Exception):
    """Base class for all backend errors."""

    pass


class BackendNotInitializedError(BackendError):
    """Raised when backend is used before initialization."""

    pass


class InvalidInputError(BackendError):
    """Raised when input data is invalid or malformed."""

    pass


class InferenceError(BackendError):
    """Raised when inference operation fails."""

    pass


class ModelLoadingError(BackendError):
    """Raised when model loading fails."""

    pass


class DeviceUnavailableError(BackendError):
    """Raised when requested device is not available."""

    pass
