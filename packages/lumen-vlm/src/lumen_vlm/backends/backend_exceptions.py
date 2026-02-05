"""
Custom exceptions for Lumen-OCR backends.
"""


class BackendError(Exception):
    """Base exception for all backend-related errors."""

    pass


class BackendNotInitializedError(BackendError):
    """Raised when an operation is attempted on an uninitialized backend."""

    pass


class ModelLoadingError(BackendError):
    """Raised when model loading fails."""

    pass


class InferenceError(BackendError):
    """Raised when model inference fails."""

    pass


class InvalidInputError(BackendError):
    """Raised when input data is invalid or malformed."""

    pass
