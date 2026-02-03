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


class BackendDependencyError(BackendError):
    """Raised when attempting to use a backend with missing optional dependencies."""

    def __init__(self, runtime: str) -> None:
        """
        Initialize error with installation instructions.

        Args:
            runtime: Runtime kind identifier (e.g., "torch", "rknn")
        """
        install_commands = {
            "torch": "pip install lumen-clip[torch]",
            "rknn": "pip install lumen-clip[rknn]",
        }
        cmd = install_commands.get(runtime, "pip install lumen-clip")
        message = (
            f"Backend '{runtime}' requires optional dependencies. Install with: {cmd}"
        )
        super().__init__(message)
        self.runtime = runtime
        self.install_command = cmd
