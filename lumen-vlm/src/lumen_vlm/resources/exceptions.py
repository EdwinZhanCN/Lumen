"""Resource-level exceptions for the Lumen VLM package.

All resource loader, manifest validation, and tokenizer provisioning errors must
raise one of the exception types defined in this module so that upstream layers
(Model Manager, Service) can distinguish resource faults from backend/runtime
faults.
"""


class ResourceError(Exception):
    """Base exception for resource-related failures."""

    pass


class ResourceNotFoundError(ResourceError):
    """Raised when a required resource (file, directory, artifact) is missing."""

    pass


class ResourceValidationError(ResourceError):
    """Raised when a resource file exists but its contents fail validation."""

    pass


class ConfigError(ResourceError):
    """Raised when a configuration entry is invalid or unsupported."""

    pass


class RuntimeNotSupportedError(ResourceError):
    """Raised when the requested runtime is absent for the selected model."""

    pass


class DatasetNotFoundError(ResourceError):
    """Raised when optional dataset artifacts cannot be located."""

    pass


class ModelInfoError(ResourceError):
    """Raised when model_info.json is absent or violates the schema."""

    pass


class TokenizerError(ResourceError):
    """Raised when tokenizer artifacts cannot be loaded or parsed."""

    pass
