"""
Resource Manager Exception Definitions

Following Lumen's contract: each layer defines its own error types.
"""


class ResourceError(Exception):
    """Base exception for all resource management operations.

    All custom exceptions in the lumen_resources package inherit from this
    base class, allowing for comprehensive error handling of resource-related
    operations including configuration, downloading, and validation.

    Example:
        try:
            # Resource operation
            pass
        except ResourceError as e:
            print(f"Resource error occurred: {e}")
    """

    pass


class ConfigError(ResourceError):
    """Raised when configuration is invalid or malformed.

    This exception is raised during configuration parsing and validation
    when the YAML configuration file contains syntax errors, missing required
    fields, invalid values, or fails schema validation.

    Example:
        try:
            config = load_and_validate_config("config.yaml")
        except ConfigError as e:
            print(f"Configuration error: {e}")
    """

    pass


class DownloadError(ResourceError):
    """Raised when resource download fails.

    This exception is raised during model download operations when
    platform adapters encounter network issues, authentication failures,
    missing repositories, or other download-related errors.

    Example:
        try:
            downloader = Downloader(config)
            results = downloader.download_all()
        except DownloadError as e:
            print(f"Download failed: {e}")
    """

    pass


class PlatformUnavailableError(ResourceError):
    """Raised when requested platform or its dependencies are not available.

    This exception is raised when attempting to use a model platform
    (HuggingFace or ModelScope) but the required SDK is not installed
    or the platform is not accessible.

    Example:
        try:
            platform = Platform(PlatformType.HUGGINGFACE, "owner")
        except PlatformUnavailableError as e:
            print(f"Platform not available: {e}")
            print("Install with: pip install huggingface_hub")
    """

    pass


class ValidationError(ResourceError):
    """Raised when model validation fails.

    This exception is raised when downloaded models fail integrity checks,
    missing required files, incompatible runtime configurations, or other
    validation issues during the download process.

    Example:
        try:
            result = downloader._download_model("clip:default", config, False)
        except ValidationError as e:
            print(f"Model validation failed: {e}")
    """

    pass


class ModelInfoError(ResourceError):
    """Raised when model_info.json is missing or invalid.

    This exception is raised when the model_info.json file is missing from
    a downloaded model directory, contains invalid JSON, or fails validation
    against the ModelInfo schema.

    Example:
        try:
            model_info = downloader._load_model_info(model_path)
        except ModelInfoError as e:
            print(f"Model info error: {e}")
    """

    pass
