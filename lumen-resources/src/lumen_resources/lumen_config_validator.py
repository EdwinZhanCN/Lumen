"""
Configuration validator for Lumen services.

Provides validation utilities for YAML configuration files against
the Lumen configuration schema.
"""

from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator
from pydantic import ValidationError

from .exceptions import ConfigError
from .lumen_config import LumenConfig


class ConfigValidator:
    """Validator for Lumen configuration files.

    Provides comprehensive validation for YAML configuration files using both
    JSON Schema and Pydantic models. Supports strict validation with custom
    validators and flexible validation for development scenarios.

    Attributes:
        schema: Loaded JSON Schema for validation.
        validator: Draft7Validator instance for JSON Schema validation.

    Example:
        >>> validator = ConfigValidator()
        >>> is_valid, errors = validator.validate_file("config.yaml")
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Validation error: {error}")
    """

    def __init__(self, schema_path: Path | None = None):
        """Initialize validator with optional custom schema.

        Args:
            schema_path: Optional path to JSON Schema file. If None, uses
                the bundled schema from docs/schemas/config-schema.yaml.

        Raises:
            FileNotFoundError: If the schema file is not found.
            yaml.YAMLError: If the schema file is invalid YAML.

        Example:
            >>> validator = ConfigValidator()  # Uses bundled schema
            >>> validator = ConfigValidator(Path("custom-schema.yaml"))  # Custom schema
        """
        if schema_path is None:
            # Use bundled schema from docs/
            schema_path = Path(__file__).parent / "schemas" / "config-schema.yaml"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = yaml.safe_load(f)

        self.validator = Draft7Validator(self.schema)

    def validate_file(
        self, config_path: Path | str, strict: bool = True
    ) -> tuple[bool, list[str]]:
        """Validate configuration file against schema.

        Performs validation of a YAML configuration file using either
        JSON Schema validation (flexible) or Pydantic validation (strict).

        Args:
            config_path: Path to configuration YAML file.
            strict: If True, use Pydantic validation with custom validators.
                If False, use JSON Schema validation only.

        Returns:
            Tuple of (is_valid, error_messages). is_valid is True if the
            configuration passes validation, False otherwise. error_messages
            contains detailed validation error messages.

        Example:
            >>> validator = ConfigValidator()
            >>> is_valid, errors = validator.validate_file("config.yaml", strict=True)
            >>> if not is_valid:
            ...     for error in errors:
            ...         print(f"Error: {error}")
        """
        config_path = Path(config_path)

        if not config_path.exists():
            return False, [f"Configuration file not found: {config_path}"]

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return False, [f"Invalid YAML syntax: {e}"]
        except Exception as e:
            return False, [f"Failed to load file: {e}"]

        if strict:
            # Use Pydantic validation (stricter, includes custom validators)
            return self._validate_with_pydantic(config_data)
        else:
            # Use JSON Schema validation only
            return self._validate_with_jsonschema(config_data)

    def _validate_with_jsonschema(
        self, config_data: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate configuration data using JSON Schema.

        Performs flexible validation using the JSON Schema specification.
        This method is less strict than Pydantic validation but provides
        good basic structural validation.

        Args:
            config_data: Parsed configuration data dictionary.

        Returns:
            Tuple of (is_valid, error_messages) where is_valid indicates
            if the data passes JSON Schema validation.

        Example:
            >>> validator = ConfigValidator()
            >>> is_valid, errors = validator._validate_with_jsonschema(data)
        """
        errors = sorted(self.validator.iter_errors(config_data), key=lambda e: e.path)

        if not errors:
            return True, []

        error_messages = []
        for error in errors:
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            error_messages.append(f"{error.message} (at: {path})")

        return False, error_messages

    def _validate_with_pydantic(
        self, config_data: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate configuration data using Pydantic models.

        Performs strict validation using Pydantic models with custom validators.
        This provides the most comprehensive validation including type checking,
        pattern matching, and business logic validation.

        Args:
            config_data: Parsed configuration data dictionary.

        Returns:
            Tuple of (is_valid, error_messages) where is_valid indicates
            if the data passes Pydantic model validation.

        Example:
            >>> validator = ConfigValidator()
            >>> is_valid, errors = validator._validate_with_pydantic(data)
        """
        try:
            LumenConfig(**config_data)
            return True, []
        except ValidationError as e:
            # Parse pydantic validation errors
            error_messages = []
            for error in e.errors():
                loc = ".".join(str(loc_part) for loc_part in error["loc"])
                msg = error["msg"]
                error_messages.append(f"{msg} (at: {loc})")
            return False, error_messages
        except Exception as e:
            return False, [f"Validation error: {e}"]

    def validate_and_load(self, config_path: Path | str) -> LumenConfig:
        """Validate and load configuration file.

        Performs strict validation using Pydantic models and returns a validated
        LumenConfig instance if successful. This is the recommended method
        for loading configuration in production code.

        Args:
            config_path: Path to configuration YAML file.

        Returns:
            Validated LumenConfig instance with all data properly typed
            and validated.

        Raises:
            ConfigError: If validation fails or file cannot be loaded.

        Example:
            >>> validator = ConfigValidator()
            >>> config = validator.validate_and_load("config.yaml")
            >>> print(config.metadata.version)
            '1.0.0'
        """
        config_path = Path(config_path)

        is_valid, errors = self.validate_file(config_path, strict=True)

        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ConfigError(error_msg)

        # Load and construct the validated configuration
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return LumenConfig(**config_data)


def validate_config_file(
    config_path: Path | str, schema_path: Path | str | None = None
) -> tuple[bool, list[str]]:
    """Convenience function to validate a configuration file.

    Simple one-line function for validating configuration files using
    the default schema or a custom schema. Uses strict validation.

    Args:
        config_path: Path to configuration YAML file.
        schema_path: Optional path to custom JSON Schema file.

    Returns:
        Tuple of (is_valid, error_messages) where is_valid is True if
        the configuration passes validation, and error_messages contains
        detailed validation errors if validation fails.

    Example:
        >>> is_valid, errors = validate_config_file("config.yaml")
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    schema_path_obj = Path(schema_path) if schema_path else None
    validator = ConfigValidator(schema_path_obj)
    return validator.validate_file(config_path, strict=True)


def load_and_validate_config(config_path: Path | str) -> LumenConfig:
    """Load and validate configuration file.

    This is the recommended way to load configuration in production.
    Combines validation and loading into a single operation for convenience
    and ensures that only validated configurations are returned.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Validated LumenConfig instance with all data properly typed and
        validated against the schema.

    Raises:
        ConfigError: If validation fails or file is not found.
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file contains invalid YAML.

    Example:
        >>> from lumen_resources.lumen_config_validator import load_and_validate_config
        >>> config = load_and_validate_config("config.yaml")
        >>> print(config.metadata.cache_dir)
        '/models'
    """
    validator = ConfigValidator()
    return validator.validate_and_load(config_path)
