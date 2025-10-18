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

from .lumen_config import LumenServicesConfiguration
from .exceptions import ConfigError


class ConfigValidator:
    """
    Validator for Lumen configuration files.

    Supports both JSON Schema validation and Pydantic model validation.
    """

    def __init__(self, schema_path: Path | None = None):
        """
        Initialize validator.

        Args:
            schema_path: Optional path to JSON Schema file.
                        If None, uses bundled schema.
        """
        if schema_path is None:
            # Use bundled schema from docs/
            package_root = Path(__file__).parent.parent.parent
            schema_path = package_root / "docs" / "config-schema.yaml"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = yaml.safe_load(f)

        self.validator = Draft7Validator(self.schema)

    def validate_file(
        self, config_path: Path | str, strict: bool = True
    ) -> tuple[bool, list[str]]:
        """
        Validate configuration file.

        Args:
            config_path: Path to configuration YAML file
            strict: If True, use Pydantic validation (stricter).
                   If False, use JSON Schema only.

        Returns:
            Tuple of (is_valid, error_messages)
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
        """Validate using JSON Schema"""
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
        """Validate using Pydantic models"""
        try:
            LumenServicesConfiguration(**config_data)
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

    def validate_and_load(self, config_path: Path | str) -> LumenServicesConfiguration:
        """
        Validate and load configuration file.

        Args:
            config_path: Path to configuration YAML file

        Returns:
            Validated LumenServicesConfiguration instance

        Raises:
            ConfigError: If validation fails
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

        return LumenServicesConfiguration(**config_data)


def validate_config_file(
    config_path: Path | str, schema_path: Path | str | None = None
) -> tuple[bool, list[str]]:
    """
    Convenience function to validate a configuration file.

    Args:
        config_path: Path to configuration YAML file
        schema_path: Optional path to schema file

    Returns:
        Tuple of (is_valid, error_messages)

    Example:
        >>> is_valid, errors = validate_config_file("config.yaml")
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    schema_path_obj = Path(schema_path) if schema_path else None
    validator = ConfigValidator(schema_path_obj)
    return validator.validate_file(config_path, strict=True)


def load_and_validate_config(config_path: Path | str) -> LumenServicesConfiguration:
    """
    Load and validate configuration file.

    This is the recommended way to load configuration in production.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Validated LumenServicesConfiguration instance

    Raises:
        ConfigError: If validation fails or file not found

    Example:
        >>> from lumen_resources.validator import load_and_validate_config
        >>> config = load_and_validate_config("config.yaml")
        >>> print(config.metadata.cache_dir)
    """
    validator = ConfigValidator()
    return validator.validate_and_load(config_path)
