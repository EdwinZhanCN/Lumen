"""
Validator for model_info.json files using JSON Schema and Pydantic.
"""

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from pydantic import ValidationError

from .model_info import ModelInfo


class ModelInfoValidator:
    """Validator for model_info.json files.

    Provides comprehensive validation for model information JSON files using
    both JSON Schema and Pydantic models. Ensures model metadata integrity
    and compliance with the Lumen model specification.

    Attributes:
        schema: Loaded JSON Schema for validation.
        validator: Draft7Validator instance for JSON Schema validation.

    Example:
        >>> validator = ModelInfoValidator()
        >>> is_valid, errors = validator.validate_file("model_info.json")
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Validation error: {error}")
    """

    def __init__(self, schema_path: str | Path | None = None):
        """Initialize validator with JSON schema.

        Args:
            schema_path: Path to model_info-schema.json file. If None,
                uses the bundled schema from docs/schemas/model_info-schema.json.

        Raises:
            FileNotFoundError: If the schema file is not found.
            json.JSONDecodeError: If the schema file contains invalid JSON.

        Example:
            >>> validator = ModelInfoValidator()  # Uses bundled schema
            >>> validator = ModelInfoValidator(Path("custom-schema.json"))
        """
        if schema_path is None:
            schema_path = Path(__file__).parent / "schemas" / "model_info-schema.json"
        else:
            schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema: dict[str, Any] = json.load(f)

        self.validator = Draft7Validator(self.schema)

    def validate_file(
        self, path: str | Path, strict: bool = True
    ) -> tuple[bool, list[str]]:
        """Validate a model_info.json file.

        Performs validation of model information JSON files using either
        JSON Schema validation (flexible) or Pydantic validation (strict).

        Args:
            path: Path to model_info.json file.
            strict: If True, use Pydantic validation with custom validators.
                If False, use JSON Schema validation only.

        Returns:
            Tuple of (is_valid, error_messages) where is_valid indicates
            if the file passes validation, and error_messages contains
            detailed validation error messages.

        Example:
            >>> validator = ModelInfoValidator()
            >>> is_valid, errors = validator.validate_file("model_info.json", strict=True)
            >>> if not is_valid:
            ...     for error in errors:
            ...         print(f"Error: {error}")
        """
        path = Path(path)
        if not path.exists():
            return False, [f"File not found: {path}"]

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]
        except Exception as e:
            return False, [f"Error reading file: {e}"]

        if strict:
            return self._validate_with_pydantic(data)
        else:
            return self._validate_with_jsonschema(data)

    def _validate_with_jsonschema(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate model info data using JSON Schema.

        Performs flexible validation using the JSON Schema specification.
        This method provides basic structural validation for model information.

        Args:
            data: Parsed model_info.json data dictionary.

        Returns:
            Tuple of (is_valid, error_messages) where is_valid indicates
            if the data passes JSON Schema validation.

        Example:
            >>> validator = ModelInfoValidator()
            >>> is_valid, errors = validator._validate_with_jsonschema(data)
        """
        errors = sorted(self.validator.iter_errors(data), key=lambda e: e.path)

        if not errors:
            return True, []

        error_messages = []
        for error in errors:
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            error_messages.append(f"{error.message} (at: {path})")

        return False, error_messages

    def _validate_with_pydantic(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate model info data using Pydantic models.

        Performs strict validation using Pydantic models with custom validators.
        This provides comprehensive validation including type checking,
        pattern matching, and model-specific business rules.

        Args:
            data: Parsed model_info.json data dictionary.

        Returns:
            Tuple of (is_valid, error_messages) where is_valid indicates
            if the data passes Pydantic model validation.

        Example:
            >>> validator = ModelInfoValidator()
            >>> is_valid, errors = validator._validate_with_pydantic(data)
        """
        try:
            ModelInfo.model_validate(data)
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

    def validate_and_load(self, path: str | Path) -> ModelInfo:
        """Validate and load model_info.json file.

        Performs strict validation using Pydantic models and returns a validated
        ModelInfo instance if successful. This is the recommended method
        for loading model information in production code.

        Args:
            path: Path to model_info.json file.

        Returns:
            Validated ModelInfo instance with all data properly typed
            and validated.

        Raises:
            ValueError: If validation fails or file cannot be loaded.
            FileNotFoundError: If the model_info.json file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.

        Example:
            >>> validator = ModelInfoValidator()
            >>> model_info = validator.validate_and_load("model_info.json")
            >>> print(model_info.name)
            'ViT-B-32'
        """
        path = Path(path)
        is_valid, errors = self.validate_file(path, strict=True)

        if not is_valid:
            error_msg = "Model info validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ValueError(error_msg)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ModelInfo.model_validate(data)


def load_and_validate_model_info(path: str | Path) -> ModelInfo:
    """Load and validate a model_info.json file.

    This is the recommended way to load model information in production.
    Combines validation and loading into a single operation for convenience
    and ensures that only validated model information is returned.

    Args:
        path: Path to model_info.json file.

    Returns:
        Validated ModelInfo instance with all data properly typed and
        validated against the model specification.

    Raises:
        ValueError: If validation fails or file cannot be loaded.
        FileNotFoundError: If the model_info.json file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    Example:
        >>> from lumen_resources.model_info_validator import load_and_validate_model_info
        >>> model_info = load_and_validate_model_info("model_info.json")
        >>> print(model_info.version)
        '1.0.0'
    """
    validator = ModelInfoValidator()
    return validator.validate_and_load(path)


def validate_file(path: str | Path, strict: bool = True) -> tuple[bool, list[str]]:
    """Convenience function to validate a model_info.json file.

    Simple one-line function for validating model information JSON files.
    Uses strict validation by default for maximum reliability.

    Args:
        path: Path to model_info.json file.
        strict: If True, use Pydantic validation with custom validators.
            If False, use JSON Schema validation only. Defaults to True.

    Returns:
        Tuple of (is_valid, error_messages) where is_valid is True if
        the file passes validation, and error_messages contains detailed
        validation errors if validation fails.

    Example:
        >>> is_valid, errors = validate_file("model_info.json")
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    validator = ModelInfoValidator()
    return validator.validate_file(path, strict=strict)
