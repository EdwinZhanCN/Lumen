"""
Validator for model_info.json files using JSON Schema and Pydantic.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import ValidationError

from .model_info import ModelConfigurationSchema, ModelInfo


class ModelInfoValidator:
    """Validator for model_info.json files."""

    def __init__(self, schema_path: str | Path | None = None):
        """
        Initialize validator with JSON schema.

        Args:
            schema_path: Path to model_info-schema.json. If None, uses default path.
        """
        if schema_path is None:
            schema_path = Path(__file__).parent / "schemas" / "model_info-schema.json"
        else:
            schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema: dict[str, Any] = json.load(f)

        self.validator: jsonschema.Draft7Validator = jsonschema.Draft7Validator(
            self.schema
        )

    def validate_json_schema(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate data against JSON schema only.

        Args:
            data: Dictionary to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        for error in self.validator.iter_errors(data):
            error_path = ".".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"{error_path}: {error.message}")

        return len(errors) == 0, errors

    def validate_pydantic(
        self, data: dict[str, Any]
    ) -> tuple[bool, ModelInfo | None, list[str]]:
        """
        Validate data using Pydantic model.

        Args:
            data: Dictionary to validate

        Returns:
            Tuple of (is_valid, model_instance or None, errors)
        """
        try:
            model = ModelConfigurationSchema.model_validate(data)
            return True, model, []
        except ValidationError as e:
            errors = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")
            return False, None, errors

    def validate(
        self, data: dict[str, Any], strict: bool = False
    ) -> tuple[bool, ModelInfo | None, list[str]]:
        """
        Validate model_info data.

        Args:
            data: Dictionary to validate
            strict: If True, use Pydantic validation (stricter).
                If False, use JSON schema only.

        Returns:
            Tuple of (is_valid, model_instance or None, errors)
        """
        if strict:
            return self.validate_pydantic(data)
        else:
            is_valid, errors = self.validate_json_schema(data)
            return is_valid, None, errors

    def validate_file(
        self, path: str | Path, strict: bool = False
    ) -> tuple[bool, ModelInfo | None, list[str]]:
        """
        Validate a model_info.json file.

        Args:
            path: Path to model_info.json file
            strict: If True, use Pydantic validation (stricter)

        Returns:
            Tuple of (is_valid, model_instance or None, errors)
        """
        path = Path(path)
        if not path.exists():
            return False, None, [f"File not found: {path}"]

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, None, [f"Invalid JSON: {e}"]
        except Exception as e:
            return False, None, [f"Error reading file: {e}"]

        return self.validate(data, strict=strict)


def load_and_validate_model_info(path: str | Path, strict: bool = True) -> ModelInfo:
    """
    Load and validate a model_info.json file.

    Returns a Pydantic model instance.

    Args:
        path: Path to model_info.json file
        strict: If True, use Pydantic validation (default).
            If False, use JSON schema only.

    Returns:
        ModelInfo instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If validation fails
        ValueError: If validation fails (non-strict mode)
    """
    if strict:
        # Use Pydantic directly for better error messages
        return ModelConfigurationSchema.from_json_file(path)
    else:
        validator = ModelInfoValidator()
        is_valid, model, errors = validator.validate_file(path, strict=False)
        if not is_valid:
            raise ValueError("Validation failed:\n" + "\n".join(errors))
        # In non-strict mode, we still return a model but only after schema validation
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ModelConfigurationSchema.model_validate(data)


def validate_file(path: str | Path, strict: bool = False) -> tuple[bool, list[str]]:
    """
    Convenience function to validate a model_info.json file.

    Args:
        path: Path to model_info.json file
        strict: If True, use Pydantic validation

    Returns:
        Tuple of (is_valid, errors)
    """
    validator = ModelInfoValidator()
    is_valid, _, errors = validator.validate_file(path, strict=strict)
    return is_valid, errors
