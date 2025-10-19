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

        self.validator = Draft7Validator(self.schema)

    def validate_file(
        self, path: str | Path, strict: bool = True
    ) -> tuple[bool, list[str]]:
        """
        Validate a model_info.json file.
        Args:
            path: Path to model_info.json file
            strict: If True, use Pydantic validation (stricter)
        Returns:
            Tuple of (is_valid, error_messages)
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
        """Validate using JSON Schema"""
        errors = sorted(self.validator.iter_errors(data), key=lambda e: e.path)

        if not errors:
            return True, []

        error_messages = []
        for error in errors:
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            error_messages.append(f"{error.message} (at: {path})")

        return False, error_messages

    def _validate_with_pydantic(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate using Pydantic models"""
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
        """
        Validate and load model_info.json file.
        Args:
            path: Path to model_info.json file
        Returns:
            Validated ModelInfo instance
        Raises:
            ValueError: If validation fails
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
    """
    Load and validate a model_info.json file.
    Args:
        path: Path to model_info.json file
    Returns:
        Validated ModelInfo instance
    Raises:
        ValueError: If validation fails or file not found
    """
    validator = ModelInfoValidator()
    return validator.validate_and_load(path)


def validate_file(path: str | Path, strict: bool = True) -> tuple[bool, list[str]]:
    """
    Convenience function to validate a model_info.json file.
    Args:
        path: Path to model_info.json file
        strict: If True, use Pydantic validation
    Returns:
        Tuple of (is_valid, errors)
    """
    validator = ModelInfoValidator()
    return validator.validate_file(path, strict=strict)
