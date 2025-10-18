"""
Unit tests for model_info validator.
"""

import json
import tempfile
from pathlib import Path

import pytest

from lumen_resources.model_info import (
    Format,
    ModelConfigurationSchema,
    ModelInfo,
)
from lumen_resources.model_info_validator import (
    ModelInfoValidator,
    load_and_validate_model_info,
    validate_file,
)


@pytest.fixture
def valid_model_info():
    """Fixture providing valid model info data."""
    return {
        "name": "test-model",
        "version": "1.0.0",
        "description": "A test model for unit tests",
        "model_type": "clip",
        "embedding_dim": 512,
        "source": {"format": "huggingface", "repo_id": "test/test-model"},
        "runtimes": {
            "torch": {"available": True, "files": ["model.pt"]},
            "onnx": {"available": False},
        },
    }


@pytest.fixture
def valid_model_info_file(valid_model_info, tmp_path):
    """Fixture providing a valid model_info.json file."""
    model_info_path = tmp_path / "model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(valid_model_info, f)
    return model_info_path


@pytest.fixture
def schema_path():
    """Fixture providing path to schema file."""
    return Path(__file__).parent.parent / "docs" / "model_info-schema.json"


class TestModelInfoValidator:
    """Test suite for ModelInfoValidator."""

    def test_validator_initialization(self, schema_path):
        """Test validator can be initialized with schema."""
        validator = ModelInfoValidator(schema_path)
        assert validator.schema is not None
        assert validator.validator is not None

    def test_validator_default_schema_path(self):
        """Test validator uses default schema path when none provided."""
        validator = ModelInfoValidator()
        assert validator.schema is not None

    def test_validate_json_schema_valid(self, valid_model_info):
        """Test JSON schema validation with valid data."""
        validator = ModelInfoValidator()
        is_valid, errors = validator.validate_json_schema(valid_model_info)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_json_schema_missing_required(self):
        """Test JSON schema validation catches missing required fields."""
        validator = ModelInfoValidator()
        invalid_data = {
            "name": "test-model",
            "version": "1.0.0",
            # Missing description, model_type, embedding_dim, source, runtimes
        }
        is_valid, errors = validator.validate_json_schema(invalid_data)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_json_schema_invalid_version_format(self):
        """Test JSON schema validation catches invalid version format."""
        validator = ModelInfoValidator()
        invalid_data = {
            "name": "test-model",
            "version": "1.0",  # Should be x.y.z format
            "description": "Test",
            "model_type": "clip",
            "embedding_dim": 512,
            "source": {"format": "huggingface", "repo_id": "test/model"},
            "runtimes": {"torch": {"available": True}},
        }
        is_valid, errors = validator.validate_json_schema(invalid_data)
        assert is_valid is False
        assert any("version" in err.lower() for err in errors)

    def test_validate_json_schema_invalid_embedding_dim(self):
        """Test JSON schema validation catches invalid embedding dimension."""
        validator = ModelInfoValidator()
        invalid_data = {
            "name": "test-model",
            "version": "1.0.0",
            "description": "Test",
            "model_type": "clip",
            "embedding_dim": 0,  # Must be >= 1
            "source": {"format": "huggingface", "repo_id": "test/model"},
            "runtimes": {"torch": {"available": True}},
        }
        is_valid, errors = validator.validate_json_schema(invalid_data)
        assert is_valid is False

    def test_validate_pydantic_valid(self, valid_model_info):
        """Test Pydantic validation with valid data."""
        validator = ModelInfoValidator()
        is_valid, model, errors = validator.validate_pydantic(valid_model_info)
        assert is_valid is True
        assert model is not None
        assert isinstance(model, ModelConfigurationSchema)
        assert len(errors) == 0
        assert model.name == "test-model"
        assert model.version == "1.0.0"
        assert model.source.format == Format.huggingface

    def test_validate_pydantic_invalid(self):
        """Test Pydantic validation with invalid data."""
        validator = ModelInfoValidator()
        invalid_data = {
            "name": "test-model",
            "version": "invalid",  # Invalid version format
            "description": "Test",
            "model_type": "clip",
            "embedding_dim": 512,
            "source": {"format": "huggingface", "repo_id": "test/model"},
            "runtimes": {"torch": {"available": True}},
        }
        is_valid, model, errors = validator.validate_pydantic(invalid_data)
        assert is_valid is False
        assert model is None
        assert len(errors) > 0

    def test_validate_file_valid(self, valid_model_info_file):
        """Test file validation with valid file."""
        validator = ModelInfoValidator()
        is_valid, model, errors = validator.validate_file(
            valid_model_info_file, strict=True
        )
        assert is_valid is True
        assert model is not None
        assert len(errors) == 0

    def test_validate_file_not_found(self):
        """Test file validation with non-existent file."""
        validator = ModelInfoValidator()
        is_valid, model, errors = validator.validate_file(
            "/nonexistent/path.json", strict=False
        )
        assert is_valid is False
        assert model is None
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_validate_file_invalid_json(self, tmp_path):
        """Test file validation with invalid JSON."""
        invalid_json_file = tmp_path / "invalid.json"
        with open(invalid_json_file, "w") as f:
            f.write("{ invalid json }")

        validator = ModelInfoValidator()
        is_valid, model, errors = validator.validate_file(invalid_json_file)
        assert is_valid is False
        assert model is None
        assert len(errors) == 1
        assert "json" in errors[0].lower()

    def test_validate_strict_vs_non_strict(self, valid_model_info):
        """Test difference between strict and non-strict validation."""
        validator = ModelInfoValidator()

        # Non-strict (JSON schema only)
        is_valid_schema, model_schema, errors_schema = validator.validate(
            valid_model_info, strict=False
        )
        assert is_valid_schema is True
        assert model_schema is None  # Non-strict doesn't return model
        assert len(errors_schema) == 0

        # Strict (Pydantic)
        is_valid_pydantic, model_pydantic, errors_pydantic = validator.validate(
            valid_model_info, strict=True
        )
        assert is_valid_pydantic is True
        assert model_pydantic is not None
        assert len(errors_pydantic) == 0


class TestModelConfigurationSchema:
    """Test suite for ModelConfigurationSchema Pydantic model."""

    def test_from_json_file(self, valid_model_info_file):
        """Test loading model info from JSON file."""
        model = ModelConfigurationSchema.from_json_file(valid_model_info_file)
        assert model.name == "test-model"
        assert model.version == "1.0.0"
        assert model.embedding_dim == 512
        assert model.source.format == Format.huggingface
        assert "torch" in model.runtimes
        assert model.runtimes["torch"].available is True

    def test_from_json_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ModelConfigurationSchema.from_json_file("/nonexistent/path.json")

    def test_to_json_file(self, valid_model_info_file, tmp_path):
        """Test saving model info to JSON file."""
        model = ModelConfigurationSchema.from_json_file(valid_model_info_file)
        output_path = tmp_path / "output.json"
        model.to_json_file(output_path)

        assert output_path.exists()

        # Verify it can be loaded back
        model2 = ModelConfigurationSchema.from_json_file(output_path)
        assert model2.name == model.name
        assert model2.version == model.version

    def test_model_validation_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(Exception):
            ModelConfigurationSchema(name="test")  # Missing required fields

    def test_model_validation_version_pattern(self):
        """Test version pattern validation."""
        with pytest.raises(Exception):
            ModelConfigurationSchema(
                name="test",
                version="1.0",  # Invalid pattern
                description="Test",
                model_type="clip",
                embedding_dim=512,
                source={"format": "huggingface", "repo_id": "test/model"},
                runtimes={"torch": {"available": True}},
            )

    def test_model_validation_embedding_dim_range(self):
        """Test embedding_dim range validation."""
        with pytest.raises(Exception):
            ModelConfigurationSchema(
                name="test",
                version="1.0.0",
                description="Test",
                model_type="clip",
                embedding_dim=0,  # Must be >= 1
                source={"format": "huggingface", "repo_id": "test/model"},
                runtimes={"torch": {"available": True}},
            )

    def test_model_alias(self):
        """Test that ModelInfo is an alias for ModelConfigurationSchema."""
        assert ModelInfo is ModelConfigurationSchema


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_load_and_validate_model_info_strict(self, valid_model_info_file):
        """Test load_and_validate_model_info with strict mode."""
        model = load_and_validate_model_info(valid_model_info_file, strict=True)
        assert isinstance(model, ModelConfigurationSchema)
        assert model.name == "test-model"

    def test_load_and_validate_model_info_non_strict(self, valid_model_info_file):
        """Test load_and_validate_model_info with non-strict mode."""
        model = load_and_validate_model_info(valid_model_info_file, strict=False)
        assert isinstance(model, ModelConfigurationSchema)
        assert model.name == "test-model"

    def test_load_and_validate_model_info_invalid_strict(self, tmp_path):
        """Test load_and_validate_model_info raises on invalid data (strict)."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            json.dump({"name": "test"}, f)  # Missing required fields

        with pytest.raises(Exception):
            load_and_validate_model_info(invalid_file, strict=True)

    def test_load_and_validate_model_info_invalid_non_strict(self, tmp_path):
        """Test load_and_validate_model_info raises on invalid data (non-strict)."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            json.dump({"name": "test"}, f)  # Missing required fields

        with pytest.raises(ValueError):
            load_and_validate_model_info(invalid_file, strict=False)

    def test_validate_file_convenience(self, valid_model_info_file):
        """Test validate_file convenience function."""
        is_valid, errors = validate_file(valid_model_info_file, strict=False)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_file_convenience_invalid(self, tmp_path):
        """Test validate_file convenience function with invalid data."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            json.dump({"name": "test"}, f)  # Missing required fields

        is_valid, errors = validate_file(invalid_file, strict=False)
        assert is_valid is False
        assert len(errors) > 0


class TestComplexModelInfoStructures:
    """Test suite for complex model_info structures."""

    def test_runtime_with_device_specific_files(self):
        """Test runtime with device-specific files structure."""
        data = {
            "name": "test-model",
            "version": "1.0.0",
            "description": "Test",
            "model_type": "clip",
            "embedding_dim": 512,
            "source": {"format": "openclip", "repo_id": "test/model"},
            "runtimes": {
                "rknn": {
                    "available": True,
                    "devices": ["rk3566", "rk3588"],
                    "files": {
                        "rk3566": ["model_rk3566.rknn"],
                        "rk3588": ["model_rk3588.rknn"],
                    },
                }
            },
        }

        validator = ModelInfoValidator()
        is_valid, model, errors = validator.validate_pydantic(data)
        assert is_valid is True
        assert model is not None
        assert isinstance(model.runtimes["rknn"].files, dict)
        assert "rk3566" in model.runtimes["rknn"].files

    def test_with_metadata_and_datasets(self):
        """Test model info with metadata and datasets."""
        data = {
            "name": "test-model",
            "version": "1.0.0",
            "description": "Test",
            "model_type": "clip",
            "embedding_dim": 768,
            "source": {"format": "modelscope", "repo_id": "test/model"},
            "runtimes": {"torch": {"available": True}},
            "datasets": {"ImageNet": "imagenet.npy", "COCO": "coco.npy"},
            "metadata": {
                "license": "MIT",
                "author": "Test Author",
                "created_at": "2025-01-01",
                "tags": ["vision", "embedding"],
            },
        }

        validator = ModelInfoValidator()
        is_valid, model, errors = validator.validate_pydantic(data)
        assert is_valid is True
        assert model is not None
        assert model.datasets is not None
        assert "ImageNet" in model.datasets
        assert model.metadata is not None
        assert model.metadata.license == "MIT"
        assert model.metadata.tags == ["vision", "embedding"]

    def test_all_source_formats(self):
        """Test all supported source formats."""
        formats = ["huggingface", "openclip", "modelscope", "custom"]

        for fmt in formats:
            data = {
                "name": "test-model",
                "version": "1.0.0",
                "description": "Test",
                "model_type": "clip",
                "embedding_dim": 512,
                "source": {"format": fmt, "repo_id": f"test/{fmt}-model"},
                "runtimes": {"torch": {"available": True}},
            }

            validator = ModelInfoValidator()
            is_valid, model, errors = validator.validate_pydantic(data)
            assert is_valid is True, f"Failed for format: {fmt}"
            assert model.source.format.value == fmt

    def test_runtime_with_requirements(self):
        """Test runtime with requirements field."""
        data = {
            "name": "test-model",
            "version": "1.0.0",
            "description": "Test",
            "model_type": "clip",
            "embedding_dim": 512,
            "source": {"format": "custom", "repo_id": "test/model"},
            "runtimes": {
                "torch": {
                    "available": True,
                    "files": ["model.pt"],
                    "requirements": {
                        "python": ">=3.10",
                        "dependencies": ["torch>=2.0", "transformers"],
                    },
                }
            },
        }

        validator = ModelInfoValidator()
        is_valid, model, errors = validator.validate_pydantic(data)
        assert is_valid is True
        assert model is not None
        assert model.runtimes["torch"].requirements is not None
        assert model.runtimes["torch"].requirements.python == ">=3.10"
        assert "torch>=2.0" in model.runtimes["torch"].requirements.dependencies
