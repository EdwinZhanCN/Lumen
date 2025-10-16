"""
Configuration validation and error handling tests.

Tests for malformed configs, missing required fields, invalid values,
and edge cases that should raise ConfigError.
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from lumen_resources import ResourceConfig
from lumen_resources.exceptions import ConfigError


class TestMissingRequiredFields:
    """Test validation of required fields."""

    def test_missing_metadata(self):
        """Test that missing metadata raises ConfigError."""
        yaml_text = """\
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Missing required field: metadata"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_missing_region(self):
        """Test that missing metadata.region raises ConfigError."""
        yaml_text = """\
metadata:
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError, match="Missing required field: metadata.region"
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_missing_cache_dir(self):
        """Test that missing metadata.cache_dir raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError, match="Missing required field: metadata.cache_dir"
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_empty_cache_dir(self):
        """Test that empty cache_dir raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: ""
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError, match="Missing required field: metadata.cache_dir"
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_missing_services(self):
        """Test that missing services section raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Missing required field: services"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_missing_model_field(self):
        """Test that missing 'model' in model spec raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError, match="Missing 'model' in service 'clip', alias 'default'"
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_missing_runtime_field(self):
        """Test that missing 'runtime' in model spec raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError,
                match="Missing 'runtime' in service 'clip', alias 'default'",
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()


class TestInvalidValues:
    """Test validation of invalid field values."""

    def test_invalid_region(self):
        """Test that invalid region value raises ConfigError."""
        yaml_text = """\
metadata:
  region: invalid_region
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError, match="Invalid region.*must be 'cn' or 'other'"
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_invalid_runtime(self):
        """Test that invalid runtime value raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "tensorflow"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Invalid runtime"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_invalid_runtime_case_sensitive(self):
        """Test that runtime is case-sensitive."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "ONNX"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Invalid runtime"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()


class TestEmptyOrMalformedStructures:
    """Test empty or malformed configuration structures."""

    def test_empty_services(self):
        """Test that empty services dict raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services: {}
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="At least one enabled service"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_all_services_disabled(self):
        """Test that all disabled services raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: false
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
  face:
    enabled: false
    models:
      default:
        model: "antelopev2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError,
                match="At least one enabled service with non-empty models is required",
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_enabled_service_with_empty_models(self):
        """Test that enabled service with empty models raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models: {}
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError, match="At least one enabled service with non-empty models"
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_enabled_service_with_null_models(self):
        """Test that enabled service with null models raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models: null
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="At least one enabled service"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_non_dict_root(self):
        """Test that non-dict root raises ConfigError."""
        yaml_text = """\
- item1
- item2
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError, match="Configuration root must be a mapping"
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_non_dict_metadata(self):
        """Test that non-dict metadata raises ConfigError."""
        yaml_text = """\
metadata: "string instead of dict"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Missing required field: metadata"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_non_dict_services(self):
        """Test that non-dict services raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services: "not a dict"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Missing required field: services"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_non_dict_model_spec(self):
        """Test that non-dict model spec raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default: "not a dict"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigError,
                match="Invalid model spec for service 'clip', alias 'default'",
            ):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()


class TestFileErrors:
    """Test file-related errors."""

    def test_file_not_found(self):
        """Test that non-existent config file raises ConfigError."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            ResourceConfig.from_yaml(Path("/nonexistent/path/config.yaml"))

    def test_invalid_yaml_syntax(self):
        """Test that invalid YAML syntax raises ConfigError."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
  unclosed: [list
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Invalid YAML format"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_yaml_with_tabs(self):
        """Test YAML with tabs (should fail or handle gracefully)."""
        yaml_text = "metadata:\n\tregion: other\n\tcache_dir: /opt/lumen/\nservices:\n\tclip:\n\t\tenabled: true\n\t\tmodels:\n\t\t\tdefault:\n\t\t\t\tmodel: test\n\t\t\t\truntime: onnx"
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Invalid YAML format"):
                ResourceConfig.from_yaml(config_path)
        finally:
            config_path.unlink()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_service_name_with_special_chars(self):
        """Test service names with hyphens and underscores."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  my-custom_service:
    enabled: true
    models:
      default:
        model: "test-model"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert "my-custom_service:default" in config.models
        finally:
            config_path.unlink()

    def test_model_alias_with_special_chars(self):
        """Test model aliases with hyphens and underscores."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      model_v2-beta:
        model: "test-model"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert "clip:model_v2-beta" in config.models
        finally:
            config_path.unlink()

    def test_very_long_model_name(self):
        """Test handling of very long model names."""
        long_name = "a" * 200
        yaml_text = f"""\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "{long_name}"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert config.models["clip:default"].model == long_name
        finally:
            config_path.unlink()

    def test_unicode_in_model_name(self):
        """Test Unicode characters in model names."""
        yaml_text = """\
metadata:
  region: cn
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "模型-中文-v1"
        runtime: "onnx"
"""
        with NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert "模型-中文-v1" in config.models["clip:default"].model
        finally:
            config_path.unlink()

    def test_extra_fields_ignored(self):
        """Test that extra/unknown fields are gracefully ignored."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
  extra_field: "ignored"
  version: "1.0"
services:
  clip:
    enabled: true
    package: "lumen-clip"
    unknown_field: "should be ignored"
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
        extra_param: "ignored too"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            # Should not raise error; extra fields are ignored
            config = ResourceConfig.from_yaml(config_path)
            assert "clip:default" in config.models
        finally:
            config_path.unlink()

    def test_whitespace_in_paths(self):
        """Test paths with whitespace."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/my lumen/models dir/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert "my lumen" in str(config.cache_dir)
        finally:
            config_path.unlink()
