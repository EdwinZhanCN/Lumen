"""
Tests for configuration parsing and validation (YAML-based unified config)
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from lumen_resources import ResourceConfig, ModelConfig, RuntimeType, PlatformType
from lumen_resources.exceptions import ConfigError


def test_valid_config():
    """Test parsing a valid configuration (YAML)."""
    yaml_text = """\
metadata:
  region: other
  cache_dir: "~/.lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP-L-14"
        runtime: "onnx"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        config_path = Path(f.name)

    try:
        config = ResourceConfig.from_yaml(config_path)

        assert config.region == "other"
        assert config.platform_type == PlatformType.HUGGINGFACE
        assert config.platform_owner == "Lumilio-Photos"
        assert "clip:default" in config.models
        assert config.models["clip:default"].model == "MobileCLIP-L-14"
        assert config.models["clip:default"].runtime == RuntimeType.ONNX
    finally:
        config_path.unlink()


def test_cn_region():
    """Test Chinese region selects ModelScope."""
    yaml_text = """\
metadata:
  region: cn
  cache_dir: "~/.lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP-L-14"
        runtime: "torch"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        config_path = Path(f.name)

    try:
        config = ResourceConfig.from_yaml(config_path)

        assert config.platform_type == PlatformType.MODELSCOPE
        assert config.platform_owner == "LumilioPhotos"
    finally:
        config_path.unlink()


def test_missing_region():
    """Test that missing metadata.region raises ConfigError."""
    yaml_text = """\
metadata:
  cache_dir: "~/.lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP-L-14"
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


def test_invalid_region():
    """Test that invalid region value raises ConfigError."""
    yaml_text = """\
metadata:
  region: invalid
  cache_dir: "~/.lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP-L-14"
        runtime: "onnx"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        config_path = Path(f.name)

    try:
        with pytest.raises(ConfigError, match="Invalid region"):
            ResourceConfig.from_yaml(config_path)
    finally:
        config_path.unlink()


def test_no_models():
    """Test that config with no enabled services/models raises ConfigError."""
    yaml_text = """\
metadata:
  region: other
  cache_dir: "~/.lumen/"
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
            ConfigError,
            match="At least one enabled service with non-empty models is required",
        ):
            ResourceConfig.from_yaml(config_path)
    finally:
        config_path.unlink()


def test_invalid_runtime():
    """Test that invalid runtime raises ConfigError."""
    yaml_text = """\
metadata:
  region: other
  cache_dir: "~/.lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP-L-14"
        runtime: "invalid_runtime"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        config_path = Path(f.name)

    try:
        with pytest.raises(ConfigError, match="Invalid runtime"):
            ResourceConfig.from_yaml(config_path)
    finally:
        config_path.unlink()


def test_rknn_device():
    """Test RKNN runtime with device specification."""
    yaml_text = """\
metadata:
  region: other
  cache_dir: "~/.lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP-L-14"
        runtime: "rknn"
        rknn_device: "rk3588"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        config_path = Path(f.name)

    try:
        config = ResourceConfig.from_yaml(config_path)

        assert config.models["clip:default"].runtime == RuntimeType.RKNN
        assert config.models["clip:default"].rknn_device == "rk3588"

        # Test runtime patterns
        patterns = config.models["clip:default"].get_runtime_patterns()
        assert "rknn/rk3588/*" in patterns
        assert "*.json" in patterns
    finally:
        config_path.unlink()


def test_dataset_specification():
    """Test model with dataset specification."""
    yaml_text = """\
metadata:
  region: other
  cache_dir: "~/.lumen/"
services:
  bioclip:
    enabled: true
    models:
      default:
        model: "bioclip-2"
        runtime: "torch"
        dataset: "TreeOfLife-200M"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        config_path = Path(f.name)

    try:
        config = ResourceConfig.from_yaml(config_path)

        assert config.models["bioclip:default"].dataset == "TreeOfLife-200M"

        # Test dataset pattern
        dataset_pattern = config.models["bioclip:default"].get_dataset_pattern()
        assert dataset_pattern == "TreeOfLife-200M.npz"
    finally:
        config_path.unlink()


def test_multiple_models():
    """Test configuration with multiple services/models."""
    yaml_text = """\
metadata:
  region: other
  cache_dir: "~/.lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP-L-14"
        runtime: "onnx"
  face:
    enabled: true
    models:
      default:
        model: "antelopev2"
        runtime: "onnx"
  bioclip:
    enabled: true
    models:
      default:
        model: "bioclip-2"
        runtime: "torch"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        config_path = Path(f.name)

    try:
        config = ResourceConfig.from_yaml(config_path)

        assert len(config.models) == 3
        assert "clip:default" in config.models
        assert "face:default" in config.models
        assert "bioclip:default" in config.models
    finally:
        config_path.unlink()


def test_repo_id_generation():
    """Test repository ID generation."""
    yaml_text = """\
metadata:
  region: other
  cache_dir: "~/.lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP-L-14"
        runtime: "onnx"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        config_path = Path(f.name)

    try:
        config = ResourceConfig.from_yaml(config_path)

        repo_id = config.get_repo_id("MobileCLIP-L-14")
        assert repo_id == "Lumilio-Photos/MobileCLIP-L-14"
    finally:
        config_path.unlink()


def test_file_not_found():
    """Test that non-existent config file raises ConfigError."""
    with pytest.raises(ConfigError, match="Configuration file not found"):
        ResourceConfig.from_yaml(Path("nonexistent.yaml"))


def test_invalid_yaml():
    """Test that invalid YAML raises ConfigError."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("metadata: [unclosed")
        config_path = Path(f.name)

    try:
        with pytest.raises(ConfigError, match="Invalid YAML format"):
            ResourceConfig.from_yaml(config_path)
    finally:
        config_path.unlink()


def test_config_patterns_include_all_json_files():
    """Test that *.json pattern includes all configuration files."""
    yaml_text = """\
metadata:
  region: cn
  cache_dir: "~/.lumen/"
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

        # Test that all runtimes include *.json pattern
        for runtime in ["torch", "onnx", "rknn"]:
            model_config = ModelConfig(
                model="test-model",
                runtime=RuntimeType(runtime),
                rknn_device="rk3588" if runtime == "rknn" else None,
            )
            patterns = model_config.get_runtime_patterns()

            # Should include *.json for all JSON config files
            assert "*.json" in patterns, (
                f"Runtime {runtime} should include *.json pattern"
            )

            # Should include runtime-specific patterns
            if runtime == "torch":
                assert "torch/*" in patterns
            elif runtime == "onnx":
                assert "onnx/*" in patterns
            elif runtime == "rknn":
                assert "rknn/rk3588/*" in patterns

    finally:
        config_path.unlink()
