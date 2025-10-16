"""
Basic configuration parsing and validation tests.

Tests fundamental config structure, metadata parsing, and single-service scenarios.
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from lumen_resources import ResourceConfig, ModelConfig, RuntimeType, PlatformType
from lumen_resources.exceptions import ConfigError


class TestBasicStructure:
    """Test basic YAML structure and metadata parsing."""

    def test_minimal_valid_config(self):
        """Test the absolute minimum valid configuration."""
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
            assert "clip:default" in config.models
        finally:
            config_path.unlink()

    def test_region_cn_selects_modelscope(self):
        """Test that region 'cn' correctly selects ModelScope platform."""
        yaml_text = """\
metadata:
  region: cn
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
            config = ResourceConfig.from_yaml(config_path)
            assert config.region == "cn"
            assert config.platform_type == PlatformType.MODELSCOPE
            assert config.platform_owner == "LumilioPhotos"
        finally:
            config_path.unlink()

    def test_region_other_selects_huggingface(self):
        """Test that region 'other' correctly selects HuggingFace platform."""
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
        finally:
            config_path.unlink()

    def test_cache_dir_expansion(self):
        """Test that tilde in cache_dir is properly expanded."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "~/custom/.lumen/"
services:
  clip:
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
            assert not str(config.cache_dir).startswith("~")
            assert "custom" in str(config.cache_dir)
        finally:
            config_path.unlink()


class TestSingleService:
    """Test single service configurations."""

    def test_single_service_single_model(self):
        """Test configuration with one service and one model."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  face:
    enabled: true
    models:
      default:
        model: "antelopev2"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert len(config.models) == 1
            assert "face:default" in config.models
            assert config.models["face:default"].model == "antelopev2"
            assert config.models["face:default"].runtime == RuntimeType.ONNX
        finally:
            config_path.unlink()

    def test_single_service_multiple_models(self):
        """Test configuration with one service but multiple models."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      small:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
      large:
        model: "MobileCLIP-L-14"
        runtime: "torch"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert len(config.models) == 2
            assert "clip:small" in config.models
            assert "clip:large" in config.models
            assert config.models["clip:small"].runtime == RuntimeType.ONNX
            assert config.models["clip:large"].runtime == RuntimeType.TORCH
        finally:
            config_path.unlink()

    def test_disabled_service_not_loaded(self):
        """Test that disabled services are not included in models."""
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
            config = ResourceConfig.from_yaml(config_path)
            assert len(config.models) == 1
            assert "clip:default" in config.models
            assert "face:default" not in config.models
        finally:
            config_path.unlink()


class TestRuntimeTypes:
    """Test different runtime type configurations."""

    def test_onnx_runtime(self):
        """Test ONNX runtime configuration."""
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
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            model = config.models["clip:default"]
            assert model.runtime == RuntimeType.ONNX
            patterns = model.get_runtime_patterns()
            assert "onnx/*" in patterns
            assert "*.json" in patterns
        finally:
            config_path.unlink()

    def test_torch_runtime(self):
        """Test PyTorch runtime configuration."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
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
            model = config.models["clip:default"]
            assert model.runtime == RuntimeType.TORCH
            patterns = model.get_runtime_patterns()
            assert "torch/*" in patterns
            assert "*.json" in patterns
        finally:
            config_path.unlink()

    def test_rknn_runtime_with_device(self):
        """Test RKNN runtime with device specification."""
        yaml_text = """\
metadata:
  region: cn
  cache_dir: "/data/lumen/"
services:
  clip:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "rknn"
        rknn_device: "rk3588"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            model = config.models["clip:default"]
            assert model.runtime == RuntimeType.RKNN
            assert model.rknn_device == "rk3588"
            patterns = model.get_runtime_patterns()
            assert "rknn/rk3588/*" in patterns
            assert "*.json" in patterns
        finally:
            config_path.unlink()

    def test_rknn_runtime_rk3576_device(self):
        """Test RKNN runtime with alternative device (RK3576)."""
        yaml_text = """\
metadata:
  region: cn
  cache_dir: "/data/lumen/"
services:
  face:
    enabled: true
    models:
      default:
        model: "antelopev2"
        runtime: "rknn"
        rknn_device: "rk3576"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            model = config.models["face:default"]
            assert model.runtime == RuntimeType.RKNN
            assert model.rknn_device == "rk3576"
            patterns = model.get_runtime_patterns()
            assert "rknn/rk3576/*" in patterns
        finally:
            config_path.unlink()


class TestDatasetSupport:
    """Test dataset specification in model configs."""

    def test_model_with_dataset(self):
        """Test model configuration with dataset specification."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
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
            model = config.models["bioclip:default"]
            assert model.dataset == "TreeOfLife-200M"
            dataset_pattern = model.get_dataset_pattern()
            assert dataset_pattern == "TreeOfLife-200M.npz"
        finally:
            config_path.unlink()

    def test_model_without_dataset(self):
        """Test that dataset is optional and None when not specified."""
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
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            model = config.models["clip:default"]
            assert model.dataset is None
            assert model.get_dataset_pattern() is None
        finally:
            config_path.unlink()


class TestRepoIdGeneration:
    """Test repository ID generation for different platforms."""

    def test_repo_id_huggingface(self):
        """Test repo ID generation for HuggingFace."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
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

    def test_repo_id_modelscope(self):
        """Test repo ID generation for ModelScope."""
        yaml_text = """\
metadata:
  region: cn
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
            config = ResourceConfig.from_yaml(config_path)
            repo_id = config.get_repo_id("MobileCLIP2-S2")
            assert repo_id == "LumilioPhotos/MobileCLIP2-S2"
        finally:
            config_path.unlink()
