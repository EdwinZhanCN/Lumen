"""
Advanced configuration tests: multi-service, filtering, and complex scenarios.

Tests for Hub aggregation mode, service filtering, model deduplication,
and complex multi-model configurations.
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from lumen_resources import ResourceConfig, RuntimeType
from lumen_resources.exceptions import ConfigError


class TestMultiService:
    """Test configurations with multiple services."""

    def test_multiple_services_all_enabled(self):
        """Test configuration with multiple enabled services."""
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
    enabled: true
    models:
      default:
        model: "antelopev2"
        runtime: "onnx"
  ocr:
    enabled: true
    models:
      default:
        model: "paddleocr-v3"
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
            assert "ocr:default" in config.models
        finally:
            config_path.unlink()

    def test_multiple_services_mixed_enabled(self):
        """Test Hub config with some services enabled and some disabled."""
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
  face:
    enabled: false
    models:
      default:
        model: "antelopev2"
        runtime: "onnx"
  ocr:
    enabled: true
    models:
      default:
        model: "paddleocr-v3"
        runtime: "torch"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            # Only enabled services should be loaded
            assert len(config.models) == 2
            assert "clip:default" in config.models
            assert "face:default" not in config.models
            assert "ocr:default" in config.models
        finally:
            config_path.unlink()

    def test_multiple_services_different_runtimes(self):
        """Test services with different runtime types."""
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
    enabled: true
    models:
      default:
        model: "antelopev2"
        runtime: "torch"
  ocr:
    enabled: true
    models:
      default:
        model: "paddleocr-v3"
        runtime: "rknn"
        rknn_device: "rk3588"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert config.models["clip:default"].runtime == RuntimeType.ONNX
            assert config.models["face:default"].runtime == RuntimeType.TORCH
            assert config.models["ocr:default"].runtime == RuntimeType.RKNN
            assert config.models["ocr:default"].rknn_device == "rk3588"
        finally:
            config_path.unlink()


class TestServiceFiltering:
    """Test filtering specific services (single module mode)."""

    def test_filter_single_service(self):
        """Test loading only a specific service (clip)."""
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
            # Single module mode: only load clip
            config = ResourceConfig.from_yaml(config_path, only_services=["clip"])
            assert len(config.models) == 1
            assert "clip:default" in config.models
            assert "face:default" not in config.models
        finally:
            config_path.unlink()

    def test_filter_multiple_services(self):
        """Test loading multiple specific services."""
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
    enabled: true
    models:
      default:
        model: "antelopev2"
        runtime: "onnx"
  ocr:
    enabled: true
    models:
      default:
        model: "paddleocr-v3"
        runtime: "torch"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            # Load only clip and face
            config = ResourceConfig.from_yaml(
                config_path, only_services=["clip", "face"]
            )
            assert len(config.models) == 2
            assert "clip:default" in config.models
            assert "face:default" in config.models
            assert "ocr:default" not in config.models
        finally:
            config_path.unlink()

    def test_filter_nonexistent_service(self):
        """Test filtering a service that doesn't exist in config."""
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
            # Try to load nonexistent service
            with pytest.raises(
                ConfigError, match="No enabled services with models found"
            ):
                ResourceConfig.from_yaml(config_path, only_services=["nonexistent"])
        finally:
            config_path.unlink()

    def test_filter_disabled_service(self):
        """Test filtering a service that exists but is disabled."""
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
            # Try to load disabled service
            with pytest.raises(
                ConfigError, match="No enabled services with models found"
            ):
                ResourceConfig.from_yaml(config_path, only_services=["clip"])
        finally:
            config_path.unlink()


class TestMultiModelPerService:
    """Test services with multiple models."""

    def test_service_with_multiple_model_aliases(self):
        """Test a service with multiple model variants."""
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
      expert:
        model: "bioclip-2"
        runtime: "torch"
        dataset: "TreeOfLife-200M"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert len(config.models) == 3
            assert "clip:small" in config.models
            assert "clip:large" in config.models
            assert "clip:expert" in config.models
            assert config.models["clip:expert"].dataset == "TreeOfLife-200M"
        finally:
            config_path.unlink()

    def test_multiple_services_multiple_models_each(self):
        """Test multiple services each with multiple models."""
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
  face:
    enabled: true
    models:
      detection:
        model: "antelopev2"
        runtime: "onnx"
      recognition:
        model: "arcface-r100"
        runtime: "onnx"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert len(config.models) == 4
            assert "clip:small" in config.models
            assert "clip:large" in config.models
            assert "face:detection" in config.models
            assert "face:recognition" in config.models
        finally:
            config_path.unlink()


class TestModelDeduplication:
    """Test model deduplication scenarios (same model, different services)."""

    def test_same_model_different_services(self):
        """Test that same model used by different services creates separate entries."""
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
  bioclip:
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
            # Should create separate entries even though same model
            assert len(config.models) == 2
            assert "clip:default" in config.models
            assert "bioclip:default" in config.models
            assert config.models["clip:default"].model == "MobileCLIP2-S2"
            assert config.models["bioclip:default"].model == "MobileCLIP2-S2"
        finally:
            config_path.unlink()

    def test_same_model_different_runtimes(self):
        """Test same model with different runtimes in different services."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip_onnx:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
  clip_torch:
    enabled: true
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "torch"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert len(config.models) == 2
            assert config.models["clip_onnx:default"].runtime == RuntimeType.ONNX
            assert config.models["clip_torch:default"].runtime == RuntimeType.TORCH
        finally:
            config_path.unlink()


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_hub_with_optional_fields(self):
        """Test Hub config with all optional fields populated."""
        yaml_text = """\
metadata:
  version: "1.0"
  region: cn
  cache_dir: "/opt/lumen/"
dependencies:
  - "lumen-clip @ git+https://github.com/user/repo.git#subdirectory=lumen-clip"
services:
  clip:
    enabled: true
    package: "lumen-clip"
    import:
      registry_class: "lumen_clip.service_registry.CLIPService"
      add_to_server: "lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
    default_model: "default"
    env:
      CLIP_BACKEND: "onnx"
    server:
      port: 50051
      mdns:
        enabled: true
        name: "CLIP-Service"
hub:
  enabled: true
  server:
    port: 50050
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            # ResourceConfig only cares about metadata and services.models
            config = ResourceConfig.from_yaml(config_path)
            assert config.region == "cn"
            assert "clip:default" in config.models
        finally:
            config_path.unlink()

    def test_mixed_runtimes_and_datasets(self):
        """Test complex config with mixed runtimes, devices, and datasets."""
        yaml_text = """\
metadata:
  region: other
  cache_dir: "/opt/lumen/"
services:
  clip:
    enabled: true
    models:
      onnx_model:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
      torch_model:
        model: "MobileCLIP-L-14"
        runtime: "torch"
  bioclip:
    enabled: true
    models:
      expert:
        model: "bioclip-2"
        runtime: "torch"
        dataset: "TreeOfLife-200M"
  face:
    enabled: true
    models:
      edge:
        model: "antelopev2"
        runtime: "rknn"
        rknn_device: "rk3588"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            config_path = Path(f.name)

        try:
            config = ResourceConfig.from_yaml(config_path)
            assert len(config.models) == 4

            # Verify ONNX
            assert config.models["clip:onnx_model"].runtime == RuntimeType.ONNX

            # Verify Torch
            assert config.models["clip:torch_model"].runtime == RuntimeType.TORCH

            # Verify dataset
            assert config.models["bioclip:expert"].dataset == "TreeOfLife-200M"

            # Verify RKNN
            assert config.models["face:edge"].runtime == RuntimeType.RKNN
            assert config.models["face:edge"].rknn_device == "rk3588"
        finally:
            config_path.unlink()

    def test_load_example_config_minimal(self):
        """Test loading the actual minimal example config."""
        example_path = Path("examples/config_minimal.yaml")
        if not example_path.exists():
            pytest.skip("Example config not found")

        config = ResourceConfig.from_yaml(example_path)
        assert config.region == "other"
        assert "clip:default" in config.models

    def test_load_example_config_hub(self):
        """Test loading the actual Hub example config."""
        example_path = Path("examples/config_hub.yaml")
        if not example_path.exists():
            pytest.skip("Example config not found")

        config = ResourceConfig.from_yaml(example_path)
        assert config.region == "cn"
        # Should load enabled services only
        assert "clip:default" in config.models or "clip:expert" in config.models
        # OCR is disabled in example
        assert not any(k.startswith("ocr:") for k in config.models.keys())

    def test_load_example_config_rknn(self):
        """Test loading the actual RKNN edge example config."""
        example_path = Path("examples/config_rknn_edge.yaml")
        if not example_path.exists():
            pytest.skip("Example config not found")

        config = ResourceConfig.from_yaml(example_path)
        assert config.region == "cn"
        # Should have RKNN models
        rknn_models = [
            m for m in config.models.values() if m.runtime == RuntimeType.RKNN
        ]
        assert len(rknn_models) > 0

    def test_load_example_config_multi_models(self):
        """Test loading the multi-model example config."""
        example_path = Path("examples/config_multi_models.yaml")
        if not example_path.exists():
            pytest.skip("Example config not found")

        config = ResourceConfig.from_yaml(example_path)
        # Should have multiple CLIP models
        clip_models = [k for k in config.models.keys() if k.startswith("clip:")]
        assert len(clip_models) > 1
