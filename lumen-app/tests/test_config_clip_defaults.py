"""Tests for region-aware CLIP model defaults."""

from lumen_resources.lumen_config import Region

from lumen_app.services.config import Config, DeviceConfig


def build_config(region: Region) -> Config:
    """Create a config instance for region default selection tests."""
    return Config(
        cache_dir="~/.lumen",
        device_config=DeviceConfig.cpu(),
        region=region,
        service_name="lumen-server",
        port=50051,
    )


def test_cn_region_prefers_cn_clip_defaults():
    """CN region should default to CN-CLIP variants."""
    config = build_config(Region.cn)

    assert config.default_light_weight_clip_model() == "CN-CLIP_ViT-B-16"
    assert config.default_basic_clip_model() == "CN-CLIP_ViT-L-14"


def test_other_region_prefers_mobileclip_defaults():
    """Non-CN regions should keep MobileCLIP defaults."""
    config = build_config(Region.other)

    assert config.default_light_weight_clip_model() == "MobileCLIP2-S2"
    assert config.default_basic_clip_model() == "MobileCLIP2-S4"
