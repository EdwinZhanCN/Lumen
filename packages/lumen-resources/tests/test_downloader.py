from lumen_resources.downloader import Downloader
from lumen_resources.lumen_config import LumenConfig
from lumen_resources.platform import PlatformType


def build_config(region: str) -> LumenConfig:
    return LumenConfig.model_validate(
        {
            "metadata": {
                "version": "1.0.0",
                "region": region,
                "cache_dir": "~/.lumen/test-cache",
            },
            "deployment": {
                "mode": "single",
                "service": "clip",
            },
            "server": {
                "port": 50051,
                "host": "127.0.0.1",
            },
            "services": {
                "clip": {
                    "enabled": True,
                    "package": "lumen_clip",
                    "import_info": {
                        "registry_class": (
                            "lumen_clip.general_clip.clip_service.GeneralCLIPService"
                        ),
                        "add_to_server": (
                            "lumen_clip.proto.ml_service_pb2_grpc"
                            ".add_InferenceServicer_to_server"
                        ),
                    },
                    "models": {
                        "general": {
                            "model": "MobileCLIP2-S2",
                            "runtime": "onnx",
                        }
                    },
                }
            },
        }
    )


def test_downloader_uses_modelscope_for_cn_region() -> None:
    downloader = Downloader(build_config("cn"), verbose=False)

    assert downloader.platform.platform_type == PlatformType.MODELSCOPE
    assert downloader.platform.owner == "LumilioPhotos"


def test_downloader_temporarily_uses_modelscope_for_other_region() -> None:
    downloader = Downloader(build_config("other"), verbose=False)

    assert downloader.platform.platform_type == PlatformType.MODELSCOPE
    assert downloader.platform.owner == "LumilioPhotos"
