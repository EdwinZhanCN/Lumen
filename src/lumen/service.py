from pathlib import Path

from lumen_resources.lumen_config import LumenConfig, Services

from .loader import ServiceLoader  # 负责动态 importlib
from .router import HubRouter
from .utils.logger import get_logger

logger = get_logger("lumen.service")


class AppService:
    def __init__(self, services: list[Services], config: LumenConfig):
        self.services = services
        self.config = config
        # 将实例映射到路由，支持多对多
        self.router = HubRouter(services)

    @classmethod
    def from_app_config(cls, config: LumenConfig):
        """
        核心工厂方法：解析 LumenConfig 并初始化所有子服务
        """
        loader = ServiceLoader()
        instances = []
        cache_dir = Path(config.metadata.cache_dir)

        # 遍历配置中定义的所有服务
        for name, svc_cfg in config.services.items():
            if not svc_cfg.enabled:
                continue

            # 1. 动态获取服务类 (例如从 lumen_ocr.registry 获取)
            # svc_cfg.import_info.registry_class 存储了类路径
            if svc_cfg.import_info is not None:
                service_cls = loader.get_class(svc_cfg.import_info.registry_class)

                # 2. 调用你重构好的 from_config 方法
                instance = service_cls.from_config(
                    service_config=svc_cfg, cache_dir=cache_dir
                )
                instances.append(instance)
                logger.info(f"Loaded service: {name} with package {svc_cfg.package}")
            else:
                raise RuntimeError(
                    f"Cannot load import_info from configuration for service:{svc_cfg.package}"
                )

        return cls(services=instances, config=config)
