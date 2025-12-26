import importlib
from typing import Any, Type

from lumen_app.utils.logger import get_logger

logger = get_logger("lumen.loader")


class ServiceLoader:
    """
    负责从字符串路径动态加载 Python 类。
    例如将 "lumen_ocr.registry.GeneralOcrService" 转换为可调用的类对象。
    """

    @staticmethod
    def get_class(class_path: str) -> Type[Any]:
        """
        根据类路径字符串获取类对象。

        Args:
            class_path: 格式为 'package.module.ClassName'
        """
        if not class_path or "." not in class_path:
            raise ValueError(f"Invalid class path: {class_path}")

        # 1. 拆分路径，例如：'lumen_ocr.registry' 和 'GeneralOcrService'
        module_path, class_name = class_path.rsplit(".", 1)

        try:
            # 2. 动态导入模块
            # 注意：此时 subprocess 必须运行在已安装该包的 micromamba 环境中
            module = importlib.import_module(module_path)

            # 3. 从模块中获取类
            cls = getattr(module, class_name)

            logger.info(f"Successfully imported class: {class_name} from {module_path}")
            return cls

        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in module {module_path}: {e}")
            raise
