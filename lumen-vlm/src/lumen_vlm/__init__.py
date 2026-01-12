from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lumen-vlm")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .cli import main
from .runtime_info import RuntimeModelInfo

__all__ = ["main", "RuntimeModelInfo"]
