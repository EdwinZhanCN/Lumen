from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lumen-vlm")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .cli import main

__all__ = ["main"]
