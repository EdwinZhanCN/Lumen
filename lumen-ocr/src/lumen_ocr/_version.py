"""Version information for lumen-ocr."""

try:
    import sys
    import os

    # Add parent directory to path to import from root
    root_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, root_path)
    import lumen_version

    __version__ = lumen_version.version
except ImportError:
    __version__ = "0.0.0"
