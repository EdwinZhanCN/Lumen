"""Main entry point for Lumen Hub application."""

import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import flet as ft

from lumen_app.ui.app import main
from lumen_app.utils.logger import initialize

# Setup logging first
log_dir = Path.home() / ".lumen" / "logs"
initialize(
    level=logging.DEBUG,  # Set to DEBUG for more verbose logging
    log_dir=log_dir,
)

logger = logging.getLogger("lumen")
logger.info("Lumen Hub application starting")


def start_app():
    """Start the flet application"""
    ft.app(target=main)


if __name__ == "__main__":
    start_app()
