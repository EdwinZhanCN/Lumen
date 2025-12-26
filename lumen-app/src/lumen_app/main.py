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
    level=logging.DEBUG,  # Set to INFO for less verbose logging
    log_dir=log_dir,
)

logger = logging.getLogger("lumen")
logger.info("Lumen Hub application starting")


def start_app(port=8000):
    """Start the flet application"""
    ft.app(target=main, port=port)


if __name__ == "__main__":
    # Check if a port is specified via command line arguments
    port = 8000  # default port
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.warning(
                f"Invalid port number: {sys.argv[1]}. Using default port 8000."
            )

    logger.info(f"Starting Lumen Hub application on port {port}")
    start_app(port)
