"""Main entry point for Lumen Hub application."""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import flet as ft

from lumen_app.ui.app import main


def start_app():
    """Start the flet application"""
    ft.app(target=main)


if __name__ == "__main__":
    start_app()
