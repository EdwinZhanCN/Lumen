"""Centralized logging configuration for Lumen applications.

This module provides a unified logging setup with colorized console output
and optional file logging. Usage:

    from lumen_app.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Application started")

Or use the predefined loggers:

    from lumen_app.utils.logger import lumen_logger

    lumen_logger.info("Using predefined logger")
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Try to import colorlog, fallback to standard logging
try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


# Color scheme for different log levels
LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    enable_colors: bool = True,
) -> None:
    """Setup root logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        enable_colors: Whether to enable colored output (requires colorlog)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if COLORLOG_AVAILABLE and enable_colors:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)8s] %(name)s:%(reset)s %(message)s",
            datefmt=DATE_FORMAT,
            log_colors=LOG_COLORS,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Suppress verbose third-party loggers
    _suppress_third_party_loggers()


def _suppress_third_party_loggers() -> None:
    """Suppress verbose debug logging from third-party libraries."""
    # List of loggers to suppress
    suppressed_loggers = {
        "flet": logging.WARNING,
        "flet_core": logging.WARNING,
        "flet.view": logging.WARNING,
        "httpx": logging.WARNING,
        "urllib3": logging.WARNING,
        "websockets": logging.WARNING,
        "asyncio": logging.WARNING,
    }

    for logger_name, logger_level in suppressed_loggers.items():
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(logger_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return logging.getLogger(name)


# Predefined loggers for common modules
def _get_module_logger(module_name: str) -> logging.Logger:
    """Get or create a logger for a specific module."""
    logger = logging.getLogger(module_name)
    # Ensure logger has a level set (root level will be used if not set)
    if not logger.setLevel:
        logger.setLevel(logging.DEBUG)
    return logger


# Export predefined loggers
lumen_logger = _get_module_logger("lumen")
config_logger = _get_module_logger("lumen.config")
env_checker_logger = _get_module_logger("lumen.env_checker")
installer_logger = _get_module_logger("lumen.installer")
ui_logger = _get_module_logger("lumen.ui")


# Auto-setup on import (can be called explicitly if needed)
def initialize(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> None:
    """Initialize logging system.

    This is called automatically on import with default settings.
    Call it explicitly to customize logging behavior.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Specific log file path
        log_dir: Directory for log files (alternative to log_file)

    Example:
        # Initialize with DEBUG level and custom log file
        initialize(
            level=logging.DEBUG,
            log_file=Path("~/lumen_logs/app.log")
        )
    """
    # Determine log file path
    if log_file is None and log_dir is not None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"lumen_{timestamp}.log"

    setup_logging(level=level, log_file=log_file, enable_colors=COLORLOG_AVAILABLE)


# Auto-initialize with defaults
initialize(level=logging.INFO)
