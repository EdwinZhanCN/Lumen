"""LogViewer component for real-time log display.

This module provides a reusable log viewer component with color-coded log levels,
timestamps, auto-scroll, and clipboard functionality.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import flet as ft


class LogLevel(Enum):
    """Log levels for color coding."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    DEBUG = "DEBUG"


class LogViewer(ft.Column):
    """A reusable log viewer component for displaying real-time logs.

    Features:
    - Auto-scrolling to latest log
    - Color-coded log levels
    - Timestamp support
    - Copy to clipboard
    - Clear logs
    - Max line limit with FIFO eviction
    - Optional file logging

    Attributes:
        max_lines (int): Maximum number of lines to keep in buffer (FIFO).
        auto_scroll (bool): Whether to auto-scroll to latest log.
        log_file (Path | None): Optional file path for log persistence.
    """

    # Color mappings for log levels
    LEVEL_COLORS = {
        LogLevel.INFO: ft.Colors.BLUE,
        LogLevel.WARNING: ft.Colors.YELLOW_700,
        LogLevel.ERROR: ft.Colors.RED,
        LogLevel.SUCCESS: ft.Colors.GREEN,
        LogLevel.DEBUG: ft.Colors.GREY,
    }

    def __init__(
        self,
        max_lines: int = 1000,
        auto_scroll: bool = True,
        log_file: Optional[Path] = None,
    ):
        super().__init__()
        self.max_lines = max_lines
        self.auto_scroll = auto_scroll
        self.log_file = log_file

        # Internal state
        self.log_buffer: list[str] = []
        self.log_file_handle = None

        # UI Components
        self.log_text = ft.TextField(
            read_only=True,
            multiline=True,
            min_lines=15,
            max_lines=15,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.GREY_900),
            text_style=ft.TextStyle(
                font_family="Consolas, Monaco, Courier New, monospace",
                size=11,
                color=ft.Colors.GREY_100,
            ),
            border_radius=4,
        )

        self.status_text = ft.Text(
            "0 lines",
            size=12,
            color=ft.Colors.GREY_600,
        )

        # Initialize layout
        self._setup_ui()
        self._open_log_file()

    def _setup_ui(self):
        """Initialize the view layout."""
        self.spacing = 10
        self.controls = [
            # Header with title and controls
            ft.Row(
                [
                    ft.Icon(ft.Icons.TEXT_SNIPPET, size=20, color=ft.Colors.PRIMARY),
                    ft.Text("Logs", size=16, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),  # Spacer replacement
                    self.status_text,
                    ft.IconButton(
                        ft.Icons.COPY,
                        icon_size=18,
                        tooltip="Copy to clipboard",
                        on_click=self._on_copy_click,
                    ),
                    ft.IconButton(
                        ft.Icons.CLEAR,
                        icon_size=18,
                        tooltip="Clear logs",
                        on_click=self._on_clear_click,
                    ),
                ],
            ),
            # Log display
            ft.Container(
                content=self.log_text,
                expand=True,
            ),
        ]

    def _open_log_file(self):
        """Open log file for writing if specified."""
        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                self.log_file_handle = open(self.log_file, "a", encoding="utf-8")
            except Exception as e:
                print(f"[LogViewer] Failed to open log file: {e}")

    def add_log(self, message: str, level: LogLevel = LogLevel.INFO):
        """Add a log entry with timestamp and level.

        Args:
            message: The log message
            level: Log level for color coding
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level.value}] {message}"

        # Add to buffer
        self.log_buffer.append(formatted_message)

        # Enforce max_lines limit
        if len(self.log_buffer) > self.max_lines:
            self.log_buffer = self.log_buffer[-self.max_lines :]

        # Update UI (join all lines)
        self.log_text.value = "\n".join(self.log_buffer)

        # Note: Auto-scroll is handled automatically by Flet when content updates

        # Update status
        self.status_text.value = f"{len(self.log_buffer)} lines"

        # Write to file if configured
        if self.log_file_handle:
            try:
                self.log_file_handle.write(formatted_message + "\n")
                self.log_file_handle.flush()
            except Exception as e:
                print(f"[LogViewer] Failed to write log: {e}")

        # Schedule UI update
        try:
            self.log_text.update()
            self.status_text.update()
        except Exception:
            # Ignore if not yet added to page
            pass

    def clear_logs(self):
        """Clear all logs from buffer and display."""
        self.log_buffer.clear()
        self.log_text.value = ""
        self.status_text.value = "0 lines"

        try:
            self.log_text.update()
            self.status_text.update()
        except Exception:
            pass

    def get_logs(self) -> str:
        """Get all logs as a single string.

        Returns:
            All logs joined by newlines
        """
        return "\n".join(self.log_buffer)

    def _on_copy_click(self, e):
        """Handle copy to clipboard button click."""
        if self.log_buffer:
            try:
                import pyperclip  # type: ignore

                pyperclip.copy(self.get_logs())
                self._show_snack_bar("Logs copied to clipboard!")
            except ImportError:
                # Fallback if pyperclip not available
                self._show_snack_bar("pyperclip not installed")
            except Exception as ex:
                self._show_snack_bar(f"Copy failed: {ex}")

    def _on_clear_click(self, e):
        """Handle clear logs button click."""
        self.clear_logs()

    def _show_snack_bar(self, message: str):
        """Show a snack bar message.

        Args:
            message: Message to display
        """
        try:
            if self.page:
                self.page.open(
                    ft.SnackBar(
                        ft.Text(message),
                        duration=2000,
                    )
                )
        except Exception:
            pass

    def will_unmount(self):
        """Cleanup when the component is unmounted."""
        if self.log_file_handle:
            try:
                self.log_file_handle.close()
            except Exception:
                pass

    def __del__(self):
        """Destructor to ensure file handle is closed."""
        if self.log_file_handle:
            try:
                self.log_file_handle.close()
            except Exception:
                pass


# --- Demo Usage ---
if __name__ == "__main__":

    def main(page: ft.Page):
        page.title = "LogViewer Demo"
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 800
        page.window.height = 600
        page.padding = 20
        page.bgcolor = ft.Colors.GREY_900

        from pathlib import Path

        log_viewer = LogViewer(
            max_lines=100,
            auto_scroll=True,
            log_file=Path("/tmp/test_logs.log"),
        )

        page.add(log_viewer)

        # Simulate log messages
        import time

        def add_test_logs():
            log_viewer.add_log("Application starting...", LogLevel.INFO)
            time.sleep(0.5)
            log_viewer.add_log("Loading configuration", LogLevel.DEBUG)
            time.sleep(0.5)
            log_viewer.add_log(
                "Warning: Deprecated config option found", LogLevel.WARNING
            )
            time.sleep(0.5)
            log_viewer.add_log("Connection established successfully", LogLevel.SUCCESS)
            time.sleep(0.5)
            log_viewer.add_log("Error: Failed to load module", LogLevel.ERROR)

        page.add(
            ft.ElevatedButton(
                "Add Test Logs",
                on_click=lambda e: add_test_logs(),
            )
        )

    ft.app(target=main)
