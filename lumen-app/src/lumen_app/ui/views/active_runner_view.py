"""Active runner view for managing the running Lumen server.

This module provides process management, log viewing, and server control.
"""

import subprocess
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import flet as ft
from lumen_resources import LumenConfig

from ...utils.env_checker import MicromambaChecker
from ..components.button_container import ButtonContainer
from ..components.log_viewer import LogLevel, LogViewer
from ..i18n_manager import t


class ServerState(Enum):
    """Server running state."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ActiveRunnerView(ft.Column):
    """Active runner view for managing the Lumen server.

    Responsibilities:
    1. Start/Stop the server in micromamba environment
    2. Capture and display server logs in real-time
    3. Display server status and uptime
    4. Handle graceful shutdown

    Attributes:
        cache_dir (str): Cache directory path
        lumen_config (LumenConfig): Lumen configuration
        button_container (ButtonContainer): Button management
        data_binding (Any): Data binding for RunnerView
    """

    def __init__(
        self,
        cache_dir: str,
        lumen_config: LumenConfig,
        button_container: Optional[ButtonContainer] = None,
        data_binding: Optional[object] = None,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir).expanduser()
        self.lumen_config = lumen_config
        self.button_container = button_container
        self.data_binding = data_binding

        # Server state
        self.server_state = ServerState.STOPPED
        self.server_process: Optional[subprocess.Popen] = None
        self.log_thread: Optional[threading.Thread] = None
        self.start_time: Optional[datetime] = None

        # Paths
        self.config_path = self.cache_dir / "lumen-config.yaml"
        self.micromamba_path: Optional[Path] = None
        self.env_path = self.cache_dir / "micromamba" / "envs" / "lumen_env"

        # UI components
        self._init_ui_components()
        self._setup_ui()

    def _init_ui_components(self):
        """Initialize UI components."""
        # Status indicator
        self.status_icon = ft.Icon(
            ft.Icons.RADIO_BUTTON_UNCHECKED,
            size=20,
            color=ft.Colors.GREY,
        )
        self.status_text = ft.Text(
            t("active_runner.status_stopped"),
            size=16,
            color=ft.Colors.GREY,
        )

        # Uptime display
        self.uptime_text = ft.Text(
            "00:00:00",
            size=14,
            color=ft.Colors.GREY_600,
        )

        # Log viewer with file output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.cache_dir / "logs"
        log_file = log_dir / f"server_{timestamp}.log"

        self.log_viewer = LogViewer(
            max_lines=1000,
            auto_scroll=True,
            log_file=log_file,
        )

        # Start/Stop button
        self.toggle_button = ft.ElevatedButton(
            t("active_runner.start_server"),
            icon=ft.Icons.PLAY_ARROW,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.GREEN,
                color=ft.Colors.WHITE,
            ),
            on_click=self._on_toggle_click,
        )

        # Restart button
        self.restart_button = ft.OutlinedButton(
            t("active_runner.restart_server"),
            icon=ft.Icons.REFRESH,
            on_click=self._on_restart_click,
        )

    def _setup_ui(self):
        """Setup the view layout."""
        self.spacing = 20
        self.scroll = ft.ScrollMode.AUTO

        self.controls = [
            ft.Text(t("views.active_runner"), size=30),
            ft.Container(
                content=ft.Column(
                    controls=[
                        # Status card
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        t("active_runner.server_status"),
                                        size=16,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Divider(),
                                    ft.Row(
                                        [
                                            self.status_icon,
                                            self.status_text,
                                            ft.Container(
                                                expand=True
                                            ),  # Spacer replacement
                                            self.uptime_text,
                                        ],
                                        spacing=10,
                                    ),
                                ],
                                spacing=10,
                            ),
                            padding=15,
                            border_radius=8,
                            bgcolor=ft.Colors.with_opacity(
                                0.05, ft.Colors.BLUE_GREY_100
                            ),
                        ),
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        # Configuration summary
                        ft.Text(
                            t("active_runner.configuration"),
                            size=16,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Container(
                            content=self._create_config_summary(),
                            padding=15,
                            border_radius=8,
                            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.GREY_100),
                        ),
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        # Logs
                        ft.Text(
                            t("active_runner.logs"),
                            size=16,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Container(
                            content=self.log_viewer,
                            height=300,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=4,
                        ),
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        # Control buttons
                        ft.Row(
                            [
                                self.toggle_button,
                                self.restart_button,
                            ],
                            spacing=15,
                        ),
                    ],
                    spacing=15,
                ),
                padding=30,
                border_radius=12,
                bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.GREY_100),
                border=ft.border.all(
                    2, ft.Colors.with_opacity(0.1, ft.Colors.GREY_300)
                ),
            ),
        ]

        # Start uptime counter
        self._start_uptime_counter()

    def _create_config_summary(self) -> ft.Column:
        """Create configuration summary display.

        Returns:
            ft.Column with configuration info
        """
        rows = []

        # Service name
        if hasattr(self.lumen_config, "metadata") and hasattr(
            self.lumen_config.metadata, "cache_dir"
        ):
            rows.append(
                self._create_info_row(
                    ft.Icons.FOLDER_OPEN, "Cache Directory", str(self.cache_dir)
                )
            )

        # Port
        if hasattr(self.lumen_config, "server") and hasattr(
            self.lumen_config.server, "port"
        ):
            rows.append(
                self._create_info_row(
                    ft.Icons.SETTINGS_INPUT_COMPONENT,  # Valid icon replacement
                    "Port",
                    str(self.lumen_config.server.port),
                )
            )

        # Region
        if hasattr(self.lumen_config, "metadata") and hasattr(
            self.lumen_config.metadata, "region"
        ):
            rows.append(
                self._create_info_row(
                    ft.Icons.PUBLIC, "Region", str(self.lumen_config.metadata.region)
                )
            )

        # Services
        if (
            hasattr(self.lumen_config, "deployment")
            and self.lumen_config.deployment is not None
            and hasattr(self.lumen_config.deployment, "services")
            and self.lumen_config.deployment.services is not None
        ):
            service_names = [s.root for s in self.lumen_config.deployment.services]
            rows.append(
                self._create_info_row(
                    ft.Icons.APPS, "Services", ", ".join(service_names)
                )
            )

        return ft.Column(rows, spacing=8)

    def _create_info_row(self, icon, label: str, value: str) -> ft.Row:
        """Create an info row with icon, label, and value.

        Args:
            icon: Icon to display
            label: Label text
            value: Value text

        Returns:
            ft.Row with the info
        """
        return ft.Row(
            [
                ft.Icon(icon, size=16, color=ft.Colors.BLUE),
                ft.Text(
                    f"{label}:",
                    size=13,
                    color=ft.Colors.GREY_700,
                ),
                ft.Text(
                    value,
                    size=13,
                    weight=ft.FontWeight.W_500,
                ),
            ],
            spacing=8,
        )

    def _on_toggle_click(self, e):
        """Handle start/stop button click."""
        if self.server_state == ServerState.RUNNING:
            self._stop_server()
        else:
            self._start_server()

    def _on_restart_click(self, e):
        """Handle restart button click."""
        if self.server_state == ServerState.RUNNING:
            # Stop then restart
            self._stop_server()
            # Wait for stop to complete, then start
            threading.Thread(target=self._restart_after_stop, daemon=True).start()

    def _restart_after_stop(self):
        """Wait for server to stop, then restart."""
        import time

        for _ in range(50):  # Wait up to 5 seconds
            if self.server_state == ServerState.STOPPED:
                self._start_server()
                return
            time.sleep(0.1)

    def _start_server(self):
        """Start the server in micromamba environment."""
        if self.server_state != ServerState.STOPPED:
            return

        self.server_state = ServerState.STARTING
        self._update_status_ui()

        # Get micromamba executable
        try:
            micromamba_exe = MicromambaChecker.get_executable_path(str(self.cache_dir))

            if not micromamba_exe or not Path(micromamba_exe).exists():
                self.log_viewer.add_log(
                    "Micromamba not found. Please run installation first.",
                    LogLevel.ERROR,
                )
                self.server_state = ServerState.ERROR
                self._update_status_ui()
                return

            # Build command
            cmd = [
                str(micromamba_exe),
                "run",
                "-p",
                str(self.env_path),
                "python",
                "-m",
                "lumen_app.main",
                "--config",
                str(self.config_path),
            ]

            self.log_viewer.add_log(
                f"Starting server with command: {' '.join(cmd)}", LogLevel.INFO
            )

            # Start subprocess
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Start log reader thread
            self.log_thread = threading.Thread(target=self._read_logs, daemon=True)
            self.log_thread.start()

            self.start_time = datetime.now()
            self.server_state = ServerState.RUNNING
            self._update_status_ui()

            self.log_viewer.add_log("Server started successfully", LogLevel.SUCCESS)

        except Exception as e:
            self.log_viewer.add_log(f"Failed to start server: {e}", LogLevel.ERROR)
            self.server_state = ServerState.ERROR
            self._update_status_ui()

    def _read_logs(self):
        """Read logs from subprocess in background thread."""
        if not self.server_process:
            return

        try:
            for line in iter(self.server_process.stdout.readline, ""):  # type: ignore
                if line:
                    self.log_viewer.add_log(line.strip(), LogLevel.INFO)
                elif self.server_state == ServerState.STOPPING:
                    break
        except Exception:
            pass

    def _stop_server(self):
        """Stop the server gracefully."""
        if self.server_state != ServerState.RUNNING:
            return

        self.server_state = ServerState.STOPPING
        self._update_status_ui()

        self.log_viewer.add_log("Stopping server...", LogLevel.INFO)

        if self.server_process:
            try:
                # Send SIGTERM
                self.server_process.terminate()

                # Wait for graceful shutdown (up to 10 seconds)
                try:
                    self.server_process.wait(timeout=10)
                    self.log_viewer.add_log(
                        "Server stopped gracefully", LogLevel.SUCCESS
                    )
                except subprocess.TimeoutExpired:
                    # Force kill
                    self.server_process.kill()
                    self.log_viewer.add_log("Server force killed", LogLevel.WARNING)

            except Exception as e:
                self.log_viewer.add_log(f"Error stopping server: {e}", LogLevel.ERROR)

        self.server_process = None
        self.server_state = ServerState.STOPPED
        self.start_time = None
        self._update_status_ui()

    def _update_status_ui(self):
        """Update the status indicator UI."""
        # Update icon and text
        if self.server_state == ServerState.RUNNING:
            icon = ft.Icons.CHECK_CIRCLE
            color = ft.Colors.GREEN
            text = t("active_runner.status_running")
            button_text = t("active_runner.stop_server")
            button_icon = ft.Icons.STOP
            button_color = ft.Colors.RED
        elif self.server_state == ServerState.STARTING:
            icon = ft.Icons.REFRESH  # Valid icon replacement
            color = ft.Colors.BLUE
            text = t("active_runner.starting")
            button_text = t("active_runner.starting")
            button_icon = ft.Icons.REFRESH  # Valid icon replacement
            button_color = ft.Colors.GREY
        elif self.server_state == ServerState.STOPPING:
            icon = ft.Icons.REFRESH  # Valid icon replacement
            color = ft.Colors.ORANGE
            text = t("active_runner.stopping")
            button_text = t("active_runner.stopping")
            button_icon = ft.Icons.REFRESH  # Valid icon replacement
            button_color = ft.Colors.GREY
        else:  # STOPPED or ERROR
            icon = ft.Icons.RADIO_BUTTON_UNCHECKED
            color = ft.Colors.GREY
            text = t("active_runner.status_stopped")
            button_text = t("active_runner.start_server")
            button_icon = ft.Icons.PLAY_ARROW
            button_color = ft.Colors.GREEN

        # Update status icon
        if hasattr(self.status_icon, "icon"):
            self.status_icon.icon = icon  # type: ignore
        if hasattr(self.status_icon, "color"):
            self.status_icon.color = color
        self.status_text.value = text
        self.status_text.color = color

        self.toggle_button.text = button_text
        self.toggle_button.icon = button_icon
        self.toggle_button.style = ft.ButtonStyle(
            bgcolor=button_color, color=ft.Colors.WHITE
        )

        # Disable restart button when not running
        self.restart_button.disabled = self.server_state != ServerState.RUNNING

        try:
            self.status_icon.update()
            self.status_text.update()
            self.toggle_button.update()
            self.restart_button.update()
        except Exception:
            pass

    def _start_uptime_counter(self):
        """Start the uptime counter update thread."""

        def update_uptime():
            import time

            while True:
                if self.start_time and self.server_state == ServerState.RUNNING:
                    uptime = datetime.now() - self.start_time
                    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)

                    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    self.uptime_text.value = uptime_str

                    try:
                        self.uptime_text.update()
                    except Exception:
                        pass

                time.sleep(1)

        uptime_thread = threading.Thread(target=update_uptime, daemon=True)
        uptime_thread.start()

    def will_unmount(self):
        """Cleanup when the view is unmounted."""
        # Stop server if running
        if self.server_state == ServerState.RUNNING:
            self._stop_server()

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self.server_process:
            try:
                self.server_process.kill()
            except Exception:
                pass


# --- Demo Usage ---
if __name__ == "__main__":

    def main(page: ft.Page):
        page.title = "Active Runner View Demo"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window.width = 1200
        page.window.height = 800
        page.padding = 30
        page.bgcolor = ft.Colors.GREY_50

        # Create test configuration
        from pathlib import Path

        from lumen_resources import Region

        from lumen_app.core.config import Config, DeviceConfig

        cache_dir = Path.home() / "test_lumen"

        device_config = DeviceConfig.cpu()
        lumen_config = Config(
            cache_dir=str(cache_dir),
            device_config=device_config,
            region=Region.other,
            service_name="lumen-test",
            port=50051,
        ).minimal()

        active_view = ActiveRunnerView(
            cache_dir=str(cache_dir),
            lumen_config=lumen_config,
        )

        page.add(active_view)

    ft.app(target=main)
