"""Installer view for managing environment and driver installations.

This module provides comprehensive installation functionality including:
- Driver detection using env_checker
- Micromamba installation
- Python environment setup
- Driver installation via micromamba
- Lumen package installation via GitHub Releases
- Configuration file persistence
"""

import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import flet as ft
from lumen_resources import LumenConfig

from ...core.config import DeviceConfig
from ...utils.env_checker import (
    DependencyInstaller,
    EnvironmentChecker,
)
from ...utils.installation import (
    InstallationVerifier,
    LumenPackageInstaller,
    MicromambaInstaller,
    PythonEnvManager,
)
from ...utils.package_resolver import (
    GitHubPackageResolver,
    LumenPackageResolver,
)
from ..components.button_container import ButtonContainer
from ..components.log_viewer import LogLevel, LogViewer
from ..components.progress_card import ProgressCard, StepStatus
from ..i18n_manager import t


class InstallerState(Enum):
    """Installation state."""

    IDLE = "idle"
    CHECKING = "checking"
    INSTALLING = "installing"
    COMPLETE = "complete"
    FAILED = "failed"


class InstallerView(ft.Column):
    """Installer view for environment and driver management.

    Responsibilities:
    1. Check system drivers based on DeviceConfig
    2. Install micromamba to cache_dir
    3. Create Python 3.11 isolated environment
    4. Install drivers (CUDA, OpenVINO, TensorRT) via micromamba
    5. Install Lumen packages via uv
    6. Verify installation and save config

    Attributes:
        cache_dir (str): Cache directory path
        device_config (DeviceConfig): Device configuration
        lumen_config (LumenConfig): Full Lumen configuration
        button_container (ButtonContainer): Button management
        data_binding (Any): Data binding for RunnerView
    """

    # Installation steps
    INSTALLATION_STEPS = [
        "Checking system drivers",
        "Installing micromamba",
        "Setting up Python 3.11 environment",
        "Installing hardware drivers",
        "Installing Lumen packages",
        "Verifying installation",
    ]

    # Mamba YAML configs mapping
    DRIVER_CONFIGS = {
        "cuda": "cuda.yaml",
        "openvino": "openvino.yaml",
        "tensorrt": "tensorrt.yaml",
    }

    def __init__(
        self,
        cache_dir: str,
        device_config: DeviceConfig,
        lumen_config: LumenConfig,
        button_container: Optional[ButtonContainer] = None,
        data_binding: Optional[Any] = None,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir).expanduser()
        self.device_config = device_config
        self.lumen_config = lumen_config
        self.button_container = button_container
        self.data_binding = data_binding

        # State
        self.state = InstallerState.IDLE
        self.installation_thread: Optional[threading.Thread] = None
        self.installation_complete = False

        # Paths
        self.micromamba_path: Optional[Path] = None
        self.log_dir = self.cache_dir / "logs"
        self.config_path = self.cache_dir / "lumen-config.yaml"

        # Initialize UI
        self._init_ui_components()
        self._setup_ui()

    def _init_ui_components(self):
        """Initialize UI components."""
        # Log viewer with file output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"installer_{timestamp}.log"

        self.log_viewer = LogViewer(
            max_lines=1000,
            auto_scroll=True,
            log_file=log_file,
        )

        # Progress card
        self.progress_card = ProgressCard(self.INSTALLATION_STEPS)

        # Buttons
        self.install_button = ft.ElevatedButton(
            t("installer.start_install"),
            icon=ft.Icons.DOWNLOAD,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.PRIMARY,
                color=ft.Colors.WHITE,
            ),
            on_click=self._on_install_click,
        )

        self.back_button = ft.OutlinedButton(
            t("installer.back_button"),
            icon=ft.Icons.ARROW_BACK,
            on_click=self._on_back_click,
        )

        # Environment status container
        self.env_status_container = ft.Container(
            padding=15,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLUE_GREY_100),
        )

    def _setup_ui(self):
        """Setup the view layout."""
        self.spacing = 20
        self.scroll = ft.ScrollMode.AUTO

        self.controls = [
            ft.Text(t("views.installer"), size=30),
            ft.Container(
                content=ft.Column(
                    controls=[
                        # Progress card
                        self.progress_card,
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        # Environment status
                        ft.Text(
                            t("installer.environment_status"),
                            size=16,
                            weight=ft.FontWeight.BOLD,
                        ),
                        self.env_status_container,
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        # Log viewer
                        ft.Text(
                            t("installer.installation_logs"),
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
                        # Action buttons
                        ft.Row(
                            [self.back_button, self.install_button],
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

        # Initial environment check
        self._check_environment()

    def _check_environment(self):
        """Check system environment and display status."""
        self.log_viewer.add_log("Checking system drivers...", LogLevel.INFO)

        try:
            env_report = EnvironmentChecker.check_device_config(self.device_config)

            # Display driver status
            self._display_env_status(env_report)

            # Update data binding
            if self.data_binding:
                self.data_binding.set_data("env_report", env_report)

            self.log_viewer.add_log(
                f"Environment check complete: {len(env_report.drivers)} drivers checked",
                LogLevel.INFO,
            )

        except Exception as e:
            self.log_viewer.add_log(f"Environment check failed: {e}", LogLevel.ERROR)

    def _display_env_status(self, env_report):
        """Display environment status in the UI.

        Args:
            env_report: Environment report from EnvironmentChecker
        """
        controls = []

        for driver_result in env_report.drivers:
            # Determine icon and color based on status
            if driver_result.status.value == "available":
                icon = ft.Icons.CHECK_CIRCLE
                color = ft.Colors.GREEN
                status_text = "Available"
            elif driver_result.status.value == "missing":
                icon = ft.Icons.CANCEL
                color = ft.Colors.RED
                status_text = "Not Installed"
            else:  # incompatible
                icon = ft.Icons.BLOCK
                color = ft.Colors.ORANGE
                status_text = "Incompatible"

            row = ft.Row(
                [
                    ft.Icon(icon, size=18, color=color),
                    ft.Text(driver_result.name, size=13, weight=ft.FontWeight.W_500),
                    ft.Container(expand=True),  # Spacer replacement
                    ft.Text(status_text, size=12, color=color),
                ],
                spacing=10,
            )
            controls.append(row)

        # If installable drivers missing, show install button
        missing_installable = [
            d
            for d in env_report.drivers
            if d.status.value == "missing" and d.installable_via_mamba
        ]

        if missing_installable:
            controls.append(ft.Divider(height=10, color=ft.Colors.GREY_300))
            controls.append(
                ft.Text(
                    f"{len(missing_installable)} drivers can be installed automatically",
                    size=12,
                    color=ft.Colors.BLUE,
                )
            )

        self.env_status_container.content = ft.Column(controls, spacing=8)

        try:
            self.env_status_container.update()
        except Exception:
            pass

    def _on_install_click(self, e):
        """Handle install button click - start installation in background thread."""
        if self.state != InstallerState.IDLE:
            return

        self.state = InstallerState.INSTALLING
        self.install_button.disabled = True
        self.install_button.text = t("installer.installing")

        try:
            self.install_button.update()
        except Exception:
            pass

        # Start installation in background thread
        self.installation_thread = threading.Thread(
            target=self._run_installation,
            daemon=True,
        )
        self.installation_thread.start()

    def _run_installation(self):
        """Run the complete installation process in background thread."""
        try:
            # Initialize installers
            # Get region from lumen_config
            region = self.lumen_config.metadata.region
            micromamba_installer = MicromambaInstaller(self.cache_dir, region=region)
            github_resolver = GitHubPackageResolver(region)

            # Step 1: Check drivers (update progress)
            self._update_step(0, StepStatus.IN_PROGRESS)
            self.log_viewer.add_log("Checking system drivers...", LogLevel.INFO)
            env_report = EnvironmentChecker.check_device_config(self.device_config)

            available_count = sum(
                1 for d in env_report.drivers if d.status.value == "available"
            )
            missing_count = len(env_report.drivers) - available_count

            self.log_viewer.add_log(
                f"Drivers: {available_count} available, {missing_count} missing",
                LogLevel.INFO if missing_count == 0 else LogLevel.WARNING,
            )

            self._update_step(0, StepStatus.COMPLETE)

            # Step 2: Install micromamba
            self._update_step(1, StepStatus.IN_PROGRESS)
            self.log_viewer.add_log("Installing micromamba...", LogLevel.INFO)
            micromamba_exe = micromamba_installer.install()
            self.log_viewer.add_log(
                f"Micromamba installed: {micromamba_exe}", LogLevel.SUCCESS
            )
            self._update_step(1, StepStatus.COMPLETE)

            # Step 3: Create Python environment
            self._update_step(2, StepStatus.IN_PROGRESS)
            self.log_viewer.add_log("Creating Python environment...", LogLevel.INFO)
            env_manager = PythonEnvManager(self.cache_dir, micromamba_exe)

            # Get yaml config from DeviceConfig
            yaml_config = self.device_config.env

            env_manager.create_env(yaml_config=yaml_config)
            self.log_viewer.add_log(
                f"Environment created: {env_manager.get_env_path()}", LogLevel.SUCCESS
            )
            self._update_step(2, StepStatus.COMPLETE)

            # Step 4: Install drivers
            self._update_step(3, StepStatus.IN_PROGRESS)
            self._install_drivers(env_report, micromamba_exe)
            self._update_step(3, StepStatus.COMPLETE)

            # Step 5: Install Lumen packages
            self._update_step(4, StepStatus.IN_PROGRESS)
            self.log_viewer.add_log("Installing Lumen packages...", LogLevel.INFO)

            packages = LumenPackageResolver.resolve_packages(self.lumen_config)
            self.log_viewer.add_log(
                f"Packages to install: {', '.join(packages)}", LogLevel.INFO
            )

            package_installer = LumenPackageInstaller(
                env_manager, github_resolver, self.log_viewer.add_log
            )
            package_installer.install_packages(
                packages, self.device_config, self.lumen_config.metadata.region
            )
            self._update_step(4, StepStatus.COMPLETE)

            # Step 6: Verify installation
            self._update_step(5, StepStatus.IN_PROGRESS)
            self.log_viewer.add_log("Verifying installation...", LogLevel.INFO)
            verifier = InstallationVerifier(env_manager)
            results = verifier.verify_imports(packages)

            # Log verification results
            all_success = True
            for pkg, success in results.items():
                if success:
                    self.log_viewer.add_log(
                        f"✓ {pkg} import successful", LogLevel.SUCCESS
                    )
                else:
                    self.log_viewer.add_log(f"✗ {pkg} import failed", LogLevel.ERROR)
                    all_success = False

            if not all_success:
                self.log_viewer.add_log(
                    "Some packages failed verification", LogLevel.WARNING
                )

            self._update_step(5, StepStatus.COMPLETE)

            # Save config
            self._save_config()

            # Mark as complete
            self.installation_complete = True
            self.state = InstallerState.COMPLETE

            self.log_viewer.add_log(
                "Installation complete! You can now start the server.",
                LogLevel.SUCCESS,
            )

            # Update UI
            self._on_install_complete_ui()

        except Exception as e:
            self.log_viewer.add_log(f"Installation failed: {e}", LogLevel.ERROR)
            self.state = InstallerState.FAILED

            # Mark current step as failed
            current_step = self.progress_card.get_current_step()
            self._update_step(current_step, StepStatus.FAILED)

            # Update UI
            try:
                self.install_button.disabled = False
                self.install_button.text = t("installer.retry")
                self.install_button.update()
            except Exception:
                pass

    def _install_drivers(self, env_report, micromamba_exe):
        """Install hardware drivers using micromamba.

        Args:
            env_report: Environment report with driver status
            micromamba_exe: Path to micromamba executable
        """
        self.log_viewer.add_log("Installing hardware drivers...", LogLevel.INFO)

        # Get missing installable drivers
        missing_drivers = [
            d.name
            for d in env_report.drivers
            if d.status.value == "missing" and d.installable_via_mamba
        ]

        if not missing_drivers:
            self.log_viewer.add_log("All drivers already installed", LogLevel.SUCCESS)
            return

        # Create dependency installer
        mamba_configs_dir = Path(__file__).parent.parent.parent / "utils" / "mamba"

        installer = DependencyInstaller(
            mamba_configs_dir=str(mamba_configs_dir),
            micromamba_path=str(micromamba_exe),
        )

        # Install each driver
        for driver_name in missing_drivers:
            self.log_viewer.add_log(f"Installing {driver_name}...", LogLevel.INFO)

            success, message = installer.install_driver(driver_name)

            if success:
                self.log_viewer.add_log(
                    f"{driver_name} installed successfully", LogLevel.SUCCESS
                )
            else:
                self.log_viewer.add_log(
                    f"{driver_name} installation skipped: {message}", LogLevel.WARNING
                )

    def _save_config(self):
        """Save lumen_config to YAML file."""
        self.log_viewer.add_log(
            f"Saving config to {self.config_path}...", LogLevel.INFO
        )

        try:
            import yaml

            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Convert to dict for YAML serialization
            config_dict = self.lumen_config.model_dump(mode="python")

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            self.log_viewer.add_log(
                f"Config saved: {self.config_path}", LogLevel.SUCCESS
            )

        except Exception as e:
            self.log_viewer.add_log(f"Failed to save config: {e}", LogLevel.ERROR)
            raise

    def _on_install_complete_ui(self):
        """Update UI when installation is complete."""
        try:
            self.install_button.disabled = False
            self.install_button.text = t("installer.complete")
            self.install_button.icon = ft.Icons.CHECK_CIRCLE
            self.install_button.bgcolor = ft.Colors.GREEN
            self.install_button.update()

            # Update data binding
            if self.data_binding:
                self.data_binding.set_data("installation_complete", True)
                self.data_binding.set_data("config_path", str(self.config_path))

        except Exception:
            pass

    def _on_back_click(self, e):
        """Handle back button click."""
        # Only allow going back if not installing
        if self.state == InstallerState.INSTALLING:
            return

        # Notify RunnerView to reset
        if self.button_container:
            # Trigger reset action
            pass

    def _update_step(self, step_index: int, status: StepStatus):
        """Update progress step status."""
        self.progress_card.update_step(step_index, status)


# --- Demo Usage ---
if __name__ == "__main__":

    def main(page: ft.Page):
        page.title = "Installer View Demo"
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

        installer_view = InstallerView(
            cache_dir=str(cache_dir),
            device_config=device_config,
            lumen_config=lumen_config,
        )

        page.add(installer_view)

    ft.app(target=main)
