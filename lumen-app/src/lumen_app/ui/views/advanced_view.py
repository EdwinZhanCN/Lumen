"""Advanced view for manual LumenConfig configuration.

This module provides a form-based view for users to manually configure
their Lumen AI services setup with full control over all fields.
"""

from typing import Any, Callable, Dict, List, Optional

import flet as ft
from lumen_resources import Region

from ...utils.logger import get_logger
from ..components.button_container import ButtonContainer
from ..i18n_manager import t

logger = get_logger("lumen.ui.advanced_view")


class AdvancedView(ft.Column):
    """An advanced configuration view with manual form input.

    This view allows users to manually configure all aspects of LumenConfig:
    - Cache directory
    - Service name and port
    - Region selection
    - Runtime selection
    - ONNX providers (manual JSON input)
    - Device-specific settings (RKNN device, batch size, precision)
    - Dynamic service management (add/remove services)

    Attributes:
        button_container (ButtonContainer | None): Continue/reset button container.
        data_binding (Any | None): Data binding object for form state.
        on_config_changed (Callable | None): Handler for configuration changes.
    """

    def __init__(
        self,
        button_container: Optional[ButtonContainer] = None,
        data_binding: Optional[Any] = None,
        on_config_changed: Optional[Callable] = None,
    ):
        super().__init__()
        self.button_container = button_container
        self.data_binding = data_binding
        self.on_config_changed = on_config_changed

        # Form state
        self.cache_dir_value: str = "~/.lumen"
        self.service_name_value: str = "lumen-app"
        self.port_value: int = 50051
        self.selected_region: Region = Region.cn
        self.selected_runtime: str = "onnx"

        # Device settings
        self.rknn_device_value: str = ""
        self.batch_size_value: int = 1
        self.precision_value: str = "fp16"

        # ONNX providers (JSON string input)
        self.onnx_providers_json: str = '["CPUExecutionProvider"]'

        # Services list
        self.services: List[Dict[str, Any]] = []

        # UI Components
        self._init_basic_fields()
        self._init_device_fields()
        self._init_service_components()

        # Initialize layout
        self._setup_ui()

    def _init_basic_fields(self):
        """Initialize basic configuration fields."""
        self.cache_dir_input = ft.TextField(
            label=t("advanced.cache_dir_label"),
            hint_text="~/.lumen",
            value=self.cache_dir_value,
            expand=True,
            on_change=self._on_form_changed,
        )

        self.service_name_input = ft.TextField(
            label=t("advanced.service_name_label"),
            hint_text="lumen-app",
            value=self.service_name_value,
            expand=True,
            on_change=self._on_form_changed,
        )

        self.port_input = ft.TextField(
            label=t("advanced.port_label"),
            hint_text="50051",
            value=str(self.port_value),
            width=150,
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=self._on_form_changed,
        )

        self.region_dropdown = ft.Dropdown(
            label=t("advanced.region_label"),
            options=[
                ft.dropdown.Option(str(Region.cn), t("region.cn")),
                ft.dropdown.Option(str(Region.other), t("region.other")),
            ],
            value=str(self.selected_region),
            width=200,
            on_change=self._on_region_changed,
        )

    def _init_device_fields(self):
        """Initialize device-specific fields."""
        self.runtime_dropdown = ft.Dropdown(
            label=t("advanced.runtime_label"),
            options=[
                ft.dropdown.Option("onnx", "ONNX"),
                ft.dropdown.Option("rknn", "RKNN (Rockchip)"),
            ],
            value=self.selected_runtime,
            width=200,
            on_change=self._on_runtime_changed,
        )

        self.rknn_device_input = ft.TextField(
            label=t("advanced.rknn_device_label"),
            hint_text="rk3588",
            value=self.rknn_device_value,
            width=200,
            visible=False,
            on_change=self._on_form_changed,
        )

        self.batch_size_input = ft.TextField(
            label=t("advanced.batch_size_label"),
            hint_text="1",
            value=str(self.batch_size_value),
            width=150,
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=self._on_form_changed,
        )

        self.precision_dropdown = ft.Dropdown(
            label=t("advanced.precision_label"),
            options=[
                ft.dropdown.Option("fp32", "FP32"),
                ft.dropdown.Option("fp16", "FP16"),
                ft.dropdown.Option("int8", "INT8"),
                ft.dropdown.Option("", "Auto"),
            ],
            value=self.precision_value,
            width=150,
            on_change=self._on_precision_changed,
        )

        # ONNX Providers JSON input
        self.onnx_providers_input = ft.TextField(
            label=t("advanced.onnx_providers_label"),
            hint_text='["CPUExecutionProvider"]',
            value=self.onnx_providers_json,
            multiline=True,
            min_lines=3,
            max_lines=5,
            on_change=self._on_form_changed,
        )

    def _init_service_components(self):
        """Initialize service management components."""
        self.service_list = ft.Column(
            controls=[],
            spacing=10,
        )

        self.service_type_dropdown = ft.Dropdown(
            label=t("advanced.service_type_label"),
            options=[
                ft.dropdown.Option("ocr", t("services.ocr.name")),
                ft.dropdown.Option("clip", t("services.clip.name")),
                ft.dropdown.Option("face", t("services.face.name")),
                ft.dropdown.Option("vlm", t("services.vlm.name")),
            ],
            width=200,
        )

        self.service_model_input = ft.TextField(
            label=t("advanced.service_model_label"),
            hint_text="PP-OCRv5",
            width=300,
        )

    def _setup_ui(self):
        """Initialize the view layout."""
        self.spacing = 20
        self.scroll = ft.ScrollMode.AUTO

        self.controls = [
            ft.Text(t("views.advanced"), size=30),
            ft.Container(
                content=ft.Column(
                    controls=[
                        # Basic Configuration
                        self._create_section_header(
                            ft.Icons.SETTINGS,
                            t("advanced.section_basic"),
                        ),
                        ft.Row(
                            controls=[
                                ft.Column(
                                    [self.cache_dir_input],
                                    expand=True,
                                ),
                                ft.Column(
                                    [self.service_name_input],
                                    expand=True,
                                ),
                                ft.Column(
                                    [self.port_input],
                                    width=150,
                                ),
                                ft.Column(
                                    [self.region_dropdown],
                                ),
                            ],
                            spacing=15,
                        ),
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        # Device Configuration
                        self._create_section_header(
                            ft.Icons.DEVELOPER_BOARD,
                            t("advanced.section_device"),
                        ),
                        ft.Row(
                            controls=[
                                ft.Column(
                                    [self.runtime_dropdown],
                                ),
                                ft.Column(
                                    [self.rknn_device_input],
                                ),
                                ft.Column(
                                    [self.batch_size_input],
                                ),
                                ft.Column(
                                    [self.precision_dropdown],
                                ),
                            ],
                            spacing=15,
                        ),
                        # ONNX Providers
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        self._create_section_header(
                            ft.Icons.SETTINGS_INPUT_COMPONENT,
                            t("advanced.section_providers"),
                        ),
                        self.onnx_providers_input,
                        ft.Text(
                            t("advanced.onnx_providers_hint"),
                            size=12,
                            color=ft.Colors.GREY_600,
                        ),
                        # Services
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        self._create_section_header(
                            ft.Icons.APPS,
                            t("advanced.section_services"),
                        ),
                        self.service_list,
                        ft.Row(
                            [
                                self.service_type_dropdown,
                                self.service_model_input,
                                ft.ElevatedButton(
                                    t("advanced.add_service_button"),
                                    icon=ft.Icons.ADD,
                                    on_click=self._on_add_service,
                                ),
                            ],
                            spacing=10,
                        ),
                        # Info
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Icon(
                                        ft.Icons.INFO_OUTLINE, color=ft.Colors.BLUE
                                    ),
                                    ft.Text(
                                        t("advanced.config_info"),
                                        size=13,
                                        color=ft.Colors.GREY_700,
                                    ),
                                ],
                                spacing=10,
                            ),
                            padding=15,
                            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_100),
                            border_radius=8,
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

    def _create_section_header(self, icon, title: str) -> ft.Row:
        """Create a section header with icon and title."""
        return ft.Row(
            [
                ft.Icon(icon, size=24, color=ft.Colors.PRIMARY),
                ft.Text(title, size=18, weight=ft.FontWeight.BOLD),
            ],
            spacing=10,
        )

    def _on_form_changed(self, e):
        """Handle form field changes."""
        self.cache_dir_value = self.cache_dir_input.value or "~/.lumen"
        self.service_name_value = self.service_name_input.value or "lumen-app"

        try:
            self.port_value = int(self.port_input.value or "50051")
        except ValueError:
            self.port_value = 50051

        self.rknn_device_value = self.rknn_device_input.value or ""
        self.onnx_providers_json = self.onnx_providers_input.value or "[]"

        try:
            self.batch_size_value = int(self.batch_size_input.value or "1")
        except ValueError:
            self.batch_size_value = 1

        self._update_external_state()

    def _on_region_changed(self, e):
        """Handle region dropdown change."""
        if self.region_dropdown.value:
            self.selected_region = Region(self.region_dropdown.value)
        self._update_external_state()

    def _on_runtime_changed(self, e):
        """Handle runtime dropdown change."""
        if self.runtime_dropdown.value:
            self.selected_runtime = self.runtime_dropdown.value

        # Show/hide RKNN device field
        self.rknn_device_input.visible = self.selected_runtime == "rknn"

        self._update_external_state()

    def _on_precision_changed(self, e):
        """Handle precision dropdown change."""
        if self.precision_dropdown.value is not None:
            self.precision_value = self.precision_dropdown.value
        self._update_external_state()

    def _on_add_service(self, e):
        """Add a service to the list."""
        service_type = self.service_type_dropdown.value
        if not service_type:
            return

        model_name = self.service_model_input.value or f"default_{service_type}"

        service_entry = {
            "root": service_type,
            "package": f"lumen_{service_type}",
            "enabled": True,
            "model": model_name,
        }

        self.services.append(service_entry)
        self._refresh_service_list()
        self._update_external_state()

        # Clear model input
        self.service_model_input.value = ""

    def _refresh_service_list(self):
        """Refresh the service list display."""
        self.service_list.controls.clear()

        if not self.services:
            self.service_list.controls.append(
                ft.Text(
                    t("advanced.no_services"),
                    color=ft.Colors.GREY_500,
                    italic=True,
                )
            )
        else:
            for i, service in enumerate(self.services):
                service_card = ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(ft.Icons.APPS, size=20, color=ft.Colors.BLUE),
                            ft.Text(
                                t(f"services.{service['root']}.name"),
                                size=14,
                                weight=ft.FontWeight.W_500,
                            ),
                            ft.Text(
                                f"model: {service['model']}",
                                size=12,
                                color=ft.Colors.GREY_600,
                            ),
                            ft.Container(expand=True),  # Spacer
                            ft.IconButton(
                                ft.Icons.DELETE,
                                icon_color=ft.Colors.RED_400,
                                on_click=lambda e, idx=i: self._remove_service(idx),
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=10,
                    bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLUE_50),
                    border_radius=8,
                )
                self.service_list.controls.append(service_card)

    def _remove_service(self, index: int):
        """Remove a service from the list."""
        if 0 <= index < len(self.services):
            self.services.pop(index)
            self._refresh_service_list()
            self._update_external_state()

    def _update_external_state(self):
        """Update external dependencies (DataBinding and ButtonContainer)."""
        is_valid = self._validate_form()

        config_dict = None
        if is_valid:
            config_dict = self._generate_config_dict()

        if self.data_binding:
            self.data_binding.set_data("cache_dir", self.cache_dir_value)
            self.data_binding.set_data("service_name", self.service_name_value)
            self.data_binding.set_data("port", self.port_value)
            self.data_binding.set_data("region", self.selected_region)
            self.data_binding.set_data("runtime", self.selected_runtime)
            self.data_binding.set_data("rknn_device", self.rknn_device_value)
            self.data_binding.set_data("batch_size", self.batch_size_value)
            self.data_binding.set_data("precision", self.precision_value)
            self.data_binding.set_data("onnx_providers_json", self.onnx_providers_json)
            self.data_binding.set_data("services", self.services)
            self.data_binding.set_data("is_valid", is_valid)
            self.data_binding.set_data("config_dict", config_dict)

        if self.button_container:
            self.button_container.continue_button.update_disabled(not is_valid)

        if self.on_config_changed:
            self.on_config_changed(config_dict)

    def _validate_form(self) -> bool:
        """Validate form inputs."""
        # Validate cache_dir
        if not self.cache_dir_value or not self.cache_dir_value.strip():
            return False

        # Validate service_name
        if not self.service_name_value or not self.service_name_value.strip():
            return False

        # Validate port
        try:
            port = int(self.port_value)
            if port < 1024 or port > 65535:
                return False
        except (ValueError, TypeError):
            return False

        # Validate RKNN device if runtime is rknn
        if self.selected_runtime == "rknn" and not self.rknn_device_value:
            return False

        # Validate at least one service
        if not self.services:
            return False

        # Validate ONNX providers JSON
        if self.selected_runtime == "onnx":
            try:
                import json

                providers = json.loads(self.onnx_providers_json)
                if not isinstance(providers, list):
                    return False
            except (json.JSONDecodeError, ValueError):
                return False

        return True

    def _generate_config_dict(self) -> Dict[str, Any]:
        """Generate configuration dictionary from form data."""
        import json

        onnx_providers = None
        if self.selected_runtime == "onnx":
            try:
                onnx_providers = json.loads(self.onnx_providers_json)
            except json.JSONDecodeError:
                onnx_providers = ["CPUExecutionProvider"]

        return {
            "cache_dir": self.cache_dir_value,
            "service_name": self.service_name_value,
            "port": self.port_value,
            "region": self.selected_region,
            "runtime": self.selected_runtime,
            "rknn_device": self.rknn_device_value
            if self.selected_runtime == "rknn"
            else None,
            "batch_size": self.batch_size_value,
            "precision": self.precision_value or None,
            "onnx_providers": onnx_providers,
            "services": self.services,
        }

    def get_config_dict(self) -> Optional[Dict[str, Any]]:
        """Get the current configuration as dictionary."""
        if self._validate_form():
            return self._generate_config_dict()
        return None


# --- Demo Usage ---
if __name__ == "__main__":

    def main(page: ft.Page):
        page.title = "Advanced View Demo"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window.width = 1400
        page.window.height = 900
        page.padding = 30
        page.bgcolor = ft.Colors.GREY_50

        class MockDataBinding:
            def __init__(self):
                self.data = {}

            def set_data(self, key: str, value: Any):
                self.data[key] = value
                logger.debug(f"[DataBinding] {key} = {value}")

        data_binding = MockDataBinding()

        def handle_config_changed(config):
            logger.debug(f"[ConfigChanged] Valid: {config is not None}")

        advanced_view = AdvancedView(
            button_container=None,  # type: ignore
            data_binding=data_binding,
            on_config_changed=handle_config_changed,
        )

        page.add(advanced_view)

    ft.app(target=main)
