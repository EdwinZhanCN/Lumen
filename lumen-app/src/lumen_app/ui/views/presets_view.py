from pathlib import Path
from typing import Any, Dict, Literal, Optional

import flet as ft
from lumen_resources import Region

from ...core.config import Config, DeviceConfig
from ...utils.logger import get_logger

# 假设这些是你项目中的依赖，保留引用
from ..components.button_container import ButtonContainer
from ..i18n_manager import t

logger = get_logger("lumen.ui.presets_view")


class PresetsView(ft.Column):
    CLIP_MODEL_MAPPING = {
        "MobileCLIP2-S2": {
            "display_name": t("m2s2.name"),
            "description": t("m2s2.description"),
        },
        "CN-CLIP_ViT-B-16": {
            "display_name": t("cnclip_b16.name"),
            "description": t("cnclip_b16.description"),
        },
        "MobileCLIP2-S4": {
            "display_name": t("m2s2.name"),
            "description": t("m2s2.description"),
        },
        "CN-CLIP_ViT-L-14": {
            "display_name": t("cnclip_b16.name"),
            "description": t("cnclip_b16.description"),
        },
    }

    def __init__(
        self,
        cache_dir: str,
        region: Region,
        device_config: DeviceConfig,
        button_container: Optional[ButtonContainer] = None,
        data_binding: Optional[Any] = None,
        on_reset: Optional[callable] = None,
    ):
        super().__init__()
        # Initialize service config first
        self.service_name: str = "lumen-app"
        self.port_value: int = 50051

        # config source
        self.config = Config(
            cache_dir=cache_dir,
            region=region,
            device_config=device_config,
            service_name=self.service_name,
            port=self.port_value,
        )

        self.button_container = button_container
        self.data_binding = data_binding
        self.cache_dir = cache_dir
        self.on_reset = on_reset

        # 组件状态
        self.selected_key: Optional[str] = None
        # 所有CLIP model输入框共用一个value，因为用户仅且仅能选择一个Preset和一个适配的CLIP model.
        self.clip_model_value: Literal[
            "MobileCLIP2-S2", "CN-CLIP_ViT-B-16", "MobileCLIP2-S4", "CN-CLIP_ViT-L-14"
        ] = "MobileCLIP2-S2"
        # 映射表：Key -> Checkbox控件实例 (用于快速访问和更新)
        self.checkbox_map: Dict[str, ft.Checkbox] = {}

        # CLIP model options for each preset
        self.clip_model_options: Dict[str, list[str]] = {
            "lightweight": ["MobileCLIP2-S2", "CN-CLIP_ViT-B-16"],
            "basic": ["MobileCLIP2-S4", "CN-CLIP_ViT-L-14"],
        }

        self.presets_mapping = [
            {
                "key": "minimal",
                "icon": ft.Icons.FILTER_NONE,
                "config_method": lambda: self.config.minimal(),
                "features": [
                    "OCR",
                    "OCR Search",
                ],
            },
            {
                "key": "lightweight",
                "icon": ft.Icons.FLIGHT_TAKEOFF,
                "config_method": lambda model: self.config.light_weight(model),
                "features": [
                    "OCR",
                    "OCR Search",
                    "Image Semantic Search",
                    "Image Classification",
                    "Face Recognition",
                ],
            },
            {
                "key": "basic",
                "icon": ft.Icons.APPS,
                "config_method": lambda model: self.config.basic(model),
                "features": [
                    "OCR",
                    "OCR Search",
                    "Image Semantic Search",
                    "Image Classification",
                    "Face Recognition",
                    "Image Captioning",
                    "Lumen Memory",
                ],
            },
            {
                "key": "brave",
                "icon": ft.Icons.ROCKET_LAUNCH,
                "config_method": lambda: self.config.brave(),
                "features": [
                    "OCR",
                    "OCR Search",
                    "Image Semantic Search",
                    "Image Classification",
                    "Face Recognition",
                    "Image Captioning",
                    "Lumen Memory",
                    "Biological Search",
                    "Biological Classification",
                    "Biological Atlas",
                ],
            },
        ]

        # 初始化 UI
        self._setup_ui()

    def _setup_ui(self):
        """初始化视图布局"""
        self.spacing = 20
        self.scroll = ft.ScrollMode.AUTO

        # 构建控件列表
        controls_list: list[ft.Control] = [
            ft.Text(t("views.presets"), size=30),
        ]

        # 如果传入了按钮容器，放在标题下方
        if self.button_container:
            # Setup reset button handler
            if self.on_reset:
                self.button_container.reset_button.get_button().on_click = (
                    lambda e: self._on_reset_clicked()
                )
            controls_list.append(self.button_container.get_container())

        # 创建卡片列表
        preset_cards = [
            self._create_preset_card(preset) for preset in self.presets_mapping
        ]

        controls_list.append(
            ft.Container(
                content=ft.Row(
                    controls=preset_cards,
                    spacing=15,
                    wrap=True,
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=20,
                border_radius=12,
                bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.GREY_100),
            )
        )

        self.controls = controls_list

    def _create_preset_card(self, preset_info: dict) -> ft.Container:
        """构建单个预设卡片 UI"""
        key = preset_info["key"]

        # 创建 Checkbox 并存入映射表
        checkbox = ft.Checkbox(
            value=False,
            # 使用 lambda 捕获当前的 key
            on_change=lambda e: self._on_preset_selected(key, e.control.value),
        )
        self.checkbox_map[key] = checkbox

        return ft.Container(
            content=ft.Column(
                [
                    # Header: Checkbox + Icon
                    ft.Row(
                        [
                            checkbox,
                            ft.Icon(
                                preset_info["icon"],
                                size=32,
                                color=ft.Colors.PRIMARY,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        spacing=10,
                    ),
                    # Title
                    ft.Text(
                        t(f"presets.{key}.title"),
                        size=18,
                        weight=ft.FontWeight.BOLD,
                    ),
                    # Description
                    ft.Text(
                        t(f"presets.{key}.description"),
                        size=14,
                        color=ft.Colors.GREY_700,
                    ),
                    # Tags
                    self._create_tags(preset_info["features"]),
                ],
                spacing=8,
                tight=True,
            ),
            width=280,
            padding=20,
            border=ft.border.all(2, ft.Colors.with_opacity(0.2, ft.Colors.GREY_300)),
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        )

    def _create_tags(self, features: list) -> ft.Column:
        """辅助方法：创建标签组"""
        return ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Container(
                            content=ft.Text(f, size=11, color=ft.Colors.WHITE),
                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                            bgcolor=ft.Colors.with_opacity(0.8, ft.Colors.BLUE),
                            border_radius=12,
                        )
                    ],
                    spacing=5,
                )
                for f in features
            ],
            spacing=3,
            tight=True,
        )

    def _on_reset_clicked(self):
        """处理重置按钮点击：清除 lumen-config.yaml 并返回 WelcomeView"""
        # 清除配置文件
        cache_path = Path(self.cache_dir).expanduser()
        config_file = cache_path / "lumen-config.yaml"

        if config_file.exists():
            try:
                config_file.unlink()
                logger.info(f"Deleted config file: {config_file}")
            except Exception as e:
                logger.error(f"Failed to delete config file: {e}")

        # 调用重置回调
        if self.on_reset:
            self.on_reset()

    def _on_preset_selected(self, key: str, is_selected: bool):
        """处理选择逻辑：实现单选互斥效果"""
        if is_selected:
            self.selected_key = key

            # 1. 互斥逻辑：取消选中其他所有 Checkbox
            for k, cb in self.checkbox_map.items():
                if k != key and cb.value:
                    cb.value = False
                    cb.update()  # 局部刷新，性能更好

            # 2. 更新外部依赖
            self._update_external_state(is_valid=True)

        else:
            # 如果用户取消了当前选中的项
            if self.selected_key == key:
                self.selected_key = None
                self._update_external_state(is_valid=False)

    def _update_external_state(self, is_valid: bool):
        """更新 DataBinding 和 ButtonContainer"""
        # 获取完整的预设信息对象
        preset_info = next(
            (p for p in self.presets_mapping if p["key"] == self.selected_key), None
        )

        # Generate LumenConfig if preset is selected
        lumen_config = None
        if preset_info and is_valid:
            # Determine the CLIP model to use based on preset
            if self.selected_key == "lightweight":
                # Ensure clip_model_value is valid for lightweight preset
                clip_model = (
                    self.clip_model_value
                    if self.clip_model_value in self.clip_model_options["lightweight"]
                    else self.clip_model_options["lightweight"][0]
                )
                lumen_config = preset_info["config_method"](clip_model)
            elif self.selected_key == "basic":
                # Ensure clip_model_value is valid for basic preset
                clip_model = (
                    self.clip_model_value
                    if self.clip_model_value in self.clip_model_options["basic"]
                    else self.clip_model_options["basic"][0]
                )
                lumen_config = preset_info["config_method"](clip_model)
            else:
                # minimal and brave don't need clip_model parameter
                lumen_config = preset_info["config_method"]()

        # 更新 DataBinding
        if self.data_binding:
            self.data_binding.set_data("selected_preset", preset_info)
            self.data_binding.set_data("preset_key", self.selected_key)
            self.data_binding.set_data("is_valid", is_valid)
            self.data_binding.set_data("lumen_config", lumen_config)
            self.data_binding.set_data("clip_model_value", self.clip_model_value)

        # 更新 按钮状态
        if self.button_container:
            self.button_container.continue_button.update_disabled(not is_valid)

    # 公共方法（如果外部需要主动获取）
    def get_selected_preset(self):
        if self.selected_key:
            return next(
                (p for p in self.presets_mapping if p["key"] == self.selected_key), None
            )
        return None


if __name__ == "__main__":
    import flet as ft
    from lumen_resources import Region

    class MockDataBinding:
        """Mock DataBinding for testing"""

        def __init__(self):
            self.data = {}

        def set_data(self, key: str, value: Any):
            self.data[key] = value
            logger.debug(f"[DataBinding] {key} = {value}")

    class MockButtonContainer:
        """Mock ButtonContainer for testing"""

        def __init__(self):
            self.continue_button = MockButton()

    class MockButton:
        """Mock Button for testing"""

        def __init__(self):
            self.disabled = False

        def update_disabled(self, disabled: bool):
            self.disabled = disabled
            logger.debug(f"[Button] disabled = {disabled}")

    def main(page: ft.Page):
        page.title = "Presets View Demo"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window.width = 1200
        page.window.height = 800
        page.padding = 30
        page.bgcolor = ft.Colors.GREY_50

        # Create mock dependencies
        data_binding = MockDataBinding()

        # Create device config (NVIDIA GPU high for demo)
        device_config = DeviceConfig.nvidia_gpu_high()

        # Create presets view
        presets_view = PresetsView(
            cache_dir="./cache",
            region=Region.cn,
            device_config=device_config,
            data_binding=data_binding,
        )

        # Add view to page
        page.add(presets_view)

        # Add info banner
        info_banner = ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Select a preset to see the configuration",
                        size=16,
                        color=ft.Colors.GREY_700,
                    ),
                    ft.Text(
                        "Check the console output for DataBinding updates",
                        size=14,
                        color=ft.Colors.GREY_500,
                    ),
                ],
                spacing=5,
            ),
            padding=20,
            bgcolor=ft.Colors.AMBER_50,
            border_radius=8,
            margin=ft.margin.only(bottom=20),
        )
        page.insert(0, info_banner)

        # Add status display
        status_text = ft.Text("No preset selected", size=14, color=ft.Colors.GREY_600)
        page.add(status_text)

        # Update status display when preset changes
        def update_status():
            if presets_view.selected_key:
                preset = presets_view.get_selected_preset()
                if preset:
                    status_text.value = f"Selected: {preset['key'].upper()}"
                    status_text.color = ft.Colors.GREEN_600
                    status_text.update()

        # Hook into the checkbox update by wrapping the original method
        original_on_preset_selected = presets_view._on_preset_selected

        def wrapped_on_preset_selected(key: str, is_selected: bool):
            original_on_preset_selected(key, is_selected)
            update_status()

        presets_view._on_preset_selected = wrapped_on_preset_selected

    ft.app(target=main)
