from typing import Any, Dict, Literal, Optional, TypedDict

import flet as ft
from lumen_resources import Region
from lumen_resources.lumen_config_validator import LumenConfig

from ...core.config import Config, DeviceConfig

# 假设这些是你项目中的依赖，保留引用
from ..components.button_container import ButtonContainer
from ..i18n_manager import t


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
    ):
        super().__init__()
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

        # 组件状态
        self.selected_key: Optional[str] = None
        # 所有CLIP model输入框共用一个value，因为用户仅且仅能选择一个Preset和一个适配的CLIP model.
        self.clip_model_value: Literal[
            "MobileCLIP2-S2", "CN-CLIP_ViT-B-16", "MobileCLIP2-S4", "CN-CLIP_ViT-L-14"
        ]
        self.service_name: str = "lumen-app"
        self.port_value: int = 50051
        # 映射表：Key -> Checkbox控件实例 (用于快速访问和更新)
        self.checkbox_map: Dict[str, ft.Checkbox] = {}

        self.presets_mapping = [
            {
                "key": "minimal",
                "icon": ft.Icons.FILTER_NONE,
                "config_method": self.config.minimal(),
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
                "config_method": self.config.brave(),
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

        # 创建卡片列表
        preset_cards = [
            self._create_preset_card(preset) for preset in self.presets_mapping
        ]

        self.controls = [
            ft.Text(t("views.presets"), size=30),
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
            ),
        ]

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

        # 更新 DataBinding
        if self.data_binding:
            self.data_binding.set_data("selected_preset", preset_info)
            self.data_binding.set_data("preset_key", self.selected_key)
            self.data_binding.set_data("is_valid", is_valid)

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
