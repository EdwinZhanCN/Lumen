from typing import Any, Dict, Optional

import flet as ft

# 假设这些是你项目中的依赖，保留引用
from ..components.button_container import ButtonContainer
from ..i18n_manager import t


class PresetsView(ft.Column):
    # 将静态配置数据移至类属性（也可以放在外部配置文件中）
    PRESETS_DATA = [
        {
            "key": "minimal",
            "icon": ft.Icons.FILTER_NONE,
            "config_method": "minimal",
            "features": ["OCR"],
        },
        {
            "key": "lightweight",
            "icon": ft.Icons.FLIGHT_TAKEOFF,
            "config_method": "light_weight",
            "features": [
                "OCR + CLIP + Face",
                "MobileCLIP2-S2",
                "Optimized Performance",
            ],
        },
        {
            "key": "basic",
            "icon": ft.Icons.APPS,
            "config_method": "basic",
            "features": [
                "OCR + CLIP + Face + VLM",
                "MobileCLIP2-S4",
                "Complete AI Toolkit",
            ],
        },
        {
            "key": "brave",
            "icon": ft.Icons.ROCKET_LAUNCH,
            "config_method": "brave",
            "features": ["Advanced AI Suite", "BioCLIP-2", "Production Grade"],
        },
    ]

    def __init__(
        self,
        button_container: Optional[ButtonContainer] = None,
        data_binding: Optional[Any] = None,
    ):
        super().__init__()
        self.button_container = button_container
        self.data_binding = data_binding

        # 组件状态
        self.selected_key: Optional[str] = None
        # 映射表：Key -> Checkbox控件实例 (用于快速访问和更新)
        self.checkbox_map: Dict[str, ft.Checkbox] = {}

        # 初始化 UI
        self._setup_ui()

    def _setup_ui(self):
        """初始化视图布局"""
        self.spacing = 20
        self.scroll = ft.ScrollMode.AUTO

        # 创建卡片列表
        preset_cards = [
            self._create_preset_card(preset) for preset in self.PRESETS_DATA
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
            (p for p in self.PRESETS_DATA if p["key"] == self.selected_key), None
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
                (p for p in self.PRESETS_DATA if p["key"] == self.selected_key), None
            )
        return None
