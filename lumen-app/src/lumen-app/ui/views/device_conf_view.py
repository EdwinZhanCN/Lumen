from typing import Any, Dict, List, Optional

import flet as ft

from ...core.config import DeviceConfig
from ..components.button_container import ButtonContainer
from ..i18n_manager import t

# --- 常量定义 ---
# 建议将静态数据移至类外部或作为类属性
DEVICE_PRESETS: Dict[str, Optional[DeviceConfig]] = {
    "Apple Silicon (M1/M2/M3)": DeviceConfig.apple_silicon(),
    "Nvidia GPU (RAM < 12GB)": DeviceConfig.nvidia_gpu(),
    "Nvidia GPU (RAM >= 12GB)": DeviceConfig.nvidia_gpu_high(),
    "Intel GPU (iGPU/Arc)": DeviceConfig.intel_gpu(),
    "AMD GPU (New Radeon)": DeviceConfig.amd_gpu(),
    "Nvidia Jetson (RAM < 12GB)": DeviceConfig.nvidia_jetson(),
    "Nvidia Jetson (RAM >= 12GB)": DeviceConfig.nvidia_jetson_high(),
    "General CPU": DeviceConfig.cpu(),
    "Rockchip NPU (自定义设备)": None,  # 特殊处理
}


class DeviceConfView(ft.Column):
    """
    设备配置视图组件
    负责管理设备选择、Rockchip 自定义输入以及配置信息的展示
    """

    def __init__(
        self,
        button_container: Optional[ButtonContainer] = None,
        data_binding: Optional[Any] = None,
    ):
        super().__init__()
        self.button_container = button_container
        self.data_binding = data_binding

        # --- 内部状态 ---
        self.selected_preset_key: str = "Nvidia GPU (RAM < 12GB)"
        self.rockchip_device_name: str = "rk3588"
        self.current_config: Optional[DeviceConfig] = None

        # --- UI 控件引用 (便于后续 update) ---
        self.device_dropdown: Optional[ft.Dropdown] = None
        self.rockchip_input: Optional[ft.TextField] = None
        self.info_container: Optional[ft.Container] = None
        self.error_text: Optional[ft.Text] = None

        # --- 初始化布局 ---
        self._setup_ui()

    def did_mount(self):
        """组件挂载到页面后，执行一次初始状态更新"""
        self._update_state()

    def _setup_ui(self):
        """构建静态 UI 结构"""
        self.spacing = 20

        # 1. 下拉菜单
        self.device_dropdown = ft.Dropdown(
            label=t("device_conf.device_preset"),
            options=[ft.dropdown.Option(name) for name in DEVICE_PRESETS.keys()],
            value=self.selected_preset_key,
            width=400,
            on_change=self._on_preset_changed,
        )

        # 2. Rockchip 输入框 (默认隐藏)
        self.rockchip_input = ft.TextField(
            label=t("device_conf.rockchip_device_name"),
            hint_text=t("device_conf.rockchip_device_hint"),
            value=self.rockchip_device_name,
            visible=False,
            width=300,
            on_change=self._on_input_changed,
        )

        # 3. 信息展示区域容器 (内容动态生成)
        self.info_container = ft.Container(
            padding=15,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLUE_GREY),
            border_radius=8,
            animate_opacity=300,  # 添加淡入淡出动画
        )

        # 4. 错误提示
        self.error_text = ft.Text("", color=ft.Colors.RED, visible=False)

        # 组装视图
        controls_list: list[ft.Control] = [
            ft.Text(t("views.device_conf"), size=30),
        ]

        # 如果传入了按钮容器，放在顶部
        if self.button_container:
            controls_list.append(self.button_container.get_container())

        # 主内容区域
        main_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        t("device_conf.device_preset"),
                        size=20,
                        weight=ft.FontWeight.BOLD,  # New API
                    ),
                    ft.Divider(),
                    self.device_dropdown,
                    self.rockchip_input,
                    ft.Divider(),
                    self.error_text,
                    self.info_container,
                ],
                spacing=15,
            ),
            padding=20,
            border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY),  # New API
        )

        controls_list.append(main_content)
        self.controls = controls_list

    def _on_preset_changed(self, e):
        assert self.rockchip_input is not None
        """处理预设切换"""
        self.selected_preset_key = e.control.value
        # 切换 Rockchip 输入框的可见性
        is_rockchip = self.selected_preset_key == "Rockchip NPU (自定义设备)"
        self.rockchip_input.visible = is_rockchip
        self.rockchip_input.update()

        self._update_state()

    def _on_input_changed(self, e):
        """处理自定义输入"""
        self.rockchip_device_name = e.control.value
        self._update_state()

    def _update_state(self):
        """核心逻辑：计算配置 -> 更新 UI -> 通知外部"""
        # 1. 计算当前配置
        config = self._calculate_config()
        self.current_config = config
        is_valid = config is not None

        # 2. 更新 UI 显示
        self._update_info_display(config)

        assert self.error_text is not None

        # 更新错误信息
        if is_valid:
            self.error_text.visible = False
        else:
            self.error_text.value = "Invalid device configuration"
            self.error_text.visible = True
        self.error_text.update()

        # 3. 通知外部 (DataBinding)
        if self.data_binding:
            self.data_binding.set_data("selected_preset", self.selected_preset_key)
            self.data_binding.set_data("rockchip_device", self.rockchip_device_name)
            self.data_binding.set_data("config", config)
            self.data_binding.set_data("is_valid", is_valid)

        # 4. 更新按钮状态
        if self.button_container:
            self.button_container.disable_all(not is_valid)

    def _calculate_config(self) -> Optional[DeviceConfig]:
        """根据当前状态生成配置对象"""
        if self.selected_preset_key == "Rockchip NPU (自定义设备)":
            device_name = self.rockchip_device_name or "rk3588"
            return DeviceConfig.rockchip(device_name)

        # 直接从字典获取，若不存在则返回 None
        return DEVICE_PRESETS.get(self.selected_preset_key)

    def _update_info_display(self, config: Optional[DeviceConfig]):
        """渲染信息卡片的内容"""
        if not config:
            content = ft.Column(
                [
                    ft.Text(
                        t("device_conf.current_config") + ":",
                        size=14,
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Text("N/A", color=ft.Colors.GREY_600),
                ]
            )
        else:
            content = ft.Column(
                [
                    ft.Text(
                        t("device_conf.current_config") + ":",
                        size=14,
                        weight=ft.FontWeight.BOLD,
                    ),
                    self._build_info_row(
                        t("device_conf.runtime"),
                        config.runtime.value if config.runtime else "N/A",
                    ),
                    self._build_info_row(
                        t("device_conf.onnx_providers"),
                        ", ".join(config.onnx_providers or ["N/A"]),
                    ),
                    self._build_info_row(
                        t("device_conf.rknn_device"), config.rknn_device or "N/A"
                    ),
                    self._build_info_row(
                        t("device_conf.batch_size"), str(config.batch_size or "N/A")
                    ),
                    self._build_info_row(
                        t("device_conf.description"), config.description
                    ),
                ],
                spacing=5,
            )
        assert self.info_container is not None

        self.info_container.content = content
        self.info_container.update()

    def _build_info_row(self, label: str, value: str) -> ft.Text:
        """辅助方法：生成信息行"""
        # 使用富文本 (Spans) 使 Label 加粗，Value 普通显示
        return ft.Text(
            spans=[
                ft.TextSpan("• ", style=ft.TextStyle(color=ft.Colors.PRIMARY)),
                ft.TextSpan(
                    f"{label}: ", style=ft.TextStyle(weight=ft.FontWeight.W_500)
                ),
                ft.TextSpan(value),
            ]
        )


# --- 辅助函数（如果外部逻辑仍需要用到）---
def get_all_presets() -> Dict[str, DeviceConfig | None]:
    return DEVICE_PRESETS.copy()
