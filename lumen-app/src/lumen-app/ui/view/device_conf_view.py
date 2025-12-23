"""
Enhanced Device Configuration View

重构后的设备配置视图，移除了按钮逻辑，使用数据绑定机制
"""

from typing import Any, Dict, Optional

import flet as ft

from ...core.config import DeviceConfig
from ..components.button_container import ButtonContainer
from ..i18n_manager import t

# 基于 DeviceConfig 类创建的完整预设映射
device_presets: Dict[str, DeviceConfig | None] = {
    "Apple Silicon (M1/M2/M3)": DeviceConfig.apple_silicon(),
    "Nvidia GPU (RAM < 12GB)": DeviceConfig.nvidia_gpu(),
    "Nvidia GPU (RAM >= 12GB)": DeviceConfig.nvidia_gpu_high(),
    "Intel GPU (iGPU/Arc)": DeviceConfig.intel_gpu(),
    "AMD GPU (New Radeon)": DeviceConfig.amd_gpu(),
    "Nvidia Jetson (RAM < 12GB)": DeviceConfig.nvidia_jetson(),
    "Nvidia Jetson (RAM >= 12GB)": DeviceConfig.nvidia_jetson_high(),
    "General CPU": DeviceConfig.cpu(),
    "Rockchip NPU": None,  # 用户需要自定义设备名称
}


def get_device_preset(
    preset_name: str, rockchip_device: str = "rk3588"
) -> DeviceConfig:
    """根据预设名称获取设备配置

    Args:
        preset_name: 预设名称
        rockchip_device: Rockchip 设备名称（仅当选择 Rockchip NPU 时使用）

    Returns:
        DeviceConfig: 对应的设备配置，如果未找到则返回默认CPU配置
    """
    if preset_name == "Rockchip NPU (自定义设备)":
        return DeviceConfig.rockchip(rockchip_device)

    # 只检查非 None 的预设
    if preset_name in device_presets:
        config = device_presets[preset_name]
        if config is not None:
            return config

    return DeviceConfig.cpu()


def get_all_presets() -> Dict[str, DeviceConfig | None]:
    """获取所有可用的设备预设

    Returns:
        Dict[str, DeviceConfig]: 所有预设的字典
    """
    return device_presets.copy()


def get_preset_list() -> list[str]:
    """获取所有预设名称列表

    Returns:
        list[str]: 预设名称列表
    """
    return list(device_presets.keys())


class DeviceConfViewData:
    """设备配置视图数据类"""

    def __init__(self):
        self.selected_preset: Optional[str] = "Nvidia GPU (RAM < 12GB)"
        self.rockchip_device: Optional[str] = "rk3588"
        self.config: Optional[DeviceConfig] = None
        self.is_valid: bool = True


def create_device_conf_view(
    button_container: Optional[ButtonContainer] = None,
    data_binding: Optional[Any] = None,
) -> ft.Column:
    """
    创建设备配置视图

    Args:
        button_container: 按钮容器，由 RunnerView 提供
        data_binding: 数据绑定对象，用于与 RunnerView 通信

    Returns:
        ft.Column: 设备配置视图
    """

    # 创建视图数据
    view_data = DeviceConfViewData()

    # 创建设备预设下拉菜单
    device_dropdown = ft.Dropdown(
        label=t("device_conf.device_preset"),
        options=[ft.dropdown.Option(name) for name in device_presets.keys()],
        value=view_data.selected_preset,
        width=400,
        expand=True,
    )

    # Rockchip NPU 设备名称输入框（初始隐藏）
    rockchip_device_field = ft.TextField(
        label=t("device_conf.rockchip_device_name"),
        hint_text=t("device_conf.rockchip_device_hint"),
        value=view_data.rockchip_device,
        visible=False,
        width=300,
    )

    # 当前选择的设备配置显示
    current_config_display = ft.Column([], spacing=5)

    # 错误信息显示（可选）
    error_text = ft.Text("", color=ft.Colors.RED, visible=False)

    def update_config_display():
        """更新配置显示"""
        selected_preset = device_dropdown.value

        # 处理 Rockchip NPU 的特殊情况
        if selected_preset == "Rockchip NPU (自定义设备)":
            rockchip_device_field.visible = True
            device_name = rockchip_device_field.value or "rk3588"
            config = DeviceConfig.rockchip(device_name)
        else:
            rockchip_device_field.visible = False
            # 使用 if 检查而不是 get 方法来避免类型问题
            if selected_preset in device_presets:
                config = device_presets[selected_preset]
            else:
                config = None

        # 更新视图数据
        view_data.selected_preset = selected_preset
        view_data.rockchip_device = rockchip_device_field.value
        view_data.config = config
        view_data.is_valid = config is not None

        # 更新数据绑定
        if data_binding:
            data_binding.set_data("selected_preset", selected_preset)
            data_binding.set_data("rockchip_device", rockchip_device_field.value)
            data_binding.set_data("config", config)
            data_binding.set_data("is_valid", view_data.is_valid)

        # 更新按钮状态
        if button_container:
            button_container.disable_all(not view_data.is_valid)

        if config:
            # 更新配置显示
            current_config_display.controls = [
                ft.Text(
                    t("device_conf.current_config") + ":",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Text(
                    f"• {t('device_conf.runtime')}: {config.runtime.value if config.runtime else 'N/A'}"
                ),
                ft.Text(
                    f"• {t('device_conf.onnx_providers')}: {', '.join(config.onnx_providers or ['N/A'])}"
                ),
                ft.Text(
                    f"• {t('device_conf.rknn_device')}: {config.rknn_device or 'N/A'}"
                ),
                ft.Text(
                    f"• {t('device_conf.batch_size')}: {config.batch_size or 'N/A'}"
                ),
                ft.Text(f"• {t('device_conf.description')}: {config.description}"),
            ]
            error_text.visible = False
        else:
            current_config_display.controls = [
                ft.Text(
                    t("device_conf.current_config") + ":",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Text("N/A", color=ft.Colors.GREY_600),
            ]
            error_text.value = "Invalid device configuration"
            error_text.visible = True

        # 更新界面
        try:
            if current_config_display.page:
                current_config_display.page.update()
            if rockchip_device_field.page:
                rockchip_device_field.page.update()
            if error_text.page:
                error_text.page.update()
        except Exception:
            pass

    def on_preset_changed(e):
        """当预设改变时更新显示"""
        update_config_display()

    def on_rockchip_device_changed(e):
        """当 Rockchip 设备名称改变时更新显示"""
        if device_dropdown.value == "Rockchip NPU (自定义设备)":
            update_config_display()

    # 绑定事件
    device_dropdown.on_change = on_preset_changed
    rockchip_device_field.on_change = on_rockchip_device_changed

    # 初始化配置显示
    update_config_display()

    # 构建视图
    # Create view controls with proper typing
    view_controls: list[ft.Control] = [
        ft.Text(t("views.device_conf"), size=30),
    ]

    # 如果有按钮容器，添加到顶部
    if button_container:
        view_controls.append(button_container.get_container())

    # 创建主要内容容器
    main_content = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    t("device_conf.device_preset"),
                    size=20,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Divider(),
                # 设备预设选择
                device_dropdown,
                # Rockchip NPU 设备名称输入（条件显示）
                rockchip_device_field,
                ft.Divider(),
                # 错误信息
                error_text,
                # 当前配置显示
                ft.Container(
                    content=current_config_display,
                    padding=15,
                    bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLUE_GREY),
                    border_radius=8,
                ),
            ],
            spacing=15,
        ),
        padding=20,
        border_radius=10,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY),
    )

    # 添加主要内容
    view_controls.append(main_content)

    return ft.Column(view_controls, spacing=20)
