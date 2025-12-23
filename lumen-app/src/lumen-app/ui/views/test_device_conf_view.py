from typing import Any, Dict, List, Optional

import flet as ft

# ==========================================
# 1. 模拟依赖项 (Mock Dependencies)
# ==========================================


# Mock 翻译函数
def t(key):
    return key.split(".")[-1].title().replace("_", " ")


# Mock DeviceConfig (为了让组件能读到属性)
class MockRuntime:
    def __init__(self, value):
        self.value = value


class DeviceConfig:
    def __init__(self, description="Mock Config", runtime="ONNX", providers=None):
        self.description = description
        self.runtime = MockRuntime(runtime)
        self.onnx_providers = providers or ["CPUExecutionProvider"]
        self.rknn_device = None
        self.batch_size = 1

    # 模拟工厂方法
    @staticmethod
    def apple_silicon():
        return DeviceConfig(
            "Apple M-Series Chip", "CoreML", ["CoreMLExecutionProvider"]
        )

    @staticmethod
    def nvidia_gpu():
        return DeviceConfig(
            "Nvidia GPU Standard", "TensorRT", ["TensorrtExecutionProvider"]
        )

    @staticmethod
    def nvidia_gpu_high():
        return DeviceConfig(
            "Nvidia GPU High Mem", "TensorRT", ["TensorrtExecutionProvider"]
        )

    @staticmethod
    def intel_gpu():
        return DeviceConfig("Intel Arc/Iris", "OpenVINO", ["OpenVINOExecutionProvider"])

    @staticmethod
    def amd_gpu():
        return DeviceConfig("AMD Radeon", "MIGraphX", ["MIGraphXExecutionProvider"])

    @staticmethod
    def nvidia_jetson():
        return DeviceConfig(
            "Jetson Orin Nano", "TensorRT", ["TensorrtExecutionProvider"]
        )

    @staticmethod
    def nvidia_jetson_high():
        return DeviceConfig(
            "Jetson AGX Orin", "TensorRT", ["TensorrtExecutionProvider"]
        )

    @staticmethod
    def cpu():
        return DeviceConfig("Standard CPU", "ONNX Runtime", ["CPUExecutionProvider"])

    @staticmethod
    def rockchip(name):
        c = DeviceConfig(f"Rockchip {name}", "RKNN", ["RKNNExecutionProvider"])
        c.rknn_device = name
        return c


# Mock 按钮容器
class ButtonContainer:
    def __init__(self):
        self.btn = ft.ElevatedButton("Next Step (Mock)", disabled=False)

    def disable_all(self, disabled: bool):
        self.btn.disabled = disabled
        self.btn.update()

    def get_container(self):
        return ft.Container(content=self.btn, alignment=ft.alignment.center_right)


# ==========================================
# 2. 你的组件代码 (DeviceConfView)
# ==========================================

# 全局预设映射
DEVICE_PRESETS: Dict[str, Optional[DeviceConfig]] = {
    "Apple Silicon (M1/M2/M3)": DeviceConfig.apple_silicon(),
    "Nvidia GPU (RAM < 12GB)": DeviceConfig.nvidia_gpu(),
    "Nvidia GPU (RAM >= 12GB)": DeviceConfig.nvidia_gpu_high(),
    "Intel GPU (iGPU/Arc)": DeviceConfig.intel_gpu(),
    "AMD GPU (New Radeon)": DeviceConfig.amd_gpu(),
    "Nvidia Jetson (RAM < 12GB)": DeviceConfig.nvidia_jetson(),
    "Nvidia Jetson (RAM >= 12GB)": DeviceConfig.nvidia_jetson_high(),
    "General CPU": DeviceConfig.cpu(),
    "Rockchip NPU (自定义设备)": None,
}


class DeviceConfView(ft.Column):
    def __init__(self, button_container=None, data_binding=None):
        super().__init__()
        self.button_container = button_container
        self.data_binding = data_binding
        self.selected_preset_key = "Nvidia GPU (RAM < 12GB)"
        self.rockchip_device_name = "rk3588"
        self._setup_ui()

    def did_mount(self):
        self._update_state()

    def _setup_ui(self):
        self.spacing = 20
        self.device_dropdown = ft.Dropdown(
            label="Device Preset",
            options=[ft.dropdown.Option(name) for name in DEVICE_PRESETS.keys()],
            value=self.selected_preset_key,
            width=400,
            on_change=self._on_preset_changed,
        )
        self.rockchip_input = ft.TextField(
            label="Rockchip Device Name",
            value=self.rockchip_device_name,
            visible=False,
            width=300,
            on_change=self._on_input_changed,
        )
        self.info_container = ft.Container(
            padding=15,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLUE_GREY),
            border_radius=8,
            animate_opacity=300,
        )
        self.error_text = ft.Text("", color=ft.Colors.RED, visible=False)

        controls_list: list[ft.Control] = [
            ft.Text("Device Configuration", size=30, weight=ft.FontWeight.BOLD)
        ]

        if self.button_container:
            controls_list.append(self.button_container.get_container())

        main_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Select Hardware", size=20, weight=ft.FontWeight.BOLD),
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
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY),
        )
        controls_list.append(main_content)
        self.controls = controls_list

    def _on_preset_changed(self, e):
        self.selected_preset_key = e.control.value
        self.rockchip_input.visible = (
            self.selected_preset_key == "Rockchip NPU (自定义设备)"
        )
        self.rockchip_input.update()
        self._update_state()

    def _on_input_changed(self, e):
        self.rockchip_device_name = e.control.value
        self._update_state()

    def _update_state(self):
        config = self._calculate_config()
        is_valid = config is not None
        self._update_info_display(config)

        self.error_text.visible = not is_valid
        if not is_valid:
            self.error_text.value = "Invalid Configuration"
        self.error_text.update()

        if self.data_binding:
            # 调用 mock binding 的 set_data
            self.data_binding.set_data("selected_preset", self.selected_preset_key)
            self.data_binding.set_data(
                "config_desc", config.description if config else "None"
            )
            self.data_binding.set_data("is_valid", is_valid)

        if self.button_container:
            self.button_container.disable_all(not is_valid)

    def _calculate_config(self):
        if self.selected_preset_key == "Rockchip NPU (自定义设备)":
            return DeviceConfig.rockchip(self.rockchip_device_name or "rk3588")
        return DEVICE_PRESETS.get(self.selected_preset_key)

    def _update_info_display(self, config):
        if not config:
            self.info_container.content = ft.Text("No Config Selected")
        else:
            self.info_container.content = ft.Column(
                [
                    ft.Text("Current Config:", weight=ft.FontWeight.BOLD),
                    ft.Text(f"• Runtime: {config.runtime.value}"),
                    ft.Text(f"• Provider: {config.onnx_providers[0]}"),
                    ft.Text(f"• Desc: {config.description}"),
                ]
            )
        self.info_container.update()


# ==========================================
# 3. 测试台 Main 方法
# ==========================================


def main(page: ft.Page):
    page.title = "Component Test Bench: DeviceConfView"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 1000
    page.window.height = 800
    page.padding = 20

    # --- 1. 创建调试面板 (右侧) ---
    debug_log = ft.Column(scroll=ft.ScrollMode.ALWAYS, expand=True)

    class TestBenchBinding:
        """用于接收组件传出的数据并显示在右侧"""

        def set_data(self, key, value):
            timestamp = import_datetime()
            log_entry = ft.Text(
                spans=[
                    ft.TextSpan(
                        f"[{timestamp}] ", style=ft.TextStyle(color=ft.Colors.GREY)
                    ),
                    ft.TextSpan(
                        f"{key}: ",
                        style=ft.TextStyle(
                            weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE
                        ),
                    ),
                    ft.TextSpan(f"{value}"),
                ],
                size=12,
                font_family="Consolas, monospace",
            )
            debug_log.controls.insert(0, log_entry)  # 最新消息在最上
            debug_log.update()

    def import_datetime():
        from datetime import datetime

        return datetime.now().strftime("%H:%M:%S")

    # --- 2. 实例化组件 ---
    binding = TestBenchBinding()
    btn_container = ButtonContainer()

    # 实例化我们要测试的目标组件
    target_view = DeviceConfView(button_container=btn_container, data_binding=binding)

    # --- 3. 布局 ---
    page.add(
        ft.Row(
            controls=[
                # 左侧：你的组件
                ft.Container(
                    content=target_view,
                    expand=3,  # 占 3/5 宽度
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=10,
                ),
                # 垂直分割线
                ft.VerticalDivider(width=1, color=ft.Colors.GREY_300),
                # 右侧：调试日志
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "Data Binding Log", size=20, weight=ft.FontWeight.BOLD
                            ),
                            ft.Divider(),
                            ft.Container(content=debug_log, expand=True),
                        ]
                    ),
                    expand=2,  # 占 2/5 宽度
                    bgcolor=ft.Colors.GREY_50,
                    padding=20,
                    border_radius=10,
                ),
            ],
            expand=True,  # 填满整个页面高度
        )
    )


if __name__ == "__main__":
    ft.app(target=main)
