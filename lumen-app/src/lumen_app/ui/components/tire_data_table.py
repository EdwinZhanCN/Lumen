from dataclasses import dataclass
from typing import List

import flet as ft


# --- 1. 数据模型层 (Data Model) ---
@dataclass
class HardwareInfo:
    tier: str
    igpu: str
    cpu_example: str
    execution_provider: str


# --- 2. 组件层 (Reusable Component) ---
class TierDataTable(ft.Container):
    """
    一个专门展示硬件层级信息的表格组件
    支持自动隐藏重复的 Tier 标签，模拟 RowSpan 效果
    """

    def __init__(self, data: List[HardwareInfo]):
        super().__init__()
        self.data = data

        # 样式配置
        self.padding = 10
        self.border = ft.border.all(1, ft.Colors.OUTLINE_VARIANT)
        self.border_radius = 10
        self.bgcolor = ft.Colors.SURFACE
        self.content = self._build_table()

    def _build_table(self):
        return ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Tier", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("iGPU Model")),
                ft.DataColumn(ft.Text("CPU/SoC Example")),
                ft.DataColumn(ft.Text("ONNXRuntime Execution Provider")),
            ],
            rows=self._generate_rows(),
            # 样式微调
            heading_row_color=ft.Colors.ON_SURFACE_VARIANT,
            heading_row_height=60,
            data_row_min_height=50,
            vertical_lines=ft.border.BorderSide(0.5, ft.Colors.OUTLINE_VARIANT),
            horizontal_lines=ft.border.BorderSide(0.5, ft.Colors.OUTLINE_VARIANT),
            column_spacing=30,
        )

    def _generate_rows(self) -> List[ft.DataRow]:
        rows = []
        last_tier = None
        assert self.data is not None

        for item in self.data:
            # --- 核心逻辑：处理行合并视觉效果 ---
            # 如果当前行的 Tier 与上一行相同，则显示为空字符串
            display_tier = item.tier if item.tier != last_tier else ""

            # 如果是新的 Tier (display_tier 不为空)，加粗显示
            tier_cell_content = (
                ft.Text(display_tier, weight=ft.FontWeight.BOLD, size=16)
                if display_tier
                else ft.Text("")
            )

            # 针对特定列的样式处理 (例如加粗显卡型号)
            gpu_cell_content = ft.Text(item.igpu, weight=ft.FontWeight.BOLD)

            # 针对 EP 列的特定样式 (高亮 OpenVINO 等关键字)
            ep_content = self._format_ep_cell(item.execution_provider)

            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(tier_cell_content),
                        ft.DataCell(gpu_cell_content),
                        ft.DataCell(ft.Text(item.cpu_example, size=13)),
                        ft.DataCell(ep_content),
                    ],
                )
            )
            last_tier = item.tier
        return rows

    def _format_ep_cell(self, text: str) -> ft.Control:
        """辅助方法：简单的关键字高亮"""
        color = ft.Colors.ON_SURFACE
        weight = ft.FontWeight.NORMAL

        if "OpenVINO" in text:
            color = ft.Colors.BLUE
            weight = ft.FontWeight.BOLD
        elif "CoreML" in text:
            color = ft.Colors.PURPLE
            weight = ft.FontWeight.BOLD
        elif "MIGraphX" in text:
            color = ft.Colors.RED
            weight = ft.FontWeight.BOLD

        return ft.Text(text, color=color, weight=weight)


# --- 3. 业务逻辑与数据注入 (Usage) ---
def main(page: ft.Page):
    page.title = "硬件分级表"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30

    # 准备数据 (通常来自数据库或 API)
    raw_data = [
        # Tier 1
        HardwareInfo(
            "Tier 1", "AMD Radeon 890/880M", "Ryzen AI 9 HX 370 / 365", "MIGraphX(2025)"
        ),
        HardwareInfo(
            "Tier 1",
            "Apple M Pro/Max/Ultra",
            "M4 Pro, M3 Max, M2 Ultra",
            "CoreML",
        ),
        HardwareInfo(
            "Tier 1",
            "Arc Graphics 140V/130V",
            "Core Ultra 7 258V, Ultra 5 226V",
            "OpenVINO",
        ),
        # Tier 2
        HardwareInfo(
            "Tier 2",
            "AMD Radeon 780/760M",
            "Ryzen 7 8845HS, 8700G, 7840U",
            "MIGraphX(2025)",
        ),
        HardwareInfo("Tier 2", "Apple M", "Apple M4 (MacBook), M3, M2", "CoreML"),
        HardwareInfo(
            "Tier 2",
            "Arc Graphics (8/7 Xe)",
            "Core Ultra 7 155H, Ultra 5 125H",
            "OpenVINO",
        ),
        # Tier 3
        HardwareInfo(
            "Tier 3", "Intel Graphics (4 Xe)", "Core Ultra 9 285K (Desktop)", "OpenVINO"
        ),
        HardwareInfo(
            "Tier 3", "Iris Xe Graphics", "Core i7-1360P, i5-1240P", "OpenVINO"
        ),
        # Tier 4
        HardwareInfo(
            "Tier 4", "UHD Graphics 770/750/730", "Core i9-14900K, i5-13400", "OpenVINO"
        ),
        HardwareInfo("Tier 4", "UHD Graphics", "Intel N100, N200, N150", "OpenVINO"),
    ]

    # 实例化组件
    hardware_table = TierDataTable(data=raw_data)

    # 布局
    page.add(
        ft.Text("Flet 进阶组件示例：硬件分级表", size=24, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        # 为了支持水平滚动（防止表格在小屏幕溢出），外部包裹 Row + Scroll
        ft.Row(controls=[hardware_table], scroll=ft.ScrollMode.ADAPTIVE),
    )


if __name__ == "__main__":
    ft.app(target=main)
