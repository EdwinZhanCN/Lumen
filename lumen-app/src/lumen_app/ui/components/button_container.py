"""
Button Container Component

按钮容器，负责布局和组合按钮
"""

import flet as ft

from .continue_button import ContinueButton
from .reset_button import ResetButton


class ButtonContainer:
    """按钮容器，负责布局"""

    def __init__(self, continue_button: ContinueButton, reset_button: ResetButton):
        """
        初始化按钮容器

        Args:
            continue_button: 继续按钮实例
            reset_button: 重置按钮实例
        """
        self.continue_button = continue_button
        self.reset_button = reset_button
        self.container = self._create_container()

    def _create_container(self):
        """创建容器控件"""
        return ft.Container(
            content=ft.Row(
                controls=[
                    self.continue_button.get_button(),
                    self.reset_button.get_button(),
                ],
                alignment=ft.MainAxisAlignment.END,
                spacing=15,
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
            margin=ft.margin.only(bottom=10),
        )

    def get_container(self):
        """获取容器控件"""
        return self.container

    def update_container(self):
        """更新容器显示"""
        try:
            self.container.update()
        except Exception:
            pass

    def disable_all(self, disabled: bool = True):
        """禁用或启用所有按钮"""
        self.continue_button.update_disabled(disabled)
        self.reset_button.update_disabled(disabled)

    def set_continue_text(self, text: str):
        """设置继续按钮文本"""
        self.continue_button.update_text(text)

    def set_reset_text(self, text: str):
        """设置重置按钮文本"""
        self.reset_button.update_text(text)
