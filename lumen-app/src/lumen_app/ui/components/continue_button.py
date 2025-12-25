"""
Continue Button Component

只负责样式的继续按钮组件，不包含业务逻辑
"""

import flet as ft

from ..i18n_manager import t


class ContinueButton:
    """只负责样式的继续按钮组件"""

    def __init__(self, text: str = None, disabled: bool = False):
        """
        初始化继续按钮

        Args:
            text: 按钮文本，如果为None则使用默认文本
            disabled: 是否禁用按钮
        """
        self.text = text or t("action_buttons.continue")
        self.disabled = disabled
        self.button = self._create_button()

    def _create_button(self):
        """创建按钮控件"""
        return ft.ElevatedButton(
            self.text,
            icon=ft.Icons.ARROW_FORWARD,
            disabled=self.disabled,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.PRIMARY,
                color=ft.Colors.WHITE,
                padding=ft.padding.symmetric(horizontal=24, vertical=12),
                shape=ft.RoundedRectangleBorder(radius=8),
                elevation=3,
            ),
        )

    def update_text(self, text: str):
        """
        更新按钮文本

        Args:
            text: 新的按钮文本
        """
        self.text = text
        self.button.text = text
        # 尝试更新按钮显示
        try:
            self.button.update()
        except:
            pass

    def update_disabled(self, disabled: bool):
        """
        更新按钮禁用状态

        Args:
            disabled: 是否禁用按钮
        """
        self.disabled = disabled
        self.button.disabled = disabled
        # 尝试更新按钮显示
        try:
            self.button.update()
        except:
            pass

    def get_button(self):
        """获取按钮控件"""
        return self.button
