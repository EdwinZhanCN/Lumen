from typing import Callable

import flet as ft

from ..i18n_manager import t


class WelcomeView(ft.Column):
    """A welcome view component for the application.

    This class represents the welcome screen of the runner, allowing users to input
    a cache directory and choose between Lumilio Photos mode or Advanced Mode.

    The lumilio-config.yaml file under cache directory will be the source for navigation router to determine if the presets_view should be skipped in lumilio navigation route.

    Attributes:
        lumilio_handler (Callable | None): Handler for Lumilio Photos button click.
        advanced_handler (Callable | None): Handler for Advanced Mode button click.
        cache_dir_input (ft.TextField): Text field for cache directory input.
    """

    def __init__(
        self,
        lumilio_handler: Callable | None = None,
        advanced_handler: Callable | None = None,
    ):
        super().__init__()
        self.lumilio_handler = lumilio_handler
        self.advanced_handler = advanced_handler

        # 1. 将需要交互的控件定义为类属性 (State)
        self.cache_dir_input = ft.TextField(
            label=t("welcome.cache_dir_input_label"),
            hint_text=t("cache_dir_input_hint"),
            value="~/.lumen",
        )

        # 2. 初始化布局
        self.setup_ui()

    def setup_ui(self):
        # 配置 Column 的属性
        self.scroll = ft.ScrollMode.AUTO
        self.controls = [
            ft.Text(t("views.welcome"), size=30),
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(t("welcome.message"), size=16),
                        ft.Divider(),
                        ft.Column(
                            controls=[
                                self.cache_dir_input,  # 使用类属性
                                ft.ElevatedButton(
                                    "For Lumilio Photos",
                                    icon=ft.Icons.ARROW_FORWARD,
                                    style=ft.ButtonStyle(
                                        bgcolor=ft.Colors.PRIMARY,
                                        color=ft.Colors.WHITE,
                                    ),
                                    # 3. 绑定内部方法，而不是直接绑定外部函数
                                    on_click=self._on_lumilio_click,
                                ),
                                ft.ElevatedButton(
                                    "Advanced Mode",
                                    icon=ft.Icons.ARROW_FORWARD,
                                    style=ft.ButtonStyle(
                                        bgcolor=ft.Colors.SECONDARY,
                                        color=ft.Colors.WHITE,
                                    ),
                                    on_click=self._on_advanced_click,
                                ),
                            ],
                            spacing=10,
                        ),
                    ],
                    spacing=10,
                ),
                padding=20,
                border_radius=10,
                bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLACK),
            ),
        ]

    # 4. 内部逻辑处理层：先获取数据，再调用外部回调
    def _on_lumilio_click(self, e):
        # 关键点：这里可以轻松获取输入框的值！
        current_path = self.cache_dir_input.value
        print(f"user input cache_dir: {current_path}")

        if self.lumilio_handler:
            # 你甚至可以将路径作为参数传给外部 handler
            self.lumilio_handler(e, current_path)

    def _on_advanced_click(self, e):
        if self.advanced_handler:
            self.advanced_handler(e)


# --- 使用方法 ---
def main(page: ft.Page):
    def handle_lumilio(e, path):
        page.open(ft.SnackBar(ft.Text(f"正在加载路径: {path}")))
        page.update()

    # 实例化组件
    welcome_view = WelcomeView(lumilio_handler=handle_lumilio)

    page.add(welcome_view)


if __name__ == "__main__":
    ft.app(target=main)
