import flet as ft

from ..i18n_manager import t


def create_monitor_view():
    """Create monitor view with system resource display"""
    return ft.Column(
        [
            ft.Text(t("views.monitor"), size=30),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("System Resources", size=20, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Row(
                            [
                                ft.Text("CPU:"),
                                ft.ProgressBar(value=0.7, width=200),
                                ft.Text("70%"),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        ft.Row(
                            [
                                ft.Text("Memory:"),
                                ft.ProgressBar(value=0.5, width=200),
                                ft.Text("50%"),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        ft.Row(
                            [
                                ft.Text("GPU:"),
                                ft.ProgressBar(value=0.3, width=200),
                                ft.Text("30%"),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        ft.Divider(),
                        ft.ElevatedButton("Refresh", icon=ft.Icons.REFRESH),
                    ],
                    spacing=10,
                ),
                padding=20,
                border_radius=10,
                bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY),
            ),
        ],
        spacing=20,
    )
