from typing import Optional

import flet as ft

from ..components.button_container import ButtonContainer
from ..i18n_manager import t


def create_active_runner_view(
    button_container: Optional[ButtonContainer] = None,
    data_binding: Optional[object] = None,
):
    """
    Create active runner view placeholder

    Args:
        button_container: Optional button container provided by RunnerView
        data_binding: Optional data binding for communication with RunnerView

    Returns:
        ft.Column: Active runner view placeholder
    """

    # Create view components with proper typing
    view_components: list[ft.Control] = [
        ft.Text(t("views.active_runner"), size=30),
        ft.Text("This view is currently under development.", size=16),
        ft.Text("It will show the active running state of the application.", size=16),
    ]

    # Add button container if provided
    if button_container:
        view_components.append(button_container.get_container())

    # Add placeholder content
    placeholder_container = ft.Container(
        content=ft.Column(
            [
                ft.Text("Status", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("● Running", color=ft.Colors.GREEN, size=16),
                ft.Text("● Active Model: Lumen-VLM", size=16),
                ft.Text("● Active Preset: Nvidia GPU (RAM < 12GB)", size=16),
                ft.Text("● Memory Usage: 3.2GB / 12GB", size=16),
                ft.Text("● Uptime: 00:15:32", size=16),
            ],
            spacing=10,
        ),
        padding=20,
        border_radius=10,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY),
    )

    view_components.append(placeholder_container)

    return ft.Column(view_components, spacing=20)
