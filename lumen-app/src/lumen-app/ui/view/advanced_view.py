from typing import Optional

import flet as ft

from ..components.button_container import ButtonContainer
from ..i18n_manager import t


def create_advanced_view(
    button_container: Optional[ButtonContainer] = None,
    data_binding: Optional[object] = None,
):
    """
    Create advanced view placeholder

    Args:
        button_container: Optional button container provided by RunnerView
        data_binding: Optional data binding for communication with RunnerView

    Returns:
        ft.Column: Advanced view placeholder
    """

    # Create view components
    # Create view components with proper typing
    view_components: list[ft.Control] = [
        ft.Text(t("views.advanced"), size=30),
        ft.Text("This view is currently under development.", size=16),
        ft.Text("It will provide advanced configuration options.", size=16),
    ]

    # Add button container if provided
    if button_container:
        view_components.append(button_container.get_container())

    # Add placeholder content
    placeholder_container = ft.Container(
        content=ft.Column(
            [
                ft.Text("Advanced Configuration", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("● Model fine-tuning parameters", size=16),
                ft.Text("● Performance optimization settings", size=16),
                ft.Text("● Custom pipeline configuration", size=16),
                ft.Text("● Debug and diagnostic tools", size=16),
                ft.Text("● API key management", size=16),
            ],
            spacing=10,
        ),
        padding=20,
        border_radius=10,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY),
    )

    view_components.append(placeholder_container)

    return ft.Column(view_components, spacing=20)
