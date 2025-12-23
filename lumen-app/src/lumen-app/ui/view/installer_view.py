from typing import Optional

import flet as ft

from ..components.button_container import ButtonContainer
from ..i18n_manager import t


def create_installer_view(
    button_container: Optional[ButtonContainer] = None,
    data_binding: Optional[object] = None,
):
    """
    Create installer view placeholder

    Args:
        button_container: Optional button container provided by RunnerView
        data_binding: Optional data binding for communication with RunnerView

    Returns:
        ft.Column: Installer view placeholder
    """

    # Create view components with proper typing
    view_components: list[ft.Control] = [
        ft.Text(t("views.installer"), size=30),
        ft.Text("This view is currently under development.", size=16),
        ft.Text(
            "It will provide module management and installation features.", size=16
        ),
    ]

    # Add button container if provided
    if button_container:
        view_components.append(button_container.get_container())

    # Add placeholder content
    placeholder_container = ft.Container(
        content=ft.Column(
            [
                ft.Text("Available Modules", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("● Lumen-VLM - Vision Language Model", size=16),
                ft.Text("● Lumen-Face - Face Recognition", size=16),
                ft.Text("● Lumen-TTS - Text to Speech", size=16),
                ft.Text("● Lumen-ASR - Automatic Speech Recognition", size=16),
            ],
            spacing=10,
        ),
        padding=20,
        border_radius=10,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY),
    )

    view_components.append(placeholder_container)

    return ft.Column(view_components, spacing=20)
