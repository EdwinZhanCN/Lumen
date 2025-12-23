from typing import Callable, Optional

import flet as ft

from ..i18n_manager import t


def create_welcome_view(
    lumilio_handler: Optional[Callable] = None,
    advanced_handler: Optional[Callable] = None,
):
    """Create welcome view with introduction message

    Args:
        lumilio_handler: Handler function for Lumilio Photos button
        advanced_handler: Handler function for Advanced Mode button
    """
    return ft.Column(
        [
            ft.Text(t("views.welcome"), size=30),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text(t("welcome.message"), size=16),
                        ft.Divider(),
                        ft.Column(
                            [
                                ft.TextField(
                                    label="Cache Directory",
                                    hint_text="Location for AI models and config files",
                                    value="~/.lumen",
                                ),
                                ft.ElevatedButton(
                                    "For Lumilio Photos",
                                    icon=ft.Icons.ARROW_FORWARD,
                                    style=ft.ButtonStyle(
                                        bgcolor=ft.Colors.PRIMARY,
                                        color=ft.Colors.WHITE,
                                    ),
                                    on_click=lumilio_handler or (lambda e: None),
                                ),
                                ft.ElevatedButton(
                                    "Advanced Mode",
                                    icon=ft.Icons.ARROW_FORWARD,
                                    style=ft.ButtonStyle(
                                        bgcolor=ft.Colors.SECONDARY,
                                        color=ft.Colors.WHITE,
                                    ),
                                    on_click=advanced_handler or (lambda e: None),
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
        ],
        scroll=ft.ScrollMode.AUTO,
    )
