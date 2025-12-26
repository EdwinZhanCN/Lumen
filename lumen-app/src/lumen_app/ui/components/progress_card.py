"""ProgressCard component for step-by-step progress tracking.

This module provides a reusable progress card with animated progress bar,
step indicators, and status tracking.
"""

from enum import Enum
from typing import List

import flet as ft


class StepStatus(Enum):
    """Status of a step in the progress card."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class ProgressCard(ft.Container):
    """A reusable progress tracking component.

    Features:
    - Step-by-step progress display
    - Animated progress bar
    - Success/failure indicators
    - Status icons per step
    - Estimated time tracking

    Attributes:
        steps (List[str]): List of step names
        current_step (int): Index of currently executing step
    """

    # Icon mappings for step statuses
    STATUS_ICONS = {
        StepStatus.PENDING: ft.Icons.RADIO_BUTTON_UNCHECKED,
        StepStatus.IN_PROGRESS: ft.Icons.REFRESH,
        StepStatus.COMPLETE: ft.Icons.CHECK_CIRCLE,
        StepStatus.FAILED: ft.Icons.CANCEL,
    }

    # Color mappings for step statuses
    STATUS_COLORS = {
        StepStatus.PENDING: ft.Colors.GREY_400,
        StepStatus.IN_PROGRESS: ft.Colors.BLUE,
        StepStatus.COMPLETE: ft.Colors.GREEN,
        StepStatus.FAILED: ft.Colors.RED,
    }

    def __init__(self, steps: List[str], title: str = "Installation Progress"):
        super().__init__()
        self.steps = steps
        self.title = title
        self.current_step = 0
        self.step_statuses: dict[int, StepStatus] = {
            i: StepStatus.PENDING for i in range(len(steps))
        }

        # UI Components
        self.progress_bar = ft.ProgressBar(
            width=400,
            height=8,
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREY_400),
            color=ft.Colors.PRIMARY,
        )

        self.progress_text = ft.Text(
            "0%",
            size=14,
            weight=ft.FontWeight.BOLD,
        )

        self.step_indicators: dict[int, ft.Icon] = {}
        self.step_labels: dict[int, ft.Text] = {}

        # Initialize layout
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the view layout."""
        self.padding = 20
        self.border_radius = 8
        self.bgcolor = ft.Colors.with_opacity(0.03, ft.Colors.GREY_100)
        self.border = ft.border.all(1, ft.Colors.with_opacity(0.1, ft.Colors.GREY_300))

        # Create step indicators
        step_rows = []
        for i, step_name in enumerate(self.steps):
            icon = ft.Icon(
                self.STATUS_ICONS[StepStatus.PENDING],
                size=20,
                color=self.STATUS_COLORS[StepStatus.PENDING],
            )

            label = ft.Text(
                step_name,
                size=13,
                color=ft.Colors.GREY_700,
            )

            self.step_indicators[i] = icon
            self.step_labels[i] = label

            step_row = ft.Row(
                [icon, label],
                spacing=10,
            )
            step_rows.append(step_row)

        # Create layout
        self.content = ft.Column(
            [
                # Header with title and progress
                ft.Row(
                    [
                        ft.Icon(ft.Icons.CHECKLIST, size=24, color=ft.Colors.PRIMARY),
                        ft.Text(
                            self.title,
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            expand=True,
                        ),
                        self.progress_text,
                    ],
                    spacing=10,
                ),
                ft.Divider(height=20, color=ft.Colors.GREY_300),
                # Progress bar
                ft.Column(
                    [
                        self.progress_bar,
                    ],
                    spacing=5,
                ),
                ft.Divider(height=20, color=ft.Colors.GREY_300),
                # Step indicators
                ft.Column(step_rows, spacing=12),
            ],
            spacing=10,
        )

    def update_step(self, step_index: int, status: StepStatus):
        """Update the status of a specific step.

        Args:
            step_index: Index of the step to update
            status: New status for the step
        """
        if step_index < 0 or step_index >= len(self.steps):
            return

        self.step_statuses[step_index] = status

        # Update icon and color
        icon = self.step_indicators.get(step_index)
        label = self.step_labels.get(step_index)

        if icon:
            if hasattr(icon, "icon"):
                icon.icon = self.STATUS_ICONS[status]  # type: ignore
            if hasattr(icon, "color"):
                icon.color = self.STATUS_COLORS[status]

        if label:
            if status == StepStatus.COMPLETE:
                label.color = ft.Colors.GREEN_700
                label.weight = ft.FontWeight.W_500
            elif status == StepStatus.FAILED:
                label.color = ft.Colors.RED_700
                label.weight = ft.FontWeight.W_500
            elif status == StepStatus.IN_PROGRESS:
                label.color = ft.Colors.PRIMARY
                label.weight = ft.FontWeight.W_500
            else:
                label.color = ft.Colors.GREY_700
                label.weight = ft.FontWeight.NORMAL

        # Update current step
        if status == StepStatus.IN_PROGRESS:
            self.current_step = step_index

        # Update progress bar and percentage
        self._update_progress()

        # Schedule UI update
        try:
            if icon:
                icon.update()
            if label:
                label.update()
            self.progress_bar.update()
            self.progress_text.update()
        except Exception:
            pass

    def _update_progress(self):
        """Update the overall progress bar and percentage text."""
        # Calculate progress based on completed steps
        completed = sum(
            1 for status in self.step_statuses.values() if status == StepStatus.COMPLETE
        )

        total = len(self.steps)
        progress_value = completed / total if total > 0 else 0

        # Update progress bar
        try:
            self.progress_bar.value = progress_value
        except AttributeError:
            # Flet ProgressBar might not support direct value assignment
            # In that case, we'll update the width
            self.progress_bar.width = int(400 * progress_value)

        # Update percentage text
        self.progress_text.value = f"{int(progress_value * 100)}%"

    def set_progress(self, value: float):
        """Set the overall progress bar value.

        Args:
            value: Progress value between 0.0 and 1.0
        """
        value = max(0.0, min(1.0, value))  # Clamp to [0, 1]

        try:
            self.progress_bar.value = value
        except AttributeError:
            self.progress_bar.width = int(400 * value)

        self.progress_text.value = f"{int(value * 100)}%"

        try:
            self.progress_bar.update()
            self.progress_text.update()
        except Exception:
            pass

    def reset(self):
        """Reset all steps to pending status."""
        for i in range(len(self.steps)):
            self.update_step(i, StepStatus.PENDING)
        self.current_step = 0

    def get_current_step(self) -> int:
        """Get the index of the currently executing step.

        Returns:
            Current step index
        """
        return self.current_step

    def is_complete(self) -> bool:
        """Check if all steps are complete.

        Returns:
            True if all steps are complete, False otherwise
        """
        return all(
            status == StepStatus.COMPLETE for status in self.step_statuses.values()
        )

    def has_failed(self) -> bool:
        """Check if any step has failed.

        Returns:
            True if any step has failed, False otherwise
        """
        return any(
            status == StepStatus.FAILED for status in self.step_statuses.values()
        )


# --- Demo Usage ---
if __name__ == "__main__":
    import time

    def main(page: ft.Page):
        page.title = "ProgressCard Demo"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window.width = 600
        page.window.height = 500
        page.padding = 30
        page.bgcolor = ft.Colors.GREY_50

        steps = [
            "Installing micromamba",
            "Setting up Python environment",
            "Installing drivers",
            "Installing packages",
            "Verifying installation",
        ]

        progress_card = ProgressCard(steps, title="Installation Progress")

        page.add(progress_card)
        page.add(
            ft.ElevatedButton(
                "Run Demo",
                on_click=lambda e: run_demo(progress_card),
            )
        )

    def run_demo(card: ProgressCard):
        """Run a demo of the progress card."""
        import threading

        def demo_thread():
            steps_count = len(card.steps)

            for i in range(steps_count):
                # Mark as in progress
                card.update_step(i, StepStatus.IN_PROGRESS)
                time.sleep(1)

                # Mark as complete (or fail last step randomly)
                if i == steps_count - 1 and False:  # Set to True to test failure
                    card.update_step(i, StepStatus.FAILED)
                    break
                else:
                    card.update_step(i, StepStatus.COMPLETE)

        thread = threading.Thread(target=demo_thread, daemon=True)
        thread.start()

    ft.app(target=main)
