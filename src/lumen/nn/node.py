"""MLNode interface for LumNN pipeline execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import MLContext
    from .packet import MLPacket


class MLNode(ABC):
    """Abstract computation node used by :class:`lumen.nn.MLPipeline`.

    Implementations should keep ``initialize()`` idempotent and accept only
    ``MLPacket`` values in ``run()``. Node authors are free to perform heavy
    setup such as model loading in ``initialize()``.
    """

    name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    device_hint: str | None = None

    @abstractmethod
    def initialize(self, ctx: MLContext) -> None:
        """Prepare runtime resources for this node."""

    @abstractmethod
    def run(
        self, ctx: MLContext, inputs: dict[str, MLPacket]
    ) -> dict[str, MLPacket]:
        """Execute this node and return packets keyed by output name."""
