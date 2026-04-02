"""Small internal test nodes used to validate LumNN behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .context import MLContext
from .node import MLNode
from .packet import MLPacket


@dataclass
class PassthroughNode(MLNode):
    """Return the input packet value unchanged under a new output name."""

    name: str = "passthrough"
    input_names: tuple[str, ...] = ("value",)
    output_names: tuple[str, ...] = ("result",)
    device_hint: str | None = None
    initialize_calls: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self, ctx: MLContext) -> None:
        """Initialize this node exactly once."""
        if self._initialized:
            return
        self.initialize_calls += 1
        self._initialized = True

    def run(self, ctx: MLContext, inputs: dict[str, MLPacket]) -> dict[str, MLPacket]:
        """Forward the configured input value."""
        packet = inputs[self.input_names[0]]
        return {
            self.output_names[0]: ctx.create_packet(
                self.output_names[0],
                value=packet.value,
                dtype=packet.dtype,
                shape=packet.shape,
                device=packet.device,
                mime=packet.mime,
                metadata=dict(packet.metadata),
            )
        }


@dataclass
class UppercaseTextNode(MLNode):
    """Uppercase a text packet."""

    name: str = "uppercase_text"
    input_names: tuple[str, ...] = ("text",)
    output_names: tuple[str, ...] = ("upper_text",)
    device_hint: str | None = None
    initialize_calls: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self, ctx: MLContext) -> None:
        """Initialize this node exactly once."""
        if self._initialized:
            return
        self.initialize_calls += 1
        self._initialized = True

    def run(self, ctx: MLContext, inputs: dict[str, MLPacket]) -> dict[str, MLPacket]:
        """Uppercase the configured text input."""
        source = inputs[self.input_names[0]]
        text = str(source.value or "")
        return {
            self.output_names[0]: ctx.create_packet(
                self.output_names[0],
                value=text.upper(),
                dtype="string",
                device=source.device,
                metadata={"transform": "uppercase"},
            )
        }


@dataclass
class JoinTextNode(MLNode):
    """Join two text packets into one output string."""

    name: str = "join_text"
    input_names: tuple[str, ...] = ("left", "right")
    output_names: tuple[str, ...] = ("joined",)
    device_hint: str | None = None
    separator: str = " "
    initialize_calls: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self, ctx: MLContext) -> None:
        """Initialize this node exactly once."""
        if self._initialized:
            return
        self.initialize_calls += 1
        self._initialized = True

    def run(self, ctx: MLContext, inputs: dict[str, MLPacket]) -> dict[str, MLPacket]:
        """Join two text values using the configured separator."""
        left = str(inputs[self.input_names[0]].value or "")
        right = str(inputs[self.input_names[1]].value or "")
        return {
            self.output_names[0]: ctx.create_packet(
                self.output_names[0],
                value=f"{left}{self.separator}{right}",
                dtype="string",
                metadata={"transform": "join"},
            )
        }


@dataclass
class NumpyTensorNode(MLNode):
    """Convert a packet value into a NumPy tensor on the requested device."""

    name: str = "numpy_tensor"
    input_names: tuple[str, ...] = ("value",)
    output_names: tuple[str, ...] = ("tensor",)
    device_hint: str | None = "cuda"
    initialize_calls: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self, ctx: MLContext) -> None:
        """Initialize this node exactly once."""
        if self._initialized:
            return
        self.initialize_calls += 1
        self._initialized = True

    def run(self, ctx: MLContext, inputs: dict[str, MLPacket]) -> dict[str, MLPacket]:
        """Materialize the source value as a NumPy float32 tensor."""
        source = inputs[self.input_names[0]]
        tensor = np.asarray(source.value, dtype=np.float32)
        return {
            self.output_names[0]: ctx.create_packet(
                self.output_names[0],
                value=tensor,
                device=source.device,
                metadata={"kind": "tensor"},
            )
        }


@dataclass
class FailingNode(MLNode):
    """Node that always fails to exercise error propagation."""

    name: str = "failing"
    input_names: tuple[str, ...] = ("value",)
    output_names: tuple[str, ...] = ("result",)
    device_hint: str | None = None
    initialize_calls: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self, ctx: MLContext) -> None:
        """Initialize this node exactly once."""
        if self._initialized:
            return
        self.initialize_calls += 1
        self._initialized = True

    def run(self, ctx: MLContext, inputs: dict[str, MLPacket]) -> dict[str, MLPacket]:
        """Always raise an error."""
        raise RuntimeError("boom")
