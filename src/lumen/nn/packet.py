"""MLPacket data container for the LumNN orchestration layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .exceptions import InvalidMLPacketError


def _freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a read-only mapping for packet metadata."""
    return cast(Mapping[str, Any], MappingProxyType(dict(metadata or {})))


@dataclass(frozen=True, slots=True)
class MLPacket:
    """Immutable data carrier passed between LumNN nodes.

    Args:
        name: Stable packet identifier inside a pipeline.
        value: Optional materialized value for this packet.
        dtype: Optional logical data type. Inferred when omitted for common inputs.
        shape: Optional logical shape. Inferred for NumPy arrays and byte/text payloads.
        device: Logical device location such as ``host`` or ``cuda:0``.
        mime: Optional MIME descriptor for serialized payloads.
        metadata: Additional read-only packet metadata.
    """

    name: str
    value: Any = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    device: str = "host"
    mime: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _pipeline_id: int | None = field(default=None, repr=False, compare=False)
    _source_kind: str = field(default="runtime", repr=False, compare=False)
    _source_node: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize inferred packet metadata and validate invariants."""
        if not self.name:
            raise InvalidMLPacketError("Packet name must be a non-empty string.")
        if not self.device:
            raise InvalidMLPacketError("Packet device must be a non-empty string.")

        inferred_dtype, inferred_shape = self._infer_from_value(self.value)

        object.__setattr__(self, "dtype", self.dtype or inferred_dtype)
        object.__setattr__(self, "shape", self.shape or inferred_shape)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @staticmethod
    def _infer_from_value(value: Any) -> tuple[str | None, tuple[int, ...] | None]:
        """Infer dtype and shape from a materialized packet value."""
        if isinstance(value, np.ndarray):
            return str(value.dtype), tuple(int(dim) for dim in value.shape)
        if isinstance(value, bytes):
            return "bytes", (len(value),)
        if isinstance(value, str):
            return "string", (len(value),)
        if value is None:
            return None, None
        return type(value).__name__, None
